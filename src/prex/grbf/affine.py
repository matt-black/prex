"""Gaussian Radial Basis Functions (GRBF)

Transform a GMM by a global rigid transform (scaling + rotation) and local Gaussian radial basis functions.
"""

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy
import optax
from jax.experimental import io_callback
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float, Int, PyTree

from ..dist import l2_distance_gmm_opt
from ._util import gaussian_rbf


@Partial(jax.jit, static_argnums=(4,))
def jacobians(
    psi: Float[Array, "n_comp n_cent"],
    diffs: Float[Array, "n_comp n_cent d"],
    affine: Float[Array, "d d"],
    rbf_weights: Float[Array, "n_cent 3"],
    bandwidth: float,
) -> Float[Array, "n_comp d d"]:
    # local contribution from each RBF center
    rbf_contribs = (
        jnp.negative(psi)[:, :, jnp.newaxis] * diffs / (bandwidth**2)
    )  # shape (n_comp, n_cent, d)
    jac_local = jnp.sum(
        rbf_weights[jnp.newaxis, :, :, jnp.newaxis]
        * rbf_contribs[:, :, jnp.newaxis, :],
        axis=1,
    )
    return affine[jnp.newaxis, :, :] + jac_local


@Partial(jax.jit, static_argnums=(4,))
def jacobian_at_point(
    x: Float[Array, " d"],
    cents: Float[Array, "n_cent d"],
    rbf_wgts: Float[Array, "n_cent d"],
    affine: Float[Array, "d d"],
    bandwidth: float,
) -> Float[Array, "d d"]:
    """Compute the Jacobian of the forward transformation at point, x.

    Args:
        x (Float[Array, " d"]): Point at which to compute Jacobian, shape (d,)
        centers (Float[Array, "n_cent d"]): RBF centers, shape (n_centers, d)
        rbf_weights (Float[Array, "n_cent d"]): GRBF coefficients, shape (n_centers, d)
        affine (Float[Array, "d d"]): global affine matrix
        bandwidth (float): RBF bandwidth

    Returns:
        Float[Array, "d d"]: Jacobian matrix
    """
    # local GRBF
    diff = x - cents
    sq_dist = jnp.sum(jnp.square(diff), axis=1)
    rbf_vals = jnp.exp(-sq_dist / (2 * bandwidth**2))

    # jacobian contribution from each RBF center
    # d(psi)/dx = -psi(r) * (x - center) / bandwidth^2
    rbf_contribs = (
        jnp.negative(rbf_vals)[:, jnp.newaxis] * diff / jnp.square(bandwidth)
    )
    jac_local = jnp.sum(
        rbf_wgts[:, :, jnp.newaxis] * rbf_contribs[:, jnp.newaxis, :], axis=0
    )
    return affine + jac_local


@Partial(jax.jit, static_argnums=(7,))
def _transform_gmm_precomputed(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_weights: Float[Array, "n_cent d"],
    psi: Float[Array, "n_comp n_cent"],
    diffs: Float[Array, "n_comp n_cent 3"],
    bandwidth: float,
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    # transform means by applying the global rigid transform -> shape (n_components, d)
    means_trans = (
        jnp.transpose(affine @ means.T)
        + translation[jnp.newaxis, :]
        + psi @ rbf_weights
    )
    # transform covariances by pre- and post-multiplication with jacobian
    jac = jacobians(psi, diffs, affine, rbf_weights, bandwidth)
    covs_trans = jnp.matmul(
        jnp.matmul(jac, covariances), jnp.matrix_transpose(jac)
    )
    return means_trans, covs_trans


@Partial(jax.jit, static_argnums=(6,))
def transform_gmm(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_cent d"],
    rbf_centers: Float[Array, "n_cent d"],
    rbf_bandwidth: float,
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    """Apply combined global (scale + rotation + translation) and local (GRBF) transformation to the means and covariances of a GMM.

    Args:
        means: GMM means, shape (n_components, d)
        covariances: GMM covariances (diagonal), shape (n_components, d)
        weights: GMM weights, shape (n_components,)
        affine: affine transformation matrix, shape (d, d)
        translation: Translation vector, shape (d,)
        rbf_wgts: GRBF coefficients, shape (n_centers, d)
        rbf_centers: GRBF centers (fixed at target GMM means), shape (n_centers, d)
        rbf_bandwidth: GRBF bandwidth

    Returns:
        transformed_means: Transformed means, shape (n_components, d)
        transformed_covs: Transformed covariances, shape (n_components, d, d)
    """

    # local deformation is weights time basis matrix
    # compute RBF basis matrix -> shape (n_components, n_centers)
    psi = gaussian_rbf(means, rbf_centers, rbf_bandwidth)
    # (n_components, n_centers) @ (n_centers, n_dim) -> (n_components, n_dims)
    local_deformation = psi @ rbf_wgts

    transformed_means = (
        jnp.transpose(affine @ means.T)
        + translation[jnp.newaxis, :]
        + local_deformation
    )

    # transform covariances using Jacobian (J @ Σ @ J.T)
    jac_at_point = Partial(
        jacobian_at_point,
        cents=rbf_centers,
        rbf_wgts=rbf_wgts,
        affine=affine,
        bandwidth=rbf_bandwidth,
    )

    def transform_single_covariance(
        mu: Float[Array, " d"], cov: Float[Array, "d d"]
    ) -> Float[Array, "d d"]:
        """Transform covariance for a single component."""
        jac = jac_at_point(mu)
        return jac @ cov @ jac.T

    transformed_covs = jax.vmap(transform_single_covariance, (0, 0), 0)(
        means, covariances
    )

    return transformed_means, transformed_covs


@Partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def unpack_params(
    flat_params: Float[Array, " p"], n_cent: int, n_dim: int
) -> tuple[
    Float[Array, "{n_dim} {n_dim}"],
    Float[Array, " {n_dim}"],
    Float[Array, "{n_cent} {n_dim}"],
]:
    affine = flat_params[: n_dim**2].reshape(n_dim, n_dim)
    translation = flat_params[n_dim**2 : n_dim**2 + n_dim]
    rbf_wgts = flat_params[n_dim**2 + n_dim :].reshape(n_cent, n_dim)
    return affine, translation, rbf_wgts


def pack_params(
    affine: Float[Array, "n_dim n_dim"],
    translation: Float[Array, " n_dim"],
    rbf_wgts: Float[Array, "n_cent n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate([affine.ravel(), translation, rbf_wgts.ravel()])


def _create_optimization_function(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    rbf_bandwidth: float,
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], dict[str, Array]]]:

    n_fixed, n_dim = means_fixed.shape
    # scale covariances for this annealing step
    scaled_covs_fixed = covs_fixed * cov_scaling
    scaled_covs_moving = covs_moving * cov_scaling

    # make partial functions for all downstream parameters that are constant over the course of optimization
    dist_fun = Partial(
        l2_distance_gmm_opt, means_fixed, scaled_covs_fixed, wgts_fixed
    )
    unpack = Partial(unpack_params, n_cent=n_fixed, n_dim=n_dim)

    # pre-compute the differences tensor and kernel matrix
    # diffs should have shape (n_comp, n_cent, 3)
    diffs = jax.vmap(
        lambda f: jax.vmap(Partial(jnp.subtract, f), 0, 0)(means_moving), 0, 0
    )(means_fixed)
    # psi is the kernel matrix, should have shape (n_comp, n_cent)
    psi = jnp.exp(
        jnp.divide(
            jnp.negative(jnp.sum(jnp.square(diffs), axis=-1)),
            (2 * rbf_bandwidth**2),
        )
    )

    def loss_l2(
        param_flat: Float[Array, " p"],
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        # unpack parameters
        affine, trans, rbf_wgts = unpack(param_flat)
        # apply transform
        means_trans, cov_trans = _transform_gmm_precomputed(
            means_moving,
            scaled_covs_moving,
            affine,
            trans,
            rbf_wgts,
            psi,
            diffs,
            rbf_bandwidth,
        )
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)
        tikhonov = jnp.linalg.norm(rbf_wgts)
        aux_data = {
            "l2": dist_l2,
            "tikhonov": tikhonov,
            "affine": affine,
            "trans": trans,
            "rbf_wgt": rbf_wgts,
        }
        loss = l2_scaling * dist_l2 + regularization_lambda * tikhonov
        return loss, aux_data

    return loss_l2


@Partial(
    jax.jit,
    static_argnums=(
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ),
)
def optimize_single_scale(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    affine: Float[Array, "d d"],
    trans: Float[Array, " d"],
    rbf_wgts: Float[Array, "f d"],
    rbf_bandwidth: float,
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
    init_pars = pack_params(affine, trans, rbf_wgts)
    loss_func = _create_optimization_function(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        rbf_bandwidth,
        cov_scaling,
        l2_scaling,
        regularization_lambda,
    )

    def loss_func_noaux(pars: Float[Array, " p"]) -> Float[Array, ""]:
        loss_val, _ = loss_func(pars)
        return loss_val

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(init_pars)

    def keep_stepping(
        x: tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            int,
        ],
    ) -> Bool:
        _, _, grad_norm, curr_loss, num_iter = x
        grad_high = grad_norm > grad_tol
        loss_high = curr_loss > loss_tol
        iter_ltmax = num_iter < max_iter
        return jnp.logical_and(
            jnp.logical_and(grad_high, loss_high), iter_ltmax
        )

    if save_path is None:

        def take_step(
            x: tuple[
                Array,
                PyTree,
                Float[Array, ""],
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, num_iter = x
            loss, grads = jax.value_and_grad(loss_func_noaux)(params)
            grad_norm = jnp.linalg.norm(grads)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
                value_fn=loss_func_noaux,
            )
            params: Array = optax.apply_updates(
                params, updates
            )  # pyright: ignore[reportAssignmentType]
            return (params, opt_state, grad_norm, loss, num_iter + 1)

    else:

        def save_step_data(aux_data: dict[str, Array], iter_num: int) -> None:
            numpy.savez(
                os.path.join(save_path, f"{iter_num:05d}.npz"),
                **aux_data,  # pyright: ignore[reportArgumentType]
            )

        def take_step(
            x: tuple[
                Array,
                PyTree,
                Float[Array, ""],
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, num_iter = x
            (loss, aux_data), grads = jax.value_and_grad(
                loss_func, has_aux=True
            )(params)
            io_callback(save_step_data, None, aux_data, num_iter)
            grad_norm = jnp.linalg.norm(grads)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
                value_fn=loss_func_noaux,
            )
            params: Array = optax.apply_updates(
                params, updates
            )  # pyright: ignore[reportAssignmentType]
            return (params, opt_state, grad_norm, loss, num_iter + 1)

    par_f, opt_state, grad_norm, final_loss, num_iter = jax.lax.while_loop(
        keep_stepping,
        take_step,
        (
            init_pars,
            opt_state,
            jnp.array(jnp.inf),
            jnp.array(jnp.inf),
            0,
        ),
    )
    return par_f, (grad_norm, final_loss, num_iter)


@Partial(
    jax.jit,
    static_argnums=(
        9,
        10,
        11,
        12,
        13,
        14,
    ),
)
def optimize_single_scale_fixediter(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    affine: Float[Array, "d d"],
    trans: Float[Array, " d"],
    rbf_wgts: Float[Array, "f d"],
    rbf_bandwidth: float,
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
    num_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], Float[Array, "{num_iter} 2"]]:
    init_pars = pack_params(affine, trans, rbf_wgts)
    loss_func = _create_optimization_function(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        rbf_bandwidth,
        cov_scaling,
        l2_scaling,
        regularization_lambda,
    )

    def loss_func_noaux(pars: Float[Array, " p"]) -> Float[Array, ""]:
        loss_val, _ = loss_func(pars)
        return loss_val

    optimizer = optax.lbfgs()
    opt_state = optimizer.init(init_pars)

    if save_path is None:

        def take_step(
            x: tuple[Array, PyTree],
            iter_num: Int[Array, ""],
        ) -> tuple[tuple[Array, PyTree], Float[Array, " 2"]]:
            params, opt_state = x
            (loss, aux_data), grads = jax.value_and_grad(
                loss_func, has_aux=True
            )(params)
            tikhonov = aux_data["tikhonov"]
            l2_abs = aux_data["l2"]
            grad_norm = jnp.linalg.norm(grads)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
                value_fn=loss_func_noaux,
            )
            params: Array = optax.apply_updates(
                params, updates
            )  # pyright: ignore[reportAssignmentType]
            return (params, opt_state), jnp.concatenate(
                [
                    grad_norm[jnp.newaxis],
                    loss[jnp.newaxis],
                    l2_abs[jnp.newaxis],
                    tikhonov[jnp.newaxis],
                ]
            )

    else:

        def save_step_data(aux_data: dict[str, Array], iter_num: int) -> None:
            numpy.savez(
                os.path.join(save_path, f"{iter_num:05d}.npz"),
                **aux_data,  # pyright: ignore[reportArgumentType]
            )

        def take_step(
            x: tuple[Array, PyTree],
            iter_num: Int[Array, ""],
        ) -> tuple[tuple[Array, PyTree], Float[Array, " 2"]]:
            params, opt_state = x
            (loss, aux_data), grads = jax.value_and_grad(
                loss_func, has_aux=True
            )(params)
            tikhonov = aux_data["tikhonov"]
            l2_abs = aux_data["l2"]
            io_callback(save_step_data, None, aux_data, iter_num)
            grad_norm = jnp.linalg.norm(grads)
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
                value_fn=loss_func_noaux,
            )
            params: Array = optax.apply_updates(
                params, updates
            )  # pyright: ignore[reportAssignmentType]
            return (params, opt_state), jnp.concatenate(
                [
                    grad_norm[jnp.newaxis],
                    loss[jnp.newaxis],
                    l2_abs[jnp.newaxis],
                    tikhonov[jnp.newaxis],
                ]
            )

    (par_f, opt_state), loss_arr = jax.lax.scan(
        take_step,
        (init_pars, opt_state),
        jnp.arange(num_iter),
    )
    return par_f, loss_arr


def optimize_multi_scale(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    affine: Float[Array, "d d"],
    trans: Float[Array, " d"],
    rbf_wgts: Float[Array, "f d"],
    rbf_bandwidth: float,
    cov_scalings: tuple[float, ...],
    l2_scaling: float,
    regularization_lambdas: tuple[float, ...],
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[
    Float[Array, "i p"],
    tuple[Float[Array, " i"], Float[Array, " i"], list[int]],
]:
    n_fixed, n_dim = means_fixed.shape
    num_scales = len(cov_scalings)

    # for multi-scale, each scale gets its own save folder
    if save_path is None:
        save_paths = tuple(
            [
                None,
            ]
            * num_scales
        )
    else:
        save_paths = tuple(
            [os.path.join(save_path, f"scale{i}") for i in range(num_scales)]
        )

    opt_single_scale = Partial(
        optimize_single_scale,
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        rbf_bandwidth=rbf_bandwidth,
        l2_scaling=l2_scaling,
        grad_tol=grad_tol,
        loss_tol=loss_tol,
        max_iter=max_iter,
    )

    pars, grads, losses, iters = list(), list(), list(), list()
    for cov_scale, reg_lambda, save_path_scale in zip(
        cov_scalings, regularization_lambdas, save_paths
    ):
        par_scale, (grad_scale, loss_scale, iter_scale) = opt_single_scale(
            affine,
            trans,
            rbf_wgts,
            cov_scaling=cov_scale,
            regularization_lambda=reg_lambda,
            save_path=save_path_scale,
        )
        pars.append(par_scale)
        grads.append(grad_scale)
        losses.append(loss_scale)
        iters.append(iter_scale)
        affine, trans, rbf_wgts = unpack_params(par_scale, n_fixed, n_dim)
    pars = jnp.stack(pars, axis=0)
    grads = jnp.concatenate(grads)
    losses = jnp.concatenate(losses)
    return pars, (grads, losses, iters)
