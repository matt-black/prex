"""Thin Plate Spline

Transform a GMM by a thin plate spline transformation.
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

from .dist import kullback_leibler_gmm_approx, l2_distance_gmm_opt
from .util import sqdist


def tps_rbf(
    x: Float[Array, "n_pts d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
) -> Float[Array, "n_pts n_ctrl"]:
    _, n_dim = x.shape
    d = jnp.sqrt(sqdist(x, ctrl_pts))
    return jax.lax.cond(
        n_dim == 2,
        Partial(_rbf_even, pwr=n_dim),
        Partial(_rbf_odd, pwr=n_dim),
        d,
    )


@Partial(jax.jit, static_argnums=(1,))
def _rbf_even(dist: Array, pwr: int) -> Array:
    d = jnp.where(dist == 0, 1, dist)
    return jnp.multiply(jnp.power(dist, pwr), jnp.log(d))


@Partial(jax.jit, static_argnums=(1,))
def _rbf_odd(dist: Array, pwr: int) -> Array:
    return jnp.negative(jnp.power(dist, pwr - 2))


@Partial(jax.jit, static_argnums=(4,))
def jacobian_at_point(
    x: Float[Array, " d"],
    affine: Float[Array, "d d"],
    rbf_wgts: Float[Array, "n_ctrl d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
    n_dim: int = 2,
) -> Float[Array, "d d"]:
    """Compute the Jacobian of the forward transformation at point, x.

    Args:
        x (Float[Array, " d"]): Point at which to compute Jacobian, shape (d,)
        centers (Float[Array, "n_cent d"]): RBF centers, shape (n_centers, d)
        rbf_weights (Float[Array, "n_cent d"]): GRBF coefficients, shape (n_centers, d)
        rotation (Float[Array, "d d"]): Rotation matrix, shape (d, d)
        bandwidth (float): RBF bandwidth
        scale (float): Scaling factor

    Returns:
        Float[Array, "d d"]: Jacobian matrix
    """
    # local RBF contributions
    diff = x - ctrl_pts
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True)
    if n_dim == 2:
        rbf_grads = _rbf_grad2(diff, dist)
    else:
        rbf_grads = _rbf_grad3(diff, dist)
    jac_local = jnp.sum(
        rbf_wgts[:, :, jnp.newaxis] * rbf_grads[:, jnp.newaxis, :], axis=0
    )
    return affine + jac_local


@Partial(jax.jit, static_argnums=(4,))
def jacobians(
    psi: Float[Array, "n_comp n_ctrl"],
    diffs: Float[Array, "n_comp n_ctrl d"],
    affine: Float[Array, "d d"],
    rbf_wgts: Float[Array, "n_ctrl d"],
    n_dim: int,
) -> Float[Array, "n_comp d"]:

    n_comp, n_ctrl = psi.shape
    dists = jnp.linalg.norm(diffs, axis=2, keepdims=True)
    grad_fun = _rbf_grad2 if n_dim == 2 else _rbf_grad3

    # shape (n_comp, n_ctrl, d)
    # NOTE: the reshape here is for compliance with _rbf_grad3
    rbf_grads = jax.vmap(grad_fun, (0, 0), 0)(
        diffs.reshape(-1, 1, n_dim), dists.reshape(-1, 1, 1)
    ).reshape(n_comp, n_ctrl, n_dim)

    jac_local = jnp.sum(
        rbf_wgts[jnp.newaxis, :, :, jnp.newaxis]
        * rbf_grads[:, :, jnp.newaxis, :],
        axis=1,
    )
    return affine[jnp.newaxis, :, :] + jac_local


@jax.jit
def _rbf_grad2(diff: Array, dist: Array) -> Array:
    d = jnp.where(dist == 0, 1, dist)
    return (2 * jnp.log(d) + 1) * diff


@jax.jit
def _rbf_grad3(diff: Array, dist: Array) -> Array:
    dist = jnp.concatenate([dist, dist, dist], axis=1)
    return jax.lax.select(dist == 0, jnp.zeros_like(dist), diff / dist)


def transform_gmm(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
    rbf_wgts: Float[Array, "n_ctrl d"],
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    _, n_dim = means.shape
    global_means = jnp.transpose(affine @ means.T) + translation[jnp.newaxis, :]
    psi = tps_rbf(means, ctrl_pts)
    local_deformation = psi @ rbf_wgts
    transformed_means = global_means + local_deformation

    jac_at_point = Partial(
        jacobian_at_point,
        affine=affine,
        rbf_wgts=rbf_wgts,
        ctrl_pts=ctrl_pts,
        n_dim=n_dim,
    )

    def transform_single_covariance(
        mu: Float[Array, " d"],
        cov: Float[Array, "d d"],
    ) -> Float[Array, "d d"]:
        jac = jac_at_point(mu)
        return jac @ cov @ jac.T

    transformed_covs = jax.vmap(transform_single_covariance, (0, 0), 0)(
        means, covariances
    )

    return transformed_means, transformed_covs


@Partial(jax.jit, static_argnums=(7,))
def _transform_gmm_precomputed(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_ctrl d"],
    psi: Float[Array, "n_comp n_ctrl"],
    diffs: Float[Array, "n_comp n_ctrl d"],
    n_dim: int,
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:

    means_trans = (
        jnp.transpose(affine @ means.T)
        + translation[jnp.newaxis, :]
        + psi @ rbf_wgts
    )
    # transform covariances by pre- and post-mult. with jacobian
    jac = jacobians(psi, diffs, affine, rbf_wgts, n_dim)
    cov_trans = jnp.matmul(
        jnp.matmul(jac, covariances), jnp.matrix_transpose(jac)
    )
    return means_trans, cov_trans


@Partial(
    jax.jit,
    static_argnums=(
        1,
        2,
    ),
)
def unpack_params(
    flat_params: Float[Array, " p"], n_ctrl: int, n_dim: int
) -> tuple[
    Float[Array, "{n_dim} {n_dim}"],
    Float[Array, " {n_dim}"],
    Float[Array, "{n_ctrl} {n_dim}"],
]:
    affine = flat_params[: n_dim**2].reshape(n_dim, n_dim)
    translation = flat_params[n_dim**2 : n_dim**2 + n_dim]
    rbf_wgts = flat_params[n_dim**2 + n_dim :].reshape(n_ctrl, n_dim)
    return affine, translation, rbf_wgts


def pack_params(
    affine: Float[Array, "n_dim n_dim"],
    translation: Float[Array, " n_dim"],
    rbf_wgts: Float[Array, "n_ctrl n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate([affine.ravel(), translation, rbf_wgts.ravel()])


def tps_bending_energy(
    K: Float[Array, "n_ctrl n_ctrl"], wgts: Float[Array, "n_ctrl n_dim"]
) -> Float[Array, ""]:
    return jnp.abs(jnp.trace(wgts.T @ K @ wgts))


def _create_optimization_function(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], dict[str, Array]]]:

    n_fixed, n_dim = means_fixed.shape
    # scale covariances for this annealing step
    scaled_covs_fixed = covs_fixed * cov_scaling
    scaled_covs_moving = covs_moving * cov_scaling

    # pre-compute differences tensor and kernel matrix, psi, between moving & fixed points
    diffs = jax.vmap(
        lambda f: jax.vmap(Partial(jnp.subtract, f), 0, 0)(means_moving), 0, 0
    )(means_fixed)
    psi = tps_rbf(means_moving, means_fixed)

    # make partial functions for all downstream parameters that are constant over the course of optimization
    trans_fun = Partial(
        _transform_gmm_precomputed,
        means_moving,
        scaled_covs_moving,
        psi=psi,
        diffs=diffs,
        n_dim=n_dim,
    )
    dist_fun = Partial(
        l2_distance_gmm_opt, means_fixed, scaled_covs_fixed, wgts_fixed
    )
    unpack = Partial(unpack_params, n_ctrl=n_fixed, n_dim=n_dim)

    # compute kernel matrix, K, to calculate bending energy of spline
    # this is an (n_fixed, n_fixed) shape matrix
    K = tps_rbf(means_fixed, means_fixed)
    tps_be_fun = Partial(tps_bending_energy, K)

    def loss_l2(
        param_flat: Float[Array, " p"],
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        # unpack parameters
        affine, trans, rbf_wgts = unpack(param_flat)
        # apply transform
        means_trans, cov_trans = trans_fun(affine, trans, rbf_wgts)
        # calculate loss terms
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)
        bending_energy = tps_be_fun(rbf_wgts)
        aux_data = {
            "l2": dist_l2,
            "bending_energy": bending_energy,
            "affine": affine,
            "trans": trans,
            "rbf_wgt": rbf_wgts,
        }
        # loss includes rescaling/weight terms
        loss = l2_scaling * dist_l2 + regularization_lambda * bending_energy
        return loss, aux_data

    return loss_l2


def _create_optimization_function_with_validation(
    means_fixed_train: Float[Array, "ft d"],
    covs_fixed_train: Float[Array, "ft d d"],
    wgts_fixed_train: Float[Array, " ft"],
    means_fixed_valid: Float[Array, "fv d"],
    covs_fixed_valid: Float[Array, "fv d d"],
    wgts_fixed_valid: Float[Array, " fv"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], dict[str, Array]]]:

    n_fixed, n_dim = means_fixed_train.shape
    # scale covariances for this annealing step
    scaled_covs_fixed = covs_fixed_train * cov_scaling
    scaled_covs_valid = covs_fixed_valid * cov_scaling
    scaled_covs_moving = covs_moving * cov_scaling

    # pre-compute differences tensor and kernel matrix, psi, between moving & fixed points
    diffs = jax.vmap(
        lambda f: jax.vmap(Partial(jnp.subtract, f), 0, 0)(means_moving), 0, 0
    )(means_fixed_train)
    psi = tps_rbf(means_moving, means_fixed_train)

    # make partial functions for all downstream parameters that are constant over the course of optimization
    trans_fun = Partial(
        _transform_gmm_precomputed,
        means_moving,
        scaled_covs_moving,
        psi=psi,
        diffs=diffs,
        n_dim=n_dim,
    )
    dist_fun = Partial(
        l2_distance_gmm_opt,
        means_fixed_train,
        scaled_covs_fixed,
        wgts_fixed_train,
    )
    kl_fun = Partial(
        kullback_leibler_gmm_approx,
        means_fixed_valid,
        scaled_covs_valid,
        wgts_fixed_valid,
    )
    unpack = Partial(unpack_params, n_ctrl=n_fixed, n_dim=n_dim)

    # compute kernel matrix, K, to calculate bending energy of spline
    # this is an (n_fixed, n_fixed) shape matrix
    K = tps_rbf(means_fixed_train, means_fixed_train)
    tps_be_fun = Partial(tps_bending_energy, K)

    def loss_l2(
        param_flat: Float[Array, " p"],
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        # unpack parameters
        affine, trans, rbf_wgts = unpack(param_flat)
        # apply transform
        means_trans, cov_trans = trans_fun(affine, trans, rbf_wgts)
        # calculate loss
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)
        bending_energy = tps_be_fun(rbf_wgts)
        # calculate validation loss
        kl_validation = kl_fun(means_trans, cov_trans, wgts_moving)
        # save auxiliary data
        aux_data = {
            "l2": dist_l2,
            "bending_energy": bending_energy,
            "kl_validation": kl_validation,
            "affine": affine,
            "trans": trans,
            "rbf_wgt": rbf_wgts,
        }
        return dist_l2 + regularization_lambda * bending_energy, aux_data

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
            Float[Array, ""],
            int,
        ],
    ) -> Bool:
        _, _, grad_norm, prev_loss, curr_loss, num_iter = x
        grad_high = grad_norm > grad_tol
        loss_high = jnp.abs(curr_loss - prev_loss) > loss_tol
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
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, prev_loss, num_iter = x
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
            return (params, opt_state, grad_norm, prev_loss, loss, num_iter + 1)

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
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, prev_loss, num_iter = x
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
            return (params, opt_state, grad_norm, prev_loss, loss, num_iter + 1)

    par_f, opt_state, grad_norm, _, final_loss, num_iter = jax.lax.while_loop(
        keep_stepping,
        take_step,
        (
            init_pars,
            opt_state,
            jnp.array(jnp.inf),
            jnp.array(0.0),
            jnp.array(jnp.inf),
            0,
        ),
    )
    return par_f, (grad_norm, final_loss, num_iter)


@Partial(
    jax.jit,
    static_argnums=(
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ),
)
def optimize_single_scale_with_validation(
    means_fixed_train: Float[Array, "ft d"],
    covs_fixed_train: Float[Array, "ft d d"],
    wgts_fixed_train: Float[Array, " ft"],
    means_fixed_valid: Float[Array, "fv d"],
    covs_fixed_valid: Float[Array, "fv d d"],
    wgts_fixed_valid: Float[Array, " fv"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    affine: Float[Array, "d d"],
    trans: Float[Array, " d"],
    rbf_wgts: Float[Array, "f d"],
    cov_scaling: float,
    l2_scaling: float,
    regularization_lambda: float,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
    init_pars = pack_params(affine, trans, rbf_wgts)
    loss_func = _create_optimization_function_with_validation(
        means_fixed_train,
        covs_fixed_train,
        wgts_fixed_train,
        means_fixed_valid,
        covs_fixed_valid,
        wgts_fixed_valid,
        means_moving,
        covs_moving,
        wgts_moving,
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
            Float[Array, ""],
            int,
        ],
    ) -> Bool:
        _, _, grad_norm, prev_loss, curr_loss, num_iter = x
        grad_high = grad_norm > grad_tol
        loss_high = jnp.abs(curr_loss - prev_loss) > loss_tol
        iter_ltmax = num_iter < max_iter
        return grad_high and loss_high and iter_ltmax

    if save_path is None:

        def take_step(
            x: tuple[
                Array,
                PyTree,
                Float[Array, ""],
                Float[Array, ""],
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, prev_loss, num_iter = x
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
            return (params, opt_state, grad_norm, prev_loss, loss, num_iter + 1)

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
                Float[Array, ""],
                int,
            ],
        ) -> tuple[
            Array,
            PyTree,
            Float[Array, ""],
            Float[Array, ""],
            Float[Array, ""],
            int,
        ]:
            params, opt_state, _, _, prev_loss, num_iter = x
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
            kl = aux_data["kl_validation"]
            return (params, opt_state, grad_norm, prev_loss, kl, num_iter + 1)

    par_f, opt_state, grad_norm, _, final_loss, num_iter = jax.lax.while_loop(
        keep_stepping,
        take_step,
        (
            init_pars,
            opt_state,
            jnp.array(jnp.inf),
            jnp.array(0.0),
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
            bending_energy = aux_data["bending_energy"]
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
                    bending_energy[jnp.newaxis],
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
                [grad_norm[jnp.newaxis], loss[jnp.newaxis]]
            )

    (par_f, opt_state), loss_arr = jax.lax.scan(
        take_step,
        (init_pars, opt_state),
        jnp.arange(num_iter),
    )
    return par_f, loss_arr
