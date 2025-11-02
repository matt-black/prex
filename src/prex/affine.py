"""Affine Transformations between GMMs

Transform a GMM by a global affine transform and translation. Also use this transform model to register one GMM onto another. Note that this assumes isotropic scaling.
"""

import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy
import optax
from jax.experimental import io_callback
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float, PyTree

from .dist import l2_distance_gmm_opt
from .util import (
    rotation_matrix_2d,
    rotation_matrix_3d,
    shear_matrix_2d,
    shear_matrix_3d,
)

__all__ = [
    "transform_gmm",
    "transform_gmm_rotangles2",
    "unpack_params",
    "pack_params",
    "optimize_single_scale",
    "optimize_multi_scale",
    "optimize_single_scale_matrix",
]


def transform_gmm(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    matrix: Float[Array, "d d"],
    translation: Float[Array, " d"],
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    """Apply forward transform (scale + rotation + translation) to GMM

    Args:
        means: GMM means, shape (n_components, d)
        covariances: GMM covariances (diagonal), shape (n_components, d)
        weights: GMM weights, shape (n_components,)
        scale: Scaling factor
        rotation: Rotation matrix, shape (d, d)
        translation: Translation vector, shape (d,)

    Returns:
        transformed_means: Transformed means, shape (n_components, d)
        transformed_covs: Transformed covariances, shape (n_components, d, d)
    """

    # transform means by applying the global rigid transform -> shape (n_components, d)
    transformed_means = jnp.add(
        jnp.transpose(matrix @ means.T), translation[jnp.newaxis, :]
    )

    # transform covariances by pre-/post-multiplication with jacobian
    # this simplifies to (s² R Σ Rᵀ)
    def transform_single_covariance(
        cov: Float[Array, "d d"],
    ) -> Float[Array, "d d"]:
        return matrix @ cov @ matrix.T

    transformed_covs = jax.vmap(transform_single_covariance, 0, 0)(covariances)

    return transformed_means, transformed_covs


def transform_gmm_rotangles2(
    means: Float[Array, "n_comp 2"],
    covariances: Float[Array, "n_comp 2 2"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    alpha: Float[Array, ""],
    k: Float[Array, ""],
    ell: Float[Array, ""],
    translation: Float[Array, " 2"],
) -> tuple[Float[Array, "n_comp 2"], Float[Array, "n_comp 2 2"]]:
    S = jnp.diag(jnp.array([scale_x, scale_y]))
    R = rotation_matrix_2d(alpha)
    Sh = shear_matrix_2d(k, ell)
    matrix = S @ R @ Sh
    return transform_gmm(
        means,
        covariances,
        matrix,
        translation,
    )


def transform_gmm_rotangles3(
    means: Float[Array, "n_comp 3"],
    covariances: Float[Array, "n_comp 3 3"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    scale_z: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
    translation: Float[Array, " 3"],
) -> tuple[Float[Array, "n_comp 3"], Float[Array, "n_comp 3 3"]]:
    S = jnp.diag(jnp.array([scale_x, scale_y, scale_z]))
    R = rotation_matrix_3d(alpha, beta, gamma)
    Sh = shear_matrix_3d(k_xy, k_xz, k_yx, k_yz, k_zx, k_zy)
    matrix = S @ R @ Sh
    return transform_gmm(
        means,
        covariances,
        matrix,
        translation,
    )


def transform_gmm_rotangles(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    scale_z: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
    translation: Float[Array, " d"],
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    _, n_dim = means.shape
    if n_dim == 2:
        return transform_gmm_rotangles2(
            means, covariances, scale_x, scale_y, alpha, k_xy, k_xz, translation
        )
    else:
        return transform_gmm_rotangles3(
            means,
            covariances,
            scale_x,
            scale_y,
            scale_z,
            alpha,
            beta,
            gamma,
            k_xy,
            k_xz,
            k_yx,
            k_yz,
            k_zx,
            k_zy,
            translation,
        )


def unpack_params(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_dim"],
]:
    scale_x = flat_params[0]
    scale_y = flat_params[1]
    scale_z = flat_params[2]
    alpha = flat_params[3]
    beta = flat_params[4]
    gamma = flat_params[5]
    k_xy = flat_params[6]
    k_xz = flat_params[7]
    k_yx = flat_params[8]
    k_yz = flat_params[9]
    k_zx = flat_params[10]
    k_zy = flat_params[11]
    trans = flat_params[12:]
    return (
        scale_x,
        scale_y,
        scale_z,
        alpha,
        beta,
        gamma,
        k_xy,
        k_xz,
        k_yx,
        k_yz,
        k_zx,
        k_zy,
        trans,
    )


def pack_params(
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    scale_z: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
    trans: Float[Array, " n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate(
        [
            scale_x[jnp.newaxis],
            scale_y[jnp.newaxis],
            scale_z[jnp.newaxis],
            alpha[jnp.newaxis],
            beta[jnp.newaxis],
            gamma[jnp.newaxis],
            k_xy[jnp.newaxis],
            k_xz[jnp.newaxis],
            k_yx[jnp.newaxis],
            k_yz[jnp.newaxis],
            k_zx[jnp.newaxis],
            k_zy[jnp.newaxis],
            trans,
        ]
    )


def _create_optimization_function(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    cov_scaling: float,
    l2_scaling: float,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], dict[str, Array]]]:

    # scale covariances for this annealing step
    scaled_covs_fixed = covs_fixed * cov_scaling
    scaled_covs_moving = covs_moving * cov_scaling

    # make partial functions for all downstream parameters that are constant over the course of optimization
    trans_fun = Partial(
        transform_gmm_rotangles,
        means_moving,
        scaled_covs_moving,
    )
    dist_fun = Partial(
        l2_distance_gmm_opt, means_fixed, scaled_covs_fixed, wgts_fixed
    )

    def loss_l2(
        param_flat: Float[Array, " p"],
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        # unpack parameters
        (
            scale_x,
            scale_y,
            scale_z,
            alpha,
            beta,
            gamma,
            k_xy,
            k_xz,
            k_yx,
            k_yz,
            k_zx,
            k_zy,
            trans,
        ) = unpack_params(param_flat)
        # apply transform
        means_trans, cov_trans = trans_fun(
            scale_x,
            scale_y,
            scale_z,
            alpha,
            beta,
            gamma,
            k_xy,
            k_xz,
            k_yx,
            k_yz,
            k_zx,
            k_zy,
            trans,
        )
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)

        aux_data = {
            "l2": dist_l2,
            "scale": jnp.array([scale_x, scale_y, scale_z]),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "shears": jnp.array([k_xy, k_xz, k_yx, k_yz, k_zx, k_zy]),
            "trans": trans,
        }
        return l2_scaling * dist_l2, aux_data

    return loss_l2


@Partial(
    jax.jit,
    static_argnums=(
        19,
        20,
        21,
        22,
        23,
        24,
    ),
)
def optimize_single_scale(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    scale_z: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
    trans: Float[Array, " d"],
    cov_scaling: float,
    l2_scaling: float,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
    init_pars = pack_params(
        scale_x,
        scale_y,
        scale_z,
        alpha,
        beta,
        gamma,
        k_xy,
        k_xz,
        k_yx,
        k_yz,
        k_zx,
        k_zy,
        trans,
    )
    loss_func = _create_optimization_function(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        cov_scaling,
        l2_scaling,
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
        grad_loss = jnp.logical_and(grad_high, loss_high)
        return jnp.logical_and(grad_loss, num_iter < max_iter)

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


def optimize_multi_scale(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    scale_z: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
    trans: Float[Array, " d"],
    cov_scalings: tuple[float, ...],
    l2_scaling: float,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[
    Float[Array, "i p"],
    tuple[Float[Array, " i"], Float[Array, " i"], list[int]],
]:
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
        l2_scaling=l2_scaling,
        grad_tol=grad_tol,
        loss_tol=loss_tol,
        max_iter=max_iter,
    )

    pars, grads, losses, iters = list(), list(), list(), list()
    for cov_scale, save_path_scale in zip(cov_scalings, save_paths):
        par_scale, (grad_scale, loss_scale, iter_scale) = opt_single_scale(
            scale_x,
            scale_y,
            scale_z,
            alpha,
            beta,
            gamma,
            k_xy,
            k_xz,
            k_yx,
            k_yz,
            k_zx,
            k_zy,
            trans,
            cov_scaling=cov_scale,
            save_path=save_path_scale,
        )
        pars.append(par_scale)
        grads.append(grad_scale)
        losses.append(loss_scale)
        iters.append(iter_scale)
        (
            scale_x,
            scale_y,
            scale_z,
            alpha,
            beta,
            gamma,
            k_xy,
            k_xz,
            k_yx,
            k_yz,
            k_zx,
            k_zy,
            trans,
        ) = unpack_params(par_scale)
    pars = jnp.stack(pars, axis=0)
    grads = jnp.concatenate(grads)
    losses = jnp.concatenate(losses)
    return pars, (grads, losses, iters)


def _create_optimization_function_matrix(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    cov_scaling: float,
    l2_scaling: float,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], dict[str, Array]]]:

    _, n_dim = means_fixed.shape
    # scale covariances for this annealing step
    scaled_covs_fixed = covs_fixed * cov_scaling
    scaled_covs_moving = covs_moving * cov_scaling

    # make partial functions for all downstream parameters that are constant over the course of optimization
    trans_fun = Partial(
        transform_gmm,
        means_moving,
        scaled_covs_moving,
    )
    dist_fun = Partial(
        l2_distance_gmm_opt, means_fixed, scaled_covs_fixed, wgts_fixed
    )

    def loss_l2(
        params_flat: Float[Array, " p"],
    ) -> tuple[Float[Array, ""], dict[str, Array]]:
        # unpack parameters
        matrix = params_flat[:-n_dim].reshape(n_dim, n_dim)
        trans = params_flat[-n_dim:]
        # apply transform
        means_trans, cov_trans = trans_fun(matrix, trans)
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)

        aux_data = {
            "l2": dist_l2,
            "matrix": matrix,
            "trans": trans,
        }
        return l2_scaling * dist_l2, aux_data

    return loss_l2


@Partial(
    jax.jit,
    static_argnums=(
        8,
        9,
        10,
        11,
        12,
        13,
    ),
)
def optimize_single_scale_matrix(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    matrix: Float[Array, "d d"],
    trans: Float[Array, " d"],
    cov_scaling: float,
    l2_scaling: float,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
    init_pars = jnp.concatenate([matrix.flatten(), trans])
    loss_func = _create_optimization_function_matrix(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        cov_scaling,
        l2_scaling,
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
        grad_loss = jnp.logical_and(grad_high, loss_high)
        return jnp.logical_and(grad_loss, num_iter < max_iter)

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
