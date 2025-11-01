"""Rigid Transformations between GMMs

Transform a GMM by a global rigid transform (scaling + rotation + translation). Also use this transform model to register one GMM onto another.
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

from .dist import l2_distance_gmm_opt
from .util import rotation_matrix_2d, rotation_matrix_3d

__all__ = [
    "jacobian",
    "transform_gmm",
    "transform_gmm_rotangles",
    "unpack_params",
    "pack_params",
    "optimize_single_scale",
    "optimize_multi_scale",
]


def jacobian(
    rotation: Float[Array, "d d"],
    scale: Float[Array, ""],
) -> Float[Array, "d d"]:
    """Compute the Jacobian of the forward transform.

    Args:
        rotation (Float[Array, "d d"]): Rotation matrix, shape (d, d)
        scale (Float[Array, ""]): Scaling factor

    Returns:
        Float[Array, "d d"]: Jacobian matrix
    """
    return scale * rotation


def transform_gmm(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale: Float[Array, ""],
    rotation: Float[Array, "d d"],
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
    transformed_means = scale * means @ rotation.T + translation[jnp.newaxis, :]

    # transform covariances by pre-/post-multiplication with jacobian
    # this simplifies to (s² R Σ Rᵀ)
    def transform_single_covariance(
        cov: Float[Array, "d d"],
    ) -> Float[Array, "d d"]:
        return jnp.square(scale) * rotation @ cov @ rotation.T

    transformed_covs = jax.vmap(transform_single_covariance, 0, 0)(covariances)

    return transformed_means, transformed_covs


def transform_gmm_rotangles(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " d"],
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    _, d = means.shape
    if d == 2:
        R = rotation_matrix_2d(alpha)
    else:
        R = rotation_matrix_3d(alpha, beta, gamma)
    return transform_gmm(
        means,
        covariances,
        scale,
        R,
        translation,
    )


def unpack_params(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_dim"],
]:
    scale = flat_params[0]
    alpha = flat_params[1]
    beta = flat_params[2]
    gamma = flat_params[3]
    trans = flat_params[4:]
    return (scale, alpha, beta, gamma, trans)


def pack_params(
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    trans: Float[Array, " n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate(
        [
            scale[jnp.newaxis],
            alpha[jnp.newaxis],
            beta[jnp.newaxis],
            gamma[jnp.newaxis],
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
    l2_scaling: float = 1.0,
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
        scale, alpha, beta, gamma, trans = unpack_params(param_flat)
        # apply transform
        means_trans, cov_trans = trans_fun(scale, alpha, beta, gamma, trans)
        dist_l2 = dist_fun(means_trans, cov_trans, wgts_moving)

        aux_data = {
            "l2": dist_l2,
            "scale": scale,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "trans": trans,
        }
        return l2_scaling * dist_l2, aux_data

    return loss_l2


@Partial(
    jax.jit,
    static_argnums=(
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
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    trans: Float[Array, " d"],
    cov_scaling: float,
    l2_scaling: float = 1.0,
    grad_tol: float = 1e-6,
    loss_tol: float = 1e-8,
    max_iter: int = 100,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
    init_pars = pack_params(scale, alpha, beta, gamma, trans)
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


@Partial(
    jax.jit,
    static_argnums=(
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
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    trans: Float[Array, " d"],
    cov_scaling: float,
    l2_scaling: float,
    num_iter: int,
    save_path: str | None = None,
) -> tuple[Float[Array, " p"], Float[Array, " {num_iter}"]]:
    init_pars = pack_params(scale, alpha, beta, gamma, trans)
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

    if save_path is None:

        def take_step(
            x: tuple[Array, PyTree],
            iter_num: Int[Array, ""],
        ) -> tuple[tuple[Array, PyTree], Float[Array, ""]]:
            params, opt_state = x
            loss, grads = jax.value_and_grad(loss_func_noaux)(params)
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
            return (params, opt_state), loss

    else:

        def save_step_data(aux_data: dict[str, Array], iter_num: int) -> None:
            numpy.savez(
                os.path.join(save_path, f"{iter_num:05d}.npz"),
                **aux_data,  # pyright: ignore[reportArgumentType]
            )

        def take_step(
            x: tuple[Array, PyTree],
            iter_num: Int[Array, ""],
        ) -> tuple[tuple[Array, PyTree], Float[Array, ""]]:
            params, opt_state = x
            (loss, aux_data), grads = jax.value_and_grad(
                loss_func, has_aux=True
            )(params)
            io_callback(save_step_data, None, aux_data, iter_num)
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
            return (params, opt_state), loss

    (par_f, opt_state), losses = jax.lax.scan(
        take_step, (init_pars, opt_state), jnp.arange(num_iter)
    )
    return par_f, losses


def optimize_multi_scale(
    means_fixed: Float[Array, "f d"],
    covs_fixed: Float[Array, "f d d"],
    wgts_fixed: Float[Array, " f"],
    means_moving: Float[Array, "m d"],
    covs_moving: Float[Array, "m d d"],
    wgts_moving: Float[Array, " m"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
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
            scale,
            alpha,
            beta,
            gamma,
            trans,
            cov_scaling=cov_scale,
            save_path=save_path_scale,
        )
        pars.append(par_scale)
        grads.append(grad_scale)
        losses.append(loss_scale)
        iters.append(iter_scale)
        scale, alpha, beta, gamma, trans = unpack_params(par_scale)
    pars = jnp.stack(pars, axis=0)
    grads = jnp.concatenate(grads)
    losses = jnp.concatenate(losses)
    return pars, (grads, losses, iters)
