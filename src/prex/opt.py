from collections.abc import Callable
from enum import Enum
from os.path import join as path_join

import jax
import jax.numpy as jnp
import numpy
import optax
from jax.experimental import io_callback
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float

from . import affine, rigid, tps
from .dist import (
    kullback_leibler_gmm_approx_spherical,
    l2_distance_gmm_opt_spherical,
)
from .grbf import affine as grbfa
from .grbf import rigid as grbfr


class AlignmentMethod(Enum):
    AFFINEM = "affine-matrix"
    AFFINE = "affine"
    RIGID = "rigid"
    GRBFR = "grbf-rigid"
    GRBFA = "grbf-affine"
    TPS = "tps"


class DistanceFunction(Enum):
    L2 = 1
    KL = 2


class Regularization(Enum):
    NONE = 0
    TPS = 1
    RIGDE = 2


type AuxiliaryData = tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, " p"]
]
type RegularizationData = tuple[Regularization, float]


def auxdata_to_npz(aux_data: AuxiliaryData, file_path: str) -> None:
    dist, reg_term, params = aux_data
    numpy.savez(
        file_path,
        dist=dist,
        reg_term=reg_term,
        params=params,
    )


def auxdata_from_npz(file_path: str) -> AuxiliaryData:
    arxiv = numpy.load(file_path)
    return arxiv["dist"], arxiv["reg_term"], arxiv["params"]


def _make_transform_function_spherical(
    means_mov: Float[Array, "m d"],
    method: AlignmentMethod,
    ctrl_pts: Float[Array, "c d"],
    grbf_bandwidth: float = 1.0,
) -> Callable[[Float[Array, " p"]], Float[Array, "m d"]]:
    _, n_dim = means_mov.shape
    if method == AlignmentMethod.AFFINEM:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                m = p[:4].reshape(2, 2)
                t = p[4:]
                return affine.transform_means(means_mov, m, t)

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                m = p[:9].reshape(3, 3)
                t = p[9:]
                return affine.transform_means(means_mov, m, t)

        else:
            raise ValueError("invalid # of dimensions")
    elif method == AlignmentMethod.AFFINE:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                return affine.transform_means_rotangles2(
                    means_mov, *affine.unpack_params_2d(p)
                )

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                return affine.transform_means_rotangles3(
                    means_mov, *affine.unpack_params_3d(p)
                )

        else:
            raise ValueError("invalid # of dimensions")
    elif method == AlignmentMethod.RIGID:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                return rigid.transform_means_rotangles2(
                    means_mov, *rigid.unpack_params_2d(p)
                )

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                return rigid.transform_means_rotangles3(
                    means_mov, *rigid.unpack_params_3d(p)
                )

        else:
            raise ValueError("invalid # of dimensions")
    elif method == AlignmentMethod.GRBFR:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                s, alpha, t, psi = grbfr.unpack_params_2d(p)
                return grbfr.transform_means_2d(
                    means_mov,
                    s,
                    alpha,
                    t,
                    psi,
                    ctrl_pts,
                    grbf_bandwidth,
                )

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                s, alpha, beta, gamma, t, psi = grbfr.unpack_params_3d(p)
                return grbfr.transform_means_3d(
                    means_mov,
                    s,
                    alpha,
                    beta,
                    gamma,
                    t,
                    psi,
                    ctrl_pts,
                    grbf_bandwidth,
                )

        else:
            raise ValueError("invalid # of dimensions")
    elif method == AlignmentMethod.GRBFA:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                aff, t, psi = grbfa.unpack_params_2d(p)
                return grbfa.transform_means(
                    means_mov, aff, t, psi, ctrl_pts, grbf_bandwidth
                )

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                aff, t, psi = grbfa.unpack_params_3d(p)
                return grbfa.transform_means(
                    means_mov, aff, t, psi, ctrl_pts, grbf_bandwidth
                )

        else:
            raise ValueError("invalid # of dimensions")
    elif method == AlignmentMethod.TPS:
        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = tps.unpack_params_2d(p)
                return tps.transform_means(means_mov, A, t, ctrl_pts, wgt)

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = tps.unpack_params_3d(p)
                return tps.transform_means(means_mov, A, t, ctrl_pts, wgt)

        else:
            raise ValueError("invalid # of dimensions")
    else:
        raise ValueError("todo")
    return transform_function


def _make_loss_function_spherical(
    means_ref: Float[Array, "r d"],
    wgts_ref: Float[Array, " r"],
    means_mov: Float[Array, "m d"],
    wgts_mov: Float[Array, " m"],
    var_ref: float,
    var_mov: float,
    method: AlignmentMethod,
    metric: DistanceFunction,
    regularization: RegularizationData,
    l2_scaling: float = 1.0,
) -> Callable[[Float[Array, " p"]], tuple[Float[Array, ""], AuxiliaryData]]:
    _, n_dim = means_ref.shape
    transform = _make_transform_function_spherical(means_mov, method, means_ref)
    reg_type, reg_const = regularization
    if reg_type == Regularization.NONE:
        if metric == DistanceFunction.KL:

            def loss_function(
                p: Float[Array, " p"],
            ) -> tuple[Float[Array, ""], AuxiliaryData]:
                means_trans = transform(p)
                dist = kullback_leibler_gmm_approx_spherical(
                    means_ref,
                    wgts_ref,
                    means_trans,
                    wgts_mov,
                    var_ref,
                    var_mov,
                    n_dim,
                )
                return dist, (dist, jnp.array(0.0), p)

        elif metric == DistanceFunction.L2:

            def loss_function(
                p: Float[Array, " p"],
            ) -> tuple[Float[Array, ""], AuxiliaryData]:
                means_trans = transform(p)
                dist = (
                    l2_distance_gmm_opt_spherical(
                        means_ref,
                        wgts_ref,
                        means_trans,
                        wgts_mov,
                        var_ref,
                        var_mov,
                        n_dim,
                    )
                    * l2_scaling
                )
                return dist, (dist, jnp.array(0.0), p)

        else:
            raise ValueError("invalid distance function for loss")
    else:
        # figure out how to get weights (which we'll need for regularization)
        if method == AlignmentMethod.GRBFA:
            if n_dim == 2:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = grbfa.unpack_params_2d(pars)
                    return wgt

            elif n_dim == 3:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = grbfa.unpack_params_3d(pars)
                    return wgt

            else:
                raise ValueError("invalid # of dimensions")
        elif method == AlignmentMethod.GRBFR:
            if n_dim == 2:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, _, wgt = grbfr.unpack_params_2d(pars)
                    return wgt

            elif n_dim == 3:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, _, _, _, wgt = grbfr.unpack_params_3d(pars)
                    return wgt

            else:
                raise ValueError("invalid # of dimensions")
        elif method == AlignmentMethod.TPS:
            if n_dim == 2:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = tps.unpack_params_2d(pars)
                    return wgt

            elif n_dim == 3:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = tps.unpack_params_3d(pars)
                    return wgt

            else:
                raise ValueError("invalid # of dimensions")
        else:
            raise ValueError("invalid alignment method for regularization")
        # make loss function, using the function to grab weights
        if reg_type == Regularization.TPS:
            calculate_bending_energy = Partial(
                tps.tps_bending_energy, tps.tps_rbf(means_ref, means_ref)
            )
            if metric == DistanceFunction.KL:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = kullback_leibler_gmm_approx_spherical(
                        means_ref,
                        wgts_ref,
                        means_trans,
                        wgts_mov,
                        var_ref,
                        var_mov,
                        n_dim,
                    )
                    bending_energy = reg_const * calculate_bending_energy(wgts)
                    return dist + bending_energy, (dist, bending_energy, p)

            elif metric == DistanceFunction.L2:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = (
                        l2_distance_gmm_opt_spherical(
                            means_ref,
                            wgts_ref,
                            means_trans,
                            wgts_mov,
                            var_ref,
                            var_mov,
                            n_dim,
                        )
                        * l2_scaling
                    )
                    bending_energy = reg_const * calculate_bending_energy(wgts)
                    return dist + bending_energy, (dist, bending_energy, p)

            else:
                raise ValueError("invalid distance function for loss")
        elif reg_type == Regularization.RIGDE:
            if metric == DistanceFunction.KL:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = kullback_leibler_gmm_approx_spherical(
                        means_ref,
                        wgts_ref,
                        means_trans,
                        wgts_mov,
                        var_ref,
                        var_mov,
                        n_dim,
                    )
                    ridge_penalty = reg_const * jnp.linalg.norm(wgts)
                    return dist + ridge_penalty, (dist, ridge_penalty, p)

            elif metric == DistanceFunction.L2:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = (
                        l2_distance_gmm_opt_spherical(
                            means_ref,
                            wgts_ref,
                            means_trans,
                            wgts_mov,
                            var_ref,
                            var_mov,
                            n_dim,
                        )
                        * l2_scaling
                    )
                    ridge_penalty = reg_const * jnp.linalg.norm(wgts)
                    return dist + ridge_penalty, (dist, ridge_penalty, p)

            else:
                raise ValueError("invalid distance function for loss")
        else:
            raise ValueError("invalid regularization method")
    return loss_function


type OptimizationState = tuple[
    Float[Array, " p"], optax.OptState, Float[Array, ""], Float[Array, ""], int
]


def spherical(
    means_ref: Float[Array, "r d"],
    wgts_ref: Float[Array, " r"],
    means_mov: Float[Array, "m d"],
    wgts_mov: Float[Array, " m"],
    method: AlignmentMethod,
    metric: DistanceFunction,
    regularization: RegularizationData,
    init_pars: Float[Array, " p"],
    var_ref: float,
    var_mov: float,
    grad_tol: float = 1e-6,
    loss_tol: float = -jnp.inf,
    max_iter: int = 100,
    save_path: str | None = None,
    l2_scaling: float = 1.0,
    **lbfgs_kwargs,
):
    loss_func = _make_loss_function_spherical(
        means_ref,
        wgts_ref,
        means_mov,
        wgts_mov,
        var_ref,
        var_mov,
        method,
        metric,
        regularization,
        l2_scaling,
    )

    def loss_func_noaux(pars: Float[Array, " p"]) -> Float[Array, ""]:
        loss_val, _ = loss_func(pars)
        return loss_val

    optimizer = optax.lbfgs(**lbfgs_kwargs)
    opt_state = optimizer.init(init_pars)

    def keep_stepping(x: OptimizationState) -> Bool:
        _, _, grad_norm, curr_loss, num_iter = x
        grad_high = grad_norm > grad_tol
        loss_high = curr_loss > loss_tol
        grad_loss = jnp.logical_and(grad_high, loss_high)
        return jnp.logical_and(grad_loss, num_iter < max_iter)

    if save_path is not None:

        def save_step_data(aux_data: AuxiliaryData, iter_num: int) -> None:
            fpath = path_join(save_path, f"{iter_num:05d}.npz")
            auxdata_to_npz(aux_data, fpath)

    else:

        def save_step_data(aux_data: AuxiliaryData, iter_num: int) -> None:
            pass

    def take_step(x: OptimizationState) -> OptimizationState:
        params, opt_state, _, _, iter_num = x
        (loss, aux_data), grads = jax.value_and_grad(loss_func, has_aux=True)(
            params
        )
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
        return params, opt_state, grad_norm, loss, iter_num + 1

    par_f, opt_state, grad_norm, loss_final, num_iter = jax.lax.while_loop(
        keep_stepping,
        take_step,
        (init_pars, opt_state, jnp.array(jnp.inf), jnp.array(jnp.inf), 0),
    )
    return par_f, (grad_norm, loss_final, num_iter)
