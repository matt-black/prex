from collections.abc import Callable
from enum import Enum
from math import floor
from os.path import join as path_join

import jax
import jax.numpy as jnp
import numpy
import optax
from jax.experimental import io_callback
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float

from . import affine, grb, rigid, tps
from .dist import (
    kullback_leibler_gmm_approx_spherical,
    kullback_leibler_gmm_approx_var_spherical,
    l2_distance_gmm_opt_spherical,
)


class AlignmentMethod(Enum):
    AFFINEM = "affine-matrix"
    AFFINE = "affine"
    RIGID = "rigid"
    GRB = "grb"
    TPS = "tps"


class DistanceFunction(Enum):
    L2 = 1
    KLG = 2
    KLV = 3


class Regularization(Enum):
    NONE = 0
    TPS = 1
    RIDGE = 2


class Optimizer(Enum):
    LBFGS = 0
    ADAM = 1
    SGD = 2


class ValidationFunction(Enum):
    HOPT = 1
    CORR = 2


# parts of AuxiliaryData: (distance, regularization, parameters)
type AuxiliaryData = tuple[
    Float[Array, ""], Float[Array, ""], Float[Array, " p"]
]
# (RegularizationEnum, scalar multiplier)
type RegularizationData = tuple[Regularization, float]
# Callable that takes in parameter vector, returns the loss value (to be minimized) and auxiliary data
type LossFunctionWithAux = Callable[
    [Float[Array, " p"]], tuple[Float[Array, ""], AuxiliaryData]
]


def auxdata_to_npz(aux_data: AuxiliaryData, file_path: str) -> None:
    dist, reg_term, params = aux_data
    numpy.savez(
        file_path,
        dist=dist,
        reg_term=reg_term,
        params=params,
    )


def auxdata_with_valid_to_npz(
    aux_data: AuxiliaryData,
    dist_valid: Float[Array, ""],
    regt_valid: Float[Array, ""],
    file_path: str,
) -> None:
    dist, reg_term, params = aux_data
    numpy.savez(
        file_path,
        dist=dist,
        reg_term=reg_term,
        params=params,
        dist_valid=dist_valid,
        reg_term_valid=regt_valid,
    )


def auxdata_from_npz(file_path: str) -> AuxiliaryData:
    arxiv = numpy.load(file_path)
    return arxiv["dist"], arxiv["reg_term"], arxiv["params"]


def auxdata_with_valid_from_npz(
    file_path: str,
) -> tuple[AuxiliaryData, numpy.ndarray, numpy.ndarray]:
    arxiv = numpy.load(file_path)
    return (
        (arxiv["dist"], arxiv["reg_term"], arxiv["params"]),
        arxiv["dist_valid"],
        arxiv["reg_term_valid"],
    )


def load_history(
    fldr_path: str, num_iter: int, valid_data: bool
) -> numpy.ndarray:
    vecs = []
    for i in range(num_iter):
        fpath = path_join(fldr_path, f"{i:05d}.npz")
        if valid_data:
            (d, rt, p), dv, rtv = auxdata_with_valid_from_npz(fpath)
        else:
            d, rt, p = auxdata_from_npz(fpath)
            dv, rtv = numpy.array(numpy.nan), numpy.array(numpy.nan)
        vec = numpy.concatenate(
            [d[None], rt[None], dv[None], rtv[None], p], axis=0
        )
        vecs.append(vec)
    return numpy.stack(vecs, axis=0)


def make_train_valid_split(
    pts: Float[Array, "n d"],
    wgts: Float[Array, " n"],
    pct_valid: float,
    prng_seed: int,
) -> tuple[
    tuple[Float[Array, "t d"], Float[Array, " t"]],
    tuple[Float[Array, "v d"], Float[Array, " v"]],
]:
    n_pts, _ = pts.shape
    n_valid = floor(n_pts * pct_valid)
    if n_valid == 0:
        raise ValueError("didnt get any points in validation set")
    key = jax.random.key(prng_seed)
    idx = jax.random.permutation(key, jnp.arange(0, n_pts))
    val_idx = idx[:n_valid]
    trn_idx = idx[n_valid:]
    return (pts[trn_idx, :], wgts[trn_idx]), (pts[val_idx, :], wgts[val_idx])


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

    elif method == AlignmentMethod.GRB:
        B, _ = grb.make_basis_kernel(means_mov, ctrl_pts, grbf_bandwidth)

        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = grb.unpack_params_2d(p)
                return grb.transform_basis(B, A, t, wgt)

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = grb.unpack_params_3d(p)
                return grb.transform_basis(B, A, t, wgt)

        else:
            raise ValueError("invalid # of dimensions")

    elif method == AlignmentMethod.TPS:
        B, _ = tps.make_basis_kernel(means_mov, ctrl_pts)

        if n_dim == 2:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = tps.unpack_params_2d(p)
                return tps.transform_basis(B, A, t, wgt)

        elif n_dim == 3:

            def transform_function(
                p: Float[Array, " p"],
            ) -> Float[Array, "m d"]:
                A, t, wgt = tps.unpack_params_3d(p)
                A = jax.lax.stop_gradient(A)
                t = jax.lax.stop_gradient(t)
                return tps.transform_basis(B, A, t, wgt)

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
    ctrl_pts: Float[Array, "c d"] | None = None,
    l2_scaling: float = 1.0,
    grbf_bandwidth: float = 1.0,
) -> LossFunctionWithAux:
    _, n_dim = means_ref.shape
    transform = _make_transform_function_spherical(
        means_mov,
        method,
        (ctrl_pts if ctrl_pts is not None else means_mov),
        grbf_bandwidth,
    )
    reg_type, reg_const = regularization
    if reg_type == Regularization.NONE:
        if metric == DistanceFunction.KLG:

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

        elif metric == DistanceFunction.KLV:

            def loss_function(
                p: Float[Array, " p"],
            ) -> tuple[Float[Array, ""], AuxiliaryData]:
                means_trans = transform(p)
                dist = kullback_leibler_gmm_approx_var_spherical(
                    means_ref,
                    wgts_ref,
                    means_trans,
                    wgts_mov,
                    var_ref,
                    var_mov,
                )
                jax.debug.print("{}", dist)
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
        if method == AlignmentMethod.GRB:
            if n_dim == 2:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = grb.unpack_params_2d(pars)
                    return wgt

            elif n_dim == 3:

                def get_weights(pars: Float[Array, " p"]) -> Float[Array, " w"]:
                    _, _, wgt = grb.unpack_params_3d(pars)
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

            if method == AlignmentMethod.TPS:
                _, K = tps.make_basis_kernel(
                    means_mov, ctrl_pts if ctrl_pts is not None else means_mov
                )
            elif method == AlignmentMethod.GRB:
                _, K = grb.make_basis_kernel(
                    means_mov,
                    ctrl_pts if ctrl_pts is not None else means_mov,
                    grbf_bandwidth,
                )
            else:
                raise ValueError(
                    "invalid alignment method for TPS regularization"
                )
            calculate_bending_energy = Partial(tps.tps_bending_energy, K)
            if metric == DistanceFunction.KLG:

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
                    e_bend = calculate_bending_energy(wgts)
                    return dist + reg_const * e_bend, (dist, e_bend, p)

            elif metric == DistanceFunction.KLV:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = kullback_leibler_gmm_approx_var_spherical(
                        means_ref,
                        wgts_ref,
                        means_trans,
                        wgts_mov,
                        var_ref,
                        var_mov,
                    )
                    e_bend = calculate_bending_energy(wgts)
                    return dist + reg_const * e_bend, (dist, e_bend, p)

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
                    e_bend = calculate_bending_energy(wgts)
                    return dist + reg_const * e_bend, (dist, e_bend, p)

            else:
                raise ValueError("invalid distance function for loss")
        elif reg_type == Regularization.RIDGE:
            if metric == DistanceFunction.KLG:

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
                    ridge_penalty = jnp.linalg.norm(wgts)
                    return dist + reg_const * ridge_penalty, (
                        dist,
                        ridge_penalty,
                        p,
                    )

            elif metric == DistanceFunction.KLV:

                def loss_function(
                    p: Float[Array, " p"],
                ) -> tuple[Float[Array, ""], AuxiliaryData]:
                    wgts = get_weights(p)
                    means_trans = transform(p)
                    dist = kullback_leibler_gmm_approx_var_spherical(
                        means_ref,
                        wgts_ref,
                        means_trans,
                        wgts_mov,
                        var_ref,
                        var_mov,
                    )
                    ridge_penalty = jnp.linalg.norm(wgts)
                    return dist + reg_const * ridge_penalty, (
                        dist,
                        ridge_penalty,
                        p,
                    )

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
                    ridge_penalty = jnp.linalg.norm(wgts)
                    return dist + reg_const * ridge_penalty, (
                        dist,
                        ridge_penalty,
                        p,
                    )

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
    l2_scaling: float = 1.0,
    grbf_bandwidth: float = 1.0,
    ctrl_pts: Float[Array, "c d"] | None = None,
    save_path: str | None = None,
    valid_pts: tuple[Float[Array, "v d"], Float[Array, " v"]] | None = None,
    optax_opt: Optimizer = Optimizer.LBFGS,
    **optim_kwargs,
) -> tuple[Float[Array, " p"], tuple[Float[Array, ""], Float[Array, ""], int]]:
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
        ctrl_pts,
        l2_scaling,
        grbf_bandwidth,
    )

    if valid_pts is None:

        def validation_callback(
            pars: Float[Array, " p"],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            return jnp.array(jnp.nan), jnp.array(jnp.nan)

    else:
        valid_func = _make_loss_function_spherical(
            valid_pts[0],
            valid_pts[1],
            means_mov,
            wgts_mov,
            var_ref,
            var_mov,
            method,
            metric,
            regularization,
            ctrl_pts,
            l2_scaling,
            grbf_bandwidth,
        )

        def validation_callback(
            pars: Float[Array, " p"],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            _, auxdata_valid = valid_func(pars)
            return auxdata_valid[0], auxdata_valid[1]

    def loss_func_noaux(pars: Float[Array, " p"]) -> Float[Array, ""]:
        loss_val, _ = loss_func(pars)
        return loss_val

    if optax_opt == Optimizer.LBFGS:
        optimizer = optax.lbfgs(**optim_kwargs)
    elif optax_opt == Optimizer.ADAM:
        optimizer = optax.adam(**optim_kwargs)
    elif optax_opt == Optimizer.SGD:
        optimizer = optax.sgd(**optim_kwargs)
    else:
        raise ValueError("invalid optimizer")
    opt_state = optimizer.init(init_pars)

    def keep_stepping(x: OptimizationState) -> Bool:
        _, _, grad_norm, curr_loss, num_iter = x
        grad_high = grad_norm > grad_tol
        loss_high = curr_loss > loss_tol
        grad_loss = jnp.logical_and(grad_high, loss_high)
        return jnp.logical_and(grad_loss, num_iter < max_iter)

    if save_path is not None:

        def save_step_data(
            aux_data: AuxiliaryData,
            valid_dist: Float[Array, ""],
            valid_regterm: Float[Array, ""],
            iter_num: int,
        ) -> None:
            fpath = path_join(save_path, f"{iter_num:05d}.npz")
            auxdata_with_valid_to_npz(
                aux_data, valid_dist, valid_regterm, fpath
            )

    else:

        def save_step_data(
            aux_data: AuxiliaryData,
            valid_dist: Float[Array, ""],
            valid_regterm: Float[Array, ""],
            iter_num: int,
        ) -> None:
            pass

    def take_step(x: OptimizationState) -> OptimizationState:
        params, opt_state, _, _, iter_num = x
        (loss, aux_data), grads = jax.value_and_grad(loss_func, has_aux=True)(
            params
        )
        valid_dist, valid_reg = validation_callback(params)
        io_callback(
            save_step_data, None, aux_data, valid_dist, valid_reg, iter_num
        )
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


def make_validation_function_hopt(
    means_valid: Float[Array, "v d"],
    wgts_valid: Float[Array, " v"],
    means_mov: Float[Array, "m d"],
    wgts_mov: Float[Array, " m"],
    method: AlignmentMethod,
    metric: DistanceFunction,
    regularization: RegularizationData,
    var_valid: float,
    var_mov: float,
    ctrl_pts: Float[Array, "c d"] | None,
    l2_scaling: float,
    grbf_bandwidth: float,
) -> LossFunctionWithAux:
    return _make_loss_function_spherical(
        means_valid,
        wgts_valid,
        means_mov,
        wgts_mov,
        var_valid,
        var_mov,
        method,
        metric,
        regularization,
        ctrl_pts,
        l2_scaling,
        grbf_bandwidth,
    )


def make_validation_function_corr(
    means_ref: Float[Array, "v d"],
    wgts_ref: Float[Array, " v"],
    means_mov: Float[Array, "m d"],
    wgts_mov: Float[Array, " m"],
    method: AlignmentMethod,
    metric: DistanceFunction,
    regularization: RegularizationData,
    var_valid: float,
    var_mov: float,
    ctrl_pts: Float[Array, "c d"] | None,
    ref_p2u_vec: Float[Array, " d"],
    mov_p2u_vec: Float[Array, " d"],
    ref_vol: Float[Array, "z y x"] | Float[Array, "y x"],
    mov_vol: Float[Array, "z y x"] | Float[Array, "y x"],
):
    raise NotImplementedError()
