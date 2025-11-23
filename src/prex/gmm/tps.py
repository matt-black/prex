"""Thin Plate Spline

Transform a GMM by a thin plate spline transformation.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ..util import gaussian_rbf_interpolate, sqdist


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
    eps = jnp.finfo(dist.dtype).eps
    return jnp.multiply(jnp.power(dist, pwr), jnp.log(dist + eps))


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


def make_basis_kernel(
    x: Float[Array, "n d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
) -> tuple[Float[Array, "n_ctrl n_ctrl"], Float[Array, "n_ctrl n_ctrl"]]:
    n_x, _ = x.shape
    n_ctrl, d = ctrl_pts.shape
    K = tps_rbf(ctrl_pts, ctrl_pts)
    U = tps_rbf(x, ctrl_pts)
    Px = jnp.c_[jnp.ones((n_x, 1)), x]
    Pc = jnp.c_[jnp.ones((n_ctrl, 1)), ctrl_pts]
    Uc, _, _ = jnp.linalg.svd(Pc)
    PP = Uc[:, d + 1 :]
    basis = jnp.c_[Px, jnp.dot(U, PP)]
    kernel = PP.T @ K @ PP
    return basis, kernel


def interpolate(
    pts: Float[Array, "m d"],
    ctrl_pts: Float[Array, "n d"],
    wgts: Float[Array, "n d"],
) -> Float[Array, "m d"]:
    m, d = pts.shape
    n, _ = ctrl_pts.shape
    k_new = tps_rbf(pts, ctrl_pts)
    P_new = jnp.c_[jnp.ones((m, 1)), pts]
    Pn = jnp.c_[jnp.ones((n, 1)), ctrl_pts]
    U, _, _ = jnp.linalg.svd(Pn)
    PP = U[:, d + 1 :]
    basis_new = jnp.c_[P_new, jnp.dot(k_new, PP)]
    return basis_new @ wgts


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


def transform_means(
    means: Float[Array, "n_comp d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
    rbf_wgts: Float[Array, "n_ctrl d"],
) -> Float[Array, "n_comp d"]:
    global_means = jnp.transpose(affine @ means.T) + translation[jnp.newaxis, :]
    psi = tps_rbf(means, ctrl_pts)
    local_deformation = psi @ rbf_wgts
    return global_means + local_deformation


def transform_basis(
    basis: Float[Array, "n_comp d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_ctrl-d d"],
) -> Float[Array, "n_comp d"]:
    par = jnp.concatenate(
        [translation[jnp.newaxis, :], affine, rbf_wgts], axis=0
    )
    return basis @ par


@Partial(
    jax.jit,
    static_argnums=(1,),
)
def unpack_params(flat_params: Float[Array, " p"], n_dim: int) -> tuple[
    Float[Array, "{n_dim} {n_dim}"],
    Float[Array, " {n_dim}"],
    Float[Array, "{n_ctrl} {n_dim}"],
]:
    affine = flat_params[: n_dim**2].reshape(n_dim, n_dim)
    translation = flat_params[n_dim**2 : n_dim**2 + n_dim]
    rbf_wgts = flat_params[n_dim**2 + n_dim :].reshape(-1, n_dim)
    return affine, translation, rbf_wgts


def pack_params(
    affine: Float[Array, "n_dim n_dim"],
    translation: Float[Array, " n_dim"],
    rbf_wgts: Float[Array, "n_ctrl n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate([affine.ravel(), translation, rbf_wgts.ravel()])


def unpack_params_2d(
    par: Float[Array, " p"],
) -> tuple[Float[Array, "2 2"], Float[Array, " 2"], Float[Array, "n_ctrl 2"]]:
    aff = par[:4].reshape(2, 2)
    trans = par[4:6]
    psi = par[6:].reshape(-1, 2)
    return aff, trans, psi


def unpack_params_3d(
    par: Float[Array, " p"],
) -> tuple[Float[Array, "3 3"], Float[Array, " 3"], Float[Array, "n_ctrl 3"]]:
    aff = par[:9].reshape(3, 3)
    trans = par[9:12]
    psi = par[12:].reshape(-1, 3)
    return aff, trans, psi


def tps_bending_energy(
    K: Float[Array, "n_ctrl n_ctrl"], wgts: Float[Array, "n_ctrl n_dim"]
) -> Float[Array, ""]:
    return jnp.trace(wgts.T @ K @ wgts)


def initialize_params(
    n_ctrl_pts: int,
    n_dim: int,
    init_aff: Float[Array, "{n_dim} {n_dim}"] | None,
    init_trans: Float[Array, " {n_dim}"] | None,
    epsilon: float = 1e-6,
) -> Float[Array, " p"]:
    if init_aff is None:
        init_aff = jnp.eye(n_dim)
    if init_trans is None:
        init_trans = jnp.zeros((n_dim,))
    init_wgt = jnp.ones((n_ctrl_pts - n_dim - 1, n_dim)) * epsilon
    return jnp.concatenate(
        [init_aff.ravel(), init_trans.ravel(), init_wgt.ravel()], axis=0
    )


def resample_volume(
    mov_vol: Float[Array, "z y x"],
    ref_grid_pts: Float[Array, "n 3"],
    ref_pix2unit: Callable[[Float[Array, " 3"]], Float[Array, " 3"]],
    mov_unit2pix: Callable[[Float[Array, " 3"]], Float[Array, " 3"]],
    out_shape: tuple[int, int, int],
    inv_pts: Float[Array, "n 3"],
    inv_vecs: Float[Array, "n 3"],
    interpolation_order: int = 0,
) -> Float[Array, "{out_shape[0]} {out_shape[1]} {out_shape[2]}"]:
    """Resample a volume using inverse thin plate spline transformation.

    Args:
        mov_vol: Moving volume to resample, shape (z, y, x)
        ref_grid_pts: Reference grid points in physical units, shape (n, 3)
        ref_pix2unit: Function that converts reference pixel coords to physical units
        mov_unit2pix: Function that converts physical units to moving pixel coords
        out_shape: Output volume shape (z, y, x)
        interpolation_mode: Interpolation mode for grid_sample ('linear' or 'nearest')
        inv_pts: Control points for inverse transform in physical units, shape (n, 3)
        inv_vecs: Inverse displacement vectors at control points, shape (n, 3)

    Returns:
        Resampled volume with shape out_shape
    """

    # Convert output grid to physical units (reference space)
    # Flatten and apply conversion function via vmap
    out_grid_flat = ref_grid_pts.reshape(-1, 3)
    out_grid_physical = jax.vmap(ref_pix2unit)(out_grid_flat)  # shape (-1, 3)

    # Interpolate inverse displacement vectors at output grid points
    inv_displacements = gaussian_rbf_interpolate(
        query_points=out_grid_physical,
        control_points=inv_pts,
        control_values=inv_vecs,
    )  # shape (-1, 3)

    # Apply inverse displacement to get coordinates in moving volume space (physical units)
    mov_coords_physical = out_grid_physical + inv_displacements

    # Convert to moving volume pixel coordinates via vmap
    mov_coords_pixels = jax.vmap(mov_unit2pix)(
        mov_coords_physical
    )  # shape (-1, 3)

    # Reshape to grid shape (z, y, x, 3)
    mov_coords_grid = mov_coords_pixels.reshape(
        out_shape[0], out_shape[1], out_shape[2], 3
    )

    # map_coordinates expects coordinates as (ndim, ...) not (..., ndim)
    # Transpose from (z, y, x, 3) to (3, z, y, x)
    coords_for_map = jnp.moveaxis(mov_coords_grid, -1, 0)

    # sample moving volume at computed coordinates
    return map_coordinates(
        mov_vol,
        coords_for_map,
        order=interpolation_order,
        mode="nearest",  # handle out-of-bounds: use nearest edge value
    )
