"""Gaussian Radial Basis Functions

Transform a GMM by a gaussian radial basis
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .util import sqdist


@Partial(jax.jit, static_argnums=(2,))
def gaussian_rbf(
    x: Float[Array, "n_pts d"],
    cents: Float[Array, "n_cent d"],
    bandwidth: float,
) -> Float[Array, "n_pts n_cent"]:
    return jnp.exp(-sqdist(x, cents) / (2 * bandwidth**2))


@Partial(jax.jit, static_argnums=(1,))
def make_basis_kernel(
    x: Float[Array, "n d"],
    ctrl_pts: Float[Array, "n_ctrl d"],
    bandwidth: float,
) -> tuple[Float[Array, "n_ctrl n_ctrl"], Float[Array, "n_ctrl n_ctrl"]]:
    n_x, _ = x.shape
    n_ctrl, d = ctrl_pts.shape
    K = gaussian_rbf(ctrl_pts, ctrl_pts, bandwidth)
    U = gaussian_rbf(x, ctrl_pts, bandwidth)
    Pc = jnp.c_[jnp.ones((n_ctrl, 1)), ctrl_pts]
    Px = jnp.c_[jnp.ones((n_x, 1)), x]
    Uc, _, _ = jnp.linalg.svd(Pc)
    PP = Uc[:, d + 1 :]
    basis = jnp.c_[Px, jnp.dot(U, PP)]
    kernel = PP.T @ K @ PP
    return basis, kernel


@Partial(jax.jit, static_argnums=(3,))
def interpolate(
    pts: Float[Array, "m d"],
    ctrl_pts: Float[Array, "n d"],
    wgts: Float[Array, "n d"],
    bandwidth: float,
) -> Float[Array, "m d"]:
    m, d = pts.shape
    n, _ = ctrl_pts.shape
    k_new = gaussian_rbf(pts, ctrl_pts, bandwidth)
    P_new = jnp.c_[jnp.ones((m, 1)), pts]
    Pn = jnp.c_[jnp.ones((n, 1)), ctrl_pts]
    U, _, _ = jnp.linalg.svd(Pn)
    PP = U[:, d + 1 :]
    basis_new = jnp.c_[P_new, jnp.dot(k_new, PP)]
    return basis_new @ wgts


def transform_basis(
    basis: Float[Array, "n_comp d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_ctrl-d d"],
) -> Float[Array, "n_comp d"]:
    par = jnp.concatenate(
        [affine, translation[jnp.newaxis, :], rbf_wgts], axis=0
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


def bending_energy(
    K: Float[Array, "n_ctrl n_ctrl"], wgts: Float[Array, "n_ctrl n_dim"]
) -> Float[Array, ""]:
    return jnp.trace(wgts.T @ K @ wgts)


def initialize_params(
    n_ctrl_pts: int,
    n_dim: int,
    init_aff: Float[Array, "{n_dim} {n_dim}"] | None,
    init_trans: Float[Array, " {n_dim}"] | None,
) -> Float[Array, " p"]:
    if init_aff is None:
        init_aff = jnp.eye(n_dim)
    if init_trans is None:
        init_trans = jnp.zeros((n_dim,))
    init_wgt = jnp.zeros((n_ctrl_pts - n_dim - 1, n_dim))
    return jnp.concatenate(
        [init_aff.ravel(), init_trans.ravel(), init_wgt.ravel()], axis=0
    )
