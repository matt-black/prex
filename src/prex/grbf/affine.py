"""Gaussian Radial Basis Functions (GRBF)

Transform a GMM by a global rigid transform (scaling + rotation) and local Gaussian radial basis functions.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

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


def transform_means(
    means: Float[Array, "n_comp d"],
    affine: Float[Array, "d d"],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_cent d"],
    rbf_centers: Float[Array, "n_cent d"],
    rbf_bandwidth: float,
) -> Float[Array, "n_comp d"]:
    psi = gaussian_rbf(means, rbf_centers, rbf_bandwidth)
    means_trans = (
        jnp.transpose(affine @ means.T)
        + translation[jnp.newaxis, :]
        + psi @ rbf_wgts
    )
    return means_trans


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


def unpack_params_2d(
    flat_params: Float[Array, " p"],
) -> tuple[Float[Array, "2 2"], Float[Array, " 2"], Float[Array, "n_comp 2"]]:
    affine = flat_params[:4].reshape(2, 2)
    trans = flat_params[4:6]
    rbf_wgts = flat_params[6:].reshape(-1, 2)
    return affine, trans, rbf_wgts


def unpack_params_3d(
    flat_params: Float[Array, " p"],
) -> tuple[Float[Array, "3 3"], Float[Array, " 3"], Float[Array, "n_comp 3"]]:
    affine = flat_params[:9].reshape(3, 3)
    trans = flat_params[9:12]
    rbf_wgts = flat_params[12:].reshape(-1, 3)
    return affine, trans, rbf_wgts
