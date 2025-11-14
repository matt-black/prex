"""Gaussian Radial Basis Functions (GRBF)

Transform a GMM by a global rigid transform (scaling + rotation) and local Gaussian radial basis functions.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ..util import rotation_matrix_2d, rotation_matrix_3d
from ._util import gaussian_rbf


@Partial(jax.jit, static_argnums=(5,))
def jacobians(
    psi: Float[Array, "n_comp n_cent"],
    diffs: Float[Array, "n_comp n_cent d"],
    scale: Float[Array, ""],
    rotation: Float[Array, "d d"],
    rbf_weights: Float[Array, "n_cent 3"],
    bandwidth: float,
) -> Float[Array, "n_comp d d"]:
    jac_glob = scale * rotation
    # local contribution from each RBF center
    rbf_contribs = (
        jnp.negative(psi)[:, :, jnp.newaxis] * diffs / (bandwidth**2)
    )  # shape (n_comp, n_cent, d)
    jac_local = jnp.sum(
        rbf_weights[jnp.newaxis, :, :, jnp.newaxis]
        * rbf_contribs[:, :, jnp.newaxis, :],
        axis=1,
    )
    return jac_glob[jnp.newaxis, :, :] + jac_local


@Partial(jax.jit, static_argnums=(5,))
def jacobian_at_point(
    x: Float[Array, " d"],
    cents: Float[Array, "n_cent d"],
    rbf_wgts: Float[Array, "n_cent d"],
    rotation: Float[Array, "d d"],
    scale: Float[Array, ""],
    bandwidth: float,
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
    # jacobian of global part is scaling * rotation
    jac_global = scale * rotation

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
    return jac_global + jac_local


@Partial(jax.jit, static_argnums=(7,))
def transform_gmm(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale: Float[Array, ""],
    rotation: Float[Array, "d d"],
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
        scale: Scaling factor
        rotation: Rotation matrix, shape (d, d)
        translation: Translation vector, shape (d,)
        rbf_wgts: GRBF coefficients, shape (n_centers, d)
        rbf_centers: GRBF centers (fixed at target GMM means), shape (n_centers, d)
        rbf_bandwidth: GRBF bandwidth

    Returns:
        transformed_means: Transformed means, shape (n_components, d)
        transformed_covs: Transformed covariances, shape (n_components, d, d)
    """

    # transform means by applying the global rigid transform -> shape (n_components, d)
    global_means = scale * means @ rotation.T + translation[jnp.newaxis, :]

    # local deformation is weights time basis matrix
    # compute RBF basis matrix -> shape (n_components, n_centers)
    psi = gaussian_rbf(means, rbf_centers, rbf_bandwidth)
    # (n_components, n_centers) @ (n_centers, n_dim) -> (n_components, n_dims)
    local_deformation = psi @ rbf_wgts

    # total transformation is just global + local
    transformed_means = global_means + local_deformation

    # transform covariances using Jacobian (J @ Σ @ J.T)
    jac_at_point = Partial(
        jacobian_at_point,
        cents=rbf_centers,
        rbf_wgts=rbf_wgts,
        rotation=rotation,
        scale=scale,
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
        9,
        10,
    ),
)
def transform_gmm_rotangles(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " d"],
    rbf_wgts: Float[Array, "n_cent d"],
    rbf_centers: Float[Array, "n_cent d"],
    rbf_bandwidth: float,
    n_dim: int,
) -> tuple[Float[Array, "n_comp d"], Float[Array, "n_comp d d"]]:
    if n_dim == 2:
        R = rotation_matrix_2d(alpha)
    else:
        R = rotation_matrix_3d(alpha, beta, gamma)
    return transform_gmm(
        means,
        covariances,
        scale,
        R,
        translation,
        rbf_wgts,
        rbf_centers,
        rbf_bandwidth,
    )


def transform_means_2d(
    means: Float[Array, "n_comp 2"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    trans: Float[Array, " 2"],
    rbf_wgts: Float[Array, "n_cent 2"],
    rbf_centers: Float[Array, "n_cent 2"],
    rbf_bandwidth: float,
) -> Float[Array, "n_comp 2"]:
    global_means = jnp.add(
        scale * means @ rotation_matrix_2d(alpha).T, trans[jnp.newaxis, :]
    )
    # local deformation is weights time basis matrix
    # compute RBF basis matrix -> shape (n_components, n_centers)
    psi = gaussian_rbf(means, rbf_centers, rbf_bandwidth)
    # (n_components, n_centers) @ (n_centers, n_dim) -> (n_components, n_dims)
    local_deformation = psi @ rbf_wgts

    return global_means + local_deformation


def transform_means_3d(
    means: Float[Array, "n_comp 2"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    trans: Float[Array, " 2"],
    rbf_wgts: Float[Array, "n_cent 2"],
    rbf_centers: Float[Array, "n_cent 2"],
    rbf_bandwidth: float,
) -> Float[Array, "n_comp 2"]:
    global_means = jnp.add(
        scale * means @ rotation_matrix_3d(alpha, beta, gamma).T,
        trans[jnp.newaxis, :],
    )
    # local deformation is weights time basis matrix
    # compute RBF basis matrix -> shape (n_components, n_centers)
    psi = gaussian_rbf(means, rbf_centers, rbf_bandwidth)
    # (n_components, n_centers) @ (n_centers, n_dim) -> (n_components, n_dims)
    local_deformation = psi @ rbf_wgts

    return global_means + local_deformation


def unpack_params_2d(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " 2"],
    Float[Array, "c 2"],
]:
    scale = flat_params[0]
    alpha = flat_params[1]
    trans = flat_params[2:5]
    rbf_wgts = flat_params[5:].reshape(-1, 2)
    return scale, alpha, trans, rbf_wgts


def unpack_params_3d(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " 3"],
    Float[Array, "c 3"],
]:
    scale = flat_params[0]
    alpha = flat_params[1]
    beta = flat_params[2]
    gamma = flat_params[3]
    trans = flat_params[4:7]
    rbf_wgts = flat_params[7:].reshape(-1, 3)
    return (scale, alpha, beta, gamma, trans, rbf_wgts)


def pack_params_2d(
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    trans: Float[Array, " 2"],
    rbf_wgts: Float[Array, "c 2"],
) -> Float[Array, " p"]:
    return jnp.concatenate(
        [
            scale[jnp.newaxis],
            alpha[jnp.newaxis],
            trans,
            rbf_wgts.ravel(),
        ]
    )


def pack_params_3d(
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    trans: Float[Array, " n_dim"],
    rbf_wgts: Float[Array, "n_cent n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate(
        [
            scale[jnp.newaxis],
            alpha[jnp.newaxis],
            beta[jnp.newaxis],
            gamma[jnp.newaxis],
            trans,
            rbf_wgts.ravel(),
        ]
    )
