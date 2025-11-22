"""Affine Transformations between GMMs

Transform a GMM by a global affine transform and translation. Also use this transform model to register one GMM onto another. Note that this assumes isotropic scaling.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ..util import (
    rotation_matrix_2d,
    rotation_matrix_3d,
    shear_matrix_2d,
    shear_matrix_3d,
)


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


def transform_means(
    means: Float[Array, "n_comp d"],
    matrix: Float[Array, "d d"],
    translation: Float[Array, " d"],
) -> Float[Array, "n_comp d"]:
    return jnp.add(jnp.transpose(matrix @ means.T), translation[jnp.newaxis, :])


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


def transform_means_rotangles2(
    means: Float[Array, "n_comp 2"],
    scale_x: Float[Array, ""],
    scale_y: Float[Array, ""],
    alpha: Float[Array, ""],
    k: Float[Array, ""],
    ell: Float[Array, ""],
    translation: Float[Array, " 2"],
) -> Float[Array, "n_comp 2"]:
    S = jnp.diag(jnp.array([scale_x, scale_y]))
    R = rotation_matrix_2d(alpha)
    Sh = shear_matrix_2d(k, ell)
    matrix = S @ R @ Sh
    return transform_means(
        means,
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


def transform_means_rotangles3(
    means: Float[Array, "n_comp 3"],
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
) -> Float[Array, "n_comp 3"]:
    S = jnp.diag(jnp.array([scale_x, scale_y, scale_z]))
    R = rotation_matrix_3d(alpha, beta, gamma)
    Sh = shear_matrix_3d(k_xy, k_xz, k_yx, k_yz, k_zx, k_zy)
    matrix = S @ R @ Sh
    return transform_means(
        means,
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


@Partial(jax.jit, static_argnums=(14,))
def transform_means_rotangles(
    means: Float[Array, "n_comp d"],
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
    n_dim: int,
) -> Float[Array, "n_comp d"]:
    if n_dim == 2:
        return transform_means_rotangles2(
            means, scale_x, scale_y, alpha, k_xy, k_xz, translation
        )
    else:
        return transform_means_rotangles3(
            means,
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


def unpack_params_2d(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_dim"],
]:
    scale_x = flat_params[0]
    scale_y = flat_params[1]
    alpha = flat_params[2]
    k = flat_params[3]
    ell = flat_params[4]
    trans = flat_params[5:]
    return (scale_x, scale_y, alpha, k, ell, trans)


def unpack_params_3d(
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


def pack_params_3d(
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
