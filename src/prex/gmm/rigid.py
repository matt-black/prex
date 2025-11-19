"""Rigid Transformations between GMMs

Transform a GMM by a global rigid transform (scaling + rotation + translation). Also use this transform model to register one GMM onto another.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .util import rotation_matrix_2d, rotation_matrix_3d


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


@Partial(jax.jit, static_argnums=(7,))
def transform_gmm_rotangles(
    means: Float[Array, "n_comp d"],
    covariances: Float[Array, "n_comp d d"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " d"],
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
    )


def transform_means(
    means: Float[Array, "n_comp d"],
    scale: Float[Array, ""],
    rotation: Float[Array, "d d"],
    translation: Float[Array, " d"],
) -> Float[Array, "n_comp d"]:
    return scale * means @ rotation.T + translation[jnp.newaxis, :]


def transform_means_rotangles3(
    means: Float[Array, "n_comp d"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " d"],
) -> Float[Array, "n_comp d"]:
    R = rotation_matrix_3d(alpha, beta, gamma)
    return transform_means(means, scale, R, translation)


def transform_means_rotangles2(
    means: Float[Array, "n_comp d"],
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    translation: Float[Array, " d"],
) -> Float[Array, "n_comp d"]:
    R = rotation_matrix_2d(alpha)
    return transform_means(means, scale, R, translation)


def unpack_params_2d(
    flat_params: Float[Array, " p"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_dim"],
]:
    scale = flat_params[0]
    alpha = flat_params[1]
    trans = flat_params[2:]
    return scale, alpha, trans


def unpack_params_3d(
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


def pack_params_2d(
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    trans: Float[Array, " n_dim"],
) -> Float[Array, " p"]:
    return jnp.concatenate(
        [
            scale[jnp.newaxis],
            alpha[jnp.newaxis],
            trans,
        ]
    )


def pack_params_3d(
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
