"""Rigid Transformations between GMMs

Transform a GMM by a global rigid transform (scaling + rotation + translation). Also use this transform model to register one GMM onto another.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ..util import rotation_matrix_2d, rotation_matrix_3d


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


def resample_volume(
    mov_vol: Float[Array, "z y x"],
    ref_grid_pts: Float[Array, "n 3"],
    ref_pix2unit: Callable[[Float[Array, " 3"]], Float[Array, " 3"]],
    mov_unit2pix: Callable[[Float[Array, " 3"]], Float[Array, " 3"]],
    out_shape: tuple[int, int, int],
    interpolation_mode: str,
    scale: Float[Array, ""],
    rotation: Float[Array, "3 3"],
    translation: Float[Array, " 3"],
) -> Float[Array, "{out_shape[0]} {out_shape[1]} {out_shape[2]}"]:
    """Resample a volume using inverse rigid transformation.

    Args:
        mov_vol: Moving volume to resample, shape (z, y, x)
        ref_grid_pts: Reference grid points in physical units, shape (n, 3)
        ref_pix2unit: Function that converts reference pixel coords to physical units
        mov_unit2pix: Function that converts physical units to moving pixel coords
        out_shape: Output volume shape (z, y, x)
        interpolation_mode: Interpolation mode ('linear' or 'nearest')
        scale: Forward transform scale factor
        rotation: Forward transform rotation matrix, shape (3, 3)
        translation: Forward transform translation vector, shape (3,)

    Returns:
        Resampled volume with shape out_shape
    """

    # Convert output grid to physical units (reference space)
    # Flatten and apply conversion function via vmap
    out_grid_flat = ref_grid_pts.reshape(-1, 3)
    out_grid_physical = jax.vmap(ref_pix2unit)(out_grid_flat)  # shape (-1, 3)

    # Compute inverse rigid transform
    # For forward: y = s * R @ x + t
    # Inverse: x = (1/s) * R^T @ (y - t)
    inv_scale = 1.0 / scale
    inv_rotation = rotation.T
    inv_translation = -inv_rotation @ translation

    # Apply inverse rigid transform to get coordinates in moving volume space (physical units)
    # mov_coords = inv_scale * (inv_rotation @ (ref_coords - translation))
    #            = inv_scale * (inv_rotation @ ref_coords + inv_translation)
    def apply_inverse_transform(
        ref_coord: Float[Array, " 3"],
    ) -> Float[Array, " 3"]:
        return inv_scale * (inv_rotation @ ref_coord + inv_translation)

    mov_coords_physical = jax.vmap(apply_inverse_transform)(
        out_grid_physical
    )  # shape (-1, 3)

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

    # Sample the moving volume at the computed coordinates
    # order: 0 = nearest, 1 = linear
    order = 1 if interpolation_mode == "linear" else 0

    return map_coordinates(
        mov_vol,
        coords_for_map,
        order=order,
        mode="nearest",  # How to handle out-of-bounds: use nearest edge value
    )
