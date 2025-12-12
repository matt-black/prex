"""Analytical gradients for variational KL divergence under rigid transformations.

This module implements analytical gradients for the variational KL divergence
approximation between two Gaussian mixture models (GMMs) where one GMM is
transformed by rigid parameters (scale, rotation, translation).
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ...util import rotation_matrix_2d, rotation_matrix_3d
from ._util import (
    compute_overlap_weights,
    compute_self_overlap_weights,
    compute_weights_alpha,
)


def gradient_translation(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q_trans: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    alpha_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, " d"]:
    """Compute gradient of variational KL divergence w.r.t. translation vector.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q_trans: Transformed moving GMM means
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        alpha_ij: Pre-computed weighting coefficients (optional)

    Returns:
        Gradient vector w.r.t. translation, shape (d,)
    """
    if alpha_ij is None:
        alpha_ij, _, _ = compute_weights_alpha(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Gradient: (1/var_q) * sum_ij alpha_ij * delta_ij
    # Shape: (d,)
    grad_t = (1.0 / var_q) * jnp.sum(
        alpha_ij[:, :, jnp.newaxis] * delta_ij, axis=(0, 1)
    )

    return grad_t


def gradient_scale(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    means_q_trans: Float[Array, "m d"],
    rotation: Float[Array, "d d"],
    var_p: float,
    var_q: float,
    n_dim: int,
    alpha_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, ""]:
    """Compute gradient of variational KL divergence w.r.t. scale parameter.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        means_q_trans: Transformed moving GMM means
        rotation: Rotation matrix
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        alpha_ij: Pre-computed weighting coefficients (optional)

    Returns:
        Gradient scalar w.r.t. scale
    """
    if alpha_ij is None:
        wgts_q = jnp.ones(means_q.shape[0]) / means_q.shape[0]  # Placeholder
        alpha_ij, _, _ = compute_weights_alpha(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Compute R * mu_q^j for all j
    # Shape: (m, d)
    R_mu_q = jax.vmap(lambda mu: rotation @ mu)(means_q)

    # Gradient: (1/var_q) * sum_ij alpha_ij * delta_ij^T * R * mu_q^j
    # Shape: scalar
    grad_s = (1.0 / var_q) * jnp.sum(
        alpha_ij * jnp.sum(delta_ij * R_mu_q[jnp.newaxis, :, :], axis=2)
    )

    return grad_s


def gradient_rotation_matrix(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    means_q_trans: Float[Array, "m d"],
    scale: Float[Array, ""],
    var_p: float,
    var_q: float,
    n_dim: int,
    alpha_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, "d d"]:
    """Compute gradient of variational KL divergence w.r.t. rotation matrix.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        means_q_trans: Transformed moving GMM means
        scale: Scale parameter
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        alpha_ij: Pre-computed weighting coefficients (optional)

    Returns:
        Gradient matrix w.r.t. rotation, shape (d, d)
    """
    if alpha_ij is None:
        wgts_q = jnp.ones(means_q.shape[0]) / means_q.shape[0]  # Placeholder
        alpha_ij, _, _ = compute_weights_alpha(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Gradient: (s/var_q) * sum_ij alpha_ij * delta_ij * (mu_q^j)^T
    # This is a sum of outer products
    # Shape: (d, d)
    grad_R = (scale / var_q) * jnp.sum(
        alpha_ij[:, :, jnp.newaxis, jnp.newaxis]
        * delta_ij[:, :, :, jnp.newaxis]
        * means_q[jnp.newaxis, :, jnp.newaxis, :],
        axis=(0, 1),
    )

    return grad_R


def gradient_rotation_angles_2d(
    grad_R: Float[Array, "2 2"],
    alpha: Float[Array, ""],
) -> Float[Array, ""]:
    """Convert rotation matrix gradient to angle gradient for 2D.

    Uses the chain rule: dL/d(alpha) = tr((dL/dR)^T * dR/d(alpha))

    Args:
        grad_R: Gradient w.r.t. rotation matrix, shape (2, 2)
        alpha: Current rotation angle

    Returns:
        Gradient w.r.t. angle alpha
    """
    # dR/d(alpha) for 2D rotation matrix
    # R = [[cos(alpha), -sin(alpha)],
    #      [sin(alpha),  cos(alpha)]]
    # dR/d(alpha) = [[-sin(alpha), -cos(alpha)],
    #                [ cos(alpha), -sin(alpha)]]
    dR_dalpha = jnp.array(
        [[-jnp.sin(alpha), -jnp.cos(alpha)], [jnp.cos(alpha), -jnp.sin(alpha)]]
    )

    # Gradient: tr(grad_R^T * dR/d(alpha))
    grad_alpha = jnp.trace(grad_R.T @ dR_dalpha)

    return grad_alpha


def gradient_rotation_angles_3d(
    grad_R: Float[Array, "3 3"],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Convert rotation matrix gradient to Euler angle gradients for 3D.

    Uses the chain rule: dL/d(angle) = tr((dL/dR)^T * dR/d(angle))

    The rotation matrix is R = Rz(gamma) @ Ry(beta) @ Rx(alpha)
    where alpha is rotation around x-axis, beta around y-axis, gamma around z-axis.

    Args:
        grad_R: Gradient w.r.t. rotation matrix, shape (3, 3)
        alpha: Current rotation angle around x-axis
        beta: Current rotation angle around y-axis
        gamma: Current rotation angle around z-axis

    Returns:
        Tuple of (grad_alpha, grad_beta, grad_gamma)
    """
    # Compute individual rotation matrices
    # Rx(alpha) - rotation around x-axis
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    Rx = jnp.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    # Ry(beta) - rotation around y-axis
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    Ry = jnp.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    # Rz(gamma) - rotation around z-axis
    cg, sg = jnp.cos(gamma), jnp.sin(gamma)
    Rz = jnp.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    # Derivatives of individual rotation matrices
    dRx_dalpha = jnp.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])

    dRy_dbeta = jnp.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])

    dRz_dgamma = jnp.array([[-sg, -cg, 0], [cg, -sg, 0], [0, 0, 0]])

    # Chain rule for composite rotation R = Rz @ Ry @ Rx
    # dR/d(alpha) = Rz @ Ry @ dRx/d(alpha)
    dR_dalpha = Rz @ Ry @ dRx_dalpha

    # dR/d(beta) = Rz @ dRy/d(beta) @ Rx
    dR_dbeta = Rz @ dRy_dbeta @ Rx

    # dR/d(gamma) = dRz/d(gamma) @ Ry @ Rx
    dR_dgamma = dRz_dgamma @ Ry @ Rx

    # Compute gradients using trace formula
    grad_alpha = jnp.trace(grad_R.T @ dR_dalpha)
    grad_beta = jnp.trace(grad_R.T @ dR_dbeta)
    grad_gamma = jnp.trace(grad_R.T @ dR_dgamma)

    return grad_alpha, grad_beta, grad_gamma


@Partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def gradient_all_2d_klv(
    means_p: Float[Array, "n 2"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 2"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    translation: Float[Array, " 2"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " 2"]]:
    """Compute all gradients for 2D rigid transformation.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality (should be 2)
        scale: Current scale parameter
        alpha: Current rotation angle
        translation: Current translation vector

    Returns:
        Tuple of (grad_scale, grad_alpha, grad_translation)
    """
    # Compute rotation matrix
    rotation = rotation_matrix_2d(alpha)

    # Transform moving means
    means_q_trans = (
        scale * jax.vmap(lambda mu: rotation @ mu)(means_q)
        + translation[jnp.newaxis, :]
    )

    # Compute alpha_ij once
    alpha_ij, _, _ = compute_weights_alpha(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute gradients
    grad_t = gradient_translation(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim, alpha_ij
    )

    grad_s = gradient_scale(
        means_p,
        wgts_p,
        means_q,
        means_q_trans,
        rotation,
        var_p,
        var_q,
        n_dim,
        alpha_ij,
    )

    grad_R = gradient_rotation_matrix(
        means_p,
        wgts_p,
        means_q,
        means_q_trans,
        scale,
        var_p,
        var_q,
        n_dim,
        alpha_ij,
    )

    grad_alpha = gradient_rotation_angles_2d(grad_R, alpha)

    return grad_s, grad_alpha, grad_t


@Partial(jax.jit, static_argnums=(6,))
def gradient_all_3d_klv(
    means_p: Float[Array, "n 3"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 3"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " 3"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " 3"],
]:
    """Compute all gradients for 3D rigid transformation.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality (should be 3)
        scale: Current scale parameter
        alpha: Current rotation angle around z-axis
        beta: Current rotation angle around y-axis
        gamma: Current rotation angle around x-axis
        translation: Current translation vector

    Returns:
        Tuple of (grad_scale, grad_alpha, grad_beta, grad_gamma, grad_translation)
    """
    # Compute rotation matrix
    rotation = rotation_matrix_3d(alpha, beta, gamma)

    # Transform moving means
    means_q_trans = (
        scale * jax.vmap(lambda mu: rotation @ mu)(means_q)
        + translation[jnp.newaxis, :]
    )

    # Compute alpha_ij once
    alpha_ij, _, _ = compute_weights_alpha(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute gradients
    grad_t = gradient_translation(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim, alpha_ij
    )

    grad_s = gradient_scale(
        means_p,
        wgts_p,
        means_q,
        means_q_trans,
        rotation,
        var_p,
        var_q,
        n_dim,
        alpha_ij,
    )

    grad_R = gradient_rotation_matrix(
        means_p,
        wgts_p,
        means_q,
        means_q_trans,
        scale,
        var_p,
        var_q,
        n_dim,
        alpha_ij,
    )

    grad_alpha, grad_beta, grad_gamma = gradient_rotation_angles_3d(
        grad_R, alpha, beta, gamma
    )

    return grad_s, grad_alpha, grad_beta, grad_gamma, grad_t


# ============================================================================
# L2 Distance Gradients
# ============================================================================


def gradient_translation_l2(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q_trans: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    overlap_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, " d"]:
    """Compute gradient of L2 distance w.r.t. translation vector.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q_trans: Transformed moving GMM means
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        overlap_ij: Pre-computed overlap weights (optional)

    Returns:
        Gradient vector w.r.t. translation, shape (d,)
    """
    if overlap_ij is None:
        overlap_ij = compute_overlap_weights(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Gradient: (-2 / (var_p + var_q)) * sum_ij overlap_ij * delta_ij
    # Shape: (d,)
    factor = 2.0 / (var_p + var_q)
    grad_t = factor * jnp.sum(
        overlap_ij[:, :, jnp.newaxis] * delta_ij, axis=(0, 1)
    )

    return grad_t


def gradient_scale_l2(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    means_q_trans: Float[Array, "m d"],
    rotation: Float[Array, "d d"],
    var_p: float,
    var_q: float,
    n_dim: int,
    overlap_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, ""]:
    """Compute gradient of L2 distance w.r.t. scale parameter.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        means_q_trans: Transformed moving GMM means
        rotation: Rotation matrix
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        overlap_ij: Pre-computed overlap weights (optional)

    Returns:
        Gradient scalar w.r.t. scale
    """
    if overlap_ij is None:
        overlap_ij = compute_overlap_weights(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Compute R * mu_q^j for all j
    # Shape: (m, d)
    R_mu_q = jax.vmap(lambda mu: rotation @ mu)(means_q)

    # Gradient: (-2 / (var_p + var_q)) * sum_ij overlap_ij * delta_ij^T * R * mu_q^j
    # Shape: scalar
    factor = 2.0 / (var_p + var_q)
    grad_cross = factor * jnp.sum(
        overlap_ij * jnp.sum(delta_ij * R_mu_q[jnp.newaxis, :, :], axis=2)
    )

    # Self-energy gradient
    # d(E_self)/ds = -s/var_q * sum_jk O_jk * ||mu_q^j - mu_q^k||^2
    overlap_jk = compute_self_overlap_weights(
        means_q_trans, wgts_q, var_q, n_dim
    )

    # general formula:
    # d(E_self)/ds = sum_j d(E_self)/d(mu_q_trans^j) * d(mu_q_trans^j)/ds
    # d(E_self)/d(mu_q_trans^j) = -1/var_q * sum_k O_jk * (mu_q_trans^j - mu_q_trans^k)
    # d(mu_q_trans^j)/ds = R * mu_q^j

    grad_self_per_point = (-1.0 / var_q) * jnp.sum(
        overlap_jk[:, :, jnp.newaxis]
        * (means_q_trans[:, jnp.newaxis, :] - means_q_trans[jnp.newaxis, :, :]),
        axis=1,
    )
    grad_self = jnp.sum(grad_self_per_point * R_mu_q)

    return grad_self + grad_cross


def gradient_rotation_matrix_l2(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    means_q_trans: Float[Array, "m d"],
    scale: Float[Array, ""],
    var_p: float,
    var_q: float,
    n_dim: int,
    overlap_ij: Float[Array, "n m"] | None = None,
) -> Float[Array, "d d"]:
    """Compute gradient of L2 distance w.r.t. rotation matrix.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        means_q_trans: Transformed moving GMM means
        scale: Scale parameter
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        overlap_ij: Pre-computed overlap weights (optional)

    Returns:
        Gradient matrix w.r.t. rotation, shape (d, d)
    """
    if overlap_ij is None:
        overlap_ij = compute_overlap_weights(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
        )

    # Compute delta_ij = mu_q_trans^j - mu_p^i for all i, j
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Gradient: (-2 * s / (var_p + var_q)) * sum_ij overlap_ij * delta_ij * (mu_q^j)^T
    # This is a sum of outer products
    # Shape: (d, d)
    factor = 2.0 * scale / (var_p + var_q)
    grad_R = factor * jnp.sum(
        overlap_ij[:, :, jnp.newaxis, jnp.newaxis]
        * delta_ij[:, :, :, jnp.newaxis]
        * means_q[jnp.newaxis, :, jnp.newaxis, :],
        axis=(0, 1),
    )

    return grad_R


@Partial(jax.jit, static_argnums=(6,))
def gradient_all_2d_l2(
    means_p: Float[Array, "n 2"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 2"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    translation: Float[Array, " 2"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, " 2"]]:
    """Compute all gradients for 2D rigid transformation using L2 distance.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality (should be 2)
        scale: Current scale parameter
        alpha: Current rotation angle
        translation: Current translation vector

    Returns:
        Tuple of (grad_scale, grad_alpha, grad_translation)
    """
    # Compute rotation matrix
    rotation = rotation_matrix_2d(alpha)

    # Transform moving means
    means_q_trans = (
        scale * jax.vmap(lambda mu: rotation @ mu)(means_q)
        + translation[jnp.newaxis, :]
    )

    # Compute overlap_ij once
    overlap_ij = compute_overlap_weights(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute gradients
    grad_t = gradient_translation_l2(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim, overlap_ij
    )

    grad_s = gradient_scale_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        means_q_trans,
        rotation,
        var_p,
        var_q,
        n_dim,
        overlap_ij,
    )

    grad_R = gradient_rotation_matrix_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        means_q_trans,
        scale,
        var_p,
        var_q,
        n_dim,
        overlap_ij,
    )

    grad_alpha = gradient_rotation_angles_2d(grad_R, alpha)

    return grad_s, grad_alpha, grad_t


@Partial(jax.jit, static_argnums=(6,))
def gradient_all_3d_l2(
    means_p: Float[Array, "n 3"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 3"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
    scale: Float[Array, ""],
    alpha: Float[Array, ""],
    beta: Float[Array, ""],
    gamma: Float[Array, ""],
    translation: Float[Array, " 3"],
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " 3"],
]:
    """Compute all gradients for 3D rigid transformation using L2 distance.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality (should be 3)
        scale: Current scale parameter
        alpha: Current rotation angle around z-axis
        beta: Current rotation angle around y-axis
        gamma: Current rotation angle around x-axis
        translation: Current translation vector

    Returns:
        Tuple of (grad_scale, grad_alpha, grad_beta, grad_gamma, grad_translation)
    """
    # Compute rotation matrix
    rotation = rotation_matrix_3d(alpha, beta, gamma)

    # Transform moving means
    means_q_trans = (
        scale * jax.vmap(lambda mu: rotation @ mu)(means_q)
        + translation[jnp.newaxis, :]
    )

    # Compute overlap_ij once
    overlap_ij = compute_overlap_weights(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute gradients
    grad_t = gradient_translation_l2(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim, overlap_ij
    )

    grad_s = gradient_scale_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        means_q_trans,
        rotation,
        var_p,
        var_q,
        n_dim,
        overlap_ij,
    )

    grad_R = gradient_rotation_matrix_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        means_q_trans,
        scale,
        var_p,
        var_q,
        n_dim,
        overlap_ij,
    )

    grad_alpha, grad_beta, grad_gamma = gradient_rotation_angles_3d(
        grad_R, alpha, beta, gamma
    )

    return grad_s, grad_alpha, grad_beta, grad_gamma, grad_t
