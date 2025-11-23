"""Analytical gradients for variational KL divergence under TPS transformations.

This module implements analytical gradients for the variational KL divergence
approximation between two Gaussian mixture models (GMMs) where one GMM is
transformed by thin plate spline (TPS) parameters.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ._util import compute_kl_divergence_spherical, compute_weights_alpha


@Partial(jax.jit, static_argnums=(7,))
def gradient_all_params_klv(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    n_dim: int,
    params: Float[Array, " p"],
) -> Float[Array, " p"]:
    """Compute gradients for all TPS parameters using basis representation.

    The TPS transformation is: mu_trans = basis @ theta
    where theta contains [affine, translation, rbf_weights] stacked for each dimension.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel, shape (m, p_per_dim)
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        params: Current TPS parameters, shape (p,) where p = n_dim * p_per_dim

    Returns:
        Gradient vector w.r.t. all parameters, shape (p,)
    """
    m, p_per_dim = basis.shape

    # Reshape parameters: (n_dim, p_per_dim)
    params_reshaped = params.reshape(n_dim, p_per_dim)

    # Transform moving means: means_trans[j, ell] = basis[j, :] @ params_reshaped[ell, :]
    means_q_trans = basis @ params_reshaped.T  # shape (m, n_dim)

    # Compute weighting coefficients alpha_ij
    alpha_ij, _, _ = compute_weights_alpha(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute residuals: delta_ij = means_q_trans[j] - means_p[i]
    # Shape: (n, m, d)
    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]

    # Flatten alpha_ij and delta_ij for efficient computation
    # alpha_flat: (n*m,)
    alpha_flat = alpha_ij.ravel()

    # Compute gradient for each dimension
    gradients = []
    for ell in range(n_dim):
        # delta_flat_ell: (n*m,) - residuals in dimension ell
        delta_flat_ell = delta_ij[:, :, ell].ravel()

        # Weighted residuals: alpha * delta
        weighted_residuals = alpha_flat * delta_flat_ell

        # Reshape to (n, m) for matrix multiplication
        weighted_residuals_matrix = weighted_residuals.reshape(alpha_ij.shape)

        # Sum over reference points (axis 0) to get (m,) vector
        weighted_residuals_per_moving = jnp.sum(
            weighted_residuals_matrix, axis=0
        )

        # Gradient: (1/var_q) * basis^T @ weighted_residuals
        # Shape: (p_per_dim,)
        grad_ell = (1.0 / var_q) * (basis.T @ weighted_residuals_per_moving)
        gradients.append(grad_ell)

    # Stack gradients for all dimensions and flatten
    # Shape: (n_dim, p_per_dim) -> (n_dim * p_per_dim,)
    return jnp.stack(gradients, axis=0).ravel()


def gradient_all_2d_klv(
    means_p: Float[Array, "n 2"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 2"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    params: Float[Array, " p"],
) -> tuple[Float[Array, "2 2"], Float[Array, " 2"], Float[Array, "n_ctrl 2"]]:
    """Compute all gradients for 2D TPS transformation.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        params: Current TPS parameters (affine, translation, rbf_weights packed)

    Returns:
        Tuple of (grad_affine, grad_translation, grad_rbf_weights)
    """
    # Compute full gradient vector (n_dim=2 is passed as literal)
    grad_params = gradient_all_params_klv(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, 2, params
    )

    # Unpack gradients
    # TPS parameters are packed as: [t_1, t_2, A_11, A_12, A_21, A_22, w_1, w_2, ...]
    # For each dimension: [t_ell, A_:ell, rbf_weights_:ell]
    # Basis is: [1, x, y, rbf_basis] so first column is for translation

    p_per_dim = basis.shape[1]
    grad_reshaped = grad_params.reshape(2, p_per_dim)

    # Extract translation gradients (first element per dimension)
    grad_translation = grad_reshaped[:, 0]  # shape (2,)

    # Extract affine gradients (elements 1:3 per dimension, i.e., for x and y)
    grad_affine = grad_reshaped[:, 1:3].T  # shape (2, 2)

    # Extract RBF weight gradients (remaining elements)
    grad_rbf_weights = grad_reshaped[:, 3:].T  # shape (n_ctrl-3, 2)

    return grad_affine, grad_translation, grad_rbf_weights


def gradient_all_3d_klv(
    means_p: Float[Array, "n 3"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 3"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    params: Float[Array, " p"],
) -> tuple[Float[Array, "3 3"], Float[Array, " 3"], Float[Array, "n_ctrl 3"]]:
    """Compute all gradients for 3D TPS transformation.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        params: Current TPS parameters (affine, translation, rbf_weights packed)

    Returns:
        Tuple of (grad_affine, grad_translation, grad_rbf_weights)
    """
    # Compute full gradient vector (n_dim=3 is passed as literal)
    grad_params = gradient_all_params_klv(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, 3, params
    )

    # Unpack gradients
    # TPS parameters are packed as: [t_1, t_2, t_3, A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33, w_1, w_2, w_3, ...]
    # For each dimension: [t_ell, A_:ell, rbf_weights_:ell]
    # Basis is: [1, x, y, z, rbf_basis] so first column is for translation

    p_per_dim = basis.shape[1]
    grad_reshaped = grad_params.reshape(3, p_per_dim)

    # Extract translation gradients (first element per dimension)
    grad_translation = grad_reshaped[:, 0]  # shape (3,)

    # Extract affine gradients (elements 1:4 per dimension, i.e., for x, y, z)
    grad_affine = grad_reshaped[:, 1:4].T  # shape (3, 3)

    # Extract RBF weight gradients (remaining elements)
    grad_rbf_weights = grad_reshaped[:, 4:].T  # shape (n_ctrl-4, 3)

    return grad_affine, grad_translation, grad_rbf_weights


# ============================================================================
# KLG Divergence Gradients
# ============================================================================


def compute_argmin_indices(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q_trans: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> Float[Array, " n"]:
    """Compute argmin indices for KLG approximation.

    For each reference component i, find the moving component j that minimizes:
    D_KL(N(mu_p[i], var_p) || N(mu_q_trans[j], var_q)) + log(w_p[i] / w_q[j])

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q_trans: Transformed moving GMM means
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality

    Returns:
        Array of argmin indices, shape (n,), where argmin_indices[i] = j*(i)
    """

    # Compute pairwise KL divergences: shape (n, m)
    def compute_kl_for_pair(mu_p_i, wgt_p_i):
        kl_vals = jax.vmap(
            lambda mu_q_j, wgt_q_j: (
                compute_kl_divergence_spherical(
                    mu_p_i, mu_q_j, var_p, var_q, n_dim
                )
                + jnp.log(wgt_p_i / wgt_q_j)
            )
        )(means_q_trans, wgts_q)
        return kl_vals

    pairwise_kl = jax.vmap(compute_kl_for_pair)(means_p, wgts_p)

    # Find argmin for each reference component
    argmin_indices = jnp.argmin(pairwise_kl, axis=1)

    return argmin_indices


@Partial(jax.jit, static_argnums=(7,))
def gradient_all_params_klg(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    n_dim: int,
    params: Float[Array, " p"],
) -> Float[Array, " p"]:
    """Compute gradients for all TPS parameters using KLG.

    This uses the KLG approximation where each reference component
    is matched to its closest moving component.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel, shape (m, p_per_dim)
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality
        params: Current TPS parameters, shape (p,) where p = n_dim * p_per_dim

    Returns:
        Gradient vector w.r.t. all parameters, shape (p,)
    """
    m, p_per_dim = basis.shape

    # Reshape parameters: (n_dim, p_per_dim)
    params_reshaped = params.reshape(n_dim, p_per_dim)

    # Transform moving means: means_trans[j, ell] = basis[j, :] @ params_reshaped[ell, :]
    means_q_trans = basis @ params_reshaped.T  # shape (m, n_dim)

    # Find argmin indices for each reference component
    argmin_indices = compute_argmin_indices(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Compute gradient for each dimension
    gradients = []
    for ell in range(n_dim):
        # Initialize weighted residual vector for this dimension
        r_min = jnp.zeros(m)

        # Accumulate weighted residuals for matched pairs
        # For each reference component i, add w_p[i] * delta[i, j*(i), ell] to r_min[j*(i)]
        def accumulate_residual(i, r):
            j_star = argmin_indices[i]
            delta_i_jstar_ell = means_q_trans[j_star, ell] - means_p[i, ell]
            return r.at[j_star].add(wgts_p[i] * delta_i_jstar_ell)

        r_min = jax.lax.fori_loop(
            0, means_p.shape[0], accumulate_residual, r_min
        )

        # Gradient: (1/var_q) * basis^T @ r_min
        # Shape: (p_per_dim,)
        grad_ell = (1.0 / var_q) * (basis.T @ r_min)
        gradients.append(grad_ell)

    # Stack gradients for all dimensions and flatten
    # Shape: (n_dim, p_per_dim) -> (n_dim * p_per_dim,)
    return jnp.stack(gradients, axis=0).ravel()


def gradient_all_2d_klg(
    means_p: Float[Array, "n 2"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 2"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    params: Float[Array, " p"],
) -> tuple[Float[Array, "2 2"], Float[Array, " 2"], Float[Array, "n_ctrl 2"]]:
    """Compute all gradients for 2D TPS using KLG.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        params: Current TPS parameters (affine, translation, rbf_weights packed)

    Returns:
        Tuple of (grad_affine, grad_translation, grad_rbf_weights)
    """
    # Compute full gradient vector (n_dim=2 is passed as literal)
    grad_params = gradient_all_params_klg(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, 2, params
    )

    # Unpack gradients (same structure as variational case)
    p_per_dim = basis.shape[1]
    grad_reshaped = grad_params.reshape(2, p_per_dim)

    # Extract translation gradients (first element per dimension)
    grad_translation = grad_reshaped[:, 0]  # shape (2,)

    # Extract affine gradients (elements 1:3 per dimension)
    grad_affine = grad_reshaped[:, 1:3].T  # shape (2, 2)

    # Extract RBF weight gradients (remaining elements)
    grad_rbf_weights = grad_reshaped[:, 3:].T  # shape (n_ctrl-3, 2)

    return grad_affine, grad_translation, grad_rbf_weights


def gradient_all_3d_klg(
    means_p: Float[Array, "n 3"],
    wgts_p: Float[Array, " n"],
    means_q: Float[Array, "m 3"],
    wgts_q: Float[Array, " m"],
    basis: Float[Array, "m p_per_dim"],
    var_p: float,
    var_q: float,
    params: Float[Array, " p"],
) -> tuple[Float[Array, "3 3"], Float[Array, " 3"], Float[Array, "n_ctrl 3"]]:
    """Compute all gradients for 3D TPS using KLG.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q: Original moving GMM means (before transformation)
        wgts_q: Moving GMM weights
        basis: Basis matrix from make_basis_kernel
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        params: Current TPS parameters (affine, translation, rbf_weights packed)

    Returns:
        Tuple of (grad_affine, grad_translation, grad_rbf_weights)
    """
    # Compute full gradient vector (n_dim=3 is passed as literal)
    grad_params = gradient_all_params_klg(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, 3, params
    )

    # Unpack gradients (same structure as variational case)
    p_per_dim = basis.shape[1]
    grad_reshaped = grad_params.reshape(3, p_per_dim)

    # Extract translation gradients (first element per dimension)
    grad_translation = grad_reshaped[:, 0]  # shape (3,)

    # Extract affine gradients (elements 1:4 per dimension)
    grad_affine = grad_reshaped[:, 1:4].T  # shape (3, 3)

    # Extract RBF weight gradients (remaining elements)
    grad_rbf_weights = grad_reshaped[:, 4:].T  # shape (n_ctrl-4, 3)

    return grad_affine, grad_translation, grad_rbf_weights
