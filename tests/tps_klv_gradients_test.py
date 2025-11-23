"""Pytest tests for TPS transformation analytical gradients (variational KL)."""

import jax
import jax.numpy as jnp

from prex.gmm.grad import rigid as grad_rigid
from prex.gmm.grad import tps as grad_tps
from prex.gmm.tps import make_basis_kernel, pack_params


def compute_variational_kl_loss(
    means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
):
    """Compute variational KL divergence loss (same as rigid test)."""
    alpha_ij, A_i, B_i = grad_rigid.compute_weights_alpha(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )
    return jnp.sum(wgts_p * (A_i / B_i))


def test_tps_gradients_2d():
    """Test 2D TPS analytical gradients against autodiff."""
    # Setup test data
    n, m = 8, 6
    n_ctrl = 4  # Small number of control points for testing

    key = jax.random.key(42)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    means_p = jax.random.normal(key1, (n, 2))
    wgts_p = jax.random.uniform(key2, (n,))
    wgts_p = wgts_p / jnp.sum(wgts_p)

    means_q = jax.random.normal(key3, (m, 2))
    wgts_q = jax.random.uniform(key4, (m,))
    wgts_q = wgts_q / jnp.sum(wgts_q)

    ctrl_pts = jax.random.normal(key5, (n_ctrl, 2))

    var_p, var_q = 0.5, 0.3

    # Create basis matrix
    basis, _ = make_basis_kernel(means_q, ctrl_pts)
    p_per_dim = basis.shape[1]

    # Initialize TPS parameters
    affine = jnp.eye(2) + jax.random.normal(jax.random.key(100), (2, 2)) * 0.1
    translation = jax.random.normal(jax.random.key(101), (2,)) * 0.1
    rbf_weights = jax.random.normal(jax.random.key(102), (n_ctrl - 3, 2)) * 0.01

    params = pack_params(affine, translation, rbf_weights)

    # Define loss function for autodiff
    def loss_fn(p):
        # Reshape parameters for each dimension
        p_reshaped = p.reshape(2, p_per_dim)
        means_q_trans = basis @ p_reshaped.T
        return compute_variational_kl_loss(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 2
        )

    # compute gradients with autodiff
    grad_autodiff = jax.grad(loss_fn)(params)

    # reshape autodiff gradients
    # TPS params are: [t_1, t_2, A_11, A_12, A_21, A_22, w_1, w_2, ...]
    grad_autodiff_reshaped = grad_autodiff.reshape(2, p_per_dim)
    grad_translation_auto = grad_autodiff_reshaped[:, 0]
    grad_affine_auto = grad_autodiff_reshaped[:, 1:3].T
    grad_rbf_weights_auto = grad_autodiff_reshaped[:, 3:].T

    # compute gradients analytically
    (
        grad_affine_analytical,
        grad_translation_analytical,
        grad_rbf_weights_analytical,
    ) = grad_tps.gradient_all_2d_klv(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, params
    )

    # check if gradients match within tolerance
    tol = 2e-2  # TODO: why does this have to be so high?
    assert jnp.linalg.norm(grad_affine_auto - grad_affine_analytical) < tol
    assert (
        jnp.linalg.norm(grad_translation_auto - grad_translation_analytical)
        < tol
    )
    assert (
        jnp.linalg.norm(grad_rbf_weights_auto - grad_rbf_weights_analytical)
        < tol
    )


def test_tps_gradients_3d():
    """Test 3D TPS analytical gradients against autodiff."""
    # Setup test data
    n, m = 8, 6
    n_ctrl = 5  # Small number of control points for testing

    key = jax.random.key(123)
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    means_p = jax.random.normal(key1, (n, 3))
    wgts_p = jax.random.uniform(key2, (n,))
    wgts_p = wgts_p / jnp.sum(wgts_p)

    means_q = jax.random.normal(key3, (m, 3))
    wgts_q = jax.random.uniform(key4, (m,))
    wgts_q = wgts_q / jnp.sum(wgts_q)

    ctrl_pts = jax.random.normal(key5, (n_ctrl, 3))

    var_p, var_q = 0.5, 0.3

    # Create basis matrix
    basis, _ = make_basis_kernel(means_q, ctrl_pts)
    p_per_dim = basis.shape[1]

    # Initialize TPS parameters
    affine = jnp.eye(3) + jax.random.normal(jax.random.key(200), (3, 3)) * 0.1
    translation = jax.random.normal(jax.random.key(201), (3,)) * 0.1
    rbf_weights = jax.random.normal(jax.random.key(202), (n_ctrl - 4, 3)) * 0.01

    params = pack_params(affine, translation, rbf_weights)

    # Define loss function for autodiff
    def loss_fn(p):
        # Reshape parameters for each dimension
        p_reshaped = p.reshape(3, p_per_dim)
        means_q_trans = basis @ p_reshaped.T
        return compute_variational_kl_loss(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 3
        )

    # Compute gradients with autodiff
    grad_autodiff = jax.grad(loss_fn)(params)

    # Reshape autodiff gradients
    # TPS params are: [t_1, t_2, t_3, A_11, A_12, A_13, A_21, A_22, A_23, A_31, A_32, A_33, w_1, w_2, w_3, ...]
    grad_autodiff_reshaped = grad_autodiff.reshape(3, p_per_dim)
    grad_translation_auto = grad_autodiff_reshaped[:, 0]
    grad_affine_auto = grad_autodiff_reshaped[:, 1:4].T
    grad_rbf_weights_auto = grad_autodiff_reshaped[:, 4:].T

    # Compute gradients analytically
    (
        grad_affine_analytical,
        grad_translation_analytical,
        grad_rbf_weights_analytical,
    ) = grad_tps.gradient_all_3d_klv(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, params
    )

    # Check if gradients match within tolerance
    # TPS has more numerical operations than rigid, so we use a slightly larger tolerance
    tol = 5e-3
    assert jnp.linalg.norm(grad_affine_auto - grad_affine_analytical) < tol
    assert (
        jnp.linalg.norm(grad_translation_auto - grad_translation_analytical)
        < tol
    )
    assert (
        jnp.linalg.norm(grad_rbf_weights_auto - grad_rbf_weights_analytical)
        < tol
    )
