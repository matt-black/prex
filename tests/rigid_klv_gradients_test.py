"""Pytest tests for rigid transformation analytical gradients."""

import jax
import jax.numpy as jnp

from prex.gmm.grad import rigid as grad_rigid
from prex.gmm.rigid import (
    transform_means_rotangles2,
    transform_means_rotangles3,
    unpack_params_2d,
    unpack_params_3d,
)


def compute_variational_kl_loss(
    means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
):
    """Compute variational KL divergence loss.

    This matches the implementation in kullback_leibler_gmm_approx_var_spherical
    but works for any dimension.
    """
    # Compute A_i and B_i using the same logic as in grad_rigid
    alpha_ij, A_i, B_i = grad_rigid.compute_weights_alpha(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, n_dim
    )

    # Loss = sum_i w_p^i * (A_i / B_i)
    return jnp.sum(wgts_p * (A_i / B_i))


def test_rigid_gradients_2d():
    """Test 2D rigid analytical gradients against autodiff."""
    # Setup test data
    n, m = 10, 8
    key = jax.random.key(42)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    means_p = jax.random.normal(key1, (n, 2))
    wgts_p = jax.random.uniform(key2, (n,))
    wgts_p = wgts_p / jnp.sum(wgts_p)

    means_q = jax.random.normal(key3, (m, 2))
    wgts_q = jax.random.uniform(key4, (m,))
    wgts_q = wgts_q / jnp.sum(wgts_q)

    var_p, var_q = 0.5, 0.3

    # Test parameters
    scale = jnp.array(1.2)
    alpha = jnp.array(0.3)
    translation = jnp.array([0.5, -0.2])

    # Define loss function for autodiff
    def loss_fn(params):
        s, a, t = unpack_params_2d(params)
        means_q_trans = transform_means_rotangles2(means_q, s, a, t)
        return compute_variational_kl_loss(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 2
        )

    # Pack parameters
    params = jnp.concatenate(
        [scale[jnp.newaxis], alpha[jnp.newaxis], translation]
    )

    # Compute gradients with autodiff
    grad_autodiff = jax.grad(loss_fn)(params)
    grad_s_auto, grad_alpha_auto, grad_t_auto = (
        grad_autodiff[0],
        grad_autodiff[1],
        grad_autodiff[2:],
    )

    # Compute gradients analytically
    grad_s_analytical, grad_alpha_analytical, grad_t_analytical = (
        grad_rigid.gradient_all_2d_klv(
            means_p,
            wgts_p,
            means_q,
            wgts_q,
            var_p,
            var_q,
            2,
            scale,
            alpha,
            translation,
        )
    )

    # Check if gradients match within tolerance
    tol = 1e-4
    assert abs(grad_s_auto - grad_s_analytical) < tol
    assert abs(grad_alpha_auto - grad_alpha_analytical) < tol
    assert jnp.linalg.norm(grad_t_auto - grad_t_analytical) < tol


def test_rigid_gradients_3d():
    """Test 3D rigid analytical gradients against autodiff."""
    # Setup test data
    n, m = 10, 8
    key = jax.random.key(123)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    means_p = jax.random.normal(key1, (n, 3))
    wgts_p = jax.random.uniform(key2, (n,))
    wgts_p = wgts_p / jnp.sum(wgts_p)

    means_q = jax.random.normal(key3, (m, 3))
    wgts_q = jax.random.uniform(key4, (m,))
    wgts_q = wgts_q / jnp.sum(wgts_q)

    var_p, var_q = 0.5, 0.3

    # Test parameters
    scale = jnp.array(1.1)
    alpha = jnp.array(0.2)
    beta = jnp.array(0.15)
    gamma = jnp.array(-0.1)
    translation = jnp.array([0.3, -0.4, 0.2])

    # Define loss function for autodiff
    def loss_fn(params):
        s, a, b, g, t = unpack_params_3d(params)
        means_q_trans = transform_means_rotangles3(means_q, s, a, b, g, t)
        return compute_variational_kl_loss(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 3
        )

    # Pack parameters
    params = jnp.concatenate(
        [
            scale[jnp.newaxis],
            alpha[jnp.newaxis],
            beta[jnp.newaxis],
            gamma[jnp.newaxis],
            translation,
        ]
    )

    # Compute gradients with autodiff
    grad_autodiff = jax.grad(loss_fn)(params)
    grad_s_auto = grad_autodiff[0]
    grad_alpha_auto = grad_autodiff[1]
    grad_beta_auto = grad_autodiff[2]
    grad_gamma_auto = grad_autodiff[3]
    grad_t_auto = grad_autodiff[4:]

    # Compute gradients analytically
    (
        grad_s_analytical,
        grad_alpha_analytical,
        grad_beta_analytical,
        grad_gamma_analytical,
        grad_t_analytical,
    ) = grad_rigid.gradient_all_3d_klv(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        3,
        scale,
        alpha,
        beta,
        gamma,
        translation,
    )

    # Check if gradients match within tolerance
    tol = 1e-4
    assert abs(grad_s_auto - grad_s_analytical) < tol
    assert abs(grad_alpha_auto - grad_alpha_analytical) < tol
    assert abs(grad_beta_auto - grad_beta_analytical) < tol
    assert abs(grad_gamma_auto - grad_gamma_analytical) < tol
    assert jnp.linalg.norm(grad_t_auto - grad_t_analytical) < tol
