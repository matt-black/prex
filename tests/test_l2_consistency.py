import jax
import jax.numpy as jnp

from prex.gmm.dist import l2_distance_gmm_opt_spherical
from prex.gmm.grad.rigid import gradient_all_2d_l2
from prex.gmm.grad.tps import gradient_all_2d_l2 as tps_grad_2d
from prex.gmm.opt import AlignmentMethod, _make_transform_function_spherical
from prex.gmm.rigid import rotation_matrix_2d
from prex.gmm.tps import (
    make_basis_kernel,
    pack_params,
    unpack_params_2d,
)


def test_l2_distance_consistency_rigid_2d():
    """Verify l2_distance_gmm_opt_spherical gradients match analytical rigid 2D gradients."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    n, m, d = 5, 5, 2
    means_p = jax.random.normal(k1, (n, d))
    means_q = jax.random.normal(k2, (m, d)) + 1.0
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (m,)))
    var_p, var_q = 0.1, 0.1

    # Parameters
    scale = 1.1
    alpha = 0.2
    translation = jnp.array([0.1, -0.1])

    # Analytical
    grad_s, grad_alpha, grad_t = gradient_all_2d_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        d,
        scale,
        alpha,
        translation,
    )

    # Autodiff
    def loss_fn(params):
        s, a, t = params[0], params[1], params[2:]
        R = rotation_matrix_2d(a)
        means_q_trans = s * (means_q @ R.T) + t
        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, d
        )

    params_flat = jnp.concatenate(
        [jnp.array([scale]), jnp.array([alpha]), translation]
    )
    grad_ad = jax.grad(loss_fn)(params_flat)

    grad_s_ad = grad_ad[0]
    grad_alpha_ad = grad_ad[1]
    grad_t_ad = grad_ad[2:]

    tol = 1e-4
    assert jnp.allclose(grad_s, grad_s_ad, atol=tol)
    assert jnp.allclose(grad_alpha, grad_alpha_ad, atol=tol)
    assert jnp.allclose(grad_t, grad_t_ad, atol=tol)


def test_l2_distance_consistency_tps_2d():
    """Verify l2_distance_gmm_opt_spherical gradients match analytical TPS 2D gradients."""
    key = jax.random.PRNGKey(1)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    n, m, d = 5, 5, 2
    n_ctrl = 9
    means_p = jax.random.normal(k1, (n, d))
    means_q = jax.random.normal(k2, (m, d))
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (m,)))
    ctrl_pts = jax.random.normal(k5, (n_ctrl, d))
    var_p, var_q = 0.1, 0.1

    # Parameters
    t = jnp.array([0.1, -0.1])
    A = jnp.eye(d) + 0.01 * jax.random.normal(key, (d, d))
    w = 0.01 * jax.random.normal(key, (n_ctrl - d - 1, d))
    params = pack_params(A, t, w)

    basis, _ = make_basis_kernel(means_q, ctrl_pts)

    # Analytical
    grad_A, grad_t, grad_w = tps_grad_2d(
        means_p, wgts_p, means_q, wgts_q, basis, var_p, var_q, params
    )

    # Autodiff
    def loss_fn(p):
        transform_fn = _make_transform_function_spherical(
            means_q, AlignmentMethod.TPS, ctrl_pts
        )
        means_q_trans = transform_fn(p)
        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, d
        )

    grad_ad = jax.grad(loss_fn)(params)
    grad_A_ad, grad_t_ad, grad_w_ad = unpack_params_2d(grad_ad)

    tol = 0.05
    assert jnp.allclose(grad_t, grad_t_ad, atol=tol)
    assert jnp.allclose(grad_A, grad_A_ad, atol=tol)
    assert jnp.allclose(grad_w, grad_w_ad, atol=tol)
