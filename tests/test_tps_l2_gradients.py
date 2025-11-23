import jax
import jax.numpy as jnp
import pytest

from prex.gmm.dist import l2_distance_gmm_opt_spherical
from prex.gmm.grad.tps import gradient_all_2d_l2, gradient_all_3d_l2
from prex.gmm.opt import AlignmentMethod, _make_transform_function_spherical
from prex.gmm.tps import (
    make_basis_kernel,
    pack_params,
    unpack_params_2d,
    unpack_params_3d,
)


@pytest.fixture
def tps_2d_setup():
    key = jax.random.PRNGKey(0)
    # Create random GMMs
    n_comp_p = 5
    n_comp_q = 5
    d = 2
    n_ctrl = 4

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    means_p = jax.random.normal(k1, (n_comp_p, d))
    means_q = jax.random.normal(k2, (n_comp_q, d)) + 0.5
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n_comp_p,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (n_comp_q,)))

    # Control points
    ctrl_pts = jax.random.normal(k5, (n_ctrl, d))

    var_p = 0.1
    var_q = 0.1

    # Initial parameters
    # TPS params: [translation, affine, rbf_weights]
    # For 2D: t (2), A (2x2), w (n_ctrl x 2)
    t = jnp.array([0.1, -0.2])
    A = jnp.eye(2) + 0.1 * jax.random.normal(key, (2, 2))
    w = 0.01 * jax.random.normal(key, (n_ctrl - d - 1, 2))

    # Pack params
    params = pack_params(A, t, w)

    # Make basis
    basis, _ = make_basis_kernel(means_q, ctrl_pts)

    return (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
        ctrl_pts,
    )


@pytest.fixture
def tps_3d_setup():
    key = jax.random.PRNGKey(1)
    # Create random GMMs
    n_comp_p = 5
    n_comp_q = 5
    d = 3
    n_ctrl = 6

    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    means_p = jax.random.normal(k1, (n_comp_p, d))
    means_q = jax.random.normal(k2, (n_comp_q, d)) + 0.5
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n_comp_p,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (n_comp_q,)))

    # Control points
    ctrl_pts = jax.random.normal(k5, (n_ctrl, d))

    var_p = 0.1
    var_q = 0.1

    # Initial parameters
    # TPS params: [translation, affine, rbf_weights]
    # For 3D: t (3), A (3x3), w (n_ctrl x 3)
    t = jnp.array([0.1, -0.2, 0.3])
    A = jnp.eye(3) + 0.1 * jax.random.normal(key, (3, 3))
    w = 0.01 * jax.random.normal(key, (n_ctrl - d - 1, 3))

    # Pack params
    params = pack_params(A, t, w)

    # Make basis
    basis, _ = make_basis_kernel(means_q, ctrl_pts)

    return (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
        ctrl_pts,
    )


def test_tps_2d_l2_gradients(tps_2d_setup):
    (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
        ctrl_pts,
    ) = tps_2d_setup

    # Analytical gradients
    grad_A, grad_t, grad_w = gradient_all_2d_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
    )

    # Autodiff gradients
    def loss_fn(p):
        transform_fn = _make_transform_function_spherical(
            means_q, AlignmentMethod.TPS, ctrl_pts
        )
        means_q_trans = transform_fn(p)

        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 2
        )

    grad_autodiff = jax.grad(loss_fn)(params)

    # Unpack autodiff gradients
    # We need to unpack them same way as analytical ones are returned
    # But wait, pack_params packs them as [t_1, t_2, A_11, A_12, A_21, A_22, w_1, w_2, ...]
    # And gradient_all_2d_l2 returns unpacked (A, t, w)

    # Let's use unpack_params from tps module to unpack the autodiff gradient vector
    grad_A_ad, grad_t_ad, grad_w_ad = unpack_params_2d(grad_autodiff)

    # Compare
    # TPS gradients involve more operations so we might need slightly higher tolerance
    tol = 0.05
    assert jnp.allclose(
        grad_t, grad_t_ad, atol=tol
    ), "Translation gradient mismatch"
    assert jnp.allclose(grad_A, grad_A_ad, atol=tol), "Affine gradient mismatch"
    assert jnp.allclose(
        grad_w, grad_w_ad, atol=tol
    ), "RBF weights gradient mismatch"


def test_tps_3d_l2_gradients(tps_3d_setup):
    (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
        ctrl_pts,
    ) = tps_3d_setup

    # Analytical gradients
    grad_A, grad_t, grad_w = gradient_all_3d_l2(
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        basis,
        var_p,
        var_q,
        params,
    )

    # Autodiff gradients
    def loss_fn(p):
        transform_fn = _make_transform_function_spherical(
            means_q, AlignmentMethod.TPS, ctrl_pts
        )
        means_q_trans = transform_fn(p)

        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 3
        )

    grad_autodiff = jax.grad(loss_fn)(params)

    # Unpack autodiff gradients
    grad_A_ad, grad_t_ad, grad_w_ad = unpack_params_3d(grad_autodiff)

    # Compare
    tol = 0.05
    assert jnp.allclose(
        grad_t, grad_t_ad, atol=tol
    ), "Translation gradient mismatch"
    assert jnp.allclose(grad_A, grad_A_ad, atol=tol), "Affine gradient mismatch"
    assert jnp.allclose(
        grad_w, grad_w_ad, atol=tol
    ), "RBF weights gradient mismatch"
