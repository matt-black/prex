import jax
import jax.numpy as jnp
import pytest

from prex.gmm.dist import l2_distance_gmm_opt_spherical
from prex.gmm.grad.rigid import gradient_all_2d_l2, gradient_all_3d_l2
from prex.gmm.opt import AlignmentMethod, _make_transform_function_spherical
from prex.gmm.rigid import pack_params_2d, pack_params_3d


@pytest.fixture
def rigid_2d_setup():
    key = jax.random.PRNGKey(0)
    # Create random GMMs
    n_comp_p = 5
    n_comp_q = 5
    d = 2

    k1, k2, k3, k4 = jax.random.split(key, 4)
    means_p = jax.random.normal(k1, (n_comp_p, d))
    means_q = jax.random.normal(k2, (n_comp_q, d)) + 0.5  # Offset slightly
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n_comp_p,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (n_comp_q,)))

    var_p = 0.1
    var_q = 0.1

    # Initial parameters
    scale = 1.1
    alpha = 0.2
    translation = jnp.array([0.1, -0.2])

    return (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        scale,
        alpha,
        translation,
    )


@pytest.fixture
def rigid_3d_setup():
    key = jax.random.PRNGKey(1)
    # Create random GMMs
    n_comp_p = 5
    n_comp_q = 5
    d = 3

    k1, k2, k3, k4 = jax.random.split(key, 4)
    means_p = jax.random.normal(k1, (n_comp_p, d))
    means_q = jax.random.normal(k2, (n_comp_q, d)) + 0.5
    wgts_p = jax.nn.softmax(jax.random.normal(k3, (n_comp_p,)))
    wgts_q = jax.nn.softmax(jax.random.normal(k4, (n_comp_q,)))

    var_p = 0.1
    var_q = 0.1

    # Initial parameters
    scale = 1.1
    alpha = 0.1  # Rx
    beta = 0.2  # Ry
    gamma = 0.3  # Rz
    translation = jnp.array([0.1, -0.2, 0.3])

    return (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        scale,
        alpha,
        beta,
        gamma,
        translation,
    )


def test_rigid_2d_l2_gradients(rigid_2d_setup):
    (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        scale,
        alpha,
        translation,
    ) = rigid_2d_setup

    # Analytical gradients
    grad_s, grad_alpha, grad_t = gradient_all_2d_l2(
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

    # Autodiff gradients
    def loss_fn(params):
        s, a, t = params[0], params[1], params[2:]
        # Reconstruct transform function
        # We need to manually construct the transform because _make_transform_function
        # expects packed parameters, but we want to differentiate w.r.t. unpacked components
        # or we can use pack_params and differentiate w.r.t. packed.
        # Let's use packed params for consistency with how we test.

        p_packed = pack_params_2d(s, a, t)
        transform_fn = _make_transform_function_spherical(
            means_q, AlignmentMethod.RIGID, jnp.zeros((1, 2))  # Dummy ctrl pts
        )
        means_q_trans = transform_fn(p_packed)

        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 2
        )

    # Pack parameters for autodiff
    params_flat = jnp.concatenate(
        [jnp.array([scale]), jnp.array([alpha]), translation]
    )
    grad_autodiff = jax.grad(loss_fn)(params_flat)

    grad_s_ad = grad_autodiff[0]
    grad_alpha_ad = grad_autodiff[1]
    grad_t_ad = grad_autodiff[2:]

    # Compare
    tol = 0.05
    assert jnp.allclose(grad_s, grad_s_ad, atol=tol), "Scale gradient mismatch"
    assert jnp.allclose(
        grad_alpha, grad_alpha_ad, atol=tol
    ), "Alpha gradient mismatch"
    assert jnp.allclose(
        grad_t, grad_t_ad, atol=tol
    ), "Translation gradient mismatch"


def test_rigid_3d_l2_gradients(rigid_3d_setup):
    (
        means_p,
        wgts_p,
        means_q,
        wgts_q,
        var_p,
        var_q,
        scale,
        alpha,
        beta,
        gamma,
        translation,
    ) = rigid_3d_setup

    # Analytical gradients
    grad_s, grad_alpha, grad_beta, grad_gamma, grad_t = gradient_all_3d_l2(
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

    # Autodiff gradients
    def loss_fn(params):
        s, a, b, g = params[0], params[1], params[2], params[3]
        t = params[4:]
        p_packed = pack_params_3d(s, a, b, g, t)
        transform_fn = _make_transform_function_spherical(
            means_q, AlignmentMethod.RIGID, jnp.zeros((1, 3))
        )
        means_q_trans = transform_fn(p_packed)

        return l2_distance_gmm_opt_spherical(
            means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, 3
        )

    params_flat = jnp.concatenate(
        [
            jnp.array([scale]),
            jnp.array([alpha]),
            jnp.array([beta]),
            jnp.array([gamma]),
            translation,
        ]
    )
    grad_autodiff = jax.grad(loss_fn)(params_flat)

    grad_s_ad = grad_autodiff[0]
    grad_alpha_ad = grad_autodiff[1]
    grad_beta_ad = grad_autodiff[2]
    grad_gamma_ad = grad_autodiff[3]
    grad_t_ad = grad_autodiff[4:]

    # Compare
    tol = 0.05
    assert jnp.allclose(grad_s, grad_s_ad, atol=tol), "Scale gradient mismatch"
    assert jnp.allclose(
        grad_alpha, grad_alpha_ad, atol=tol
    ), "Alpha gradient mismatch"
    assert jnp.allclose(
        grad_beta, grad_beta_ad, atol=tol
    ), "Beta gradient mismatch"
    assert jnp.allclose(
        grad_gamma, grad_gamma_ad, atol=tol
    ), "Gamma gradient mismatch"
    assert jnp.allclose(
        grad_t, grad_t_ad, atol=tol
    ), "Translation gradient mismatch"


def test_unweighted_case_matches_paper():
    """Verify that unweighted case matches paper equation (14).

    Paper says G_j = -2/(nm(var_p + var_q)) * sum_i N(delta_ij) * delta_ij
    when weights are 1/n and 1/m.
    """
    key = jax.random.PRNGKey(2)
    n, m, d = 3, 4, 2

    means_p = jax.random.normal(key, (n, d))
    means_q_trans = jax.random.normal(key, (m, d))  # Already transformed

    # Uniform weights
    wgts_p = jnp.ones(n) / n
    wgts_q = jnp.ones(m) / m

    var_p = 0.1
    var_q = 0.1
    var_combined = var_p + var_q

    # Compute gradients using our implementation (via translation gradient logic)
    # gradient_translation_l2 computes: -2/(var_p+var_q) * sum_ij O_ij * delta_ij
    # where O_ij = w_p w_q N(delta_ij)

    # Let's compute the "G" matrix (gradient w.r.t. means_q_trans) manually using our formula
    # G_j = -2/(var_p+var_q) * sum_i O_ij * delta_ij

    from prex.gmm.grad._util import compute_overlap_weights

    overlap_ij = compute_overlap_weights(
        means_p, wgts_p, means_q_trans, wgts_q, var_p, var_q, d
    )

    delta_ij = means_q_trans[jnp.newaxis, :, :] - means_p[:, jnp.newaxis, :]
    factor = -2.0 / var_combined

    # Our computed G matrix (m x d)
    # G_j = factor * sum_i overlap_ij[i, j] * delta_ij[i, j]
    G_ours = factor * jnp.sum(overlap_ij[:, :, jnp.newaxis] * delta_ij, axis=0)

    # Paper formula implementation
    # G_j = -2/(nm * var_combined) * sum_i N(delta_ij) * delta_ij
    norm_const = 1.0 / ((2.0 * jnp.pi * var_combined) ** (d / 2.0))

    def compute_paper_G_j(j):
        # sum over i
        diffs = means_q_trans[j] - means_p  # (n, d)
        sq_dists = jnp.sum(diffs**2, axis=1)
        gaussians = norm_const * jnp.exp(-sq_dists / (2.0 * var_combined))

        weighted_diffs = gaussians[:, jnp.newaxis] * diffs
        sum_weighted_diffs = jnp.sum(weighted_diffs, axis=0)

        return (-2.0 / (n * m * var_combined)) * sum_weighted_diffs

    G_paper = jax.vmap(compute_paper_G_j)(jnp.arange(m))

    tol = 1e-6
    assert jnp.allclose(
        G_ours, G_paper, atol=tol
    ), "Unweighted case does not match paper formula"
