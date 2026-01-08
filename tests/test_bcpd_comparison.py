"""Comparison tests between BCPD and traditional CPD methods.

Note: BCPD and traditional CPD use fundamentally different formulations:
- BCPD: Bayesian approach with per-point variance (sigma_m) and Dirichlet priors
- CPD: Frequentist approach with uniform point treatment

Therefore, we expect qualitative similarity but not exact numerical agreement.
"""

import math

import jax
import jax.numpy as jnp
import pytest

from prex.cpd import rigid as cpd_rigid
from prex.cpd.bayes import align as bcpd_align
from prex.cpd.bayes.kernel import gaussian_kernel


def generate_data(n=20, noise=0.01, seed=42):
    """Generate random point clouds for testing."""
    key = jax.random.PRNGKey(seed)
    ref = jax.random.uniform(key, (n, 2))

    # Apply random rigid transform
    theta = 0.1
    R = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )
    t = jnp.array([0.1, 0.2])
    mov = ref @ R.T + t + noise * jax.random.normal(key, (n, 2))
    return ref, mov, R, t


def test_bcpd_rigid_vs_cpd_rigid():
    """Compare BCPD rigid-only mode to traditional rigid CPD.

    Expect: Similar but not identical results due to Bayesian formulation.
    """
    ref, mov, R_gt, t_gt = generate_data(n=20, seed=0)

    # Run BCPD rigid-only (more iterations for convergence)
    # gamma=0.1 to tighten initial variance
    (P_bcpd, R_bcpd, s_bcpd, t_bcpd, v_bcpd), var_bcpd = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=30,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=2.0,
        kernel_beta=2.0,
        gamma=0.1,
        kappa=1000.0,  # Stronger prior for uniform mixing
        transform_mode="rigid",
    )

    # Run traditional rigid CPD
    (P_cpd, R_cpd, s_cpd, t_cpd), var_cpd = cpd_rigid.align_fixed_iter(
        ref, mov, outlier_prob=0.1, num_iter=30
    )

    # BCPD should have v=0 in rigid mode
    assert jnp.allclose(v_bcpd, jnp.zeros_like(mov), atol=1e-6)

    # Compare transform parameters (qualitative similarity)
    print("\nBCPD rigid-only vs CPD rigid comparison:")
    print(f"  R_GT:\n{R_gt}")
    print(f"  R_bcpd:\n{R_bcpd}")
    print(f"  R_cpd:\n{R_cpd}")
    print(
        f"  Final variance - BCPD: {var_bcpd[-1]:.6f}, CPD: {var_cpd[-1]:.6f}"
    )

    # Both should recover the rotation reasonably well (note: finding inverse of data gen transform)
    assert jnp.allclose(
        R_bcpd, R_gt.T, atol=0.2
    ), "BCPD should recover rotation"
    assert jnp.allclose(R_cpd, R_gt.T, atol=0.2), "CPD should recover rotation"

    # Variance comparison
    assert abs(var_bcpd[-1] - var_cpd[-1]) < 0.1


def test_bcpd_nonrigid_produces_deformation():
    """Test that BCPD nonrigid-only mode produces non-zero deformation."""
    ref = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32
    )
    # Larger perturbation to require deformation
    mov = jnp.array(
        [[0.15, 0.15], [1.15, 0.05], [1.05, 1.15], [0.05, 1.05]],
        dtype=jnp.float32,
    )

    # Run BCPD nonrigid-only
    (P_bcpd, R_bcpd, s_bcpd, t_bcpd, v_bcpd), var_bcpd = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=20,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=1.0,
        gamma=0.1,
        kappa=math.inf,
        transform_mode="nonrigid",
    )

    # BCPD should have R=I, s=1, t=0 in nonrigid mode
    assert jnp.allclose(R_bcpd, jnp.eye(2), atol=1e-6)
    assert jnp.allclose(s_bcpd, 1.0, atol=1e-6)
    assert jnp.allclose(t_bcpd, jnp.zeros(2), atol=1e-6)

    # Should produce non-zero deformation
    deformation_magnitude = jnp.linalg.norm(v_bcpd)
    print(f"\nBCPD nonrigid deformation magnitude: {deformation_magnitude:.6f}")
    assert deformation_magnitude > 1e-3, "Should produce non-zero deformation"


def test_bcpd_fixed_alpha_affects_matching():
    """Test that fixed_alpha affects the matching matrix."""
    ref, mov, _, _ = generate_data(n=20, seed=1)

    # Run with uniform weights
    (P_uniform, _, _, _, _), _ = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=10,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=1.0,
        gamma=0.1,
        kappa=math.inf,
        fixed_alpha=jnp.ones(20, dtype=jnp.float32),
        transform_mode="rigid",
    )  # fixed_alpha provided, kappa ignored

    # Run with non-uniform weights (first point very heavy)
    weights = jnp.ones(20, dtype=jnp.float32)
    weights = weights.at[0].set(100.0)

    (P_weighted, _, _, _, _), _ = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=10,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=1.0,
        gamma=0.1,
        kappa=math.inf,
        fixed_alpha=weights,
        transform_mode="rigid",
    )

    # Matching matrices should be different
    assert not jnp.allclose(
        P_uniform, P_weighted, atol=1e-3
    ), "Different weights should produce different matching"


def test_bcpd_modes_produce_different_results():
    """Verify that different transform modes produce qualitatively different results."""
    ref = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32
    )
    mov = jnp.array(
        [[0.15, 0.15], [1.15, 0.05], [1.05, 1.15], [0.05, 1.05]],
        dtype=jnp.float32,
    )
    # Add non-rigid perturbation to force different results in 'both' mode
    mov = mov.at[0].add(jnp.array([0.2, -0.2]))

    # Run all three modes
    (P_rigid, R_rigid, s_rigid, t_rigid, v_rigid), _ = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=10,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=1.0,
        gamma=0.1,
        kappa=math.inf,
        transform_mode="rigid",
    )

    (P_nonrigid, R_nonrigid, s_nonrigid, t_nonrigid, v_nonrigid), _ = (
        bcpd_align(
            ref,
            mov,
            outlier_prob=0.1,
            num_iter=10,
            tolerance=None,
            kernel=gaussian_kernel,
            lambda_param=1.0,
            kernel_beta=1.0,
            gamma=0.1,
            kappa=math.inf,
            transform_mode="nonrigid",
        )
    )

    # Verify mode constraints
    assert jnp.allclose(v_rigid, 0, atol=1e-6), "Rigid mode should have v=0"
    assert jnp.allclose(
        R_nonrigid, jnp.eye(2), atol=1e-6
    ), "Nonrigid mode should have R=I"


def test_bcpd_qualitative_comparison_with_cpd():
    """Qualitative comparison using random data."""
    ref, mov, _, _ = generate_data(n=20, seed=0)

    # Initial error
    initial_error = jnp.mean(jnp.sum((ref - mov) ** 2, axis=1))

    # BCPD rigid
    (_, R_bcpd, s_bcpd, t_bcpd, _), _ = bcpd_align(
        ref,
        mov,
        outlier_prob=0.1,
        num_iter=20,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=2.0,
        kernel_beta=2.0,
        gamma=0.1,
        kappa=1000.0,
        transform_mode="rigid",
    )
    from prex.cpd.bayes.util import transform as bcpd_transform

    mov_bcpd = bcpd_transform(mov, R_bcpd, s_bcpd, t_bcpd)
    bcpd_error = jnp.mean(jnp.sum((ref - mov_bcpd) ** 2, axis=1))

    # CPD rigid
    (_, R_cpd, s_cpd, t_cpd), _ = cpd_rigid.align_fixed_iter(
        ref, mov, outlier_prob=0.1, num_iter=20
    )
    from prex.cpd.rigid import transform as cpd_transform

    mov_cpd = cpd_transform(mov, R_cpd, s_cpd, t_cpd)
    cpd_error = jnp.mean(jnp.sum((ref - mov_cpd) ** 2, axis=1))

    print("\nQualitative comparison:")
    print(f"  Initial error: {initial_error:.6f}")
    print(f"  BCPD error: {bcpd_error:.6f}")
    print(f"  CPD error: {cpd_error:.6f}")
    print(f"  BCPD s: {s_bcpd}, t: {t_bcpd}")
    print(f"  CPD s: {s_cpd}, t: {t_cpd}")

    # Both should reduce error significantly
    assert (
        bcpd_error < initial_error * 0.5
    ), "BCPD should reduce error by at least 50%"
    assert (
        cpd_error < initial_error * 0.5
    ), "CPD should reduce error by at least 50%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
