import math

import jax.numpy as jnp
import pytest

from prex.cpd.bayes import align as bcpd_align
from prex.cpd.bayes import align_with_ic, initialize
from prex.cpd.bayes.kernel import gaussian_kernel
from prex.cpd.bayes.normalization import (
    denormalize_parameters,
    normalize_points,
)
from prex.cpd.bayes.util import transform as bcpd_transform


def generate_grid(n_side=10):
    """Generate a 2D grid of points."""
    x = jnp.linspace(-1.0, 1.0, n_side)
    y = jnp.linspace(-1.0, 1.0, n_side)
    X, Y = jnp.meshgrid(x, y)
    return jnp.column_stack([X.ravel(), Y.ravel()])


def test_pure_nonrigid_recovery():
    """
    Test 1: Verify that a non-zero nonrigid vector field is created when
    registering a source to a non-rigidly deformed version of itself.
    """
    # 1. Setup
    source = generate_grid(n_side=15)

    # Create non-rigid deformation (sine wave warp)
    v_gt = jnp.zeros_like(source)
    # Apply sine wave to y based on x
    v_gt = v_gt.at[:, 1].set(0.3 * jnp.sin(3.0 * source[:, 0]))

    target = source + v_gt

    # 2. Action: Register source to target using nonrigid mode
    # We expect BCPD to find a v that maps source -> target
    (P, R, s, t, v_hat), _ = bcpd_align(
        target,
        source,
        outlier_prob=0.0,
        num_iter=100,
        tolerance=1e-5,
        kernel=gaussian_kernel,
        lambda_param=2.0,
        kernel_beta=2.0,
        gamma=1.0,
        kappa=1000.0,
        transform_mode="nonrigid",  # Check pure nonrigid first
        normalize_input=True,
    )

    # 3. Validation

    # A) Check that v_hat is non-zero
    v_norm = jnp.linalg.norm(v_hat)
    print(f"\n[Pure Nonrigid] Norm of recovered v: {v_norm:.4f}")
    assert v_norm > 0.1, "Vector field should be non-zero and significant"

    # B) Check alignment quality
    y_hat = source + v_hat
    mse = jnp.mean(jnp.sum((target - y_hat) ** 2, axis=1))
    print(f"[Pure Nonrigid] MSE: {mse:.6f}")
    assert mse < 0.01, "Alignment should be good"

    # C) Correlation check
    v_diff = jnp.linalg.norm(v_hat - v_gt) / jnp.linalg.norm(v_gt)
    print(f"[Pure Nonrigid] Relative error in v: {v_diff:.4f}")


def test_combined_rigid_nonrigid_recovery():
    """
    Test 2: Verify combined rigid + non-rigid recovery using a lambda sweep.
    (1) Manual normalization for coordinate consistency.
    (2) Stage 1: Rigid-only capture.
    (3) Stage 2: Sweep lambda to demonstrate trade-offs between MSE and rigid recovery.
    """
    # 1. Setup
    source = generate_grid(n_side=15)

    v_gt = jnp.zeros_like(source)
    v_gt = v_gt.at[:, 1].set(0.3 * jnp.sin(3.0 * source[:, 0]))
    deformed_source = source + v_gt

    theta = math.radians(30)
    c, s = math.cos(theta), math.sin(theta)
    R_gt = jnp.array([[c, -s], [s, c]])
    s_gt = 1.2
    t_gt = jnp.array([0.5, -0.2])
    target = bcpd_transform(
        deformed_source, R_gt, s_gt, t_gt
    )  # pyright: ignore[reportArgumentType]

    # Manual Normalization
    target_norm, mu_x, sigma_x = normalize_points(target)
    source_norm, mu_y, sigma_y = normalize_points(source)

    # 2. Stage 1: Rigid-only alignment in normalized space
    (P1, R1, s1, t1, v1), res1 = bcpd_align(
        target_norm,
        source_norm,
        outlier_prob=0.0,
        num_iter=100,
        tolerance=1e-5,
        kernel=gaussian_kernel,
        lambda_param=2.0,
        kernel_beta=2.0,
        gamma=0.1,
        kappa=1000.0,
        transform_mode="rigid",
        normalize_input=False,
    )

    # 3. Stage 2 Refinement Sweep
    lambdas = [0.1, 2.0, 10.0]
    results = []

    print("\n[Lambda Sweep Results]")
    for lam in lambdas:
        G, alpha0, sigma0, _ = initialize(
            target_norm, source_norm, gaussian_kernel, 2.0, 1.0
        )

        # We use a warm variance (0.1) for refinement search reach
        (P2, R2, s2, t2, v2), res2 = align_with_ic(
            target_norm,
            source_norm,
            outlier_prob=0.0,
            num_iter=150,
            tolerance=1e-5,
            lambda_param=lam,
            kappa=jnp.inf,
            G=G,
            R=R1,
            s=s1,
            t=t1,
            v=v1,
            sigma_m=sigma0,
            alpha_m=alpha0,
            var_i=jnp.array(0.05),
            transform_mode="both",
            normalize_input=False,
        )

        # Denormalize to original space
        R_opt, s_opt, t_opt, v_opt = denormalize_parameters(
            R2, s2, t2, v2, mu_y, sigma_y, mu_x, sigma_x
        )

        # Metrics
        y_full = bcpd_transform(source + v_opt, R_opt, s_opt, t_opt)
        mse = jnp.mean(jnp.sum((target - y_full) ** 2, axis=1))
        scale_err = jnp.abs(s_opt - s_gt)

        results.append(
            {
                "lambda": lam,
                "mse": mse,
                "scale": s_opt,
                "scale_err": scale_err,
                "v_norm": jnp.linalg.norm(v_opt),
            }
        )

        print(
            f"  Lambda={lam:4.1f}: MSE={mse:.6f}, Scale={s_opt:.3f}, ScaleErr={scale_err:.4f}, v_norm={jnp.linalg.norm(v_opt):.4f}"
        )

    # 4. Validation & Assertions

    # (1) MSE should increase as lambda increases (stiffening reduces fit quality)
    for i in range(len(results) - 1):
        assert (
            results[i]["mse"] <= results[i + 1]["mse"] + 1e-6
        ), f"MSE should be non-decreasing with lambda: {results[i]['mse']} vs {results[i+1]['mse']}"

    # (2) v_norm should decrease as lambda increases (stiffening reduces deformation)
    for i in range(len(results) - 1):
        assert (
            results[i]["v_norm"] >= results[i + 1]["v_norm"] - 1e-4
        ), f"Vector field norm should be non-increasing with lambda: {results[i]['v_norm']} vs {results[i+1]['v_norm']}"

    # (3) Find "Best" performers
    best_mse_idx = jnp.argmin(jnp.array([r["mse"] for r in results]))
    best_scale_idx = jnp.argmin(jnp.array([r["scale_err"] for r in results]))

    print("\n[Summary]")
    print(
        f"  Best MSE: Lambda={results[best_mse_idx]['lambda']} (MSE={results[best_mse_idx]['mse']:.6f})"
    )
    print(
        f"  Best Scale Recovery: Lambda={results[best_scale_idx]['lambda']} (Scale={results[best_scale_idx]['scale']:.3f})"
    )

    # Assert that at least one recovery is "good enough" for scale (e.g. at lambda=2.0)
    # Note: lam=2.0 is expected to be a good balance.
    s_best = results[best_scale_idx]["scale"]
    assert jnp.allclose(
        s_best, s_gt, atol=0.1
    ), f"Best scale recovery {s_best} is not close enough to {s_gt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
