import jax
import jax.numpy as jnp

from prex.cpd.rigid import align, transform


def test_rigid_cpd_exact_match():
    """Test rigid CPD with an exact rigid transformation."""
    key = jax.random.PRNGKey(0)
    n, d = 100, 3

    # Generate reference points
    ref = jax.random.normal(key, (n, d))

    # Create ground truth transformation
    theta = jnp.pi / 4
    R_gt = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    s_gt = 1.5
    t_gt = jnp.array([0.5, -0.2, 1.0])

    # Transform points to create moving set (inverse transform logic for alignment)
    # If we want mov aligned to ref, ref = s * (mov @ R.T) + t
    # So mov = (ref - t) @ R / s
    # But usually we generate mov and transform it to match ref.
    # Let's say ref is the target. mov is the source.
    # We want to find R, s, t such that s * (mov @ R.T) + t ~ ref.

    # Let's generate mov first.
    mov = jax.random.normal(key, (n, d))

    # Apply transform to get ref
    ref = s_gt * (mov @ R_gt.T) + t_gt

    # Run CPD
    (P, R, s, t), (var, iter_num) = align(
        ref=ref, mov=mov, outlier_prob=0.0, max_iter=100, tolerance=1e-5
    )

    # Verify parameters
    print(f"R_gt:\n{R_gt}")
    print(f"R:\n{R}")
    print(f"s_gt: {s_gt}, s: {s}")
    print(f"t_gt: {t_gt}, t: {t}")

    assert jnp.allclose(R, R_gt, atol=2e-3)
    assert jnp.allclose(s, s_gt, atol=2e-3)
    assert jnp.allclose(t, t_gt, atol=2e-3)

    # Verify alignment
    mov_aligned = transform(mov, R, s, t)
    err = jnp.mean((mov_aligned - ref) ** 2)
    print(f"MSE: {err}")
    assert jnp.allclose(mov_aligned, ref, atol=2e-3)


def test_rigid_cpd_noisy_match():
    """Test rigid CPD with noise."""
    key = jax.random.PRNGKey(1)
    n, d = 100, 3

    mov = jax.random.normal(key, (n, d))

    theta = jnp.pi / 6
    R_gt = jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta), 0],
            [jnp.sin(theta), jnp.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    s_gt = 1.2
    t_gt = jnp.array([0.1, 0.2, 0.3])

    ref = s_gt * (mov @ R_gt.T) + t_gt

    # Add noise
    noise = 0.01 * jax.random.normal(key, ref.shape)
    ref_noisy = ref + noise

    (P, R, s, t), _ = align(
        ref=ref_noisy, mov=mov, outlier_prob=0.0, max_iter=100, tolerance=1e-5
    )

    # Parameters should be close but not exact
    assert jnp.allclose(R, R_gt, atol=0.05)
    assert jnp.allclose(s, s_gt, atol=0.05)
    assert jnp.allclose(t, t_gt, atol=0.05)

    # Alignment error should be comparable to noise level
    mov_aligned = transform(mov, R, s, t)
    mse = jnp.mean((mov_aligned - ref_noisy) ** 2)
    assert mse < 0.001  # Noise variance is ~1e-4


def test_rigid_cpd_outliers():
    """Test rigid CPD robustness to outliers."""
    key = jax.random.PRNGKey(2)
    n, d = 100, 2
    n_outliers = 20

    # Generate moving points
    mov = jax.random.normal(key, (n, d))
    ref = mov  # identity transform

    # Add outliers to BOTH sets (CPD outlier model works better this way)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    outliers_ref = jax.random.normal(subkey1, (n_outliers, d)) + 5.0
    outliers_mov = jax.random.normal(subkey2, (n_outliers, d)) + 5.0

    ref_with_outliers = jnp.concatenate([ref, outliers_ref], axis=0)
    mov_with_outliers = jnp.concatenate([mov, outliers_mov], axis=0)

    (P, R, s, t), _ = align(
        ref=ref_with_outliers,
        mov=mov_with_outliers,
        outlier_prob=0.21,
        max_iter=100,
        tolerance=1e-5,
    )
    # should recover identity transform
    assert jnp.allclose(R, jnp.eye(d), atol=0.1)
    assert jnp.allclose(s, 1.0, atol=0.1)
    assert jnp.allclose(t, jnp.zeros(d), atol=0.1)
