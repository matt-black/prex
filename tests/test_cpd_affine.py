import jax
import jax.numpy as jnp

from prex.cpd.affine import align, transform


def test_affine_cpd_exact_match():
    """Test affine CPD with an exact affine transformation."""
    key = jax.random.PRNGKey(0)
    n, d = 100, 3

    # Generate reference points
    ref = jax.random.normal(key, (n, d))

    # ground truth transformation
    A_gt = jnp.array([[1.2, 0.1, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.5]])
    t_gt = jnp.array([0.5, -0.2, 1.0])

    mov = (ref - t_gt) @ jnp.linalg.inv(A_gt)

    (P, A, t), _ = align(
        ref=ref, mov=mov, outlier_prob=0.0, max_iter=100, tolerance=1e-5
    )

    assert jnp.allclose(A, A_gt, atol=2e-3)
    assert jnp.allclose(t, t_gt, atol=2e-3)

    mov_aligned = transform(mov, A, t)
    assert jnp.allclose(mov_aligned, ref, atol=2e-3)


def test_affine_cpd_noisy_match():
    """Test affine CPD with noise."""
    key = jax.random.PRNGKey(1)
    n, d = 100, 3

    mov = jax.random.normal(key, (n, d))

    A_gt = jnp.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 1.0]])
    t_gt = jnp.array([0.1, 0.2, 0.3])

    ref = mov @ A_gt + t_gt

    # add random noise
    noise = 0.01 * jax.random.normal(key, ref.shape)
    ref_noisy = ref + noise

    (P, A, t), _ = align(
        ref=ref_noisy, mov=mov, outlier_prob=0.0, max_iter=100, tolerance=1e-5
    )

    assert jnp.allclose(A, A_gt, atol=0.05)
    assert jnp.allclose(t, t_gt, atol=0.05)

    mov_aligned = transform(mov, A, t)
    mse = jnp.mean((mov_aligned - ref_noisy) ** 2)
    assert mse < 0.001
