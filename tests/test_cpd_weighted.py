"""Test weighted CPD functionality."""

import jax.numpy as jnp
import pytest

from prex.cpd import affine, nonrigid, rigid


def test_weighted_expectation_relative_weights():
    """Test that uniformly scaled weights produce the same result (weights are relative)."""
    from prex.cpd._matching import expectation_weighted

    x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    y = jnp.array([[0.1, 0.1], [1.1, 1.1]])
    var = jnp.array(0.1)
    w = 0.1

    # Test with weights that sum to 2.0
    weights1 = jnp.array([1.0, 1.0])
    P1 = expectation_weighted(x, y, var, w, weights1)

    # Test with weights that sum to 4.0 (uniformly scaled by 2)
    weights2 = jnp.array([2.0, 2.0])
    P2 = expectation_weighted(x, y, var, w, weights2)

    # The matching matrices should be identical (weights are relative)
    assert jnp.allclose(P1, P2, atol=1e-6)

    # Test with non-uniform weights
    weights3 = jnp.array([2.0, 1.0])  # First point has 2x weight
    P3 = expectation_weighted(x, y, var, w, weights3)

    # This should be different from uniform weights
    assert not jnp.allclose(P1, P3, atol=1e-6)


def test_weighted_affine_zero_weight():
    """Test that zero weight effectively removes a point."""
    # Create simple point clouds
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    mov = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [5.0, 5.0]]
    )  # Last point is outlier

    # Run with near-zero weight on the outlier
    weights = jnp.array([1.0, 1.0, 1.0, 1e-10])
    (P, A, t), var = affine.align_fixed_iter(
        ref, mov, outlier_prob=0.1, num_iter=10, moving_weights=weights
    )

    # The last point should have very low matching probabilities
    assert jnp.all(P[3, :] < 1e-5)


def test_weighted_nonrigid_nonuniform_weights():
    """Test that non-uniform weights produce different results."""
    # Create simple point clouds
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    mov = jnp.array([[0.0, 0.0], [1.0, 0.0]])

    # Test with uniform weights
    weights1 = jnp.array([1.0, 1.0])
    (P1, W1, G1), var1 = nonrigid.align_fixed_iter(
        ref,
        mov,
        outlier_prob=0.1,
        regularization_param=1.0,
        kernel_stddev=1.0,
        num_iter=3,
        moving_weights=weights1,
    )

    # Test with non-uniform weights
    weights2 = jnp.array([2.0, 1.0])
    (P2, W2, G2), var2 = nonrigid.align_fixed_iter(
        ref,
        mov,
        outlier_prob=0.1,
        regularization_param=1.0,
        kernel_stddev=1.0,
        num_iter=3,
        moving_weights=weights2,
    )

    # Results should be different
    assert not jnp.allclose(P1, P2, atol=1e-6)


def test_weighted_rigid_with_weights():
    """Test that rigid CPD works with weights."""
    # Create simple point clouds
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [1.1, 1.1]])

    # Test with weights
    weights = jnp.array([1.0, 1.0, 0.5])
    (P, R, s, t), var = rigid.align_fixed_iter(
        ref, mov, outlier_prob=0.1, num_iter=5, moving_weights=weights
    )

    # Should complete without errors and produce reasonable results
    assert P.shape == (3, 3)
    assert R.shape == (2, 2)
    assert jnp.isfinite(s)
    assert t.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
