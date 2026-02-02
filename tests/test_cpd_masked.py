"""Test masked CPD functionality."""

import jax.numpy as jnp
import pytest

from prex import cpd


def test_masked_affine():
    """Test that masking works for affine alignment."""
    # Create points: ref is a 1x1 square, mov is shifted
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [1.1, 1.1], [0.1, 1.1]])

    # Run with identity mask (full matching)
    mask_full = jnp.ones((4, 4))
    (P_full, A_full, t_full), _ = cpd.align(
        ref,
        mov,
        method="affine",
        outlier_prob=0.0,
        num_iter=10,
        tolerance=1e-6,
        mask=mask_full,
    )

    # Run with diagonal mask (forcing 1-1 matching)
    mask_diag = jnp.eye(4)
    (P_diag, A_diag, t_diag), _ = cpd.align(
        ref,
        mov,
        method="affine",
        outlier_prob=0.0,
        num_iter=10,
        tolerance=1e-6,
        mask=mask_diag,
    )

    # P_diag should be identity because of mask
    assert jnp.allclose(P_diag, jnp.eye(4), atol=1e-2)


def test_masked_rigid():
    """Test that masking works for rigid alignment."""
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]])

    mask = jnp.eye(3)
    (P, R, s, t), _ = cpd.align(
        ref,
        mov,
        method="rigid",
        outlier_prob=0.0,
        num_iter=10,
        tolerance=1e-6,
        mask=mask,
    )

    # Mask should be enforced
    assert jnp.all(P[mask == 0] == 0)
    assert jnp.all(P[mask == 1] > 0.9)


def test_masked_nonrigid():
    """Test that masking works for nonrigid alignment."""
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]])

    mask = jnp.eye(3)
    (P, W, G), _ = cpd.align(
        ref,
        mov,
        method="nonrigid",
        outlier_prob=0.1,
        num_iter=5,
        tolerance=1e-6,
        mask=mask,
    )

    # Mask should be enforced
    assert jnp.all(P[mask == 0] == 0)
    assert jnp.all(P[mask == 1] > 0.5)  # Lower threshold due to outlier_prob


def test_mask_exclusion():
    """Test that mask effectively excludes matching."""
    ref = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    mov = jnp.array([[0.1, 0.0], [1.1, 1.0]])  # Shifted by 0.1

    # Mask that forbids matching point 0 to point 0
    mask = jnp.array([[0, 1], [1, 0]])

    (P, A, t), _ = cpd.align(
        ref,
        mov,
        method="affine",
        outlier_prob=0.0,
        num_iter=5,
        tolerance=1e-6,
        mask=mask,
    )

    # Diagonal should be 0 despite points being identical
    assert jnp.all(jnp.diag(P) == 0)
    # Off-diagonal should be 1
    assert P[0, 1] > 0.99
    assert P[1, 0] > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
