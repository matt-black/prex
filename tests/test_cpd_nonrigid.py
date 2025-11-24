import jax
import jax.numpy as jnp

from prex.cpd.nonrigid import align, interpolate, transform
from prex.util import sqdist


def test_nonrigid_cpd_deformable_match():
    """Test non-rigid CPD with a known deformation."""

    # Create a line of points
    n = 50
    x = jnp.linspace(-2, 2, n)
    y = jnp.zeros(n)
    mov = jnp.stack([x, y], axis=1)

    # deform into a sine wave
    ref_y = jnp.sin(x * jnp.pi)
    ref = jnp.stack([x, ref_y], axis=1)

    beta = 0.5
    lmbda = 10.0

    (P, W, G), _ = align(
        ref=ref,
        mov=mov,
        outlier_prob=0.0,
        regularization_param=lmbda,
        kernel_stddev=beta,
        max_iter=100,
        tolerance=1e-5,
    )

    mov_aligned = transform(mov, G, W)
    mse = jnp.mean((mov_aligned - ref) ** 2)

    assert mse < 0.01


def test_nonrigid_cpd_interpolation():
    """Test interpolation of the deformation field."""
    key_mov, key_W, key_new = jax.random.split(jax.random.PRNGKey(0), 3)
    n = 20
    d = 2

    mov = jax.random.normal(key_mov, (n, d))

    beta = 1.0
    W = jax.random.normal(key_W, (n, d)) * 0.1

    interp_pts = mov

    G = jnp.exp(-sqdist(mov, mov) / (2 * beta**2))

    transformed = transform(mov, G, W)
    displacement = transformed - mov

    interpolated_displacement = interpolate(mov, interp_pts, W, beta)

    assert jnp.allclose(displacement, interpolated_displacement, atol=1e-5)

    new_pts = jax.random.normal(key_new, (5, d))
    interp_new = interpolate(mov, new_pts, W, beta)

    G_new = jnp.exp(-sqdist(new_pts, mov) / (2 * beta**2))
    expected_new = G_new @ W

    assert jnp.allclose(interp_new, expected_new, atol=1e-5)
