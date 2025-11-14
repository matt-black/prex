import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from prex.grbf import rigid as grbf
from prex.rigid import transform_gmm as transform_gmm_rigid


@jax.jit
def create_diagonal_covariances(
    variances: Float[Array, " d"],
) -> Float[Array, "d d"]:
    """Create diagonal covariance matrices from variance vectors.

    Args:
        variances: Variance for each dimension, shape (n_components, d)

    Returns:
        Diagonal covariance matrices, shape (n_components, d, d)
    """
    n_components, d = variances.shape
    covs = jnp.zeros((n_components, d, d))
    indices = jnp.diag_indices(d)
    return covs.at[:, indices[0], indices[1]].set(variances)


def make_random_gmm(n_comp: int, n_dim: int, key: PRNGKeyArray):
    subkeys = jr.split(key, 3)
    means = jax.random.normal(subkeys[0], (n_comp, n_dim))
    vars = jax.random.uniform(subkeys[1], (n_comp, n_dim)) + 0.1
    weights = jax.random.uniform(subkeys[2], (n_comp,))
    weights = weights / jnp.sum(weights)
    covs = create_diagonal_covariances(vars)
    return means, covs, weights


def test_transformation_runs():
    """Test the transformation function with fake data, make sure it runs without error."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # sample data parameters
    n_components = 5
    d = 3
    n_centers = 4

    means, covariances, _ = make_random_gmm(n_components, d, key)

    # Transformation parameters
    scale = jnp.array(1.5)
    rotation = jnp.eye(d)  # Identity rotation for testing
    translation = jnp.array([1.0, -0.5, 2.0])

    # GRBF parameters
    rbf_centers = jax.random.normal(subkey, (n_centers, d))
    key, subkey = jax.random.split(key)
    rbf_weights = jax.random.normal(subkey, (n_centers, d)) * 0.1
    rbf_bandwidth = 1.0

    # Apply transformation
    mu, cov = grbf.transform_gmm(
        means,
        covariances,
        scale,
        rotation,
        translation,
        rbf_centers,
        rbf_weights,
        rbf_bandwidth,
    )

    assert jnp.logical_not(jnp.any(jnp.isnan(mu)))
    assert jnp.logical_not(jnp.any(jnp.isnan(cov)))


def test_transform_identity():
    key = jr.PRNGKey(1001)
    mu, cov, _ = make_random_gmm(10, 3, key)
    mu2, cov2 = grbf.transform_gmm(
        mu,
        cov,
        jnp.array(1.0),
        jnp.eye(3),
        jnp.zeros((3,)),
        jnp.zeros_like(mu),
        mu,
        1.0,
    )
    assert jnp.allclose(mu, mu2)
    assert jnp.allclose(cov, cov2)


def test_transform_zero_rbf_affine():
    mu, cov, _ = make_random_gmm(5, 3, jr.PRNGKey(21))
    scale = jnp.array(1.125)
    rot = jnp.eye(3)
    trans = jnp.ones((3,)) * 1.25
    rbf_bandwidth = 1.0
    mu2, cov2 = transform_gmm_rigid(mu, cov, scale, rot, trans)
    mu3, cov3 = grbf.transform_gmm(
        mu, cov, scale, rot, trans, jnp.zeros_like(mu), mu, rbf_bandwidth
    )
    assert jnp.allclose(mu2, mu3)
    assert jnp.allclose(cov2, cov3)
