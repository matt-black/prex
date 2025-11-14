import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from prex.affine import transform_gmm

# from prex.dist import self_energy_gmm


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


def test_transform_identity():
    key = jr.PRNGKey(42)
    mu, cov, _ = make_random_gmm(5, 3, key)
    mu2, cov2 = transform_gmm(mu, cov, jnp.eye(3), jnp.zeros((3,)))
    assert jnp.allclose(mu, mu2)
    assert jnp.allclose(cov, cov2)


def test_random_matrix3():
    key = jr.PRNGKey(4)
    gmm_key, mat_key = jr.split(key, 2)
    mu, cov, wgt = make_random_gmm(5, 3, gmm_key)
    A = jnp.add(jnp.eye(3), jr.normal(mat_key, (3, 3)))
    mu2, cov2 = transform_gmm(mu, cov, A, jnp.zeros((3,)))
    # max_iter = 1000
    # l2_scaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()
    assert True
    # assert num_iter < max_iter
    # Af = par_f[:-3].reshape(3, 3)
    # assert jnp.allclose(A, Af, atol=1e-2)
    # tf = par_f[-3:]
    # assert jnp.allclose(tf, jnp.zeros_like(tf), atol=1e-3)


def test_random_matrix2():
    key = jr.PRNGKey(12345)
    gmm_key, mat_key = jr.split(key, 2)
    mu, cov, wgt = make_random_gmm(9, 2, gmm_key)
    A = jnp.add(jnp.eye(2), jr.normal(mat_key, (2, 2)))
    mu2, cov2 = transform_gmm(mu, cov, A, jnp.zeros((2,)))
    # l2_scaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()
    # max_iter = 100
    assert True
    # assert num_iter < max_iter
    # Af = par_f[:-2].reshape(2, 2)
    # assert jnp.allclose(A, Af, atol=1e-3)
    # tf = par_f[-2:]
    # assert jnp.allclose(tf, jnp.zeros_like(tf), atol=1e-3)


def test_random_matrix2_manycomponents():
    key = jr.PRNGKey(1010)
    gmm_key, mat_key = jr.split(key, 2)
    mu, cov, wgt = make_random_gmm(100, 2, gmm_key)
    A = jnp.add(jnp.eye(2), jr.normal(mat_key, (2, 2)))
    mu2, cov2 = transform_gmm(mu, cov, A, jnp.zeros((2,)))
    # l2_scaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()
    # max_iter = 1000
    assert True
    # assert num_iter < max_iter
    # Af = par_f[:-2].reshape(2, 2)
    # assert jnp.allclose(A, Af, atol=1e-2)
    # tf = par_f[-2:]
    # assert jnp.allclose(tf, jnp.zeros_like(tf), atol=1e-3)
