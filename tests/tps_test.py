import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from prex import tps
from prex.affine import transform_gmm as transform_gmm_affine


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
    mu2, cov2 = tps.transform_gmm(
        mu, cov, jnp.eye(3), jnp.zeros((3,)), mu, jnp.zeros_like(mu)
    )
    assert jnp.allclose(mu, mu2)
    assert jnp.allclose(cov, cov2)


def test_transform_rbf_notnan3():
    key = jr.PRNGKey(42)
    mu, cov, _ = make_random_gmm(5, 3, key)
    mu2, cov2 = tps.transform_gmm(
        mu, cov, jnp.eye(3), jnp.zeros((3,)), mu, jnp.ones_like(mu) / 0.1
    )
    assert jnp.logical_not(jnp.any(jnp.isnan(mu2)))
    assert jnp.logical_not(jnp.any(jnp.isnan(cov2)))


def test_transform_rbf_notnan2():
    key = jr.PRNGKey(42)
    mu, cov, _ = make_random_gmm(5, 2, key)
    mu2, cov2 = tps.transform_gmm(
        mu, cov, jnp.eye(2), jnp.zeros((2,)), mu, jnp.ones_like(mu) / 0.1
    )
    assert jnp.logical_not(jnp.any(jnp.isnan(mu2)))
    assert jnp.logical_not(jnp.any(jnp.isnan(cov2)))


def test_transform_zero_rbf_affine():
    key = jr.PRNGKey(21)
    gmm_key, mat_key = jr.split(key, 2)
    mu, cov, _ = make_random_gmm(5, 3, gmm_key)
    A = jnp.add(jnp.eye(3), jr.normal(mat_key, (3, 3)))
    mu2, cov2 = transform_gmm_affine(mu, cov, A, jnp.zeros((3,)))
    mu3, cov3 = tps.transform_gmm(
        mu, cov, A, jnp.zeros((3,)), mu, jnp.zeros_like(mu)
    )
    assert jnp.allclose(mu2, mu3)
    assert jnp.allclose(cov2, cov3)


# def test_random_converges():
#     key = jr.PRNGKey(81)
#     gmm_key, mat_key = jr.split(key, 2)
#     mu, cov, wgt = make_random_gmm(5, 3, gmm_key)
#     A = jnp.add(jnp.eye(3), jr.normal(mat_key, (3, 3)))
#     mu2, cov2 = transform_gmm_affine(mu, cov, A, jnp.ones((3,))*0.25)
#     l2_rescaling = 1.0 / self_energy_gmm(mu2, cov2, wgt).item()
#     max_iter = 100
#     lambda_ = 0.0
#     par_f, (_, loss_final, num_iter) = optimize_single_scale(
#         mu2, cov2, wgt, mu, cov, wgt, jnp.eye(3),
#         jnp.zeros((3,)), jnp.ones_like(mu2)*0.001,
#         1.0, l2_rescaling, lambda_,
#         max_iter=max_iter
#     )
#     _, _, rbf_wgts = unpack_params(par_f, mu2.shape[0], 3)
#     K = tps_rbf(mu2, mu2)
#     tps_E = tps_bending_energy(K, rbf_wgts)
#     assert num_iter < max_iter
#     assert loss_final
