import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from prex.gmm import dist

jax.config.update("jax_platform_name", "cpu")


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
    means = jr.normal(subkeys[0], (n_comp, n_dim))
    vars = jr.uniform(subkeys[1], (n_comp, n_dim)) + 0.1
    weights = jr.uniform(subkeys[2], (n_comp,))
    weights = weights / jnp.sum(weights)
    covs = create_diagonal_covariances(vars)
    return means, covs, weights


def test_pdf_logpdf_match_self():
    key = jr.PRNGKey(11)
    keys = jr.split(key, 10)

    def test_pair(_: None, key: PRNGKeyArray) -> Bool:
        mu, cov, wgt = make_random_gmm(10, 3, key)
        pdf = dist.self_energy_gmm_pdf(mu, cov, wgt)
        logpdf = dist.self_energy_gmm_logpdf(mu, cov, wgt)
        jax.debug.print("{:.3f}, {:.3f}", pdf, logpdf)
        return None, jnp.isclose(pdf, logpdf)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)


def test_pdf_logpdf_match_cross():
    keys = jr.split(jr.PRNGKey(12), (2, 10))

    def test_pair(_: None, ks: PRNGKeyArray) -> Bool:
        mu1, cov1, wgt1 = make_random_gmm(10, 3, ks[0])
        mu2, cov2, wgt2 = make_random_gmm(10, 3, ks[1])
        pdf = dist.cross_energy_gmms_pdf(mu1, cov1, wgt1, mu2, cov2, wgt2)
        logpdf = pdf = dist.cross_energy_gmms_logpdf(
            mu1, cov1, wgt1, mu2, cov2, wgt2
        )
        return None, jnp.isclose(pdf, logpdf)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)


def test_cross_scan_match():
    keys = jr.split(jr.PRNGKey(12), (2, 10))

    def test_pair(_: None, ks: PRNGKeyArray) -> Bool:
        mu1, cov1, wgt1 = make_random_gmm(10, 3, ks[0])
        mu2, cov2, wgt2 = make_random_gmm(10, 3, ks[1])
        e_ref = dist.cross_energy_gmms_pdf(mu1, cov1, wgt1, mu2, cov2, wgt2)
        e_scn = dist.cross_energy_gmms_scan(mu1, cov1, wgt1, mu2, cov2, wgt2)
        return None, jnp.isclose(e_ref, e_scn)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)


def test_self_scan_match():
    keys = jr.split(jr.PRNGKey(12), (2, 10))

    def test_pair(_: None, ks: PRNGKeyArray) -> Bool:
        mu1, cov1, wgt1 = make_random_gmm(10, 3, ks[0])
        e_ref = dist.self_energy_gmm(mu1, cov1, wgt1)
        e_scn = dist.self_energy_gmm(mu1, cov1, wgt1)
        return None, jnp.isclose(e_ref, e_scn)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)


def test_self_spherical_match():
    keys = jr.split(jr.PRNGKey(12), (2, 10))
    var = 1.0
    cov = jnp.repeat(jnp.eye(3)[None, ...], 10, axis=0)

    def test_pair(_: None, ks: PRNGKeyArray) -> Bool:
        mu1, _, wgt1 = make_random_gmm(10, 3, ks[0])
        e_ref = dist.self_energy_gmm(mu1, cov, wgt1)
        e_sph = dist.self_energy_gmm_spherical(mu1, wgt1, var, 3)
        return None, jnp.isclose(e_ref, e_sph)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)


def test_cross_spherical_match():
    keys = jr.split(jr.PRNGKey(12), (2, 10))
    var = 1.0
    cov = jnp.repeat(jnp.eye(3)[None, ...], 10, axis=0)

    def test_pair(_: None, ks: PRNGKeyArray) -> Bool:
        mu1, _, wgt1 = make_random_gmm(10, 3, ks[0])
        mu2, _, wgt2 = make_random_gmm(10, 3, ks[1])
        e_ref = dist.cross_energy_gmms(mu1, cov, wgt1, mu2, cov, wgt2)
        e_sph = dist.cross_energy_gmms_spherical(
            mu1, wgt1, mu2, wgt2, var, var, 3
        )
        return None, jnp.isclose(e_ref, e_sph)

    _, tests = jax.lax.scan(test_pair, None, keys)
    assert jnp.all(tests)
