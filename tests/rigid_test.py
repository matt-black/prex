from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from prex.dist import self_energy_gmm, self_energy_gmm_spherical
from prex.rigid import (
    optimize_single_scale,
    optimize_single_scale_spherical,
    transform_gmm,
    transform_gmm_rotangles,
    transform_means_rotangles,
    unpack_params,
)
from prex.util import rotation_matrix_3d

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
    means = jax.random.normal(subkeys[0], (n_comp, n_dim))
    vars = jax.random.uniform(subkeys[1], (n_comp, n_dim)) + 0.1
    weights = jax.random.uniform(subkeys[2], (n_comp,))
    weights = weights / jnp.sum(weights)
    covs = create_diagonal_covariances(vars)
    return means, covs, weights


def test_rigid_transform_identity():
    key = jr.PRNGKey(42)
    mu, cov, _ = make_random_gmm(5, 3, key)
    rot = rotation_matrix_3d(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    mu2, cov2 = transform_gmm(mu, cov, jnp.array(1.0), rot, jnp.zeros((3,)))
    assert jnp.allclose(mu, mu2)
    assert jnp.allclose(cov, cov2)


def test_rigid_transform_trans():
    key = jr.PRNGKey(41)
    mu, cov, _ = make_random_gmm(5, 3, key)
    rot = rotation_matrix_3d(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
    mu2, cov2 = transform_gmm(mu, cov, jnp.array(1.0), rot, jnp.ones((3,)))
    assert jnp.allclose(mu + 1, mu2)
    assert jnp.allclose(cov, cov2)


def test_rigid_opt_rot():
    key = jr.PRNGKey(40)
    mu, cov, wgt = make_random_gmm(5, 3, key)
    # define fwd transform
    alpha = jnp.array(jnp.pi / 8)
    beta = jnp.array(0.0)
    gamma = jnp.array(jnp.pi / 10)
    trans = jnp.array([0.0, 0.0, 0.0])
    scale = jnp.array(1.0)
    mu2, cov2 = transform_gmm_rotangles(
        mu, cov, scale, alpha, beta, gamma, trans, 3
    )
    rescaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()
    max_iter = 100
    par_f, (_, l2_final, num_iter) = optimize_single_scale(
        mu2,
        cov2,
        wgt,
        mu,
        cov,
        wgt,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.zeros((3,)),
        1.0,
        rescaling,
        grad_tol=1e-8,
        loss_tol=-1 + 1e-8,
        max_iter=max_iter,
    )
    assert num_iter < max_iter
    scale_f, alpha_f, beta_f, gamma_f, trans_f = unpack_params(par_f)
    assert jnp.isclose(l2_final, -1, atol=1e-2)
    assert jnp.isclose(scale, scale_f, atol=1e-3)
    assert jnp.isclose(alpha, alpha_f, atol=1e-3)
    assert jnp.isclose(beta, beta_f, atol=1e-3)
    assert jnp.isclose(gamma, gamma_f, atol=1e-3)
    assert jnp.allclose(trans, trans_f, atol=1e-3)


def test_rigid_opt_rot_scale():
    key = jr.PRNGKey(40)
    mu, cov, wgt = make_random_gmm(5, 3, key)
    # define fwd transform
    alpha = jnp.array(jnp.pi / 8)
    beta = jnp.array(0.0)
    gamma = jnp.array(jnp.pi / 10)
    trans = jnp.array([0.0, 0.0, 0.0])
    scale = jnp.array(2.0)
    mu2, cov2 = transform_gmm_rotangles(
        mu, cov, scale, alpha, beta, gamma, trans, 3
    )
    rescaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()
    max_iter = 100
    par_f, (_, _, num_iter) = optimize_single_scale(
        mu2,
        cov2,
        wgt,
        mu,
        cov,
        wgt,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.zeros((3,)),
        1.0,
        rescaling,
        loss_tol=-1 + 1e-8,
        max_iter=max_iter,
    )

    assert num_iter < max_iter
    scale_f, alpha_f, beta_f, gamma_f, trans_f = unpack_params(par_f)
    assert jnp.isclose(scale, scale_f, atol=1e-3)
    assert jnp.isclose(alpha, alpha_f, atol=1e-3)
    assert jnp.isclose(beta, beta_f, atol=1e-3)
    assert jnp.isclose(gamma, gamma_f, atol=1e-3)
    assert jnp.allclose(trans, trans_f, atol=1e-2)


def test_rigid_opt_rot_scale_trans():
    key = jr.PRNGKey(40)
    mu, cov, wgt = make_random_gmm(5, 3, key)
    # define fwd transform
    alpha = jnp.array(jnp.pi / 8)
    beta = jnp.array(0.0)
    gamma = jnp.array(jnp.pi / 10)
    trans = jnp.array([0.5, 0.5, 0.5])
    scale = jnp.array(2.0)
    mu2, cov2 = transform_gmm_rotangles(
        mu, cov, scale, alpha, beta, gamma, trans, 3
    )
    rescaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()

    max_iter = 100
    par_f, (_, _, num_iter) = optimize_single_scale(
        mu2,
        cov2,
        wgt,
        mu,
        cov,
        wgt,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.zeros((3,)),
        1.0,
        rescaling,
        loss_tol=-1 + 1e-8,
        max_iter=max_iter,
    )

    assert num_iter < max_iter
    scale_f, alpha_f, beta_f, gamma_f, trans_f = unpack_params(par_f)
    assert jnp.isclose(scale, scale_f, atol=1e-3)
    assert jnp.isclose(alpha, alpha_f, atol=1e-3)
    assert jnp.isclose(beta, beta_f, atol=1e-3)
    assert jnp.isclose(gamma, gamma_f, atol=1e-3)
    assert jnp.allclose(trans, trans_f, atol=1e-2)


def test_rigid_opt_rst_withsave():
    """Test that saving intermediate results doesn't mess up optimization"""
    key = jr.PRNGKey(40)
    mu, cov, wgt = make_random_gmm(5, 3, key)
    # define fwd transform
    alpha = jnp.array(jnp.pi / 8)
    beta = jnp.array(0.0)
    gamma = jnp.array(jnp.pi / 10)
    trans = jnp.array([0.5, 0.5, 0.5])
    scale = jnp.array(2.0)
    mu2, cov2 = transform_gmm_rotangles(
        mu, cov, scale, alpha, beta, gamma, trans, 3
    )
    rescaling = 1 / self_energy_gmm(mu2, cov2, wgt).item()

    max_iter = 100
    par_f1, (_, _, num_iter1) = optimize_single_scale(
        mu2,
        cov2,
        wgt,
        mu,
        cov,
        wgt,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.zeros((3,)),
        1.0,
        rescaling,
        max_iter=max_iter,
    )
    with TemporaryDirectory() as save_path:
        par_f2, (_, _, num_iter2) = optimize_single_scale(
            mu2,
            cov2,
            wgt,
            mu,
            cov,
            wgt,
            jnp.array(1.0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.zeros((3,)),
            1.0,
            rescaling,
            max_iter=max_iter,
            save_path=save_path,
        )
    assert num_iter1 == num_iter2
    assert jnp.allclose(par_f1, par_f2)


def test_rigid_opt_rot_scale_trans_spherical():
    key = jr.PRNGKey(1029)
    n_dim = 2
    mu, _, wgt = make_random_gmm(5, n_dim, key)
    # define fwd transform
    alpha = jnp.array(jnp.pi / 8)
    beta = jnp.array(0.0)
    gamma = jnp.array(jnp.pi / 10)
    trans = jnp.array(
        [
            0.5,
        ]
        * n_dim
    )
    scale = jnp.array(2.0)
    mu2 = transform_means_rotangles(mu, scale, alpha, beta, gamma, trans, n_dim)
    rescaling = 1 / self_energy_gmm_spherical(mu2, wgt, 1.0, n_dim).item()
    max_iter = 100
    par_f, (_, _, num_iter) = optimize_single_scale_spherical(
        mu2,
        wgt,
        mu,
        wgt,
        jnp.array(1.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.array(0.0),
        jnp.zeros((n_dim,)),
        var_fixed=1.0,
        var_moving=1.0,
        l2_scaling=rescaling,
        loss_tol=-1 + 1e-8,
        max_iter=max_iter,
    )
    assert num_iter < max_iter
    scale_f, alpha_f, _, _, trans_f = unpack_params(par_f)
    assert jnp.isclose(scale, scale_f, atol=1e-3)
    assert jnp.isclose(alpha, alpha_f, atol=1e-3)
    assert jnp.allclose(trans, trans_f, atol=1e-2)
