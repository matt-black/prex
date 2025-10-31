"""Distance metrics between GMMs"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float

__all__ = [
    "mixture_component_overlap",
    "self_energy_gmm",
    "cross_energy_gmms",
    "l2_distance_gmm",
    "l2_distance_gmm_opt",
]


def mixture_component_overlap(
    mean1: Float[Array, " d"],
    cov1: Float[Array, "d d"],
    wgt1: Float[Array, ""] | float,
    mean2: Float[Array, " d"],
    cov2: Float[Array, "d d"],
    wgt2: Float[Array, ""] | float,
) -> Float[Array, ""]:
    """Compute the integral of the product of two components of GMMs.

    Args:
        mean1 (Float[Array, " d"]): mean of first Gaussian component
        cov1 (Float[Array, "d d"]): covariance matrix of first Gaussian component
        wgt1 (Float[Array, ""]|float): scalar weight of first Gaussian component
        mean2 (Float[Array, " d"]): mean of first Gaussian component
        cov2 (Float[Array, "d d"]): covariance matrix of first Gaussian component
        wgt2 (Float[Array, ""]|float): scalar weight of first Gaussian component

    Returns:
        Float[Array, ""]: scalar overlap value, weight1 * weight2 * ∫ N(x|mean1,cov1) N(x|mean2,cov2) dx
    """
    dif = mean1 - mean2
    cov = cov1 + cov2
    return wgt1 * wgt2 * multivariate_normal.pdf(dif, jnp.zeros_like(dif), cov)


def self_energy_gmm(
    means: Float[Array, "n_comp d"],
    covs: Float[Array, "n_comp d d"],
    wgts: Float[Array, " n_comp"],
) -> Float[Array, ""]:
    """Compute the self-energy for a Gaussian Mixture Model

    Args:
        means (Float[Array, " d"]): mean of Gaussian components
        covs (Float[Array, "d d"]): covariance matrix of Gaussian components
        wgts (Float[Array, ""]|float): scalar weights of Gaussian component

    Returns:
        Float[Array, ""]: self energy
    """
    return jnp.sum(
        jax.vmap(
            lambda mu0, cov0, w0: jax.vmap(
                lambda mu1, cov1, w1: mixture_component_overlap(
                    mu0, cov0, w0, mu1, cov1, w1
                )
            )(means, covs, wgts)
        )(means, covs, wgts)
    )


def cross_energy_gmms(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the cross-energy betwen two Gaussian Mixture Models

    Args:
        means1 (Float[Array, " d"]): mean of Gaussian components for 1st GMM
        covs1 (Float[Array, "d d"]): covariance matrix of Gaussian components for 1st GMM
        wgts1 (Float[Array, ""]|float): scalar weights of Gaussian component for 1st GMM
        means2 (Float[Array, " d"]): mean of Gaussian components for 2nd GMM
        covs2 (Float[Array, "d d"]): covariance matrix of Gaussian components for 2nd GMM
        wgts2 (Float[Array, ""]|float): scalar weights of Gaussian component for 2nd GMM

    Returns:
        Float[Array, ""]: cross energy
    """
    return jnp.sum(
        jax.vmap(
            lambda m0, c0, w0: jax.vmap(
                lambda m1, c1, w1: mixture_component_overlap(
                    m0, c0, w0, m1, c1, w1
                )
            )(means1, covs1, wgts1)
        )(means2, covs2, wgts2)
    )


def l2_distance_gmm(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the L2 distance between 2 Gaussian mixture models.

    Args:
        means1 (Float[Array, "nc1 d"]): means of each component, shape (n, d)
        covs1 (Float[Array, "nc1 d d"]): covariance matrix of each component, shape (n, d, d)
        wgts1 (Float[Array, " nc1"]): weights of each component, shape (n,)
        means2 (Float[Array, "nc1 d"]): means of each component, shape (m, d)
        covs2 (Float[Array, "nc1 d d"]): covariance matrix of each component, shape (m, d, d)
        wgts2 (Float[Array, " nc1"]): weights of each component, shape (m,)

    Returns:
        Float[Array, ""]: scalar L2 distance between 2 GMMs
    """
    self1 = self_energy_gmm(means1, covs1, wgts1)
    self2 = self_energy_gmm(means2, covs2, wgts2)
    cross = cross_energy_gmms(means1, covs1, wgts1, means2, covs2, wgts2)
    return self1 - 2 * cross + self2


def l2_distance_gmm_opt(
    means_fixed: Float[Array, "nc1 d"],
    covs_fixed: Float[Array, "nc1 d d"],
    wgts_fixed: Float[Array, " nc1"],
    means_moving: Float[Array, "nc2 d"],
    covs_moving: Float[Array, "nc2 d d"],
    wgts_moving: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the L2 distance between 2 Gaussian mixture models, excluding the constant term for the fixed GMM.

    Args:
        means_fixed (Float[Array, "nc1 d"]): means of each component, shape (n, d)
        covs_fixed (Float[Array, "nc1 d d"]): covariance matrix of each component, shape (n, d, d)
        wgts_fixed (Float[Array, " nc1"]): weights of each component, shape (n,)
        means_moving (Float[Array, "nc1 d"]): means of each component, shape (m, d)
        covs_moving (Float[Array, "nc1 d d"]): covariance matrix of each component, shape (m, d, d)
        wgts_moving (Float[Array, " nc1"]): weights of each component, shape (m,)

    Returns:
        Float[Array, ""]: scalar L2 distance between 2 GMMs, excluding the self-energy of the fixed distribution.

    Notes:
        Use this when optimizing, since the self-energy of the fixed GMM is fixed and we can save computation time by just not computing it. Also note that since its constant it doesn't affect the optimization (gradient of constant is zero...)
    """
    self_energy = self_energy_gmm(means_moving, covs_moving, wgts_moving)
    cross_energy = cross_energy_gmms(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
    )
    return self_energy - 2 * cross_energy
