"""Distance metrics between GMMs"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax.tree_util import Partial
from jaxtyping import Array, Float


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


@Partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def mixture_component_overlap_spherical(
    mean1: Float[Array, " d"],
    wgt1: Float[Array, ""] | float,
    mean2: Float[Array, " d"],
    wgt2: Float[Array, ""] | float,
    var1: float,
    var2: float,
    n_dim: int,
) -> Float[Array, ""]:
    cov1 = jnp.eye(n_dim) * var1
    cov2 = jnp.eye(n_dim) * var2
    return mixture_component_overlap(mean1, cov1, wgt1, mean2, cov2, wgt2)


def log_mixture_component_overlap(
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
    return (
        jnp.log(wgt1)
        + jnp.log(wgt2)
        + multivariate_normal.logpdf(dif, jnp.zeros_like(dif), cov)
    )


@Partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def log_mixture_component_overlap_spherical(
    mean1: Float[Array, " d"],
    wgt1: Float[Array, ""] | float,
    mean2: Float[Array, " d"],
    wgt2: Float[Array, ""] | float,
    var1: float,
    var2: float,
    n_dim: int,
) -> Float[Array, ""]:
    cov1 = jnp.eye(n_dim) * var1
    cov2 = jnp.eye(n_dim) * var2
    return log_mixture_component_overlap(mean1, cov1, wgt1, mean2, cov2, wgt2)


def self_energy_gmm_pdf(
    means: Float[Array, "n_comp d"],
    covs: Float[Array, "n_comp d d"],
    wgts: Float[Array, " n_comp"],
) -> Float[Array, ""]:
    """Compute the self-energy for a Gaussian Mixture Model

    Args:
        means (Float[Array, "n_comp d"]): mean of Gaussian components
        covs (Float[Array, "n_comp d d"]): covariance matrix of Gaussian components
        wgts (Float[Array, " n_comp"]): scalar weights of Gaussian components

    Returns:
        Float[Array, ""]: self energy
    """
    return jnp.sum(
        jax.vmap(
            lambda mu0, cov0, w0: jax.vmap(
                lambda mu1, cov1, w1: mixture_component_overlap(
                    mu0, cov0, w0, mu1, cov1, w1
                ),
                (0, 0, 0),
                0,
            )(means, covs, wgts),
            (0, 0, 0),
            0,
        )(means, covs, wgts)
    )


def self_energy_gmm_logpdf(
    means: Float[Array, "n_comp d"],
    covs: Float[Array, "n_comp d d"],
    wgts: Float[Array, " n_comp"],
) -> Float[Array, ""]:
    """Compute the self-energy for a Gaussian Mixture Model

    Args:
        means (Float[Array, "n_comp d"]): mean of Gaussian components
        covs (Float[Array, "n_comp d d"]): covariance matrix of Gaussian components
        wgts (Float[Array, " n_comp"]): scalar weights of Gaussian components

    Returns:
        Float[Array, ""]: self energy
    """
    return jnp.exp(
        jax.nn.logsumexp(
            jax.vmap(
                lambda mu0, cov0, w0: jax.vmap(
                    lambda mu1, cov1, w1: log_mixture_component_overlap(
                        mu0, cov0, w0, mu1, cov1, w1
                    ),
                    (0, 0, 0),
                    0,
                )(means, covs, wgts),
                (0, 0, 0),
                0,
            )(means, covs, wgts)
        )
    )


def self_energy_gmm_scan(
    means: Float[Array, "n_comp d"],
    covs: Float[Array, "n_comp d d"],
    wgts: Float[Array, " n_comp"],
) -> Float[Array, ""]:
    """Compute the cross-energy betwen two Gaussian Mixture Models

    Args:
        means1 (Float[Array, "nc1 d"]): mean of Gaussian components for 1st GMM
        covs1 (Float[Array, "nc1 d d"]): covariance matrix of Gaussian components for 1st GMM
        wgts1 (Float[Array, " nc1"]): scalar weights of Gaussian component for 1st GMM
        means2 (Float[Array, "nc2 d"]): mean of Gaussian components for 2nd GMM
        covs2 (Float[Array, "nc2 d d"]): covariance matrix of Gaussian components for 2nd GMM
        wgts2 (Float[Array, " nc2"]): scalar weights of Gaussian component for 2nd GMM

    Returns:
        Float[Array, ""]: cross energy
    """

    def outer_scan(
        e_total: Float[Array, ""],
        outer_comp_tup: tuple[
            Float[Array, " d"], Float[Array, "d d"], Float[Array, ""]
        ],
    ):
        mu1, cov1, wgt1 = outer_comp_tup

        def inner_scan(
            e_inner: Float[Array, ""],
            inner_comp_tup: tuple[
                Float[Array, " d"], Float[Array, "d d"], Float[Array, ""]
            ],
        ) -> tuple[Float[Array, ""], None]:
            mu2, cov2, wgt2 = inner_comp_tup
            overlap = mixture_component_overlap(
                mu1, cov1, wgt1, mu2, cov2, wgt2
            )
            return jnp.add(e_inner, overlap), None

        e_inner, _ = jax.lax.scan(
            inner_scan, jnp.array(0.0), (means, covs, wgts)
        )
        return jnp.add(e_total, e_inner), None

    energy, _ = jax.lax.scan(
        jax.checkpoint(outer_scan),  # pyright: ignore[reportPrivateImportUsage]
        jnp.array(0.0),
        (means, covs, wgts),
    )
    return energy


@Partial(jax.jit, static_argnums=(3,))
def self_energy_gmm(
    means: Float[Array, "n_comp d"],
    covs: Float[Array, "n_comp d d"],
    wgts: Float[Array, " n_comp"],
    use_scan: bool = False,
) -> Float[Array, ""]:
    if use_scan:
        return self_energy_gmm_scan(means, covs, wgts)
    else:
        return self_energy_gmm_logpdf(means, covs, wgts)


def cross_energy_gmms_pdf(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the cross-energy betwen two Gaussian Mixture Models

    Args:
        means1 (Float[Array, "nc1 d"]): mean of Gaussian components for 1st GMM
        covs1 (Float[Array, "nc1 d d"]): covariance matrix of Gaussian components for 1st GMM
        wgts1 (Float[Array, " nc1"]): scalar weights of Gaussian component for 1st GMM
        means2 (Float[Array, "nc2 d"]): mean of Gaussian components for 2nd GMM
        covs2 (Float[Array, "nc2 d d"]): covariance matrix of Gaussian components for 2nd GMM
        wgts2 (Float[Array, " nc2"]): scalar weights of Gaussian component for 2nd GMM

    Returns:
        Float[Array, ""]: cross energy
    """
    return jnp.sum(
        jax.vmap(
            lambda m0, c0, w0: jax.vmap(
                lambda m1, c1, w1: mixture_component_overlap(
                    m0, c0, w0, m1, c1, w1
                ),
                (0, 0, 0),
                0,
            )(means1, covs1, wgts1),
            (0, 0, 0),
            0,
        )(means2, covs2, wgts2)
    )


def cross_energy_gmms_logpdf(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the cross-energy betwen two Gaussian Mixture Models

    Args:
        means1 (Float[Array, "nc1 d"]): mean of Gaussian components for 1st GMM
        covs1 (Float[Array, "nc1 d d"]): covariance matrix of Gaussian components for 1st GMM
        wgts1 (Float[Array, " nc1"]): scalar weights of Gaussian component for 1st GMM
        means2 (Float[Array, "nc2 d"]): mean of Gaussian components for 2nd GMM
        covs2 (Float[Array, "nc2 d d"]): covariance matrix of Gaussian components for 2nd GMM
        wgts2 (Float[Array, " nc2"]): scalar weights of Gaussian component for 2nd GMM

    Returns:
        Float[Array, ""]: cross energy
    """
    return jnp.exp(
        jax.nn.logsumexp(
            jax.vmap(
                lambda m0, c0, w0: jax.vmap(
                    lambda m1, c1, w1: log_mixture_component_overlap(
                        m0, c0, w0, m1, c1, w1
                    ),
                    (0, 0, 0),
                    0,
                )(means1, covs1, wgts1),
                (0, 0, 0),
                0,
            )(means2, covs2, wgts2)
        )
    )


def cross_energy_gmms_scan(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
) -> Float[Array, ""]:
    """Compute the cross-energy betwen two Gaussian Mixture Models

    Args:
        means1 (Float[Array, "nc1 d"]): mean of Gaussian components for 1st GMM
        covs1 (Float[Array, "nc1 d d"]): covariance matrix of Gaussian components for 1st GMM
        wgts1 (Float[Array, " nc1"]): scalar weights of Gaussian component for 1st GMM
        means2 (Float[Array, "nc2 d"]): mean of Gaussian components for 2nd GMM
        covs2 (Float[Array, "nc2 d d"]): covariance matrix of Gaussian components for 2nd GMM
        wgts2 (Float[Array, " nc2"]): scalar weights of Gaussian component for 2nd GMM

    Returns:
        Float[Array, ""]: cross energy
    """

    def outer_scan(
        e_total: Float[Array, ""],
        outer_comp_tup: tuple[
            Float[Array, " d"], Float[Array, "d d"], Float[Array, ""]
        ],
    ):
        mu1, cov1, wgt1 = outer_comp_tup

        def inner_scan(
            e_inner: Float[Array, ""],
            inner_comp_tup: tuple[
                Float[Array, " d"], Float[Array, "d d"], Float[Array, ""]
            ],
        ) -> tuple[Float[Array, ""], None]:
            mu2, cov2, wgt2 = inner_comp_tup
            overlap = mixture_component_overlap(
                mu1, cov1, wgt1, mu2, cov2, wgt2
            )
            return e_inner + overlap, None

        e_inner, _ = jax.lax.scan(
            inner_scan, jnp.array(0.0), (means2, covs2, wgts2)
        )
        return e_total + e_inner, None

    energy, _ = jax.lax.scan(
        jax.checkpoint(outer_scan),  # pyright: ignore[reportPrivateImportUsage]
        jnp.array(0.0),
        (means1, covs1, wgts1),
    )
    return energy


@Partial(jax.jit, static_argnums=(6,))
def cross_energy_gmms(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
    use_scan: bool = False,
) -> Float[Array, ""]:
    if use_scan:
        return cross_energy_gmms_scan(
            means1, covs1, wgts1, means2, covs2, wgts2
        )
    else:
        return cross_energy_gmms_logpdf(
            means1, covs1, wgts1, means2, covs2, wgts2
        )


@Partial(
    jax.jit,
    static_argnums=(
        2,
        3,
    ),
)
def self_energy_gmm_spherical(
    means: Float[Array, "n_comp d"],
    wgts: Float[Array, " n_comp"],
    var: float,
    n_dim: int,
) -> Float[Array, ""]:
    pfun = Partial(
        log_mixture_component_overlap_spherical, var1=var, var2=var, n_dim=n_dim
    )
    return jnp.exp(
        jax.nn.logsumexp(
            jax.vmap(
                lambda m0, w0: jax.vmap(
                    lambda m1, w1: pfun(m0, w0, m1, w1), (0, 0), 0
                )(means, wgts),
                (0, 0),
                0,
            )(means, wgts)
        )
    )
    # pfun = Partial(mixture_component_overlap_spherical,
    #                var1=var, var2=var, n_dim=n_dim)
    # return jnp.sum(
    #     jax.vmap(
    #         lambda mu0, w0: jax.vmap(
    #             lambda mu1, w1: pfun(mu0, w0, mu1, w1), (0, 0), 0
    #         )(means, wgts), (0, 0), 0
    #     )(means, wgts)
    # )


@Partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def cross_energy_gmms_spherical(
    means1: Float[Array, "n_comp d"],
    wgts1: Float[Array, " n_comp"],
    means2: Float[Array, "n_comp d"],
    wgts2: Float[Array, " n_comp"],
    var1: float,
    var2: float,
    n_dim: int,
) -> Float[Array, ""]:
    pfun = Partial(
        log_mixture_component_overlap_spherical,
        var1=var1,
        var2=var2,
        n_dim=n_dim,
    )
    return jnp.exp(
        jax.nn.logsumexp(
            jax.vmap(
                lambda mu0, w0: jax.vmap(
                    lambda mu1, w1: pfun(mu0, w0, mu1, w1), (0, 0), 0
                )(means1, wgts1),
                (0, 0),
                0,
            )(means2, wgts2)
        )
    )


@Partial(jax.jit, static_argnums=(6,))
def l2_distance_gmm(
    means1: Float[Array, "nc1 d"],
    covs1: Float[Array, "nc1 d d"],
    wgts1: Float[Array, " nc1"],
    means2: Float[Array, "nc2 d"],
    covs2: Float[Array, "nc2 d d"],
    wgts2: Float[Array, " nc2"],
    use_scan: bool = False,
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
    self1 = self_energy_gmm(means1, covs1, wgts1, use_scan)
    self2 = self_energy_gmm(means2, covs2, wgts2, use_scan)
    cross = cross_energy_gmms(
        means1, covs1, wgts1, means2, covs2, wgts2, use_scan
    )
    return self1 - 2 * cross + self2


@Partial(jax.jit, static_argnums=(6,))
def l2_distance_gmm_opt(
    means_fixed: Float[Array, "nc1 d"],
    covs_fixed: Float[Array, "nc1 d d"],
    wgts_fixed: Float[Array, " nc1"],
    means_moving: Float[Array, "nc2 d"],
    covs_moving: Float[Array, "nc2 d d"],
    wgts_moving: Float[Array, " nc2"],
    use_scan: bool = False,
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
    self_energy = self_energy_gmm(
        means_moving, covs_moving, wgts_moving, use_scan
    )
    cross_energy = cross_energy_gmms(
        means_fixed,
        covs_fixed,
        wgts_fixed,
        means_moving,
        covs_moving,
        wgts_moving,
        use_scan,
    )
    return self_energy - 2 * cross_energy


@Partial(
    jax.jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def l2_distance_gmm_opt_spherical(
    means_fixed: Float[Array, "nc1 d"],
    wgts_fixed: Float[Array, " nc1"],
    means_moving: Float[Array, "nc2 d"],
    wgts_moving: Float[Array, " nc2"],
    var_fixed: float,
    var_moving: float,
    n_dim: int,
) -> Float[Array, ""]:
    self_energy = self_energy_gmm_spherical(
        means_moving, wgts_moving, var_moving, n_dim
    )
    cross_energy = cross_energy_gmms_spherical(
        means_fixed,
        wgts_fixed,
        means_moving,
        wgts_moving,
        var_fixed,
        var_moving,
        n_dim,
    )
    return self_energy - 2 * cross_energy


def kullback_leibler_gaussian(
    mu_p: Float[Array, " d"],
    cov_p: Float[Array, "d d"],
    mu_q: Float[Array, " d"],
    cov_q: Float[Array, "d d"],
) -> Float[Array, ""]:
    """Calculate the Kullback-Leibler divergence between two multivariate Gaussians.

    Args:
        mu_p (Float[Array, " d"]): mean vector of reference
        cov_p (Float[Array, "d d"]): covariance matrix of reference
        mu_q (Float[Array, " d"]): mean vector of other distribution
        cov_q (Float[Array, "d d"]): covariance matrix of other

    Returns:
        Float[Array, ""]: scalar KL-divergence
    """
    cov_q_inv = jnp.linalg.inv(cov_q)
    n_dim = mu_p.shape[0]
    return 0.5 * (
        jnp.log(jnp.linalg.det(cov_q) / jnp.linalg.det(cov_p))
        - n_dim
        + jnp.trace(cov_q_inv @ cov_p)
        + (mu_q - mu_p).T @ cov_q_inv @ (mu_q - mu_p)
    )


@Partial(jax.jit, static_argnums=(2, 3, 4))
def kullback_leibler_gaussian_spherical(
    mu_p: Float[Array, " d"],
    mu_q: Float[Array, " d"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> Float[Array, ""]:
    cov_inv_q = jnp.diag(
        jnp.array(
            [
                1.0 / var_q,
            ]
            * n_dim
        )
    )
    diff = mu_q - mu_p
    return (
        0.5
        * (
            jnp.log(var_q**n_dim / var_p**n_dim)
            - n_dim
            + n_dim * (var_p / var_q)
            + diff[jnp.newaxis, :] @ cov_inv_q @ diff[:, jnp.newaxis]
        )[0]
    )


def kullback_leibler_gmm_approx(
    mu_p: Float[Array, "n d"],
    cov_p: Float[Array, "n d d"],
    wgt_p: Float[Array, " n"],
    mu_q: Float[Array, "m d"],
    cov_q: Float[Array, "m d d"],
    wgt_q: Float[Array, " m"],
) -> Float[Array, ""]:
    """Calculate the approximate Kullback-Leibler divergence between two multi-component multivariate Gaussian mixture distributions.

    Args:
        mu_p (Float[Array, "n d"]): mean vectors of reference distribution components
        cov_p (Float[Array, "n d d"]): covariance matrices of reference distribution components
        wgt_p (Float[Array, " n"]): weights of components of reference
        mu_q (Float[Array, "n d"]): mean vectors of other distribution components
        cov_q (Float[Array, "n d d"]): covariance matrices of reference distribution components
        wgt_q (Float[Array, " n"]): weights of components of other

    Returns:
        Float[Array, ""]: scalar KL-divergence
    """
    pairwise_kl = jax.vmap(
        lambda mp, cp, wp: jax.vmap(
            lambda mq, cq, wq: jnp.add(
                kullback_leibler_gaussian(mp, cp, mq, cq), jnp.log(wp / wq)
            ),
            (0, 0, 0),
            0,
        )(mu_q, cov_q, wgt_q),
        (0, 0, 0),
        0,
    )(mu_p, cov_p, wgt_p)
    min_kl = jnp.amin(pairwise_kl, axis=1)
    return jnp.sum(jnp.multiply(wgt_p, min_kl))


@Partial(jax.jit, static_argnums=(4, 5, 6))
def kullback_leibler_gmm_approx_spherical(
    mu_p: Float[Array, "n d"],
    wgt_p: Float[Array, " n"],
    mu_q: Float[Array, "m d"],
    wgt_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> Float[Array, ""]:
    """Calculate the approximate Kullback-Leibler divergence between two multi-component multivariate Gaussian mixture distributions.

    Args:
        mu_p (Float[Array, "n d"]): mean vectors of reference distribution components
        cov_p (Float[Array, "n d d"]): covariance matrices of reference distribution components
        wgt_p (Float[Array, " n"]): weights of components of reference
        mu_q (Float[Array, "n d"]): mean vectors of other distribution components
        cov_q (Float[Array, "n d d"]): covariance matrices of reference distribution components
        wgt_q (Float[Array, " n"]): weights of components of other

    Returns:
        Float[Array, ""]: scalar KL-divergence
    """
    kl_fun = Partial(
        kullback_leibler_gaussian_spherical,
        var_p=var_p,
        var_q=var_q,
        n_dim=n_dim,
    )
    pairwise_kl = jax.vmap(
        lambda mp, wp: jax.vmap(
            lambda mq, wq: jnp.add(kl_fun(mp, mq), jnp.log(wp / wq)),
            (0, 0),
            0,
        )(mu_q, wgt_q),
        (0, 0),
        0,
    )(mu_p, wgt_p)
    min_kl = jnp.amin(pairwise_kl, axis=1)
    return jnp.sum(jnp.multiply(wgt_p, min_kl))
