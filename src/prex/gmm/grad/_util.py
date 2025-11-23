import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float


def compute_kl_divergence_spherical(
    mu_p: Float[Array, " d"],
    mu_q: Float[Array, " d"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> Float[Array, ""]:
    """Compute KL divergence between two spherical Gaussians.

    Args:
        mu_p: Mean of reference Gaussian
        mu_q: Mean of moving Gaussian (transformed)
        var_p: Variance of reference Gaussian (isotropic)
        var_q: Variance of moving Gaussian (isotropic)
        n_dim: Dimensionality

    Returns:
        KL divergence D_KL(N(mu_p, var_p*I) || N(mu_q, var_q*I))
    """
    # Constant term
    c_pq = 0.5 * (
        n_dim * jnp.log(var_q / var_p) + n_dim * (var_p / var_q) - n_dim
    )

    # Distance term
    delta = mu_q - mu_p
    dist_term = 0.5 * jnp.sum(delta**2) / var_q

    return c_pq + dist_term


def compute_weights_alpha(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q_trans: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> tuple[Float[Array, "n m"], Float[Array, " n"], Float[Array, " n"]]:
    """Compute the weighting coefficients alpha_ij for gradient computation.

    Args:
        means_p: Reference GMM means (fixed)
        wgts_p: Reference GMM weights
        means_q_trans: Transformed moving GMM means
        wgts_q: Moving GMM weights
        var_p: Reference variance (isotropic)
        var_q: Moving variance (isotropic)
        n_dim: Dimensionality

    Returns:
        alpha_ij: Weighting coefficients, shape (n, m)
        A_i: Self-energy terms for reference GMM, shape (n,)
        B_i: Cross-energy terms, shape (n,)
    """

    # Compute A_i (self-energy of reference GMM, constant w.r.t. transform params)
    def compute_A_i(
        mu_p_i: Float[Array, " d"], wgt_p_i: float
    ) -> Float[Array, ""]:
        # Sum over k: w_p^k * exp(-D_KL(N(mu_p^i, var_p) || N(mu_p^k, var_p)))
        kl_vals = jax.vmap(
            lambda mu_p_k: compute_kl_divergence_spherical(
                mu_p_i, mu_p_k, var_p, var_p, n_dim
            )
        )(means_p)
        return jnp.sum(wgts_p * jnp.exp(-kl_vals))

    A_i = jax.vmap(compute_A_i)(means_p, wgts_p)

    # Compute B_i (cross-energy with transformed moving GMM)
    def compute_B_i(mu_p_i: Float[Array, " d"]) -> Float[Array, ""]:
        # Sum over j: w_q^j * exp(-D_KL(N(mu_p^i, var_p) || N(mu_q_trans^j, var_q)))
        kl_vals = jax.vmap(
            lambda mu_q_j: compute_kl_divergence_spherical(
                mu_p_i, mu_q_j, var_p, var_q, n_dim
            )
        )(means_q_trans)
        return jnp.sum(wgts_q * jnp.exp(-kl_vals))

    B_i = jax.vmap(compute_B_i)(means_p)

    # Compute alpha_ij = (w_p^i * A_i * w_q^j * exp(-D_KL^ij)) / B_i^2
    def compute_alpha_ij(
        mu_p_i: Float[Array, " d"],
        wgt_p_i: float,
        A_i_val: float,
        B_i_val: float,
    ) -> Float[Array, " m"]:
        kl_vals = jax.vmap(
            lambda mu_q_j: compute_kl_divergence_spherical(
                mu_p_i, mu_q_j, var_p, var_q, n_dim
            )
        )(means_q_trans)
        exp_neg_kl = jnp.exp(-kl_vals)
        return (wgt_p_i * A_i_val * wgts_q * exp_neg_kl) / (B_i_val**2)

    alpha_ij = jax.vmap(compute_alpha_ij)(means_p, wgts_p, A_i, B_i)

    return alpha_ij, A_i, B_i


def compute_overlap_weights(
    means_p: Float[Array, "n d"],
    wgts_p: Float[Array, " n"],
    means_q_trans: Float[Array, "m d"],
    wgts_q: Float[Array, " m"],
    var_p: float,
    var_q: float,
    n_dim: int,
) -> Float[Array, "n m"]:
    """Compute overlap weights O_ij for L2 distance gradient.

    O_ij = w_p^i * w_q^j * N(mu_p^i - mu_q^j; 0, (var_p + var_q)I)

    Args:
        means_p: Reference GMM means
        wgts_p: Reference GMM weights
        means_q_trans: Transformed moving GMM means
        wgts_q: Moving GMM weights
        var_p: Reference variance
        var_q: Moving variance
        n_dim: Dimensionality

    Returns:
        Overlap weights O_ij, shape (n, m)
    """
    # Combined variance for convolution
    var_combined = var_p + var_q

    # Normalization constant for spherical Gaussian
    norm_const = 1.0 / ((2.0 * jnp.pi * var_combined) ** (n_dim / 2.0))

    def compute_O_ij(
        mu_p_i: Float[Array, " d"],
        wgt_p_i: float,
    ) -> Float[Array, " m"]:
        # compute squared Euclidean distances ||mu_p^i - mu_q^j||^2
        diffs = means_q_trans - mu_p_i
        sq_dists = jnp.sum(diffs**2, axis=1)

        # gaussian term: exp(-||delta||^2 / (2 * var_combined))
        gaussian_vals = jnp.exp(-sq_dists / (2.0 * var_combined))

        # combine: w_p^i * w_q^j * norm_const * gaussian_val
        return wgt_p_i * wgts_q * norm_const * gaussian_vals

    return jax.vmap(compute_O_ij)(means_p, wgts_p)


@Partial(jax.jit, static_argnums=(3,))
def compute_self_overlap_weights(
    means: Float[Array, "m d"],
    wgts: Float[Array, " m"],
    var: float,
    n_dim: int,
) -> Float[Array, "m m"]:
    """Compute overlap weights for self-energy gradient.

    O_jk = w_j * w_k * N(mu_j - mu_k; 0, 2*var*I)
    """
    var_combined = 2.0 * var
    norm_const = 1.0 / ((2.0 * jnp.pi * var_combined) ** (n_dim / 2.0))

    def compute_O_jk(
        mu_j: Float[Array, " d"],
        wgt_j: float,
    ) -> Float[Array, " m"]:
        diffs = means - mu_j
        sq_dists = jnp.sum(diffs**2, axis=1)
        gaussian_vals = jnp.exp(-sq_dists / (2.0 * var_combined))
        return wgt_j * wgts * norm_const * gaussian_vals

    return jax.vmap(compute_O_jk)(means, wgts)
