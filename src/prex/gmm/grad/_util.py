import jax
import jax.numpy as jnp
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
