import math

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ...util import sqdist
from .._matching import MatchingMatrix
from ..nonrigid import KernelMatrix
from ..rigid import RotationMatrix, ScalingTerm, Translation
from .util import dimension_bounds
from .util import transform as apply_T
from .util import transform_inverse as apply_Tinv

__all__ = [
    "update_matching",
    "update_rigid",
    "update_nonrigid",
    "update_variance",
]


@Partial(jax.jit, static_argnums=(6,))
def update_matching(
    x: Float[Array, "n d"],
    y_hat: Float[Array, "m d"],
    sigma_m: Float[Array, " m"],
    alpha_m: Float[Array, " m"],
    s: ScalingTerm,
    var: Float[Array, ""],
    outlier_prob: float,
) -> tuple[
    MatchingMatrix,
    Float[Array, " m"],
    Float[Array, " n"],
    Float[Array, ""],
    Float[Array, "m d"],
]:
    _, d = x.shape
    eps = jnp.finfo(x.dtype).eps
    bnds = jax.vmap(dimension_bounds, 1, 1)(x)
    v = jnp.prod(jnp.ptp(bnds))
    d_t = sqdist(y_hat, x)
    # m_term is shape (m,1)
    m_term = jnp.expand_dims(
        alpha_m * jnp.exp(-(s**2) / (2 * var) * sigma_m * d), 1
    )
    # shape (m, n)
    gauss = jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    numer = (1 - outlier_prob) * m_term * gauss
    denom = jnp.add(
        outlier_prob / v,
        (1 - outlier_prob) * jnp.sum(m_term * gauss, axis=0, keepdims=True),
    )
    p_mn = numer / denom
    nu = jnp.clip(jnp.sum(p_mn, axis=1), eps)
    nu_p = jnp.clip(jnp.sum(p_mn, axis=0), eps)
    n_hat = jnp.sum(nu)
    x_hat = jnp.diag(jnp.divide(1.0, nu)) @ p_mn @ x
    return p_mn, nu, nu_p, n_hat, x_hat


@Partial(jax.jit, static_argnums=(10, 11))
def update_nonrigid(
    x_hat: Float[Array, "m d"],
    y: Float[Array, "m d"],
    G: KernelMatrix,
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    v_hat: Float[Array, "m d"],
    nu: Float[Array, " m"],
    var: Float[Array, ""],
    n_hat: Float[Array, ""],
    kappa: float,
    lambda_: float,
    fixed_alpha: Float[Array, " m"] | None = None,
) -> tuple[
    Float[Array, "m d"],
    Float[Array, "m d"],
    Float[Array, " m"],
    Float[Array, " m"],
]:
    m, _ = y.shape
    eps = jnp.finfo(y.dtype).eps

    # Clip s to prevent division by zero during back-projection
    s_safe = jnp.maximum(s, 1e-6)

    # Back-project x_hat to source space: (x_hat - t) @ R / s
    resid_projected = apply_Tinv(x_hat, R, s_safe, t) - y

    # Stabilization: bcpd.c uses G1 = G + (lambda * var / s^2) * diag(1/nu)
    # We solve: (G * diag(nu) + cc * I) W = resid_projected
    # Then v = G * diag(nu) * W
    cc = lambda_ * var / (jnp.square(s_safe) + eps)

    # Symmetric formulation: (G + cc * diag(1/nu))
    # To avoid 1/nu when nu is zero, we solve for W = (G + cc * diag(1/nu))^-1 resid
    # This is equivalent to solving (diag(nu) @ G + cc * I) W = diag(nu) @ resid
    # However, to maintain symmetry for JAX/LA solvers:
    # Let K = G + cc * diag(1/nu).
    # v = G @ K^-1 @ resid

    G_reg = G + cc * jnp.diag(1.0 / (nu + eps))
    W = jnp.linalg.solve(G_reg, resid_projected)

    v_hat = G @ W
    u_hat = y + v_hat

    # Update alpha_m: use fixed values if provided, otherwise compute
    if fixed_alpha is not None:
        alpha_m = fixed_alpha
    elif math.isinf(kappa):
        alpha_m = nu / (n_hat + eps)
    else:
        alpha_m = jnp.exp(digamma(kappa + nu) - digamma(kappa * m + n_hat))

    # sigma_m: diagonal of posterior covariance
    # Ref: bcpd.c:333 -> sgm[m] = G2[m,m] * SQ(*r/(*s))/w[m];
    # where G2 = inv(G + cc*diag(1/nu)) @ G and SQ(r/s) = cc/lambda
    G_inv_G = jnp.linalg.solve(G_reg, G)
    sigma_m = jnp.maximum(
        jnp.diagonal(G_inv_G) * (cc / lambda_) / (nu + eps), 0.0
    )

    return v_hat, u_hat, sigma_m, alpha_m


def update_rigid(
    y: Float[Array, "m d"],
    x_hat: Float[Array, "m d"],
    u_hat: Float[Array, "m d"],
    sigma_m: Float[Array, " m"],
    nu: Float[Array, " m"],
    n_hat: Float[Array, ""],
    debias: bool = False,
) -> tuple[RotationMatrix, ScalingTerm, Translation, Float[Array, ""]]:
    _, d = y.shape
    x_bar = jnp.divide(jnp.sum(nu[:, None] * x_hat, axis=0), n_hat)
    var_bar = jnp.divide(jnp.sum(nu * sigma_m), n_hat)
    u_bar = jnp.divide(jnp.sum(nu[:, None] * u_hat, axis=0), n_hat)
    x_diff = (x_hat - x_bar[None, :])[..., None]  # (n, 3, 1)
    u_diff = (u_hat - u_bar[None, :])[..., None]  # (n, 3, 1)
    # S_xu is like an average (d,d) matrix over residuals
    S_xu = (
        jnp.sum(
            jnp.multiply(
                nu[:, None, None], jnp.matmul(x_diff, u_diff.transpose(0, 2, 1))
            ),
            axis=0,
        )
        / n_hat
    )
    # S_uu is similar to S_xu, but only over moved points
    S_uu = (
        jnp.sum(
            jnp.multiply(
                nu[:, None, None], jnp.matmul(u_diff, u_diff.transpose(0, 2, 1))
            ),
            axis=0,
        )
        / n_hat
    )

    if debias:
        S_uu += jnp.eye(d) * var_bar

    big_phi, _, big_psiT = jnp.linalg.svd(S_xu)
    mid = jnp.diag(
        jnp.concatenate(
            [jnp.ones((d - 1,)), jnp.linalg.det(big_phi @ big_psiT)[None]],
            axis=0,
        )
    )
    R = big_phi @ mid @ big_psiT
    s = jnp.linalg.trace(R.T @ S_xu) / jnp.trace(S_uu)
    t = (x_bar[None, :] - apply_T(u_bar[None, :], R, s, jnp.array(0.0)))[0]
    return R, s, t, var_bar


def update_variance(
    x: Float[Array, "n d"],
    y_hat: Float[Array, "m d"],
    P: MatchingMatrix,
    s: ScalingTerm,
    nu: Float[Array, "m d"],
    nu_p: Float[Array, "n d"],
    n_hat: Float[Array, ""],
    var_bar: Float[Array, ""],
) -> Float[Array, ""]:
    _, d = x.shape
    var = (
        jnp.divide(
            jnp.sum(
                x.T @ jnp.diag(nu_p) @ x
                - 2 * x.T @ P.T @ y_hat
                + y_hat.T @ jnp.diag(nu) @ y_hat
            ),
            n_hat * d,
        )
        + jnp.square(s) * var_bar
    )
    return jax.lax.select(var > 1e-6, var, 1e-6)
