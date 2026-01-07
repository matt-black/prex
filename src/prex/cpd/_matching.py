import jax.numpy as jnp
from jaxtyping import Array, Float

from ..util import sqdist

__all__ = ["MatchingMatrix", "expectation"]


type MatchingMatrix = Float[Array, "m n"]


def expectation(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    var: Float[Array, ""],
    w: float,
) -> MatchingMatrix:
    """Do a single expectation step of the CPD algorithm. This essentially just updates the matching matrix.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source (moving) point set
        var (Float[Array, ""]): variance of the Gaussian kernel
        w (float): outlier probability

    Returns:
        MatchingMatrix: (m x n) matrix of matching probabilities.
    """
    n, d = x.shape
    m, _ = y_t.shape
    d_t = sqdist(x, y_t).transpose()
    top = jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
        jnp.float_power(2 * jnp.pi * var, d / 2) * m, n
    )
    bot = jnp.add(
        jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
        outl_term,
    )
    return jnp.divide(top, bot)


def expectation_weighted(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    var: Float[Array, ""],
    w: float,
    alpha_m: Float[Array, " m"],
) -> MatchingMatrix:
    """Do a single expectation step of the CPD algorithm, with per-point weightings for the source point set.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source (moving) point set
        var (Float[Array, ""]): variance of the Gaussian kernel
        w (float): outlier probability
        alpha_m (Float[Array, " m"]): per-point weightings for the source points (arbitrary positive values)

    Returns:
        MatchingMatrix: (m x n) matrix of matching probabilities.
    """
    n, d = x.shape
    m, _ = y_t.shape
    d_t = sqdist(x, y_t).transpose()  # (m, n)
    top = alpha_m[:, None] * jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    # Use sum of weights instead of m to support arbitrary scaled weights
    alpha_sum = jnp.sum(alpha_m)
    outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
        jnp.float_power(2 * jnp.pi * var, d / 2) * alpha_sum, n
    )
    bot = jnp.add(
        jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
        outl_term,
    )
    return jnp.divide(top, bot)
