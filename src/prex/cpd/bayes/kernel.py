from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

__all__ = [
    "KernelFunction",
    "gaussian_kernel",
    "inverse_multiquadratic_kernel",
    "rational_quadratic_kernel",
]


type KernelFunction = Callable[
    [Float[Array, "1 d"], Float[Array, "1 d"], float],
    Float[Array, ""],
]


@Partial(jax.jit, static_argnums=(2,))
def gaussian_kernel(
    a: Float[Array, "1 d"],
    b: Float[Array, "1 d"],
    beta: float,
) -> Float[Array, ""]:
    d = jnp.sum(jnp.square(jnp.subtract(a, b)))
    return jnp.exp(jnp.negative(jnp.divide(d, 2 * beta)))


@Partial(jax.jit, static_argnums=(2,))
def inverse_multiquadratic_kernel(
    a: Float[Array, "1 d"],
    b: Float[Array, "1 d"],
    beta: float,
) -> Float[Array, ""]:
    d = jnp.sum(jnp.square(jnp.subtract(a, b)))
    return jnp.divide(1.0, jnp.sqrt(d + beta))


@Partial(jax.jit, static_argnums=(2,))
def rational_quadratic_kernel(
    a: Float[Array, "1 d"],
    b: Float[Array, "1 d"],
    beta: float,
) -> Float[Array, ""]:
    d = jnp.sum(jnp.square(jnp.subtract(a, b)))
    return jnp.subtract(1.0, jnp.divide(d, d + beta))
