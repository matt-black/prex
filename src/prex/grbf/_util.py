import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from ..util import sqdist


@Partial(jax.jit, static_argnums=(2,))
def gaussian_rbf(
    x: Float[Array, "n_pts d"],
    cents: Float[Array, "n_cent d"],
    bandwidth: float,
) -> Float[Array, "n_pts n_cent"]:
    return jnp.exp(-sqdist(x, cents) / (2 * bandwidth**2))
