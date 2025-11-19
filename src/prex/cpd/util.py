import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["sqdist", "decompose_affine_transform"]


def sqdist(
    x: Float[Array, "n d"], y: Float[Array, "m d"]
) -> Float[Array, "n m"]:
    """Compute the squared distance between all pairs of two sets of input points.

    Args:
        x (Float[Array, "n d"]): set of points
        y (Float[Array, "m d"]): another set of poitns

    Returns:
        Float[Array, "n m"]: kernel matrix of all squared distances between pairs of points.
    """
    return jax.vmap(
        lambda x1: jax.vmap(lambda y1: _squared_distance(x1, y1))(y)
    )(x)


def _squared_distance(x: Float[Array, " d"], y: Float[Array, " d"]):
    return jnp.sum(jnp.square(jnp.subtract(x, y)))


def decompose_affine_transform(
    A: Float[Array, "d d"],
) -> tuple[Float[Array, "d d"], Float[Array, " d"], Float[Array, " d"]]:
    """Decompose an affine transform matrix into a rotation matrix and diagonal matrices corresponding to shear and scaling.

    Args:
        A (Float[Array, "d d"]): square, affine matrix

    Returns:
        tuple[Float[Array, "d d"], Float[Array, "d"], Float[Array, " d"]]: rotation matrix and vectors corresponding to per-axis scaling and shear, respectively
    """
    ZS = jnp.linalg.cholesky(A.T @ A).T
    Z = jnp.diag(ZS).copy()
    shears = ZS / Z[:, jnp.newaxis]
    n = len(Z)
    S = shears[jnp.triu(jnp.ones((n, n)), 1).astype(bool)]
    R = jnp.dot(A, jnp.linalg.inv(ZS))
    if jnp.linalg.det(R) < 0:
        Z = Z.at[0].multiply(-1)
        ZS = ZS.at[0].multiply(-1)
        R = jnp.dot(A, jnp.linalg.inv(ZS))
    return R, Z, S


def z_score(
    x: Float[Array, "n d"],
) -> tuple[Float[Array, "n d"], tuple[Float[Array, "d d"], Float[Array, " d"]]]:
    mu = jnp.mean(x[:, [0, 1]], axis=0, keepdims=True)
    sd = jnp.std(x, axis=0, keepdims=True)
    S = jnp.diag(sd[0])
    t = mu
    return (x - mu) / sd, (S, t)


def undo_z_score(
    x: Float[Array, "n d"],
    S: Float[Array, "d d"],
    t: Float[Array, " d"],
) -> Float[Array, "n d"]:
    return S @ x.T - t
