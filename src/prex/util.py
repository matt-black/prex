import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = [
    "sqdist",
    "decompose_affine_transform",
    "rotation_matrix_2d",
    "rotation_matrix_3d",
]


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


def shear_matrix_2d(
    k: Float[Array, ""], ell: Float[Array, ""]
) -> Float[Array, "2 2"]:
    return jnp.array([[1, k], [0, 1]]) @ jnp.array([[1, 0], [ell, 1]])


def rotation_matrix_2d(alpha: Float[Array, ""]) -> Float[Array, "2 2"]:
    return jnp.array(
        [[jnp.cos(alpha), -jnp.sin(alpha)], [jnp.sin(alpha), jnp.cos(alpha)]]
    )


def shear_matrix_3d(
    k_xy: Float[Array, ""],
    k_xz: Float[Array, ""],
    k_yx: Float[Array, ""],
    k_yz: Float[Array, ""],
    k_zx: Float[Array, ""],
    k_zy: Float[Array, ""],
):
    M_xy_xz = jnp.array([[1, k_xy, k_xz], [0, 1, 0], [0, 0, 1]])
    M_yx_yz = jnp.array([[1, 0, 0], [k_yx, 1, k_yz], [0, 0, 1]])
    M_zx_zy = jnp.array([[1, 0, 0], [0, 1, 0], [k_zx, k_zy, 1]])
    return M_xy_xz @ M_yx_yz @ M_zx_zy


def rotation_matrix_3d(
    alpha: Float[Array, ""], beta: Float[Array, ""], gamma: Float[Array, ""]
) -> Float[Array, "3 3"]:
    Rx = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(alpha), -jnp.sin(alpha)],
            [0, jnp.sin(alpha), jnp.cos(alpha)],
        ]
    )
    Ry = jnp.array(
        [
            [jnp.cos(beta), 0, jnp.sin(beta)],
            [0, 1, 0],
            [-jnp.sin(beta), 0, jnp.cos(beta)],
        ]
    )
    Rz = jnp.array(
        [
            [jnp.cos(gamma), -jnp.sin(gamma), 0],
            [jnp.sin(gamma), jnp.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    return Rz @ Ry @ Rx
