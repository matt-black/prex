import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..util import sqdist
from ._matching import (
    MatchingMatrix,
    expectation,
    expectation_masked,
    expectation_weighted,
)

__all__ = [
    "AffineMatrix",
    "Translation",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "maximization",
]


type AffineMatrix = Float[Array, "d d"]
type Translation = Float[Array, " d"]
type TransformParams = tuple[MatchingMatrix, AffineMatrix, Translation]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    max_iter: int,
    tolerance: float,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align the moving points onto the reference points by affine transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

    def cond_fun(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fun(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> tuple[
        tuple[AffineMatrix, Translation, MatchingMatrix],
        tuple[Float[Array, ""], int],
    ]:
        (A, t, P), (var, iter_num) = a
        mov_t = transform(mov, A, t)
        if moving_weights is None:
            if mask is None:
                P = expectation(ref, mov_t, var, outlier_prob)
            else:
                P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
        else:
            P = expectation_weighted(
                ref, mov_t, var, outlier_prob, moving_weights
            )
        (A, t), new_var = maximization(ref, mov, P, tolerance)
        return (A, t, P), (new_var, iter_num + 1)

    (A, t, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), (var_i, 0)),
    )

    return (P, A, t), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by affine transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): # of iterations to optimize for.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

    def scan_funa(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        _,
    ):
        (A, t, P), var = a
        mov_t = transform(mov, A, t)
        if moving_weights is None:
            if mask is None:
                P = expectation(ref, mov_t, var, outlier_prob)
            else:
                P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
        else:
            P = expectation_weighted(
                ref, mov_t, var, outlier_prob, moving_weights
            )
        (A, t), new_var = maximization(ref, mov, P, 1e-6)
        return ((A, t, P), new_var), new_var

    ((A, t, P), _), varz = jax.lax.scan(
        scan_funa,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), var_i),
        None,
        length=num_iter,
    )

    return (P, A, t), varz


def transform(
    y: Float[Array, "m d"], A: Float[Array, "d d"], t: Float[Array, " d"]
) -> Float[Array, "m d"]:
    """Transform the input points by affine transform.

    Args:
        y (Float[Array, "m d"]): `d`-dimensional points to be transformed
        A (Float[Array, "d d"]): `d`-dimensional affine transform matrix
        t (Float[Array, " d"]): translation

    Returns:
        Float[Array, "m d"]: transformed points, `y @ A + t`
    """
    return y @ A + t[None, :]


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
) -> tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    """Do a single M-step.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source point set
        P (MatchingMatrix): matching matrix
        tolerance (float): termination tolerance

    Returns:
        tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]: updated transform parameters, and variance.
    """
    A, t = update_transform(x, y, P)
    y_t = transform(y, A, t)
    var = update_variance(x, y_t, P, tolerance)
    return (A, t), var


def update_transform(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
) -> tuple[Float[Array, "d d"], Float[Array, " d"]]:
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    mu_x = jnp.divide(x.T @ Pt1, N)
    mu_y = jnp.divide(y.T @ P1, N)
    x_hat, y_hat = x - mu_x, y - mu_y
    B = jnp.dot(jnp.dot(x_hat.T, P.T), y_hat)
    ypy = jnp.dot(jnp.dot(y_hat.T, jnp.diag(P1)), y_hat)
    A = jnp.linalg.solve(ypy.T, B.T)
    t = mu_x.T - A.T @ mu_y.T
    return (A, t)


def update_variance(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
) -> Float[Array, ""]:
    _, d = x.shape
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    val = (
        jnp.trace(x.T @ jnp.diag(Pt1) @ x)
        - 2 * jnp.trace((P @ x).T @ y_t)
        + jnp.trace(y_t.T @ jnp.diag(P1) @ y_t)
    )
    new = jnp.divide(val, N * d)
    return jax.lax.select(new > 0, new, tolerance - 2 * jnp.finfo(x.dtype).eps)
