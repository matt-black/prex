import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..util import sqdist
from ._matching import MatchingMatrix, expectation

__all__ = [
    "KernelMatrix",
    "CoeffMatrix",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "interpolate",
    "maximization",
]


type KernelMatrix = Float[Array, "m m"]
type CoeffMatrix = Float[Array, "m d"]
type TransformParams = tuple[MatchingMatrix, KernelMatrix, CoeffMatrix]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_stddev: float,
    max_iter: int,
    tolerance: float,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align the moving points onto the reference points by a nonrigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_stddev (float): standard deviation of Gaussian kernel function.
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix and the kernel and coefficient matrices) along with the final variance and the number of iterations that the algorithm was run for.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

    # compute gaussian kernel matrix
    G = jnp.exp(
        jnp.negative(jnp.divide(sqdist(mov, mov), 2 * kernel_stddev**2))
    )

    def cond_fund(
        a: tuple[
            tuple[Float[Array, "m d"], Float[Array, "m m"]],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fund(
        a: tuple[
            tuple[CoeffMatrix, MatchingMatrix], tuple[Float[Array, ""], int]
        ],
    ) -> tuple[
        tuple[CoeffMatrix, MatchingMatrix], tuple[Float[Array, ""], int]
    ]:
        (W, _), (var, iter_num) = a
        mov_t = transform(mov, G, W)
        P = expectation(ref, mov_t, var, outlier_prob)
        W, new_var = maximization(
            ref, mov, P, G, var, regularization_param, tolerance
        )
        return ((W, P), (new_var, iter_num + 1))

    (W, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fund,
        body_fund,
        ((jnp.zeros_like(mov), jnp.zeros((m, n))), (var_i, 0)),
    )
    return (P, W, G), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_stddev: float,
    num_iter: int,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by a nonrigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_stddev (float): standard deviation of Gaussian kernel function.
        num_iter (int): # of iterations to optimize for.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix and the kernel and coefficient matrices) along with the variance at each step of the optimization.
    """
    n, d = ref.shape
    m, _ = mov.shape
    # compute gaussian kernel
    G = jnp.exp(
        jnp.negative(jnp.divide(sqdist(mov, mov), 2 * kernel_stddev**2))
    )
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

    def scan_fun(
        a: tuple[tuple[MatchingMatrix, CoeffMatrix], Float[Array, ""]],
        _,
    ):
        (_, W), var = a
        mov_t = transform(mov, G, W)
        P = expectation(ref, mov_t, var, outlier_prob)
        W, new_var = maximization(
            ref, mov, P, G, var, regularization_param, 0.0
        )
        return ((P, W), new_var), new_var

    ((P, W), _), varz = jax.lax.scan(
        scan_fun,
        ((jnp.zeros((m, n)), jnp.zeros_like(mov)), var_i),
        length=num_iter,
    )

    return (P, W, G), varz


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
    tolerance: float,
) -> tuple[CoeffMatrix, Float[Array, ""]]:
    """Do a single M-step.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source point set
        P (MatchingMatrix): matching matrix
        G (KernelMatrix): matrix of kernel values between points in the source point set
        var (Float[Array, ""]): current variance
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        tolerance (float): termination tolerance

    Returns:
        tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    """

    W = update_transform(x, y, P, G, var, regularization_param)
    y_t = transform(y, G, W)
    new_var = update_variance(x, y_t, P, tolerance)
    return W, new_var


def transform(
    y: Float[Array, "m d"],
    G: KernelMatrix,
    W: CoeffMatrix,
) -> Float[Array, "m d"]:
    """Transform the input points by nonrigid warping.

    Args:
        y (Float[Array, "m d"]): `d`-dimensional points to be transformed
        G (KernelMatrix): matrix of kernel values between points (should be m x m)
        W (CoeffMatrix): fitted coefficient matrix (should be same shape as `y`)

    Returns:
        Float[Array, "m d"]: warped points, `y + G @ W`
    """
    return jnp.add(y, G @ W)


def update_transform(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
) -> CoeffMatrix:
    m, _ = P.shape
    P1 = jnp.sum(P, axis=1)
    A = jnp.diag(P1) @ G + regularization_param * var * jnp.eye(m)
    B = jnp.matmul(P, x) - jnp.diag(P1) @ y
    return jnp.linalg.solve(A, B)


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


def interpolate(
    mov: Float[Array, "m d"],
    interp: Float[Array, "n d"],
    W: Float[Array, "m d"],
    kernel_stddev: float,
) -> Float[Array, "n d"]:
    """Interpolate values of vector field at points outside of the original fit domain.

    Args:
        mov (Float[Array, "m d"]): "moving" point cloud that was aligned
        interp (Float[Array, "n d"]): points to interpolate
        W (Float[Array, "m d"]): fitted transform coefficients
        kernel_stddev (float): standard deviation of kernel

    Returns:
        Float[Array, "n d"]: interpolated vector values at specified points

    Notes:
        This assumes a "Gaussian process-like" interpretation of the fitting coefficients whereby the deformation field can be interpolated by computing a Gram matrix between the interpolated points and the moving points used during fitting, then the fitted weights are used to calculate deformation vectors at the interpolation coordinates.
    """
    # calculate kernel matrix b/t moving & interpolating points
    G_im = jnp.exp(
        jnp.negative(jnp.divide(sqdist(interp, mov), 2 * kernel_stddev**2))
    )
    return G_im @ W
