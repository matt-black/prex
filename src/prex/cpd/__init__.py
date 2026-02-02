"""Coherent point drift and related point cloud matching algorithms.

References
---
[1] A. Myronenko, X. Song, and M. Á. Carreira-Perpiñán, “Non-rigid point set registration: Coherent point drift,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2006, pp. 1009–1016.

[2] A. Myronenko and X. Song, “Point set registration: Coherent point drift,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 32, no. 12, pp. 2262–2275, Dec. 2010.
"""

from typing import Union

from jaxtyping import Array, Float

from .affine import TransformParams as AffineTransformParams
from .affine import align as align_affine
from .affine import align_fixed_iter as align_fixed_iter_affine
from .nonrigid import TransformParams as CoherentMotionTransformParams
from .nonrigid import align as align_nonrigid
from .nonrigid import align_fixed_iter as align_fixed_iter_nonrigid
from .rigid import TransformParams as RigidTransformParams
from .rigid import align as align_rigid
from .rigid import align_fixed_iter as align_fixed_iter_rigid

type TransformParams = Union[
    RigidTransformParams, AffineTransformParams, CoherentMotionTransformParams
]


__all__ = [
    "align",
    "align_rigid",
    "align_affine",
    "align_nonrigid",
]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    method: str,
    outlier_prob: float,
    num_iter: int,
    tolerance: float | None,
    regularization_param_nonrigid: float = 1.0,
    kernel_stddev_nonrigid: float = 1.0,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[
    TransformParams,
    Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]],
]:
    """Align the moving points onto the reference points by the specified transform method.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        method (str): the transform that is to be fit/used to register the two sets of points.
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): maximum # of iterations to optimize for. if tolerance is `None`, this is the number of iterations that will be optimized for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate. If `None`, a fixed number of iterations is used.
        regularization_param_nonrigid (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_stddev_nonrigid (float): standard deviation of Gaussian kernel function. only used if `method=nonrigid`.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    if method == "nonrigid":
        if tolerance is None:
            return align_fixed_iter_nonrigid(
                ref,
                mov,
                outlier_prob,
                regularization_param_nonrigid,
                kernel_stddev_nonrigid,
                num_iter,
                mask=mask,
            )
        else:
            return align_nonrigid(
                ref,
                mov,
                outlier_prob,
                regularization_param_nonrigid,
                kernel_stddev_nonrigid,
                num_iter,
                tolerance,
                mask=mask,
            )
    elif method == "affine":
        if tolerance is None:
            return align_fixed_iter_affine(
                ref, mov, outlier_prob, num_iter, mask=mask
            )
        else:
            return align_affine(
                ref, mov, outlier_prob, num_iter, tolerance, mask=mask
            )
    elif method == "rigid":
        if tolerance is None:
            return align_fixed_iter_rigid(
                ref, mov, outlier_prob, num_iter, mask=mask
            )
        else:
            return align_rigid(
                ref, mov, outlier_prob, num_iter, tolerance, mask=mask
            )
    else:
        raise ValueError("invalid method")
