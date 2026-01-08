import jax.numpy as jnp
from jaxtyping import Array, Float

# Type aliases
RotationMatrix = Float[Array, "d d"]
ScalingTerm = Float[Array, ""]
Translation = Float[Array, " d"]


def normalize_points(
    points: Float[Array, "n d"],
) -> tuple[Float[Array, "n d"], Float[Array, " d"], Float[Array, ""]]:
    """Normalize point cloud to zero mean and unit variance.

    Args:
        points: Input point cloud of shape (N, D).

    Returns:
        tuple: (normalized_points, mean, scale)
        - normalized_points: (points - mean) / scale
        - mean: Centroid of the points (1, D) or (D,)
        - scale: Standard deviation of the points (scalar)
    """
    mean = jnp.mean(points, axis=0)
    centered = points - mean
    variance = jnp.mean(centered**2)
    scale = jnp.sqrt(variance)

    scale = jnp.where(scale < 1e-12, 1.0, scale)  # avoid division by zero

    normalized_points = centered / scale

    return normalized_points, mean, scale


def denormalize_parameters(
    R_n: RotationMatrix,
    s_n: ScalingTerm,
    t_n: Translation,
    v_n: Float[Array, "m d"],
    mu_y: Float[Array, " d"],
    sigma_y: Float[Array, ""],
    mu_x: Float[Array, " d"],
    sigma_x: Float[Array, ""],
) -> tuple[RotationMatrix, ScalingTerm, Translation, Float[Array, "m d"]]:
    """Convert alignment parameters from normalized space back to original space.

    The mapping is:
    s = s_n * (sigma_x / sigma_y)
    R = R_n
    v = v_n * sigma_y
    t = sigma_x * t_n + mu_x - s * (mu_y @ R.T)

    Args:
        R_n: Rotation matrix from normalized alignment.
        s_n: Scale factor from normalized alignment.
        t_n: Translation vector from normalized alignment.
        v_n: Deformation field from normalized alignment.
        mu_y: Mean of original source points (Mov).
        sigma_y: Scale of original source points (Mov).
        mu_x: Mean of original target points (Ref).
        sigma_x: Scale of original target points (Ref).

    Returns:
        tuple: (R, s, t, v) in original space.
    """
    # 1. Scale
    s = s_n * (sigma_x / sigma_y)

    # 2. Rotation (unchanged)
    R = R_n

    # 3. Deformation
    v = v_n * sigma_y

    # 4. Translation
    # t = sigma_x * t_n + mu_x - s * (mu_y @ R.T)
    # Note on dimensions: mu_y is (D,), R is (D, D). mu_y @ R.T is (D,).
    t = (sigma_x * t_n) + mu_x - s * (mu_y @ R.T)

    return R, s, t, v
