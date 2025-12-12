import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float, Int

from ._tree import build_tree, query_neighbors


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


def inv_zscore_coordinates(
    coordz: Float[Array, "n d"], mu: Float[Array, " d"], sd: Float[Array, ""]
) -> Float[Array, "n d"]:
    return (coordz * sd) + mu[jnp.newaxis, :]


def meshgrid_for_volume(
    ndep: int,
    nrow: int,
    ncol: int,
    normalize: bool,
) -> Int[Array, "z y x 3"]:
    """Create coordinate grid for a Float[Array, "z y x"] of specified dimensions.

    Args:
        ndep (int): number of planes in the Float[Array, "z y x"].
        nrow (int): number of rows in the Float[Array, "z y x"].
        ncol (int): number of columns in the Float[Array, "z y x"].
        normalize (bool): normalize coordinates to range [-1, 1]

    Returns:
        Int[Array, "z y x 3"]: array of coordinates
    """
    zs = jnp.linspace(0, ndep - 1, ndep)
    ys = jnp.linspace(0, nrow - 1, nrow)
    xs = jnp.linspace(0, ncol - 1, ncol)
    if normalize:
        # div = max(nrow, ncol, ndep)
        xs = (xs / (ncol - 1) - 0.5) * 2.0
        ys = (ys / (nrow - 1) - 0.5) * 2.0
        zs = (zs / (ndep - 1) - 0.5) * 2.0
    return jnp.stack(jnp.meshgrid(zs, ys, xs, indexing="ij"), axis=-1)


def grid_sample(
    image: Float[Array, "y x"] | Float[Array, "z y x"],
    coords: Array,
    mode: str = "linear",
    index: str = "ij",
) -> Array:
    """Sample an image at arbitrary coordinates.

    Args:
        image: 2D or 3D array to sample
        coords: Array of shape [h, w, 2] or [d, h, w, 3] containing coordinates in [0, dim] range
        mode: Interpolation mode ('linear'/'bilinear'/'trilinear' or 'nearest')

    Returns:
        Interpolated values of shape [B, h, w, C] or [B, h, w, d, C]
    """
    spatial_dims = image.shape
    if index == "xy":
        coords = jnp.concatenate(
            (coords[..., 1:2], coords[..., 0:1], coords[..., 2:]), -1
        )
    elif index != "ij":
        raise ValueError(f"Unsuported indexing type: {index}")
    if mode in {"linear", "bilinear", "trilinear"}:
        # corner coordinates (floor'd are _0, then get other corner with +1)
        # NOTE: index these using negative indices so that we don't have to repeat the logic for volumes
        i0 = jnp.floor(coords[..., -2]).astype(jnp.int32)
        j0 = jnp.floor(coords[..., -1]).astype(jnp.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        # clip to valid range (edge of image)
        # note that this is an implicit padding s.t. out of bounds values just get the value at the edge
        i0 = jnp.clip(i0, 0, spatial_dims[-2] - 1)
        i1 = jnp.clip(i1, 0, spatial_dims[-2] - 1)
        j0 = jnp.clip(j0, 0, spatial_dims[-1] - 1)
        j1 = jnp.clip(j1, 0, spatial_dims[-1] - 1)

        # calculate interpolation weights
        wi = coords[..., -2] - i0
        wj = coords[..., -1] - j0
        if len(spatial_dims) == 2:  # image
            output = (
                image[i0, j0] * (1 - wi) * (1 - wj)
                + image[i1, j0] * wi * (1 - wj)
                + image[i0, j1] * (1 - wi) * wj
                + image[i1, j1] * wi * wj
            )
        elif len(spatial_dims) == 3:
            # need to repeat above procedure for the leading dimension
            k0 = jnp.floor(coords[..., 0]).astype(jnp.int32)
            k1 = k0 + 1
            k0 = jnp.clip(k0, 0, spatial_dims[0] - 1)
            k1 = jnp.clip(k1, 0, spatial_dims[0] - 1)
            wk = coords[..., 0] - k0
            output = (
                image[k0, i0, j0] * (1 - wi) * (1 - wj) * (1 - wk)
                + image[k0, i1, j0] * wi * (1 - wj) * (1 - wk)
                + image[k0, i0, j1] * (1 - wi) * wj * (1 - wk)
                + image[k1, i0, j0] * (1 - wi) * (1 - wj) * wk
                + image[k1, i1, j0] * wi * (1 - wj) * wk
                + image[k1, i0, j1] * (1 - wi) * wj * wk
                + image[k0, i1, j1] * wi * wj * (1 - wk)
                + image[k1, i1, j1] * wi * wj * wk
            )
        else:
            raise ValueError("invalid input, must be 2- or 3-D")
    elif mode == "nearest":
        # Round coordinates to nearest integer
        y = jnp.clip(
            jnp.round(coords[..., -2]).astype(jnp.int32),
            0,
            spatial_dims[-2] - 1,
        )
        x = jnp.clip(
            jnp.round(coords[..., -1]).astype(jnp.int32),
            0,
            spatial_dims[-1] - 1,
        )
        if len(spatial_dims) == 2:
            output = image[y, x]
        else:
            z = jnp.clip(
                jnp.round(coords[..., 0]).astype(jnp.int32),
                0,
                spatial_dims[0] - 1,
            )
            output = image[z, y, x]
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    return output


def normalized_cross_correlation(
    vol1: Float[Array, "z y x"] | Float[Array, "y x"],
    vol2: Float[Array, "z y x"] | Float[Array, "y x"],
) -> Float[Array, ""]:
    """Compute normalized cross-correlation between two volumes or images.

    Args:
        vol1: First volume/image
        vol2: Second volume/image (must have same shape as vol1)

    Returns:
        Normalized cross-correlation coefficient (scalar)
    """
    # Flatten volumes
    v1_flat = vol1.ravel()
    v2_flat = vol2.ravel()

    # Normalize (zero mean, unit variance)
    v1_mean = jnp.mean(v1_flat)
    v2_mean = jnp.mean(v2_flat)
    v1_centered = v1_flat - v1_mean
    v2_centered = v2_flat - v2_mean

    v1_std = jnp.std(v1_flat)
    v2_std = jnp.std(v2_flat)

    # Avoid division by zero
    v1_std = jnp.maximum(v1_std, 1e-8)
    v2_std = jnp.maximum(v2_std, 1e-8)

    v1_norm = v1_centered / v1_std
    v2_norm = v2_centered / v2_std

    # Compute correlation
    ncc = jnp.mean(v1_norm * v2_norm)
    return ncc


@Partial(jax.jit, static_argnums=(3, 4, 5))
def gaussian_rbf_interpolate(
    query_pts: Float[Array, "m d"],
    ctrl_pts: Float[Array, "n d"],
    ctrl_vals: Float[Array, "n d"],
    bandwidth: float | None = None,
    k_neighbors: int = 50,
    scan: bool = False,
):
    """Do sparse gaussian RBF interpolation using k-nearest neighbors.

    This uses a KD tree to find k nearest neighbors for each query point,
    then performs local RBF interpolation. Much faster than dense RBF for
    large point sets (1000-10000 points).

    Args:
        query_pts: (M, d) array of points where you want to interpolate
        ctrl_pts: (N, d) array of control point locations (your q points)
        ctrl_vals: (N, d) array of values at control points (your w vectors)
        bandwidth: Gaussian bandwidth σ (if None, auto-computed as median distance)
        k_neighbors: number of nearest neighbors to use for local interpolation (default: 50)

    Returns:
        (M, d) array of interpolated values
    """

    # Auto-compute bandwidth if not provided
    if bandwidth is None:
        # Use median pairwise distance heuristic
        sample_size = min(1000, ctrl_pts.shape[0])
        indices = jax.random.permutation(
            jax.random.PRNGKey(0), ctrl_pts.shape[0]
        )[:sample_size]
        sample_points = ctrl_pts[indices]
        dists = jnp.linalg.norm(
            sample_points[:, None, :] - sample_points[None, :, :], axis=-1
        )
        bandwidth = jnp.median(dists)  # pyright: ignore[reportAssignmentType]

    # Build KD tree for control points
    tree = build_tree(ctrl_pts, optimize=True)

    # Find k nearest neighbors for each query point
    k = min(k_neighbors, ctrl_pts.shape[0])
    neighbor_inds, _ = query_neighbors(tree, query_pts, k=k)

    # Perform local RBF interpolation for each query point
    def interpolate_single_point(query_idx: Int) -> Float[Array, " d"]:
        # Get k nearest control points and their values
        nn_indices = neighbor_inds[query_idx]  # (k,)
        nn_points = ctrl_pts[nn_indices]  # (k, d)
        nn_values = ctrl_vals[nn_indices]  # (k, d)

        # Compute RBF matrix for local neighborhood (k x k)
        # Using Gaussian RBF: φ(r) = exp(-r²/(2σ²))
        dist_matrix = jnp.linalg.norm(
            nn_points[:, None, :] - nn_points[None, :, :], axis=-1
        )
        phi_matrix = jnp.exp(
            -(dist_matrix**2) / (2 * bandwidth**2)
        )  # pyright: ignore[reportOptionalOperand]

        # Solve for local weights (k x d)
        # Add regularization for numerical stability
        local_weights = jnp.linalg.solve(
            phi_matrix + 1e-8 * jnp.eye(k), nn_values
        )

        # Evaluate at query point
        query_point = query_pts[query_idx]
        query_dist = jnp.linalg.norm(
            nn_points - query_point[None, :], axis=-1
        )  # (k,)
        query_phi = jnp.exp(
            -(query_dist**2) / (2 * bandwidth**2)
        )  # pyright: ignore[reportOptionalOperand] # (k,)

        # Interpolated value (d,)
        return query_phi @ local_weights

    # Vectorize over all query points
    if scan:

        def interp(_: None, query_idx: Int) -> tuple[None, Float[Array, " d"]]:
            return None, interpolate_single_point(query_idx)

        _, vecs = jax.lax.scan(interp, None, jnp.arange(query_pts.shape[0]))
        return vecs
    else:
        return jax.vmap(interpolate_single_point, 0, 0)(
            jnp.arange(query_pts.shape[0])
        )
