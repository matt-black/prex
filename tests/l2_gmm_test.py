import jax
import jax.numpy as jnp

from prex.gmm.dist import l2_distance_gmm, l2_distance_gmm_opt

jax.config.update("jax_platform_name", "cpu")


def test_identical_gmm():
    """Test that identical GMMs have zero L2 distance."""
    # Create a simple GMM
    means = jnp.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    covs = jnp.array([jnp.eye(2) * 0.1, jnp.eye(2) * 0.2, jnp.eye(2) * 0.15])
    weights = jnp.array([0.3, 0.4, 0.3])

    # Distance between identical GMMs
    distance = l2_distance_gmm(means, covs, weights, means, covs, weights)
    assert jnp.isclose(distance, 0.0)


def test_single_gaussian():
    """Test that metric works for single Gaussian distributions."""

    # Two identical single Gaussians
    mean1 = jnp.array([[0.0, 0.0]])
    cov1 = jnp.array([jnp.eye(2) * 0.1])
    weights1 = jnp.array([1.0])

    distance_same = l2_distance_gmm(
        mean1, cov1, weights1, mean1, cov1, weights1
    )

    # Two different single Gaussians
    mean2 = jnp.array([[1.0, 1.0]])
    cov2 = jnp.array([jnp.eye(2) * 0.2])
    weights2 = jnp.array([1.0])

    distance_diff = l2_distance_gmm(
        mean1, cov1, weights1, mean2, cov2, weights2
    )

    assert jnp.isclose(distance_same, 0.0) and distance_same < distance_diff


def test_known_analytical_case():
    """Test with a case that has known analytical solution."""
    # Two 1D Gaussians
    means1 = jnp.array([[0.0]])  # shape (1, 1)
    means2 = jnp.array([[1.0]])  # shape (1, 1)
    covs1 = jnp.array([jnp.eye(1) * 1.0])  # variance = 1
    covs2 = jnp.array([jnp.eye(1) * 1.0])  # variance = 1
    weights1 = jnp.array([1.0])
    weights2 = jnp.array([1.0])

    # Analytical calculation for L2 distance between two 1D Gaussians
    # E_f = ∫ N(x|0,1)² dx = N(0|0,2) = 1/√(4π)
    # C_gf = ∫ N(x|0,1)N(x|1,1) dx = N(0|1,2) = 1/√(4π) * exp(-0.5*(1²/2))
    # L2 = E_f - 2*C_gf
    E_f = 1.0 / jnp.sqrt(4 * jnp.pi)
    C_gf = 1.0 / jnp.sqrt(4 * jnp.pi) * jnp.exp(-0.25)  # exp(-0.5 * (1²/2))
    analytical_l2 = E_f - 2 * C_gf

    computed_l2 = l2_distance_gmm_opt(
        means1, covs1, weights1, means2, covs2, weights2
    )

    assert jnp.isclose(analytical_l2, computed_l2)


def test_weights_effect():
    """Test that weights affect the distance calculation."""

    means = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    covs = jnp.array([jnp.eye(2) * 0.1, jnp.eye(2) * 0.1])

    # Different weight distributions
    weights1 = jnp.array([0.9, 0.1])  # Mostly first component
    weights2 = jnp.array([0.1, 0.9])  # Mostly second component
    weights3 = jnp.array([0.5, 0.5])  # Equal weights

    distance_1vs2 = l2_distance_gmm(
        means, covs, weights1, means, covs, weights2
    )
    assert distance_1vs2 > 0
    distance_1vs3 = l2_distance_gmm(
        means, covs, weights1, means, covs, weights3
    )
    assert distance_1vs3 > 0
    distance_2vs3 = l2_distance_gmm(
        means, covs, weights2, means, covs, weights3
    )
    assert distance_2vs3 > 0


def test_covariance_effect():
    """Test that covariance matrices affect the distance."""
    print("Test 5: Covariance sensitivity")

    means = jnp.array([[0.0, 0.0]])

    # Different covariance matrices
    covs_small = jnp.array([jnp.eye(2) * 0.1])
    covs_large = jnp.array([jnp.eye(2) * 1.0])
    weights = jnp.array([1.0])

    distance_same_small = l2_distance_gmm(
        means, covs_small, weights, means, covs_small, weights
    )
    distance_same_large = l2_distance_gmm(
        means, covs_large, weights, means, covs_large, weights
    )
    distance_mixed = l2_distance_gmm(
        means, covs_small, weights, means, covs_large, weights
    )

    assert jnp.isclose(distance_same_small, 0.0)
    assert jnp.isclose(distance_same_large, 0.0)
    assert distance_mixed > 0


def test_symmetry():
    """Test that the distance function is symmetric."""

    means1 = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    means2 = jnp.array([[0.5, 0.5], [-0.5, -0.5]])
    covs1 = jnp.array([jnp.eye(2) * 0.1, jnp.eye(2) * 0.2])
    covs2 = jnp.array([jnp.eye(2) * 0.15, jnp.eye(2) * 0.25])
    weights1 = jnp.array([0.6, 0.4])
    weights2 = jnp.array([0.7, 0.3])

    distance_1to2 = l2_distance_gmm(
        means1, covs1, weights1, means2, covs2, weights2
    )
    distance_2to1 = l2_distance_gmm(
        means2, covs2, weights2, means1, covs1, weights1
    )

    assert jnp.isclose(distance_1to2, distance_2to1)


def test_non_negativity():
    """Test that the distance is always non-negative."""

    # Test multiple random GMM pairs
    key = jax.random.PRNGKey(42)
    n_tests = 10
    distances = []
    for _ in range(n_tests):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Random GMM parameters
        n_comp1, n_comp2, d = 3, 4, 2
        means1 = jax.random.normal(subkey1, (n_comp1, d))
        means2 = jax.random.normal(subkey2, (n_comp2, d))

        # Positive definite covariances
        covs1 = jnp.stack(
            [
                jnp.eye(d) * (0.1 + jax.random.uniform(subkey1, ()) * 0.9)
                for _ in range(n_comp1)
            ]
        )
        covs2 = jnp.stack(
            [
                jnp.eye(d) * (0.1 + jax.random.uniform(subkey2, ()) * 0.9)
                for _ in range(n_comp2)
            ]
        )

        # Valid weight distributions
        weights1 = jax.random.uniform(subkey1, (n_comp1,))
        weights1 = weights1 / jnp.sum(weights1)
        weights2 = jax.random.uniform(subkey2, (n_comp2,))
        weights2 = weights2 / jnp.sum(weights2)

        distance = l2_distance_gmm(
            means1, covs1, weights1, means2, covs2, weights2
        )
        distances.append(distance)
    distances = jnp.array(distances)
    assert jnp.all(distances > -1e-10)


def test_jit():
    """Test that JIT compilation works correctly, distance is JIT-compatible."""
    # Create sample data
    means1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    means2 = jnp.array([[0.5, 0.5], [-0.5, -0.5]])
    covs1 = jnp.array([jnp.eye(2) * 0.1, jnp.eye(2) * 0.2])
    covs2 = jnp.array([jnp.eye(2) * 0.15, jnp.eye(2) * 0.25])
    weights1 = jnp.array([0.6, 0.4])
    weights2 = jnp.array([0.7, 0.3])

    d_jit = jax.jit(l2_distance_gmm)(
        means1, covs1, weights1, means2, covs2, weights2
    )
    d_reg = l2_distance_gmm(means1, covs1, weights1, means2, covs2, weights2)
    assert jnp.isclose(d_jit, d_reg)
