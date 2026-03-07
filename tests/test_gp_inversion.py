import jax
import jax.numpy as jnp

from prex.cpd.bayes.kernel import gaussian_kernel
from prex.cpd.bayes.util import invert_gp_mapping


def test_gp_inversion_simple():
    # Setup control points
    mov = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Setup coefficients (some displacement)
    W = jnp.array([[0.1, 0.1], [-0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])

    beta = 1.0

    # Points in source space
    x_true = jnp.array([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])

    # Forward mapping
    def forward(x_pts):
        # G(x, mov) is (n, m)
        def affinity(x1, y1):
            return gaussian_kernel(x1[None, :], y1[None, :], beta)

        G = jax.vmap(lambda x: jax.vmap(lambda y: affinity(x, y))(mov))(x_pts)
        return x_pts + G @ W

    y_target = forward(x_true)

    # Invert mapping
    x_recovered = invert_gp_mapping(
        y_target, mov, W, gaussian_kernel, beta, max_iter=20, tol=1e-7
    )

    print("\nTrue x:\n", x_true)
    print("Recovered x:\n", x_recovered)

    # Verify
    mse = jnp.mean(jnp.square(x_true - x_recovered))
    print(f"MSE: {mse}")

    assert mse < 1e-6


if __name__ == "__main__":
    test_gp_inversion_simple()
