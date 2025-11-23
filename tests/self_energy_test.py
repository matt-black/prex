import jax
import jax.numpy as jnp

from prex.gmm.grad._util import compute_self_overlap_weights


def test_debug_sum():
    key = jax.random.PRNGKey(0)
    m, d = 5, 2
    means = jax.random.normal(key, (m, d))
    wgts = jax.nn.softmax(jax.random.normal(key, (m,)))
    var = 0.1
    n_dim = 2

    overlap_jk = compute_self_overlap_weights(means, wgts, var, n_dim)

    # r_self_j = sum_k O_jk * (mu_j - mu_k)
    r_self = jnp.sum(
        overlap_jk[:, :, jnp.newaxis]
        * (means[:, jnp.newaxis, :] - means[jnp.newaxis, :, :]),
        axis=1,
    )

    total_sum = jnp.sum(r_self, axis=0)
    assert jnp.allclose(total_sum, 0)

    # Check symmetry of O_jk
    assert jnp.allclose(overlap_jk, overlap_jk.T)

    # Check antisymmetry of diffs
    diffs = means[:, jnp.newaxis, :] - means[jnp.newaxis, :, :]
    assert jnp.allclose(diffs, -jnp.transpose(diffs, (1, 0, 2)))
