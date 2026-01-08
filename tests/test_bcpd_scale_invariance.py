import math

import jax
import jax.numpy as jnp

from prex.cpd.bayes import align as bcpd_align
from prex.cpd.bayes.kernel import gaussian_kernel
from prex.cpd.bayes.util import transform as bcpd_transform


def test_bcpd_large_scale_and_deformation():
    """
    Verify that BCPD can recover a shape that has undergone both
    significant non-rigid deformation AND large scale change.

    This tests whether the 'Rigid First' initialization strategy successfully
    resolves the scale/variance ambiguity.
    """
    jax.config.update("jax_enable_x64", True)

    # 1. Generate Ref (Standard Grid)
    steps = 10
    x = jnp.linspace(-1.0, 1.0, steps)
    y = jnp.linspace(-1.0, 1.0, steps)
    X, Y = jnp.meshgrid(x, y)
    ref = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (100, 2)

    # 2. Apply Non-Rigid Deformation (Sine Wave)
    # v_gt = 0.1 * sin(2*pi*x)
    v_gt = jnp.zeros_like(ref)
    v_gt = v_gt.at[:, 1].set(0.2 * jnp.sin(2 * math.pi * ref[:, 0]))

    deformed_ref = ref + v_gt

    # 3. Apply Large Rigid Transform
    # Scale = 5.0 (Significant scaling)
    # Rotation = 45 degrees
    theta = math.radians(45)
    c, s = math.cos(theta), math.sin(theta)
    R_gt = jnp.array([[c, -s], [s, c]])
    s_gt = 5.0
    t_gt = jnp.array([0.0, 0.0])

    # mov = s * (deformed_ref @ R) + t
    # Note: Our util.transform computes s * (pts @ R.T) + t.
    # To use it for generation, we pass R_gt.T so it does pts @ R_gt
    mov = bcpd_transform(
        deformed_ref, R_gt.T, s_gt, t_gt
    )  # pyright: ignore[reportArgumentType]

    # 4. Align Mov -> Ref
    # The algorithm needs to reverse the rigid transform and then (or simultaneously)
    # reverse the non-rigid deformation (or find the deformation from ref to mov?)
    # Wait, BCPD usually models: Ref = T(Mov) + v ?? OR Mov = T(Ref + v)?
    #
    # Let's clarify the BCPD model in my library:
    # align(ref, mov) -> returns parameters (P, R, s, t, v)
    # y_hat = transform(mov + v, R, s, t)
    # It tries to match y_hat to ref.
    #
    # So the model is: Ref ~ s * (Mov + v) * R^T + t
    #
    # The data generation above did: Mov = s_gt * (Ref + v_gt) * R_gt + t_gt
    # This is slightly different.
    #
    # To test nicely, let's look at it from the algorithm's perspective:
    # We want Mov to be the source.
    # We want Ref to be the target.
    #
    # Let's generate Mov as the "clean" shape, and Ref as the "warped + scaled" shape?
    # No, typically "Ref" is a template, and "Mov" is observed data.
    #
    # Let's stick to the user request: "apply a known combined rigid+nonrigid transform to a toy dataset"
    # and align "the transformed dataset [Mov] to the original one [Ref]".
    #
    # So:
    # Original (Ref): The Grid.
    # Transformed (Mov): Grid + Deformation + Rigid Body.
    #
    # Algorithm aligns Mov -> Ref.
    # Model: Transformed_Mov = s * (Mov + v) @ R.T + t  ~~ Ref
    #
    # If Mov = s_gt * (Ref + v_gt) ... this is hard to invert perfectly with the standard model if v applies first.
    #
    # Let's generate data consistent with the BCPD model structure to verify parameters,
    # OR generate "realistic" data and just check alignment error.
    #
    # Let's use the BCPD model structure for generation to check param recovery:
    # Ref is Target. Mov is Source.
    # Data Gen: Ref = s_real * (Mov + v_real) @ R_real.T + t_real
    #
    # So start with Mov (e.g. Grid).
    # Add deformation v_real.
    # Apply rigid transform to get Ref.
    #
    # Wait, usually we align a "Subject" (Mov) to a "Atlas" (Ref).
    # If I scale the Subject by 5x, I want to shrink it back to the Atlas.
    #
    # Let's generate:
    # Ref = Grid (Scale ~ 1)
    # Mov = Grid + Deformation, then Scaled up by 5.0.
    # Mov = s_gt * (Ref + v_gt) ...
    #
    # If BCPD aligns Mov -> Ref:
    # It does y_hat = s * (Mov + v) + t.
    # It needs to find s ~ 1/5.

    # Let's generate exactly that:
    # Mov = Grid
    # Ref = s_gt * (Mov + v_gt) @ R_gt.T + t_gt
    #
    # Then running align(ref, mov) is weird because Ref is usually the fixed coordinate system.
    #
    # Standard Registration Task:
    # Ref: The target shape (e.g. Atlas).
    # Mov: The source shape (e.g. Scanned Mesh).
    # Goal: Move Mov to match Ref.
    #
    # Test Case:
    # Ref = Grid.
    # Mov = (Grid + v_warp) transformed by (s=0.2, etc).
    # Mov is a "miniature deformed version" of Ref.
    #
    # Goal: BCPD should scale Mov UP (s=5.0), un-rotate it, and un-deform it to match Ref.
    # Model: y_hat = s * (Mov + v) @ R.T + t.
    #
    # So if Mov is the "base" scaling, and Ref is "base", then s=1.
    # But if Mov is 0.2 scale:
    # y_hat = 5.0 * (Mov + v) ... -> Ref.
    #
    # So:
    # 1. Create Clean Grid (Base).
    # 2. Create Mov = 0.2 * (Base + v_warp) + t_shift
    # 3. Ref = Base.
    # 4. Align Mov -> Ref.
    # 5. Expect s ~ 5.0.

    # Setup Data
    # Base: Grid
    base = ref  # (100, 2) from earlier

    # Deformation
    v_base = jnp.zeros_like(base)
    # A bulge in the middle
    v_base = v_base.at[:, 2].set(0.0) if base.shape[1] == 3 else v_base
    # Simple sine warp
    v_base = v_base.at[:, 1].set(0.3 * jnp.sin(3 * base[:, 0]))

    # Distorted shape (before rigid)
    distorted_base = base + v_base

    # Apply Rigid Transform to create Mov
    # We want Mov to be "small" and "rotated".
    # Mov = s_gen * distorted_base @ R_gen.T + t_gen
    s_gen = 0.2
    theta_gen = math.radians(30)
    c, s = math.cos(theta_gen), math.sin(theta_gen)
    R_gen = jnp.array([[c, -s], [s, c]])  # To rotate defined points
    t_gen = jnp.array([5.0, 5.0])

    # Mov construction
    # We use our transform util: s * (x @ R.T) + t
    # We want to rotate by +30 deg. So allow R_gen to be the rotation matrix.
    # transform uses R^T. So if we pass R_gen.T, it applies (R^T)^T = R.
    mov = bcpd_transform(
        distorted_base, R_gen.T, s_gen, t_gen
    )  # pyright: ignore[reportArgumentType]

    # Align Mov -> Ref
    # Expected result:
    # s_estimated ~ 1 / s_gen = 5.0
    # R_estimated ~ R_gen^T (to undo rotation?) or R_gen?
    # y_hat = s * (mov + v) @ R.T + t
    # This aligns Mov to Ref.
    # Ideally, y_hat ~= Ref.

    print("Test Generation:")
    print(f"  Scale Factor: {s_gen} (Mov is 0.2x of Ref)")
    print(f"  Expected Recovery Scale: {1.0/s_gen}")

    # Step A: Validate Rigid-Only Alignment first
    print("\n--- Step A: Rigid Only Alignment ---")
    (P_r, R_r, s_r, t_r, v_r), _ = bcpd_align(
        ref,
        mov,
        outlier_prob=0.0,
        num_iter=50,
        tolerance=None,
        kernel=gaussian_kernel,
        lambda_param=2.0,
        kernel_beta=2.0,
        gamma=1.0,
        kappa=1000.0,
        transform_mode="rigid",
        normalize_input=True,
    )
    mov_r = bcpd_transform(mov, R_r, s_r, t_r)
    mse_r = jnp.mean(jnp.sum((ref - mov_r) ** 2, axis=1))
    print(f"  Rigid MSE: {mse_r:.6f}")
    print(f"  Rigid s: {s_r:.4f}")

    # Step B: Both Mode
    print("\n--- Step B: Both Mode Alignment ---")
    (P, R_opt, s_opt, t_opt, v_opt), result = bcpd_align(
        ref,
        mov,
        outlier_prob=0.0,
        num_iter=100,
        tolerance=1e-5,
        kernel=gaussian_kernel,
        lambda_param=10000.0,  # High regularization to force near-rigid behavior
        kernel_beta=2.0,  # Kernel width for normalized space (unit variance)
        gamma=1.0,  # Initial variance scaling.
        kappa=1000.0,  # Effectively uniform mixing
        transform_mode="both",
        normalize_input=True,
    )

    # Apply transform
    aligned_mov = bcpd_transform(mov + v_opt, R_opt, s_opt, t_opt)

    # Metrics
    mse = jnp.mean(jnp.sum((ref - aligned_mov) ** 2, axis=1))
    print("Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  s_opt: {s_opt:.4f} (Expected ~5.0)")

    # Assertions
    # 1. Alignment quality
    assert (
        mse < 0.1
    ), f"Alignment failed, MSE {mse} too high (Rigid MSE was {mse_r})"

    # 2. Scale recovery
    assert s_opt > 3.0, f"Failed to recover large scale factor, got {s_opt}"
    assert (
        abs(s_opt - 5.0) < 1.0
    ), f"Scale parameter {s_opt} unreasonably far from 5.0"


if __name__ == "__main__":
    test_bcpd_large_scale_and_deformation()
