"""
Blend shape computations for SMPL and SMPL-X.

Three types of blend shapes are supported:
  - Shape blend shapes  — driven by identity coefficients (betas)
  - Expression blend shapes — driven by expression coefficients (SMPL-X only)
  - Pose blend shapes   — driven by per-joint rotation matrices
"""

import jax.numpy as jnp


def shape_blend_shapes(
    v_template: jnp.ndarray,
    shapedirs: jnp.ndarray,
    betas: jnp.ndarray,
) -> jnp.ndarray:
    """Apply shape blend shapes to the template mesh.

    Args:
        v_template: (V, 3)            template vertices.
        shapedirs:  (V, 3, num_betas) blend shape directions.
        betas:      (B, num_betas)    shape coefficients.

    Returns:
        (B, V, 3) shaped vertices.
    """
    blend = jnp.einsum("vcp,bp->bvc", shapedirs, betas)            # (B, V, 3)
    return v_template[None] + blend


def expression_blend_shapes(
    v_shaped: jnp.ndarray,
    exprdirs: jnp.ndarray,
    expression: jnp.ndarray,
) -> jnp.ndarray:
    """Apply expression blend shapes (SMPL-X only).

    Args:
        v_shaped:   (B, V, 3)            shape-blended vertices.
        exprdirs:   (V, 3, num_expr)     expression blend shape directions.
        expression: (B, num_expr)        expression coefficients.

    Returns:
        (B, V, 3) shape + expression blended vertices.
    """
    blend = jnp.einsum("vcp,bp->bvc", exprdirs, expression)        # (B, V, 3)
    return v_shaped + blend


def pose_blend_shapes(
    rotmats: jnp.ndarray,
    posedirs: jnp.ndarray,
) -> jnp.ndarray:
    """Apply pose-dependent corrective blend shapes.

    The pose feature vector is (R_i − I) for each non-root joint, flattened.
    posedirs maps this feature to per-vertex displacements.

    Args:
        rotmats:  (B, J, 3, 3) local rotation matrices for all joints.
        posedirs: (V*3, P)     blend shape directions;  P ≤ (J-1)*9.

    Returns:
        (B, V, 3) pose corrective displacements.
    """
    B, J = rotmats.shape[:2]
    P = posedirs.shape[1]
    V = posedirs.shape[0] // 3

    I = jnp.eye(3, dtype=rotmats.dtype)

    # Feature: flatten (R_i − I) for all non-root joints → (B, (J-1)*9)
    pose_feat = (rotmats[:, 1:] - I).reshape(B, -1)

    # Trim to the number of pose directions stored in the model
    pose_feat = pose_feat[:, :P]                                    # (B, P)

    # correctives: (B, P) @ (P, V*3) → (B, V*3) → (B, V, 3)
    correctives = (pose_feat @ posedirs.T).reshape(B, V, 3)
    return correctives
