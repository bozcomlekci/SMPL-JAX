"""
Linear Blend Skinning (LBS) for SMPL / SMPL-X.

The LBS pipeline is:
  1. Compute per-joint relative transforms from FK globals and bind-pose joints.
  2. Blend the per-joint transforms at each vertex using skinning weights.
  3. Apply the blended transform to the shaped + corrected vertex positions.

Transforms are stored as compact (3, 4) = [R | t] matrices throughout to avoid
the overhead of the constant [0,0,0,1] bottom row and homogeneous coordinates.
"""

import jax.numpy as jnp


def lbs_transforms(
    G: jnp.ndarray,
    joints: jnp.ndarray,
) -> jnp.ndarray:
    """Compute per-joint LBS transforms relative to the bind pose.

    For each joint i:
        M_i = G_i  @  G_i_bind_inv
            = [ R_global_i  |  t_i − R_global_i @ j_i ]

    Args:
        G:      (B, J, 4, 4)  global FK transforms.
        joints: (B, J, 3)     bind-pose joint positions (world space).

    Returns:
        (B, J, 3, 4) compact [R | t_rel] blend transforms.
        The implicit bottom row is [0, 0, 0, 1].
    """
    R = G[..., :3, :3]                                             # (B, J, 3, 3)
    t = G[..., :3, 3]                                              # (B, J, 3)

    # t_rel_i = t_i − R_i @ j_i
    t_rel = t - jnp.einsum("bjrc,bjc->bjr", R, joints)            # (B, J, 3)

    return jnp.concatenate([R, t_rel[..., None]], axis=-1)         # (B, J, 3, 4)


def lbs(
    v_shaped: jnp.ndarray,
    pose_correctives: jnp.ndarray,
    M: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Linear Blend Skinning.

    Args:
        v_shaped:         (B, V, 3)  shape (+ expression) blended vertices.
        pose_correctives: (B, V, 3)  pose corrective displacements.
        M:                (B, J, 3, 4) per-joint compact [R | t] blend transforms.
        weights:          (V, J)     skinning weights (should sum to 1 per vertex).

    Returns:
        (B, V, 3) posed vertices.
    """
    v = v_shaped + pose_correctives                                 # (B, V, 3)

    # Blend rotation and translation separately — avoids the (B, V, 4, 4)
    # intermediate and homogeneous coordinate overhead of the naive approach.
    R_v = jnp.einsum("vj,bjrc->bvrc", weights, M[..., :3])        # (B, V, 3, 3)
    t_v = jnp.einsum("vj,bjr->bvr",  weights, M[..., 3])          # (B, V, 3)

    # Apply blended rigid transform: R_v @ v + t_v
    return jnp.einsum("bvrc,bvc->bvr", R_v, v) + t_v              # (B, V, 3)
