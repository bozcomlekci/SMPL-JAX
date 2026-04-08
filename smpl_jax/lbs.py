"""
Linear Blend Skinning (LBS) for SMPL / SMPL-X.

The LBS pipeline is:
  1. Compute per-joint relative transforms from FK globals and bind-pose joints.
  2. Blend the per-joint transforms at each vertex using skinning weights.
  3. Apply the blended transform to the shaped + corrected vertex positions.
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
              [      0       |           1              ]

    Args:
        G:      (B, J, 4, 4)  global FK transforms.
        joints: (B, J, 3)     bind-pose joint positions (world space).

    Returns:
        (B, J, 4, 4) relative blend transforms.
    """
    R = G[..., :3, :3]                                             # (B, J, 3, 3)
    t = G[..., :3, 3]                                              # (B, J, 3)

    # t_rel_i = t_i − R_i @ j_i
    t_rel = t - jnp.einsum("bjrc,bjc->bjr", R, joints)            # (B, J, 3)

    B, J = R.shape[:2]
    bottom = jnp.concatenate(
        [jnp.zeros((B, J, 1, 3)), jnp.ones((B, J, 1, 1))], axis=-1
    )                                                               # (B, J, 1, 4)
    top = jnp.concatenate([R, t_rel[..., None]], axis=-1)          # (B, J, 3, 4)
    return jnp.concatenate([top, bottom], axis=-2)                 # (B, J, 4, 4)


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
        M:                (B, J, 4, 4) per-joint relative blend transforms.
        weights:          (V, J)     skinning weights (should sum to 1 per vertex).

    Returns:
        (B, V, 3) posed vertices.
    """
    v = v_shaped + pose_correctives                                 # (B, V, 3)

    # Weighted sum of per-joint transforms at each vertex
    # T[b, v] = Σ_j  w[v, j] * M[b, j]
    T = jnp.einsum("vj,bjrc->bvrc", weights, M)                   # (B, V, 4, 4)

    # Homogeneous vertex coordinates
    v_hom = jnp.concatenate(
        [v, jnp.ones(v.shape[:-1] + (1,))], axis=-1
    )                                                               # (B, V, 4)

    # Apply blended transform
    v_posed = jnp.einsum("bvrc,bvc->bvr", T, v_hom)               # (B, V, 4)
    return v_posed[..., :3]
