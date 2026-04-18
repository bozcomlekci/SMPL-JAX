"""
Forward kinematics (FK) for SMPL / SMPL-X via jax.lax.scan.

Joints must be in topological order (root first, parents before children).
The implementation uses a sequential scan over the joint list so that each
joint's global transform is available when its children are processed.

The FK transform for joint i is:
    G_i = G_{parent_i} ∘ T(j_i − j_{parent_i}) ∘ R_i     (non-root)
    G_0 =               T(j_0)                 ∘ R_0     (root)

where ∘ denotes rigid-body composition and T(t) is pure translation by t.

Each global transform is stored internally as a compact (3, 4) = [R | t] matrix
(the bottom [0,0,0,1] row is implied), which halves the carry memory and
replaces 4×4 matrix multiplications with a cheaper 3×3 @ 3×3 + 3-vector op.
The final output is promoted to (4, 4) for API compatibility.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# single-sample FK  (vmap-able)
# ---------------------------------------------------------------------------

def fk_forward(
    rotmats: jnp.ndarray,
    joints: jnp.ndarray,
    parents: jnp.ndarray,
) -> jnp.ndarray:
    """Compute global rigid-body transforms for all joints.

    Args:
        rotmats: (J, 3, 3)  local rotation matrices (body frame).
        joints:  (J, 3)     bind-pose joint positions in world space.
        parents: (J,)       parent indices; parents[0] must be -1 (root).

    Returns:
        (J, 4, 4) global transforms in SE(3).
        G[i][:3, 3] gives the world-space position of joint i after posing.
    """
    J = rotmats.shape[0]

    # Local translation offsets in the bind pose:
    #   root   → absolute joint position
    #   others → offset from parent joint
    safe_parents = jnp.maximum(parents, 0)
    local_t = jnp.where(
        (parents >= 0)[:, None],
        joints - joints[safe_parents],
        joints,
    )                                                               # (J, 3)

    # Represent each local transform as (3, 4) = [R | t]
    # Avoids building 4×4 matrices; the bottom row [0,0,0,1] is implicit.
    local_Rts = jnp.concatenate(
        [rotmats, local_t[:, :, None]], axis=-1
    )                                                               # (J, 3, 4)

    def step(G_all: jnp.ndarray, i: jnp.ndarray):
        """Compose parent global transform with local joint transform."""
        R_l = local_Rts[i, :, :3]              # (3, 3)  local rotation
        t_l = local_Rts[i, :, 3]               # (3,)    local translation

        parent_Rt = G_all[safe_parents[i]]      # (3, 4)  parent global [R|t]
        R_p = parent_Rt[:, :3]                  # (3, 3)
        t_p = parent_Rt[:, 3]                   # (3,)

        # Rigid-body composition: [R_p|t_p] ∘ [R_l|t_l] = [R_p@R_l | R_p@t_l+t_p]
        # 3×3 @ 3×3 + 3-vector = 15 multiply-adds  vs  64 for full 4×4 matmul
        R_g = R_p @ R_l
        t_g = R_p @ t_l + t_p

        is_root = parents[i] < 0
        R_g = jnp.where(is_root, R_l, R_g)
        t_g = jnp.where(is_root, t_l, t_g)

        G_i = jnp.concatenate([R_g, t_g[:, None]], axis=-1)       # (3, 4)
        return G_all.at[i].set(G_i), None

    G34, _ = jax.lax.scan(
        step, jnp.zeros((J, 3, 4), dtype=rotmats.dtype), jnp.arange(J)
    )

    # Promote (J, 3, 4) → (J, 4, 4) by appending the constant [0,0,0,1] row
    bottom = jnp.zeros((J, 1, 4), dtype=rotmats.dtype).at[:, 0, 3].set(1.0)
    return jnp.concatenate([G34, bottom], axis=1)                  # (J, 4, 4)


# ---------------------------------------------------------------------------
# batched FK
# ---------------------------------------------------------------------------

def fk_forward_batched(
    rotmats: jnp.ndarray,
    joints: jnp.ndarray,
    parents: jnp.ndarray,
) -> jnp.ndarray:
    """Batched FK via vmap over the leading batch dimension.

    Args:
        rotmats: (B, J, 3, 3)
        joints:  (B, J, 3)
        parents: (J,)  — shared across the batch

    Returns:
        (B, J, 4, 4)
    """
    return jax.vmap(lambda r, j: fk_forward(r, j, parents))(rotmats, joints)
