"""
Forward kinematics (FK) for SMPL / SMPL-X via jax.lax.scan.

Joints must be in topological order (root first, parents before children).
The implementation uses a sequential scan over the joint list so that each
joint's global transform is available when its children are processed.

The FK transform for joint i is:
    G_i = G_{parent_i} @ T(j_i − j_{parent_i}) @ R_i     (non-root)
    G_0 =                T(j_0)                 @ R_0     (root)

where T(t) is the 4×4 pure-translation matrix for offset t, and R_i is the
3×3 local rotation matrix embedded in a 4×4 rigid body transform.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_rigid(R: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Pack R (3,3) and t (3,) into a 4×4 rigid body transform.

    The resulting matrix maps local-frame points p to world as: R @ p + t.
    """
    bottom = jnp.zeros((1, 4)).at[0, 3].set(1.0)
    top = jnp.concatenate([R, t[:, None]], axis=1)                 # (3, 4)
    return jnp.concatenate([top, bottom], axis=0)                  # (4, 4)


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
        (J, 4, 4) global transforms.
        G[i][:3, 3] gives the world-space position of joint i.
    """
    J = rotmats.shape[0]

    # Precompute local translation offsets in the bind pose:
    #   root   → absolute joint position
    #   others → offset from parent joint
    safe_parents = jnp.maximum(parents, 0)                         # clamp root index
    parent_joints = joints[safe_parents]                           # (J, 3)
    local_t = jnp.where(
        (parents >= 0)[:, None],
        joints - parent_joints,
        joints,
    )                                                               # (J, 3)

    # Build all J local 4×4 transforms at once (vectorised over joints)
    local_Ts = jax.vmap(_make_rigid)(rotmats, local_t)             # (J, 4, 4)

    # Sequential scan: carry = accumulated global transforms array
    def step(G_all: jnp.ndarray, i: jnp.ndarray):
        parent_idx = safe_parents[i]
        parent_G = G_all[parent_idx]                               # (4, 4)
        is_root = parents[i] < 0
        G_i = jnp.where(is_root, local_Ts[i], parent_G @ local_Ts[i])
        G_all = G_all.at[i].set(G_i)
        return G_all, None

    G_all, _ = jax.lax.scan(step, jnp.zeros((J, 4, 4)), jnp.arange(J))
    return G_all


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
