"""
Differentiable rotation representations.

All functions are compatible with jax.jit, jax.vmap, and jax.grad.

Representations:
  axis-angle  — 3D vector; magnitude = rotation angle (radians)
  rotmat      — 3×3 SO(3) rotation matrix
  6D          — first two columns of a rotation matrix (Zhou et al., 2019)
                recovered via Gram-Schmidt orthogonalisation
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# axis-angle ↔ rotation matrix
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(aa: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle to rotation matrix via Rodrigues' formula.

    Numerically stable everywhere (including zero-angle), with well-defined
    gradients at aa = 0 via a squared-norm + epsilon formulation.

    Args:
        aa: (..., 3) axis-angle vectors.  Magnitude encodes the angle.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    # sqrt(||aa||^2 + eps) avoids the NaN gradient of ||aa|| at zero.
    angle = jnp.sqrt(
        jnp.sum(aa * aa, axis=-1, keepdims=True) + 1e-12
    )                                                               # (..., 1)
    axis = aa / angle                                              # (..., 3) unit vector

    cos_a = jnp.cos(angle)   # (..., 1)
    sin_a = jnp.sin(angle)   # (..., 1)

    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    zero = jnp.zeros_like(x)

    # Skew-symmetric cross-product matrix K  [[0,-z,y],[z,0,-x],[-y,x,0]]
    K = jnp.concatenate(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(aa.shape[:-1] + (3, 3))

    # Outer product n ⊗ n
    outer = axis[..., :, None] * axis[..., None, :]               # (..., 3, 3)

    I = jnp.eye(3, dtype=aa.dtype)
    return (
        cos_a[..., None] * I
        + (1.0 - cos_a[..., None]) * outer
        + sin_a[..., None] * K
    )


def rotmat_to_axis_angle(R: jnp.ndarray) -> jnp.ndarray:
    """Convert rotation matrix to axis-angle.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) axis-angle vectors.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_a = jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cos_a)                                      # (...)

    # Axis from skew-symmetric part  (R - R.T) / (2 sin θ)
    axis = jnp.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    ) / (2.0 * jnp.sin(angle)[..., None] + 1e-8)

    aa = axis * angle[..., None]
    return jnp.where(jnp.abs(angle)[..., None] < 1e-8, jnp.zeros_like(aa), aa)


# ---------------------------------------------------------------------------
# rotation matrix ↔ 6D continuous representation
# ---------------------------------------------------------------------------

def rotmat_to_6d(R: jnp.ndarray) -> jnp.ndarray:
    """Rotation matrix → 6D continuous representation.

    Extracts the first two columns of R.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6)
    """
    return jnp.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rotation_6d_to_rotmat(r6d: jnp.ndarray) -> jnp.ndarray:
    """6D representation → rotation matrix via Gram-Schmidt.

    Args:
        r6d: (..., 6)

    Returns:
        (..., 3, 3) proper rotation matrices (det = +1).
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:]

    # sqrt(sum²+ε) instead of linalg.norm: gradient is defined everywhere,
    # including at zero (linalg.norm has 0/0 gradient at the origin).
    b1 = a1 / jnp.sqrt(jnp.sum(a1 * a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / jnp.sqrt(jnp.sum(b2 * b2, axis=-1, keepdims=True) + 1e-12)
    b3 = jnp.cross(b1, b2)

    return jnp.stack([b1, b2, b3], axis=-1)
