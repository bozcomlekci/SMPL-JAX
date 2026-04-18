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
# shared normalisation primitive
# ---------------------------------------------------------------------------

def safe_normalize(
    x: jnp.ndarray,
    axis: int | tuple = -1,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """L2-normalise *x* along *axis* with a gradient-safe epsilon.

    Unlike ``jnp.linalg.norm``, the gradient is finite everywhere including at
    **x = 0** because the squared norm is shifted by *eps* before the sqrt:
    ``x / sqrt(sum(x²) + eps)``.

    Args:
        x:    input array (any shape).
        axis: axis or axes to reduce over (default: last axis).
        eps:  small constant added inside the sqrt (default: 1e-12).

    Returns:
        Array with the same shape as *x*, unit-norm along *axis*.
    """
    return x / jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


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

    b1 = safe_normalize(a1)
    b2 = safe_normalize(a2 - jnp.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = jnp.cross(b1, b2)

    return jnp.stack([b1, b2, b3], axis=-1)
