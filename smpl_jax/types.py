"""
Typed parameter and output containers for SMPL and SMPL-X.

All containers are NamedTuples so they are valid JAX pytrees and
work transparently with jax.jit, jax.vmap, and jax.grad.
"""

from typing import NamedTuple

import jax.numpy as jnp


class SMPLXParams(NamedTuple):
    """Batched input parameters for SMPL-X forward pass.

    All arrays have a leading batch dimension B.
    """

    betas: jnp.ndarray          # (B, num_betas)   shape coefficients
    body_pose: jnp.ndarray      # (B, 63)           21 body joints × 3 axis-angle
    global_orient: jnp.ndarray  # (B, 3)            root joint axis-angle
    transl: jnp.ndarray         # (B, 3)            global translation
    expression: jnp.ndarray     # (B, num_expr)     expression coefficients
    left_hand_pose: jnp.ndarray  # (B, 45)          15 left-hand joints × 3 axis-angle
    right_hand_pose: jnp.ndarray  # (B, 45)         15 right-hand joints × 3 axis-angle


class SMPLXOutput(NamedTuple):
    """Output of SMPL-X forward pass."""

    vertices: jnp.ndarray  # (B, 10475, 3)  posed mesh vertices
    joints: jnp.ndarray    # (B, J, 3)      posed joint positions


class SMPLParams(NamedTuple):
    """Batched input parameters for SMPL forward pass."""

    betas: jnp.ndarray          # (B, num_betas)
    body_pose: jnp.ndarray      # (B, 69)  23 body joints × 3 axis-angle
    global_orient: jnp.ndarray  # (B, 3)
    transl: jnp.ndarray         # (B, 3)


class SMPLOutput(NamedTuple):
    """Output of SMPL forward pass."""

    vertices: jnp.ndarray  # (B, 6890, 3)
    joints: jnp.ndarray    # (B, J, 3)
