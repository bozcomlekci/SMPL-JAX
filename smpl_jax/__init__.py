"""
SMPL-JAX — fully differentiable, JIT-compiled SMPL and SMPL-X in JAX.

Public API
----------
Models:
    SMPLXModel   — SMPL-X forward pass (10 475 vertices, expression + hands)
    SMPLModel    — SMPL forward pass   (6 890 vertices)

Parameter / output containers (JAX pytrees):
    SMPLXParams, SMPLXOutput
    SMPLParams,  SMPLOutput

Pose representations:
    axis_angle_to_rotmat    — Rodrigues' formula
    rotmat_to_axis_angle    — inverse via SVD / atan2
    rotmat_to_6d            — first two columns
    rotation_6d_to_rotmat   — Gram-Schmidt

Inverse kinematics:
    inverse_lbs             — analytical init + Adam refinement
"""

from .types import SMPLXParams, SMPLXOutput, SMPLParams, SMPLOutput
from .smplx import SMPLXModel
from .smpl import SMPLModel
from .inverse_lbs import inverse_lbs
from .rotations import (
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
    rotmat_to_6d,
    rotation_6d_to_rotmat,
)

__all__ = [
    # models
    "SMPLXModel",
    "SMPLModel",
    # containers
    "SMPLXParams",
    "SMPLXOutput",
    "SMPLParams",
    "SMPLOutput",
    # inverse kinematics
    "inverse_lbs",
    # rotation utilities
    "axis_angle_to_rotmat",
    "rotmat_to_axis_angle",
    "rotmat_to_6d",
    "rotation_6d_to_rotmat",
]
