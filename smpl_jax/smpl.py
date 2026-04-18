"""
SMPL: Skinned Multi-Person Linear Model in JAX.

SMPL has 6 890 vertices, 24 joints (root + 23 body), and 10 shape components
by default.  It does not have expression blend shapes or hand articulation.

Reference: Loper et al., "SMPL: A Skinned Multi-Person Linear Model",
SIGGRAPH Asia 2015.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import SMPLParams, SMPLOutput
from .model_io import load_model_data
from .rotations import axis_angle_to_rotmat
from ._base import _SMPLBase


class SMPLModel(_SMPLBase):
    """JAX port of SMPL.

    Usage::

        model = SMPLModel.load("data/smpl/SMPL_NEUTRAL.pkl")

        params = SMPLParams(
            betas=jnp.zeros((8, 10)),
            body_pose=jnp.zeros((8, 69)),
            global_orient=jnp.zeros((8, 3)),
            transl=jnp.zeros((8, 3)),
        )

        forward = jax.jit(model)
        output  = forward(params)
        # output.vertices.shape == (8, 6890, 3)
        # output.joints.shape   == (8, 24, 3)
    """

    _output_cls = SMPLOutput

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str, num_betas: int = 10) -> SMPLModel:
        """Load a SMPL model from a .pkl file."""
        data = load_model_data(path)
        return cls(
            v_template=data["v_template"],
            shapedirs=data["shapedirs"],
            posedirs=data["posedirs"],
            J_regressor=data["J_regressor"],
            parents=data["parents"],
            weights=data["weights"],
            faces=data["faces"],
            num_betas=num_betas,
        )

    # ------------------------------------------------------------------
    # hook: rotation assembly
    # ------------------------------------------------------------------

    def _build_rotmats(self, params: SMPLParams, B: int) -> jnp.ndarray:
        """Assemble (B, J, 3, 3) from global_orient + body_pose."""
        n_body = self.num_joints - 1                                # dynamic, typically 23
        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        R_body = jax.vmap(jax.vmap(axis_angle_to_rotmat))(
            params.body_pose.reshape(B, n_body, 3)
        )
        return jnp.concatenate([R_root, R_body], axis=1)           # (B, J, 3, 3)
