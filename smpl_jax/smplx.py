"""
SMPL-X: Expressive Human Body Model in JAX.

Reference: Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and
Body from a Single Image", CVPR 2019.

SMPL-X joint convention (55 total, indexed 0–54):
    0        pelvis            ← global_orient
    1–21     body joints       ← body_pose (63D = 21×3)
    22       jaw               ← identity (not exposed in params)
    23       left eye          ← identity
    24       right eye         ← identity
    25–39    left hand         ← left_hand_pose (45D = 15×3)
    40–54    right hand        ← right_hand_pose (45D = 15×3)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .types import SMPLXParams, SMPLXOutput
from .model_io import load_model_data
from .rotations import axis_angle_to_rotmat
from .blend_shapes import expression_blend_shapes
from ._base import _SMPLBase


_NUM_BODY_JOINTS = 21
_NUM_FACE_JOINTS = 3    # jaw, left eye, right eye (fixed at identity)
_NUM_HAND_JOINTS = 15
_SMPLX_JOINTS = 1 + _NUM_BODY_JOINTS + _NUM_FACE_JOINTS + 2 * _NUM_HAND_JOINTS  # 55


class SMPLXModel(_SMPLBase):
    """JAX port of SMPL-X.

    Model arrays are stored as JAX device arrays so they can be captured in
    jit closures and treated as compile-time constants.

    Usage::

        model = SMPLXModel.load("data/smplx/SMPLX_NEUTRAL.pkl")

        params = SMPLXParams(
            betas=jnp.zeros((8, 10)),
            body_pose=jnp.zeros((8, 63)),
            global_orient=jnp.zeros((8, 3)),
            transl=jnp.zeros((8, 3)),
            expression=jnp.zeros((8, 10)),
            left_hand_pose=jnp.zeros((8, 45)),
            right_hand_pose=jnp.zeros((8, 45)),
        )

        forward = jax.jit(model)
        output  = forward(params)
        # output.vertices.shape == (8, 10475, 3)
        # output.joints.shape   == (8, 55, 3)
    """

    _output_cls = SMPLXOutput

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        v_template: np.ndarray,
        shapedirs: np.ndarray,
        exprdirs: np.ndarray,          # (V, 3, num_expr)
        posedirs: np.ndarray,
        J_regressor: np.ndarray,
        parents: np.ndarray,
        weights: np.ndarray,
        faces: np.ndarray,
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
    ) -> None:
        super().__init__(
            v_template=v_template,
            shapedirs=shapedirs,
            posedirs=posedirs,
            J_regressor=J_regressor,
            parents=parents,
            weights=weights,
            faces=faces,
            num_betas=num_betas,
        )
        self.num_expression_coeffs = num_expression_coeffs
        self.exprdirs = jnp.array(
            exprdirs[..., :num_expression_coeffs], dtype=jnp.float32
        )

    @classmethod
    def load(
        cls,
        path: str,
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
    ) -> SMPLXModel:
        """Load an SMPL-X model from a .pkl file.

        Args:
            path: path to SMPLX_NEUTRAL.pkl (or MALE / FEMALE).
            num_betas: number of shape components to use (max in model).
            num_expression_coeffs: number of expression components to use.
        """
        data = load_model_data(path)
        if data["exprdirs"] is None:
            raise ValueError(
                f"{path} does not contain expression blend shapes. "
                "Use SMPLModel for plain SMPL."
            )
        return cls(
            v_template=data["v_template"],
            shapedirs=data["shapedirs"],
            exprdirs=data["exprdirs"],
            posedirs=data["posedirs"],
            J_regressor=data["J_regressor"],
            parents=data["parents"],
            weights=data["weights"],
            faces=data["faces"],
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
        )

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------

    def _vertex_blend_shapes(self, params: SMPLXParams) -> jnp.ndarray:
        """Shape + expression blend shapes → (B, V, 3)."""
        v_shaped = super()._vertex_blend_shapes(params)
        return expression_blend_shapes(v_shaped, self.exprdirs, params.expression)

    def _build_rotmats(self, params: SMPLXParams, B: int) -> jnp.ndarray:
        """Assemble (B, J, 3, 3) for the 55-joint SMPL-X skeleton."""

        def aa_block(aa_flat: jnp.ndarray, n: int) -> jnp.ndarray:
            """(B, n*3) → (B, n, 3, 3)"""
            return jax.vmap(jax.vmap(axis_angle_to_rotmat))(
                aa_flat.reshape(B, n, 3)
            )

        R_root  = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]  # (B,1,3,3)
        R_body  = aa_block(params.body_pose,       _NUM_BODY_JOINTS)              # (B,21,3,3)
        R_lhand = aa_block(params.left_hand_pose,  _NUM_HAND_JOINTS)              # (B,15,3,3)
        R_rhand = aa_block(params.right_hand_pose, _NUM_HAND_JOINTS)              # (B,15,3,3)

        # Face joints (jaw, left-eye, right-eye) → identity rotation.
        # These are not exposed as input parameters in the base SMPL-X model.
        I_face = jnp.broadcast_to(
            jnp.eye(3, dtype=jnp.float32)[None, None],
            (B, _NUM_FACE_JOINTS, 3, 3),
        )

        # Joint order: root(0), body(1-21), face(22-24), lhand(25-39), rhand(40-54)
        R_all = jnp.concatenate(
            [R_root, R_body, I_face, R_lhand, R_rhand], axis=1
        )                                                           # (B, ≤55, 3, 3)

        # Trim or pad to match the model's actual joint count
        J = self.num_joints
        if R_all.shape[1] < J:
            pad = jnp.broadcast_to(
                jnp.eye(3, dtype=jnp.float32)[None, None],
                (B, J - R_all.shape[1], 3, 3),
            )
            R_all = jnp.concatenate([R_all, pad], axis=1)
        elif R_all.shape[1] > J:
            R_all = R_all[:, :J]

        return R_all                                               # (B, J, 3, 3)
