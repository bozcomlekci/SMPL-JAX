"""
SMPL-X: Expressive Human Body Model in JAX.

Reference: Pavlakos et al., "Expressive Body Capture: 3D Hands, Face, and
Body from a Single Image", CVPR 2019.

SMPL-X joint convention (55 total, indexed 0–54):
    0        pelvis            ← global_orient
    1–21     body joints       ← body_pose (63D = 21×3)
    22       jaw               ← zero (not exposed in params)
    23       left eye          ← zero
    24       right eye         ← zero
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
from .blend_shapes import shape_blend_shapes, expression_blend_shapes, pose_blend_shapes
from .kinematics import fk_forward_batched
from .lbs import lbs_transforms, lbs


_NUM_BODY_JOINTS = 21
_NUM_FACE_JOINTS = 3   # jaw, left eye, right eye
_NUM_HAND_JOINTS = 15
_SMPLX_JOINTS = 1 + _NUM_BODY_JOINTS + _NUM_FACE_JOINTS + 2 * _NUM_HAND_JOINTS  # 55


class SMPLXModel:
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

    def __init__(
        self,
        v_template: np.ndarray,       # (V, 3)
        shapedirs: np.ndarray,        # (V, 3, num_betas)
        exprdirs: np.ndarray,         # (V, 3, num_expr)
        posedirs: np.ndarray,         # (V*3, P)
        J_regressor: np.ndarray,      # (J, V)  — J ≥ num_joints
        parents: np.ndarray,          # (num_joints,)  int
        weights: np.ndarray,          # (V, num_joints)
        faces: np.ndarray,            # (F, 3)  int
        num_betas: int = 10,
        num_expression_coeffs: int = 10,
    ) -> None:
        self.num_joints = int(weights.shape[1])
        self.num_betas = num_betas
        self.num_expression_coeffs = num_expression_coeffs

        # Store integer arrays as numpy for Python-level queries
        self._parents_np = parents
        self._faces_np = faces

        # Device arrays — captured as constants inside jax.jit
        self.v_template = jnp.array(v_template)
        self.shapedirs = jnp.array(shapedirs[..., :num_betas])
        self.exprdirs = jnp.array(exprdirs[..., :num_expression_coeffs])
        self.posedirs = jnp.array(posedirs)
        # Use the first num_joints rows of J_regressor for FK / LBS joints
        self.J_regressor = jnp.array(J_regressor[:self.num_joints])
        self.parents = jnp.array(parents)
        self.weights = jnp.array(weights)
        self.faces = jnp.array(faces)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

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
    # forward pass
    # ------------------------------------------------------------------

    def __call__(self, params: SMPLXParams) -> SMPLXOutput:
        return self.forward(params)

    def forward(self, params: SMPLXParams) -> SMPLXOutput:
        """SMPL-X forward pass.

        Args:
            params: SMPLXParams with (B, ...) batched arrays.

        Returns:
            SMPLXOutput with
                vertices (B, V, 3) and
                joints   (B, num_joints, 3).
        """
        B = params.betas.shape[0]

        # 1. Shape blend shapes  →  (B, V, 3)
        v_shaped = shape_blend_shapes(
            self.v_template, self.shapedirs, params.betas
        )

        # 2. Expression blend shapes  →  (B, V, 3)
        v_shaped = expression_blend_shapes(
            v_shaped, self.exprdirs, params.expression
        )

        # 3. Regress bind-pose joint positions  →  (B, J, 3)
        joints = jnp.einsum("jv,bvd->bjd", self.J_regressor, v_shaped)

        # 4. Assemble rotation matrices for all joints  →  (B, J, 3, 3)
        rotmats = self._assemble_rotmats(
            params.global_orient,
            params.body_pose,
            params.left_hand_pose,
            params.right_hand_pose,
            B,
        )

        # 5. Forward kinematics  →  (B, J, 4, 4)
        G = fk_forward_batched(rotmats, joints, self.parents)

        # 6. Pose corrective blend shapes  →  (B, V, 3)
        pose_corr = pose_blend_shapes(rotmats, self.posedirs)

        # 7. Relative LBS transforms  →  (B, J, 4, 4)
        M = lbs_transforms(G, joints)

        # 8. Linear blend skinning  →  (B, V, 3)
        vertices = lbs(v_shaped, pose_corr, M, self.weights)

        # 9. Global translation
        vertices = vertices + params.transl[:, None, :]
        posed_joints = G[..., :3, 3] + params.transl[:, None, :]

        return SMPLXOutput(vertices=vertices, joints=posed_joints)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _assemble_rotmats(
        self,
        global_orient: jnp.ndarray,  # (B, 3)
        body_pose: jnp.ndarray,      # (B, 63)
        lhand_pose: jnp.ndarray,     # (B, 45)
        rhand_pose: jnp.ndarray,     # (B, 45)
        B: int,
    ) -> jnp.ndarray:
        """Assemble the full (B, J, 3, 3) rotation matrix array."""

        def aa_block(aa_flat: jnp.ndarray, n: int) -> jnp.ndarray:
            """(B, n*3) → (B, n, 3, 3)"""
            return jax.vmap(jax.vmap(axis_angle_to_rotmat))(
                aa_flat.reshape(B, n, 3)
            )

        R_root  = jax.vmap(axis_angle_to_rotmat)(global_orient)[:, None]   # (B,1,3,3)
        R_body  = aa_block(body_pose, _NUM_BODY_JOINTS)                     # (B,21,3,3)
        R_lhand = aa_block(lhand_pose, _NUM_HAND_JOINTS)                    # (B,15,3,3)
        R_rhand = aa_block(rhand_pose, _NUM_HAND_JOINTS)                    # (B,15,3,3)

        # Face joints (jaw, left-eye, right-eye) → identity rotation
        I_face = jnp.broadcast_to(
            jnp.eye(3)[None, None],
            (B, _NUM_FACE_JOINTS, 3, 3),
        )

        # Concatenate in joint order: 0(root), 1-21(body), 22-24(face),
        # 25-39(left hand), 40-54(right hand)
        R_all = jnp.concatenate(
            [R_root, R_body, I_face, R_lhand, R_rhand], axis=1
        )                                                           # (B, ≤55, 3, 3)

        J = self.num_joints
        if R_all.shape[1] < J:
            # Pad extra joints with identity
            pad = jnp.broadcast_to(
                jnp.eye(3)[None, None],
                (B, J - R_all.shape[1], 3, 3),
            )
            R_all = jnp.concatenate([R_all, pad], axis=1)
        elif R_all.shape[1] > J:
            R_all = R_all[:, :J]

        return R_all                                               # (B, J, 3, 3)
