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
import numpy as np

from .types import SMPLParams, SMPLOutput
from .model_io import load_model_data
from .rotations import axis_angle_to_rotmat
from .blend_shapes import shape_blend_shapes, pose_blend_shapes
from .kinematics import fk_forward_batched
from .lbs import lbs_transforms, lbs


_NUM_BODY_JOINTS = 23   # joints 1–23; joint 0 is the root (global_orient)


class SMPLModel:
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

    def __init__(
        self,
        v_template: np.ndarray,    # (V, 3)
        shapedirs: np.ndarray,     # (V, 3, num_betas)
        posedirs: np.ndarray,      # (V*3, P)
        J_regressor: np.ndarray,   # (J, V)
        parents: np.ndarray,       # (J,)  int
        weights: np.ndarray,       # (V, J)
        faces: np.ndarray,         # (F, 3)  int
        num_betas: int = 10,
    ) -> None:
        self.num_joints = int(weights.shape[1])
        self.num_betas = num_betas

        self._parents_np = parents
        self._faces_np = faces

        self.v_template = jnp.array(v_template)
        self.shapedirs = jnp.array(shapedirs[..., :num_betas])
        self.posedirs = jnp.array(posedirs)
        self.J_regressor = jnp.array(J_regressor)
        self.parents = jnp.array(parents)
        self.weights = jnp.array(weights)
        self.faces = jnp.array(faces)

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
    # forward pass
    # ------------------------------------------------------------------

    def __call__(self, params: SMPLParams) -> SMPLOutput:
        return self.forward(params)

    def forward(self, params: SMPLParams) -> SMPLOutput:
        """SMPL forward pass.

        Args:
            params: SMPLParams with (B, ...) batched arrays.

        Returns:
            SMPLOutput with
                vertices (B, 6890, 3) and
                joints   (B, J, 3).
        """
        B = params.betas.shape[0]

        # 1. Shape blend shapes  →  (B, V, 3)
        v_shaped = shape_blend_shapes(
            self.v_template, self.shapedirs, params.betas
        )

        # 2. Regress bind-pose joint positions  →  (B, J, 3)
        joints = jnp.einsum("jv,bvd->bjd", self.J_regressor, v_shaped)

        # 3. Rotation matrices for all joints  →  (B, J, 3, 3)
        n_body = self.num_joints - 1                                # e.g. 23 for full SMPL
        R_root = jax.vmap(axis_angle_to_rotmat)(params.global_orient)[:, None]
        R_body = jax.vmap(jax.vmap(axis_angle_to_rotmat))(
            params.body_pose.reshape(B, n_body, 3)
        )
        rotmats = jnp.concatenate([R_root, R_body], axis=1)        # (B, J, 3, 3)

        # 4. Forward kinematics  →  (B, J, 4, 4)
        G = fk_forward_batched(rotmats, joints, self.parents)

        # 5. Pose corrective blend shapes  →  (B, V, 3)
        pose_corr = pose_blend_shapes(rotmats, self.posedirs)

        # 6. Relative LBS transforms  →  (B, J, 4, 4)
        M = lbs_transforms(G, joints)

        # 7. Linear blend skinning  →  (B, V, 3)
        vertices = lbs(v_shaped, pose_corr, M, self.weights)

        # 8. Global translation
        vertices = vertices + params.transl[:, None, :]
        posed_joints = G[..., :3, 3] + params.transl[:, None, :]

        return SMPLOutput(vertices=vertices, joints=posed_joints)
