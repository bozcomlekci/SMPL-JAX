"""
Abstract base class shared by SMPLModel and SMPLXModel.

Both models execute the same seven-step LBS pipeline:

    1. Vertex blend shapes  (shape + optional expression)
    2. Bind-pose joint regression
    3. Rotation matrix assembly
    4. Forward kinematics
    5. Pose corrective blend shapes
    6. LBS transform computation + linear blend skinning
    7. Global translation

The only differences between SMPL and SMPL-X are:
  • step 1: SMPL-X adds an expression blend-shape pass
  • step 3: SMPL-X assembles joints for a 55-joint skeleton (body + face + hands)
  • output container type (SMPLOutput vs SMPLXOutput)

Subclasses override ``_vertex_blend_shapes`` and ``_build_rotmats``; all other
logic lives here and is compiled once by ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .blend_shapes import shape_blend_shapes, pose_blend_shapes
from .kinematics import fk_forward_batched
from .lbs import lbs_transforms, lbs


class _SMPLBase:
    """Common forward-pass backbone for SMPL and SMPL-X.

    Subclasses must set the class attribute ``_output_cls`` and implement
    ``_vertex_blend_shapes`` and ``_build_rotmats``.
    """

    # Set to SMPLOutput / SMPLXOutput in concrete subclasses
    _output_cls = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        v_template: np.ndarray,       # (V, 3)
        shapedirs: np.ndarray,        # (V, 3, num_betas)
        posedirs: np.ndarray,         # (V*3, P)
        J_regressor: np.ndarray,      # (≥J, V)  — sliced to first J rows
        parents: np.ndarray,          # (J,)  int, parents[0] == -1
        weights: np.ndarray,          # (V, J)
        faces: np.ndarray,            # (F, 3)  int
        num_betas: int = 10,
    ) -> None:
        self.num_joints = int(weights.shape[1])
        self.num_betas = num_betas

        # Kept as numpy for Python-level mesh queries (non-differentiable)
        self._parents_np = np.asarray(parents)
        self._faces_np = np.asarray(faces)

        # Device arrays (float32 throughout to avoid silent float64 promotion)
        self.v_template  = jnp.array(v_template,  dtype=jnp.float32)
        self.shapedirs   = jnp.array(shapedirs[..., :num_betas], dtype=jnp.float32)
        self.posedirs    = jnp.array(posedirs,     dtype=jnp.float32)
        # Slice to exactly num_joints rows; handles models where J_regressor
        # contains extra landmark-joint rows (e.g. SMPL-X has 127 rows).
        self.J_regressor = jnp.array(J_regressor[:self.num_joints], dtype=jnp.float32)
        self.parents     = jnp.array(parents)
        self.weights     = jnp.array(weights,  dtype=jnp.float32)
        self.faces       = jnp.array(faces)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------

    def __call__(self, params):
        return self.forward(params)

    def forward(self, params):
        """Run the full LBS forward pass.

        Accepts both batched params ``(B, ...)`` and unbatched params ``(...)``.
        Unbatched inputs are promoted to ``B=1`` internally and the leading
        dimension is removed from the output, enabling ``jax.vmap(model)``.

        Args:
            params: model-specific ``*Params`` NamedTuple.

        Returns:
            model-specific ``*Output`` NamedTuple with shapes
            ``(B, V, 3)`` / ``(B, J, 3)`` for batched input or
            ``(V, 3)`` / ``(J, 3)`` for unbatched input.
        """
        unbatched = params.betas.ndim == 1
        if unbatched:
            params = jax.tree_util.tree_map(lambda x: x[None], params)

        B = params.betas.shape[0]

        # ── 1. Vertex blend shapes ──────────────────────────────────────
        v_shaped = self._vertex_blend_shapes(params)                # (B, V, 3)

        # ── 2. Bind-pose joint regression ──────────────────────────────
        joints = jnp.einsum("jv,bvd->bjd", self.J_regressor, v_shaped)  # (B, J, 3)

        # ── 3. Rotation matrices ────────────────────────────────────────
        rotmats = self._build_rotmats(params, B)                    # (B, J, 3, 3)

        # ── 4. Forward kinematics ───────────────────────────────────────
        G = fk_forward_batched(rotmats, joints, self.parents)       # (B, J, 4, 4)

        # ── 5. Pose corrective blend shapes ─────────────────────────────
        pose_corr = pose_blend_shapes(rotmats, self.posedirs)       # (B, V, 3)

        # ── 6. Linear blend skinning ────────────────────────────────────
        M        = lbs_transforms(G, joints)                        # (B, J, 3, 4)
        vertices = lbs(v_shaped, pose_corr, M, self.weights)        # (B, V, 3)

        # ── 7. Global translation ───────────────────────────────────────
        vertices    = vertices + params.transl[:, None, :]
        posed_joints = G[..., :3, 3] + params.transl[:, None, :]

        out = self._output_cls(vertices=vertices, joints=posed_joints)
        if unbatched:
            out = self._output_cls(vertices=out.vertices[0], joints=out.joints[0])
        return out

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    def _vertex_blend_shapes(self, params) -> jnp.ndarray:
        """Apply shape (and optionally expression) blend shapes.

        Base implementation applies shape blend shapes only.
        SMPL-X overrides this to add expression blend shapes.
        """
        return shape_blend_shapes(self.v_template, self.shapedirs, params.betas)

    def _build_rotmats(self, params, B: int) -> jnp.ndarray:
        """Assemble the ``(B, J, 3, 3)`` rotation matrix tensor.

        Must be implemented by each subclass because SMPL and SMPL-X have
        different joint counts and parameter names.
        """
        raise NotImplementedError
