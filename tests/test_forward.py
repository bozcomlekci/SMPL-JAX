"""
Tests for SMPL and SMPL-X forward passes using synthetic model data.

All tests use the fixtures defined in conftest.py and do not require
downloading actual model weights.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smpl_jax.types import SMPLParams, SMPLXParams


# ---------------------------------------------------------------------------
# SMPL tests
# ---------------------------------------------------------------------------

class TestSMPLForward:
    def test_output_shapes(self, smpl_model):
        B = 4
        params = SMPLParams(
            betas=jnp.zeros((B, 10)),
            body_pose=jnp.zeros((B, 69)),
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
        )
        out = smpl_model(params)
        assert out.vertices.shape == (B, smpl_model.v_template.shape[0], 3)
        assert out.joints.shape == (B, smpl_model.num_joints, 3)

    def test_jit_forward(self, smpl_model):
        B = 2
        params = SMPLParams(
            betas=jnp.zeros((B, 10)),
            body_pose=jnp.zeros((B, 69)),
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
        )
        forward = jax.jit(smpl_model)
        out = forward(params)
        assert out.vertices.shape[0] == B

    def test_translation_applied_correctly(self, smpl_model):
        B = 2
        t = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        base = SMPLParams(
            betas=jnp.zeros((B, 10)),
            body_pose=jnp.zeros((B, 69)),
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
        )
        shifted = base._replace(transl=t)

        out0 = smpl_model(base)
        out1 = smpl_model(shifted)

        diff = out1.vertices - out0.vertices            # (B, V, 3)
        expected = jnp.broadcast_to(t[:, None, :], diff.shape)
        np.testing.assert_allclose(diff, expected, atol=1e-5)

    def test_gradient_flows_through_betas(self, smpl_model):
        def loss(betas):
            params = SMPLParams(
                betas=betas,
                body_pose=jnp.zeros((1, 69)),
                global_orient=jnp.zeros((1, 3)),
                transl=jnp.zeros((1, 3)),
            )
            return jnp.sum(smpl_model(params).vertices ** 2)

        betas = jnp.zeros((1, 10))
        grad = jax.grad(loss)(betas)
        assert grad.shape == (1, 10)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_flows_through_pose(self, smpl_model):
        def loss(body_pose):
            params = SMPLParams(
                betas=jnp.zeros((1, 10)),
                body_pose=body_pose,
                global_orient=jnp.zeros((1, 3)),
                transl=jnp.zeros((1, 3)),
            )
            return jnp.sum(smpl_model(params).vertices ** 2)

        pose = jnp.zeros((1, 69))
        grad = jax.grad(loss)(pose)
        assert grad.shape == (1, 69)
        assert jnp.all(jnp.isfinite(grad))

    def test_output_is_finite(self, smpl_model):
        key = jax.random.PRNGKey(42)
        B = 3
        betas = jax.random.normal(key, (B, 10)) * 0.5
        pose = jax.random.normal(jax.random.PRNGKey(1), (B, 69)) * 0.3
        params = SMPLParams(
            betas=betas,
            body_pose=pose,
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
        )
        out = smpl_model(params)
        assert jnp.all(jnp.isfinite(out.vertices))
        assert jnp.all(jnp.isfinite(out.joints))


# ---------------------------------------------------------------------------
# SMPL-X tests
# ---------------------------------------------------------------------------

class TestSMPLXForward:
    def _params(self, B: int) -> SMPLXParams:
        return SMPLXParams(
            betas=jnp.zeros((B, 10)),
            body_pose=jnp.zeros((B, 63)),
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
            expression=jnp.zeros((B, 10)),
            jaw_pose=jnp.zeros((B, 3)),
            leye_pose=jnp.zeros((B, 3)),
            reye_pose=jnp.zeros((B, 3)),
            left_hand_pose=jnp.zeros((B, 45)),
            right_hand_pose=jnp.zeros((B, 45)),
        )

    def test_output_shapes(self, smplx_model):
        B = 4
        out = smplx_model(self._params(B))
        assert out.vertices.shape == (B, smplx_model.v_template.shape[0], 3)
        assert out.joints.shape == (B, smplx_model.num_joints, 3)

    def test_jit_forward(self, smplx_model):
        B = 2
        forward = jax.jit(smplx_model)
        out = forward(self._params(B))
        assert out.vertices.shape[0] == B

    def test_translation_applied_correctly(self, smplx_model):
        B = 2
        t = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        p0 = self._params(B)
        p1 = p0._replace(transl=t)

        diff = smplx_model(p1).vertices - smplx_model(p0).vertices
        expected = jnp.broadcast_to(t[:, None, :], diff.shape)
        np.testing.assert_allclose(diff, expected, atol=1e-5)

    def test_gradient_flows_through_expression(self, smplx_model):
        def loss(expr):
            p = SMPLXParams(
                betas=jnp.zeros((1, 10)),
                body_pose=jnp.zeros((1, 63)),
                global_orient=jnp.zeros((1, 3)),
                transl=jnp.zeros((1, 3)),
                expression=expr,
                jaw_pose=jnp.zeros((1, 3)),
                leye_pose=jnp.zeros((1, 3)),
                reye_pose=jnp.zeros((1, 3)),
                left_hand_pose=jnp.zeros((1, 45)),
                right_hand_pose=jnp.zeros((1, 45)),
            )
            return jnp.sum(smplx_model(p).vertices ** 2)

        g = jax.grad(loss)(jnp.zeros((1, 10)))
        assert g.shape == (1, 10)
        assert jnp.all(jnp.isfinite(g))

    def test_output_is_finite_with_random_pose(self, smplx_model):
        B = 2
        key = jax.random.PRNGKey(0)
        p = SMPLXParams(
            betas=jax.random.normal(key, (B, 10)) * 0.3,
            body_pose=jax.random.normal(jax.random.PRNGKey(1), (B, 63)) * 0.3,
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
            expression=jax.random.normal(jax.random.PRNGKey(2), (B, 10)) * 0.3,
            jaw_pose=jnp.zeros((B, 3)),
            leye_pose=jnp.zeros((B, 3)),
            reye_pose=jnp.zeros((B, 3)),
            left_hand_pose=jax.random.normal(jax.random.PRNGKey(3), (B, 45)) * 0.3,
            right_hand_pose=jax.random.normal(jax.random.PRNGKey(4), (B, 45)) * 0.3,
        )
        out = smplx_model(p)
        assert jnp.all(jnp.isfinite(out.vertices))
        assert jnp.all(jnp.isfinite(out.joints))
