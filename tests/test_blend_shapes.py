"""Tests for smpl_jax.blend_shapes."""

import jax
import jax.numpy as jnp
import numpy as np

from smpl_jax.blend_shapes import (
    shape_blend_shapes,
    expression_blend_shapes,
    pose_blend_shapes,
)
from smpl_jax.rotations import axis_angle_to_rotmat


class TestShapeBlendShapes:
    def test_zero_betas_returns_template(self):
        V, K, B = 20, 5, 3
        v_template = jnp.ones((V, 3))
        shapedirs = jnp.ones((V, 3, K))
        betas = jnp.zeros((B, K))

        out = shape_blend_shapes(v_template, shapedirs, betas)
        np.testing.assert_allclose(out, jnp.ones((B, V, 3)), atol=1e-6)

    def test_shape(self):
        V, K, B = 20, 10, 4
        key = jax.random.PRNGKey(0)
        v_template = jax.random.normal(key, (V, 3))
        shapedirs = jax.random.normal(jax.random.PRNGKey(1), (V, 3, K))
        betas = jax.random.normal(jax.random.PRNGKey(2), (B, K))

        out = shape_blend_shapes(v_template, shapedirs, betas)
        assert out.shape == (B, V, 3)

    def test_linearity_in_betas(self):
        V, K = 10, 3
        v_template = jnp.zeros((V, 3))
        shapedirs = jax.random.normal(jax.random.PRNGKey(0), (V, 3, K))
        beta_a = jax.random.normal(jax.random.PRNGKey(1), (1, K))
        beta_b = jax.random.normal(jax.random.PRNGKey(2), (1, K))

        out_a = shape_blend_shapes(v_template, shapedirs, beta_a)
        out_b = shape_blend_shapes(v_template, shapedirs, beta_b)
        out_sum = shape_blend_shapes(v_template, shapedirs, beta_a + beta_b)
        np.testing.assert_allclose(out_a + out_b, out_sum, atol=1e-5)


class TestExpressionBlendShapes:
    def test_zero_expression_unchanged(self):
        V, E, B = 10, 5, 2
        v_shaped = jnp.ones((B, V, 3))
        exprdirs = jnp.ones((V, 3, E))
        expr = jnp.zeros((B, E))

        out = expression_blend_shapes(v_shaped, exprdirs, expr)
        np.testing.assert_allclose(out, v_shaped, atol=1e-6)

    def test_shape(self):
        V, E, B = 15, 10, 3
        v_shaped = jnp.zeros((B, V, 3))
        exprdirs = jax.random.normal(jax.random.PRNGKey(0), (V, 3, E))
        expr = jax.random.normal(jax.random.PRNGKey(1), (B, E))

        out = expression_blend_shapes(v_shaped, exprdirs, expr)
        assert out.shape == (B, V, 3)


class TestPoseBlendShapes:
    def test_identity_rotations_give_zero_correctives(self):
        B, J, V = 2, 10, 50
        P = (J - 1) * 9
        rotmats = jnp.broadcast_to(jnp.eye(3)[None, None], (B, J, 3, 3))
        posedirs = jnp.ones((V * 3, P))

        corr = pose_blend_shapes(rotmats, posedirs)
        np.testing.assert_allclose(corr, jnp.zeros((B, V, 3)), atol=1e-6)

    def test_shape(self):
        B, J, V = 3, 8, 30
        P = (J - 1) * 9
        aa = jax.random.normal(jax.random.PRNGKey(0), (B, J, 3))
        rotmats = jax.vmap(jax.vmap(axis_angle_to_rotmat))(aa)
        posedirs = jax.random.normal(jax.random.PRNGKey(1), (V * 3, P))

        corr = pose_blend_shapes(rotmats, posedirs)
        assert corr.shape == (B, V, 3)

    def test_posedirs_trimming(self):
        """posedirs with fewer directions than (J-1)*9 should still work."""
        B, J, V = 2, 10, 20
        P_full = (J - 1) * 9
        P_trim = P_full // 2   # use only half the directions
        aa = jax.random.normal(jax.random.PRNGKey(0), (B, J, 3))
        rotmats = jax.vmap(jax.vmap(axis_angle_to_rotmat))(aa)
        posedirs = jax.random.normal(jax.random.PRNGKey(1), (V * 3, P_trim))

        corr = pose_blend_shapes(rotmats, posedirs)
        assert corr.shape == (B, V, 3)
