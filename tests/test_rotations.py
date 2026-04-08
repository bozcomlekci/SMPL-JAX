"""
Tests for smpl_jax.rotations.

Covers:
  - axis_angle_to_rotmat: zero-angle, orthogonality, determinant
  - rotmat_to_axis_angle: inverse of above
  - rotmat_to_6d / rotation_6d_to_rotmat: roundtrip, orthogonality
  - jax.jit and jax.vmap compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smpl_jax.rotations import (
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
    rotation_6d_to_rotmat,
    rotmat_to_6d,
)


class TestAxisAngleToRotmat:
    def test_zero_angle_is_identity(self):
        R = axis_angle_to_rotmat(jnp.zeros(3))
        np.testing.assert_allclose(R, jnp.eye(3), atol=1e-6)

    def test_output_shape_single(self):
        R = axis_angle_to_rotmat(jnp.ones(3))
        assert R.shape == (3, 3)

    def test_output_shape_batched(self):
        aa = jnp.ones((5, 3))
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        assert R.shape == (5, 3, 3)

    def test_orthogonality(self):
        key = jax.random.PRNGKey(0)
        aa = jax.random.normal(key, (8, 3))
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        B = R.shape[0]
        RRt = jnp.einsum("bij,bkj->bik", R, R)
        np.testing.assert_allclose(RRt, np.tile(np.eye(3), (B, 1, 1)), atol=1e-5)

    def test_determinant_one(self):
        key = jax.random.PRNGKey(1)
        aa = jax.random.normal(key, (8, 3))
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        dets = jnp.linalg.det(R)
        np.testing.assert_allclose(dets, jnp.ones(8), atol=1e-5)

    def test_jit_compatible(self):
        f = jax.jit(axis_angle_to_rotmat)
        R = f(jnp.array([0.1, 0.2, 0.3]))
        assert R.shape == (3, 3)

    def test_90deg_around_z(self):
        aa = jnp.array([0.0, 0.0, jnp.pi / 2])
        R = axis_angle_to_rotmat(aa)
        expected = jnp.array([[0.0, -1.0, 0.0],
                               [1.0,  0.0, 0.0],
                               [0.0,  0.0, 1.0]])
        np.testing.assert_allclose(R, expected, atol=1e-6)


class TestRotmatToAxisAngle:
    def test_identity_to_zero(self):
        aa = rotmat_to_axis_angle(jnp.eye(3))
        np.testing.assert_allclose(aa, jnp.zeros(3), atol=1e-6)

    def test_roundtrip(self):
        key = jax.random.PRNGKey(2)
        aa = jax.random.normal(key, (6, 3)) * 0.5   # small angles
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        aa_rec = jax.vmap(rotmat_to_axis_angle)(R)
        np.testing.assert_allclose(aa, aa_rec, atol=1e-5)


class TestRotation6D:
    def test_6d_shape(self):
        key = jax.random.PRNGKey(3)
        aa = jax.random.normal(key, (4, 3))
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        r6d = jax.vmap(rotmat_to_6d)(R)
        assert r6d.shape == (4, 6)

    def test_roundtrip(self):
        key = jax.random.PRNGKey(4)
        aa = jax.random.normal(key, (6, 3))
        R = jax.vmap(axis_angle_to_rotmat)(aa)
        R_rec = jax.vmap(rotation_6d_to_rotmat)(jax.vmap(rotmat_to_6d)(R))
        np.testing.assert_allclose(R, R_rec, atol=1e-5)

    def test_random_6d_gives_valid_rotmat(self):
        key = jax.random.PRNGKey(5)
        r6d = jax.random.normal(key, (8, 6))
        R = jax.vmap(rotation_6d_to_rotmat)(r6d)
        # det ≈ +1
        np.testing.assert_allclose(jnp.linalg.det(R), np.ones(8), atol=1e-5)
        # R @ R.T ≈ I
        RRt = jnp.einsum("bij,bkj->bik", R, R)
        np.testing.assert_allclose(RRt, np.tile(np.eye(3), (8, 1, 1)), atol=1e-5)

    def test_grad_through_6d(self):
        def f(r6d):
            return jnp.sum(rotation_6d_to_rotmat(r6d) ** 2)

        r6d = jnp.ones(6)
        g = jax.grad(f)(r6d)
        assert g.shape == (6,)
        assert jnp.all(jnp.isfinite(g))
