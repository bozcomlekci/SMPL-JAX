"""Tests for smpl_jax.kinematics (FK via lax.scan)."""

import jax
import jax.numpy as jnp
import numpy as np

from smpl_jax.kinematics import fk_forward, fk_forward_batched


class TestFKForward:
    def test_identity_pose_translations_equal_joints(self):
        """With identity rotations, G[i][:3,3] should equal the bind-pose joint."""
        J = 5
        parents = jnp.array([-1, 0, 1, 2, 3])
        joints = jnp.stack([jnp.array([float(i), 0.0, 0.0]) for i in range(J)])
        rotmats = jnp.broadcast_to(jnp.eye(3), (J, 3, 3))

        G = fk_forward(rotmats, joints, parents)

        assert G.shape == (J, 4, 4)
        np.testing.assert_allclose(G[:, :3, 3], joints, atol=1e-5)

    def test_root_rotation_propagates_to_children(self):
        """90° rotation at root should swing all descendants to the y-axis."""
        J = 3
        parents = jnp.array([-1, 0, 1])
        joints = jnp.array([[0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [2.0, 0.0, 0.0]])

        # 90° around z: x → y
        R_z90 = jnp.array([[0.0, -1.0, 0.0],
                            [1.0,  0.0, 0.0],
                            [0.0,  0.0, 1.0]])
        rotmats = jnp.stack([R_z90, jnp.eye(3), jnp.eye(3)])

        G = fk_forward(rotmats, joints, parents)

        np.testing.assert_allclose(G[0, :3, 3], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(G[1, :3, 3], [0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(G[2, :3, 3], [0.0, 2.0, 0.0], atol=1e-5)

    def test_output_is_rigid_body_transforms(self):
        """Each G[i] should be a valid SE(3) matrix (upper-left 3×3 is orthogonal)."""
        J = 6
        parents = jnp.array([-1, 0, 0, 1, 1, 2])
        joints = jax.random.normal(jax.random.PRNGKey(0), (J, 3))
        aa = jax.random.normal(jax.random.PRNGKey(1), (J, 3)) * 0.3
        from smpl_jax.rotations import axis_angle_to_rotmat
        rotmats = jax.vmap(axis_angle_to_rotmat)(aa)

        G = fk_forward(rotmats, joints, parents)

        R_global = G[:, :3, :3]
        RRt = jnp.einsum("jab,jcb->jac", R_global, R_global)
        np.testing.assert_allclose(RRt, np.tile(np.eye(3), (J, 1, 1)), atol=1e-5)
        np.testing.assert_allclose(G[:, 3, :], np.tile([0.0, 0.0, 0.0, 1.0], (J, 1)), atol=1e-5)

    def test_jit_compatible(self):
        J = 4
        parents = jnp.array([-1, 0, 1, 2])
        joints = jnp.eye(J, 3)
        rotmats = jnp.broadcast_to(jnp.eye(3), (J, 3, 3))

        f = jax.jit(fk_forward, static_argnums=())
        G = f(rotmats, joints, parents)
        assert G.shape == (J, 4, 4)

    def test_batched_shape(self):
        B, J = 4, 6
        parents = jnp.array([-1, 0, 1, 2, 3, 4])
        joints = jax.random.normal(jax.random.PRNGKey(0), (B, J, 3))
        rotmats = jnp.broadcast_to(jnp.eye(3), (B, J, 3, 3))

        G = fk_forward_batched(rotmats, joints, parents)
        assert G.shape == (B, J, 4, 4)
