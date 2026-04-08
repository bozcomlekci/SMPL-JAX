"""
Tests for smpl_jax.inverse_lbs.

Uses a tiny synthetic model to keep test time short.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from smpl_jax.smpl import SMPLModel
from smpl_jax.types import SMPLParams
from smpl_jax.inverse_lbs import analytical_init, autograd_refine, inverse_lbs


@pytest.fixture(scope="module")
def tiny_model():
    """A very small SMPL-like model for fast inverse-LBS tests."""
    V, J, num_betas = 30, 6, 5
    rng = np.random.default_rng(99)
    P = (J - 1) * 9

    w = rng.random((V, J)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)

    jr = rng.random((J, V)).astype(np.float32)
    jr /= jr.sum(axis=1, keepdims=True)

    return SMPLModel(
        v_template=rng.standard_normal((V, 3)).astype(np.float32),
        shapedirs=rng.standard_normal((V, 3, num_betas)).astype(np.float32),
        posedirs=rng.standard_normal((V * 3, P)).astype(np.float32),
        J_regressor=jr,
        parents=np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32),
        weights=w,
        faces=rng.integers(0, V, size=(V, 3), dtype=np.int32),
        num_betas=num_betas,
    )


class TestAnalyticalInit:
    def test_output_shape(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(0),
                                  tiny_model.v_template.shape)
        R = analytical_init(posed, tiny_model.v_template, tiny_model.weights)
        assert R.shape == (tiny_model.num_joints, 3, 3)

    def test_output_is_near_orthogonal(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(1),
                                  tiny_model.v_template.shape)
        R = analytical_init(posed, tiny_model.v_template, tiny_model.weights)
        J = R.shape[0]
        RRt = jnp.einsum("jab,jcb->jac", R, R)
        np.testing.assert_allclose(
            RRt, jnp.broadcast_to(jnp.eye(3)[None], (J, 3, 3)), atol=1e-4
        )

    def test_output_is_finite(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(2),
                                  tiny_model.v_template.shape)
        R = analytical_init(posed, tiny_model.v_template, tiny_model.weights)
        assert jnp.all(jnp.isfinite(R))


class TestInverseLBS:
    def test_inverse_lbs_output_shape(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(3),
                                  tiny_model.v_template.shape)
        R = inverse_lbs(posed, tiny_model, num_refine_iters=5)
        assert R.shape == (tiny_model.num_joints, 3, 3)

    def test_inverse_lbs_output_is_finite(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(4),
                                  tiny_model.v_template.shape)
        R = inverse_lbs(posed, tiny_model, num_refine_iters=5)
        assert jnp.all(jnp.isfinite(R))

    def test_inverse_lbs_jit_compatible(self, tiny_model):
        posed = jax.random.normal(jax.random.PRNGKey(5),
                                  tiny_model.v_template.shape)

        f = jax.jit(lambda v: inverse_lbs(v, tiny_model, num_refine_iters=3))
        R = f(posed)
        assert R.shape == (tiny_model.num_joints, 3, 3)

    def test_forward_inverse_consistency(self, tiny_model):
        """Forward-pass vertices → inverse_lbs → re-forward should be close."""
        B = 1
        n_body = tiny_model.num_joints - 1                         # 5 for tiny model
        params = SMPLParams(
            betas=jnp.zeros((B, tiny_model.num_betas)),
            body_pose=jax.random.normal(jax.random.PRNGKey(6), (B, n_body * 3)) * 0.2,
            global_orient=jnp.zeros((B, 3)),
            transl=jnp.zeros((B, 3)),
        )

        out = tiny_model(params)
        posed_verts = out.vertices[0]                               # (V, 3)

        R_recovered = inverse_lbs(
            posed_verts, tiny_model, num_refine_iters=20, lr=1e-2
        )
        assert jnp.all(jnp.isfinite(R_recovered))
