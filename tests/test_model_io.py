"""
Tests for model loading and the SMPL-X hand-pose convention.

Uses small synthetic model files written to a tmp .npz / .pkl, so no real
SMPL / SMPL-X weights are required.
"""

import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from smpl_jax.model_io import load_model_data
from smpl_jax.smplx import SMPLXModel
from smpl_jax.types import SMPLXParams


V, J, F = 40, 55, 20
NUM_SHAPE = 300          # official SMPL-X identity basis width
NUM_EXPR = 100           # official SMPL-X expression basis width


def _synthetic_model(num_shape_components: int) -> dict:
    """Build a tiny SMPL-X-shaped model dict with a packed `shapedirs` block."""
    rng = np.random.default_rng(0)

    w = rng.random((V, J)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)

    jr = rng.random((J, V)).astype(np.float32)
    jr /= jr.sum(axis=1, keepdims=True)

    kintree = np.zeros((2, J), dtype=np.int64)
    kintree[0] = np.array([0] + list(range(J - 1)))
    kintree[1] = np.arange(J)

    return dict(
        v_template=rng.standard_normal((V, 3)).astype(np.float32),
        shapedirs=rng.standard_normal((V, 3, num_shape_components)).astype(np.float32),
        posedirs=rng.standard_normal((V, 3, (J - 1) * 9)).astype(np.float32),
        J_regressor=jr,
        kintree_table=kintree,
        weights=w,
        f=rng.integers(0, V, size=(F, 3)).astype(np.int32),
        hands_meanl=rng.standard_normal(45).astype(np.float32) * 0.2,
        hands_meanr=rng.standard_normal(45).astype(np.float32) * 0.2,
    )


def _write(tmp_path, data: dict, ext: str) -> str:
    path = tmp_path / f"model.{ext}"
    if ext == "npz":
        np.savez(path, **data)
    else:
        with open(path, "wb") as f:
            pickle.dump(data, f)
    return str(path)


def _params(B: int = 2, **overrides) -> SMPLXParams:
    base = dict(
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
    base.update(overrides)
    return SMPLXParams(**base)


class TestPackedBasisSplit:
    """Official SMPL-X files pack shape + expression into one `shapedirs`."""

    @pytest.mark.parametrize("ext", ["npz", "pkl"])
    def test_split_applied(self, tmp_path, ext):
        path = _write(tmp_path, _synthetic_model(NUM_SHAPE + NUM_EXPR), ext)
        data = load_model_data(path)
        assert data["shapedirs"].shape == (V, 3, NUM_SHAPE)
        assert data["exprdirs"].shape == (V, 3, NUM_EXPR)

    @pytest.mark.parametrize("ext", ["npz", "pkl"])
    def test_smpl_sized_basis_is_not_split(self, tmp_path, ext):
        """A plain SMPL file (300 shape components, no expression) is untouched."""
        path = _write(tmp_path, _synthetic_model(NUM_SHAPE), ext)
        data = load_model_data(path)
        assert data["shapedirs"].shape == (V, 3, NUM_SHAPE)
        assert data["exprdirs"] is None

    def test_explicit_exprdirs_key_wins(self, tmp_path):
        raw = _synthetic_model(NUM_SHAPE + NUM_EXPR)
        raw["expr_dirs"] = np.zeros((V, 3, 7), dtype=np.float32)
        data = load_model_data(_write(tmp_path, raw, "npz"))
        assert data["exprdirs"].shape == (V, 3, 7)
        # shapedirs must not be split when the file already separates the bases
        assert data["shapedirs"].shape == (V, 3, NUM_SHAPE + NUM_EXPR)

    @pytest.mark.parametrize("ext", ["npz", "pkl"])
    def test_smplx_model_loads(self, tmp_path, ext):
        path = _write(tmp_path, _synthetic_model(NUM_SHAPE + NUM_EXPR), ext)
        model = SMPLXModel.load(path, num_betas=10, num_expression_coeffs=10)
        assert model.shapedirs.shape == (V, 3, 10)
        assert model.exprdirs.shape == (V, 3, 10)
        assert model(_params()).vertices.shape == (2, V, 3)

    def test_hand_means_are_loaded(self, tmp_path):
        raw = _synthetic_model(NUM_SHAPE + NUM_EXPR)
        data = load_model_data(_write(tmp_path, raw, "npz"))
        np.testing.assert_allclose(data["hands_meanl"], raw["hands_meanl"])
        np.testing.assert_allclose(data["hands_meanr"], raw["hands_meanr"])


class TestFlatHandMean:
    """`flat_hand_mean` selects the hand-pose convention; default is flat."""

    @pytest.fixture(scope="class")
    def path(self, tmp_path_factory):
        return _write(tmp_path_factory.mktemp("m"),
                      _synthetic_model(NUM_SHAPE + NUM_EXPR), "npz")

    def test_default_is_flat(self, path):
        model = SMPLXModel.load(path)
        assert model.flat_hand_mean is True
        np.testing.assert_allclose(np.asarray(model.hands_meanl), 0.0)
        np.testing.assert_allclose(np.asarray(model.hands_meanr), 0.0)

    def test_non_flat_uses_model_mean(self, path):
        raw = load_model_data(path)
        model = SMPLXModel.load(path, flat_hand_mean=False)
        np.testing.assert_allclose(np.asarray(model.hands_meanl), raw["hands_meanl"])

    def test_conventions_differ_at_zero_hand_pose(self, path):
        flat = SMPLXModel.load(path, flat_hand_mean=True)(_params())
        relaxed = SMPLXModel.load(path, flat_hand_mean=False)(_params())
        assert float(jnp.max(jnp.abs(flat.vertices - relaxed.vertices))) > 1e-4

    def test_conventions_agree_when_offset_by_the_mean(self, path):
        """pose=0 under flat == pose=-mean under the relaxed convention."""
        mean_l = load_model_data(path)["hands_meanl"]
        mean_r = load_model_data(path)["hands_meanr"]
        flat = SMPLXModel.load(path, flat_hand_mean=True)(_params())
        relaxed = SMPLXModel.load(path, flat_hand_mean=False)(
            _params(
                left_hand_pose=jnp.tile(-jnp.asarray(mean_l), (2, 1)),
                right_hand_pose=jnp.tile(-jnp.asarray(mean_r), (2, 1)),
            )
        )
        np.testing.assert_allclose(
            np.asarray(relaxed.vertices), np.asarray(flat.vertices), atol=1e-5
        )
