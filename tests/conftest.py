"""
Shared test fixtures and helpers.

All fixtures build synthetic (random) model data so that tests can run
without downloading the actual SMPL / SMPL-X .pkl files.
"""

import numpy as np
import pytest

from smpl_jax.smpl import SMPLModel
from smpl_jax.smplx import SMPLXModel


def _make_smpl_data(V: int = 100, J: int = 24, num_betas: int = 10) -> dict:
    rng = np.random.default_rng(0)
    P = (J - 1) * 9

    w = rng.random((V, J)).astype(np.float32)
    w /= w.sum(axis=1, keepdims=True)

    jr = rng.random((J, V)).astype(np.float32)
    jr /= jr.sum(axis=1, keepdims=True)

    return dict(
        v_template=rng.standard_normal((V, 3)).astype(np.float32),
        shapedirs=rng.standard_normal((V, 3, num_betas)).astype(np.float32),
        posedirs=rng.standard_normal((V * 3, P)).astype(np.float32),
        J_regressor=jr,
        parents=np.array([-1] + list(range(J - 1)), dtype=np.int32),
        weights=w,
        faces=rng.integers(0, V, size=(V * 2, 3), dtype=np.int32),
    )


def _make_smplx_data(
    V: int = 100,
    J: int = 55,
    num_betas: int = 10,
    num_expr: int = 10,
) -> dict:
    data = _make_smpl_data(V=V, J=J, num_betas=num_betas)
    rng = np.random.default_rng(1)
    data["exprdirs"] = rng.standard_normal((V, 3, num_expr)).astype(np.float32)
    return data


@pytest.fixture(scope="session")
def smpl_model() -> SMPLModel:
    data = _make_smpl_data()
    return SMPLModel(**data)


@pytest.fixture(scope="session")
def smplx_model() -> SMPLXModel:
    data = _make_smplx_data()
    return SMPLXModel(**data)
