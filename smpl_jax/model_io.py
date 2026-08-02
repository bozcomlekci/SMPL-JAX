"""
Model loading utilities for SMPL and SMPL-X model files (.pkl and .npz).

Normalises the various array shapes found in different model versions into a
single, consistent dict that the model constructors can consume.
"""

from __future__ import annotations

import pickle

import numpy as np


# Official SMPL-X files ship shape and expression blend shapes concatenated
# into a single `shapedirs` block: the first 300 columns are the identity
# (beta) basis and everything after is the expression basis.  There is no
# `expr_dirs` key in those files, so the split has to be applied on load.
_SMPLX_SHAPE_COMPONENTS = 300


def _read_raw(path: str):
    """Read a model file as either a pickle or an .npz archive."""
    if path.lower().endswith(".npz"):
        with np.load(path, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}
    try:
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except (pickle.UnpicklingError, UnicodeDecodeError):
        with np.load(path, allow_pickle=True) as npz:
            return {k: npz[k] for k in npz.files}


def load_model_data(path: str) -> dict:
    """Load a SMPL or SMPL-X model file and return standardised numpy arrays.

    The returned dict contains:
        v_template   (V, 3)          float32
        shapedirs    (V, 3, K)       float32
        posedirs     (V*3, P)        float32
        J_regressor  (J, V)          float32   dense
        parents      (J,)            int32     parents[0] == -1
        weights      (V, J)          float32
        faces        (F, 3)          int32
        exprdirs     (V, 3, E)       float32   or None (SMPL-X only)
        hands_meanl  (45,)           float32   or None (SMPL-X / SMPL-H only)
        hands_meanr  (45,)           float32   or None
    """
    raw = _read_raw(path)

    def get(key, default=None):
        if isinstance(raw, dict):
            return raw.get(key, default)
        return getattr(raw, key, default)

    # ---- v_template ---------------------------------------------------
    v_template = np.array(get("v_template"), dtype=np.float32)     # (V, 3)
    V = v_template.shape[0]

    # ---- shapedirs ----------------------------------------------------
    shapedirs = np.array(get("shapedirs"), dtype=np.float32)
    if shapedirs.ndim == 2:
        # (V*3, K) → (V, 3, K)
        shapedirs = shapedirs.reshape(V, 3, -1)
    # shapedirs is now (V, 3, K)

    # ---- posedirs -----------------------------------------------------
    posedirs_raw = np.array(get("posedirs"), dtype=np.float32)
    if posedirs_raw.ndim == 3:
        # (V, 3, P) → (V*3, P)
        posedirs = posedirs_raw.reshape(V * 3, -1)
    elif posedirs_raw.shape[0] == V * 3:
        posedirs = posedirs_raw                                     # already (V*3, P)
    else:
        # Likely (P, V*3) — transpose to (V*3, P)
        if posedirs_raw.ndim != 2:
            raise ValueError(
                f"Cannot interpret posedirs shape {posedirs_raw.shape}: "
                "expected a 2-D array in (V*3, P) or (P, V*3) layout."
            )
        posedirs = posedirs_raw.T

    # ---- J_regressor --------------------------------------------------
    J_reg = get("J_regressor")
    try:
        J_regressor = np.array(J_reg.todense(), dtype=np.float32)
    except AttributeError:
        J_regressor = np.array(J_reg, dtype=np.float32)            # (J, V)

    # ---- kintree_table → parents --------------------------------------
    kintree = np.array(get("kintree_table"), dtype=np.int32)       # (2, J)
    parents = kintree[0].copy()
    parents[0] = -1                                                 # root has no parent

    # ---- weights & faces ----------------------------------------------
    weights = np.array(get("weights"), dtype=np.float32)           # (V, J)
    faces = np.array(get("f"), dtype=np.int32)                     # (F, 3)

    # ---- expression blend shapes (SMPL-X only) ------------------------
    exprdirs_raw = get("expr_dirs")
    if exprdirs_raw is None:
        exprdirs_raw = get("exprdirs")
    exprdirs: np.ndarray | None = None
    if exprdirs_raw is not None:
        exprdirs = np.array(exprdirs_raw, dtype=np.float32)
        if exprdirs.ndim == 2:
            exprdirs = exprdirs.reshape(V, 3, -1)
        # exprdirs is now (V, 3, E)
    elif shapedirs.shape[-1] > _SMPLX_SHAPE_COMPONENTS:
        # Official SMPL-X layout: split the packed basis into shape + expression.
        exprdirs = shapedirs[..., _SMPLX_SHAPE_COMPONENTS:]
        shapedirs = shapedirs[..., :_SMPLX_SHAPE_COMPONENTS]

    # ---- MANO hand pose means (SMPL-X / SMPL-H only) ------------------
    def _hand_mean(key):
        v = get(key)
        return None if v is None else np.array(v, dtype=np.float32).reshape(-1)

    return dict(
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        weights=weights,
        faces=faces,
        exprdirs=exprdirs,
        hands_meanl=_hand_mean("hands_meanl"),
        hands_meanr=_hand_mean("hands_meanr"),
    )
