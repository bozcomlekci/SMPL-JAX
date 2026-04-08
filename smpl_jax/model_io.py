"""
Model loading utilities for SMPL and SMPL-X .pkl files.

Normalises the various array shapes found in different model versions into a
single, consistent dict that the model constructors can consume.
"""

from __future__ import annotations

import pickle

import numpy as np


def load_model_data(path: str) -> dict:
    """Load a SMPL or SMPL-X .pkl file and return standardised numpy arrays.

    The returned dict contains:
        v_template   (V, 3)          float32
        shapedirs    (V, 3, K)       float32
        posedirs     (V*3, P)        float32
        J_regressor  (J, V)          float32   dense
        parents      (J,)            int32     parents[0] == -1
        weights      (V, J)          float32
        faces        (F, 3)          int32
        exprdirs     (V, 3, E)       float32   or None (SMPL-X only)
    """
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

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
    exprdirs_raw = get("expr_dirs") or get("exprdirs")
    exprdirs: np.ndarray | None = None
    if exprdirs_raw is not None:
        exprdirs = np.array(exprdirs_raw, dtype=np.float32)
        if exprdirs.ndim == 2:
            exprdirs = exprdirs.reshape(V, 3, -1)
        # exprdirs is now (V, 3, E)

    return dict(
        v_template=v_template,
        shapedirs=shapedirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        weights=weights,
        faces=faces,
        exprdirs=exprdirs,
    )
