#!/usr/bin/env bash
# Shared environment resolution for the comparison-render drivers
# (run.sh, run_multi.sh). Sourced, not executed.
#
# Everything here is overridable from the caller's environment so the scripts
# carry no machine-specific absolute paths:
#
#   PY               interpreter used for every stage   (default: python3 on PATH)
#   TORCH_CUDA_LIBS  dir holding the CUDA 13 NVRTC libs that torch/smplxpp need
#                    (default: auto-detected from the interpreter's site-packages)
#   SEQ              input mocap clip
#   OUT              scratch dir for intermediate .npz files
#
# torch/smplxpp need CUDA 13 NVRTC while JAX ships CUDA 12, so the stages run as
# separate subprocesses with different LD_LIBRARY_PATHs.

# Repo root = two levels up from this file, regardless of where it is invoked.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CR="$REPO/tools/compare_render"

PY="${PY:-$(command -v python3 || command -v python)}"
if [[ -z "$PY" ]]; then
    echo "ERROR: no python interpreter found; set PY=/path/to/python" >&2
    exit 2
fi

# Locate the pip-installed NVIDIA CUDA 13 runtime libraries that torch needs for
# NVRTC. Override by exporting TORCH_CUDA_LIBS if they live elsewhere.
if [[ -z "${TORCH_CUDA_LIBS:-}" ]]; then
    TORCH_CUDA_LIBS="$(
        "$PY" - <<'PYEOF' 2>/dev/null || true
import pathlib, re, sysconfig
# Bundled NVIDIA wheels install as nvidia/cu<major>/lib (e.g. cu12, cu13).
# Match the version-numbered dirs only -- names like "cusparselt" must not win.
best = None
for key in ("purelib", "platlib"):
    root = pathlib.Path(sysconfig.get_paths()[key]) / "nvidia"
    for candidate in root.glob("cu*/lib"):
        m = re.fullmatch(r"cu(\d+)", candidate.parent.name)
        if m and candidate.is_dir():
            version = int(m.group(1))
            if best is None or version > best[0]:
                best = (version, candidate)
if best:
    print(best[1])
PYEOF
    )"
fi
if [[ -z "${TORCH_CUDA_LIBS:-}" ]]; then
    echo "WARNING: could not auto-detect CUDA libs for torch; set TORCH_CUDA_LIBS=... if the torch stage fails" >&2
fi

BENCH_BATCH=${BENCH_BATCH:-2048}
WARMUP=${WARMUP:-10}
REPEATS=${REPEATS:-50}
MATMUL_PRECISION=${MATMUL_PRECISION:-tf32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Real SOMA-dataset SMPL-X mocap clip (override with SEQ=...).
SEQ=${SEQ:-$REPO/datasets/SOMA/soma_subject1/dance_001_stageii.npz}
if [[ ! -f "$SEQ" ]]; then
    echo "ERROR: mocap clip not found: $SEQ" >&2
    echo "       Point SEQ=... at an AMASS/SOMA *_stageii.npz clip." >&2
    exit 2
fi
