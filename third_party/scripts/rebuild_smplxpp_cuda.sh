#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMPLXPP_DIR="${ROOT_DIR}/third_party/smplxpp"
PATCH_FILE="${ROOT_DIR}/third_party/patches/smplxpp_cuda_build_fix.patch"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
CC="${CC:-/usr/bin/gcc-13}"
CXX="${CXX:-/usr/bin/g++-13}"
CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"
CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-body}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "Error: conda executable not found in PATH. Set CONDA_BIN explicitly." >&2
  exit 1
fi

if [[ ! -d "${SMPLXPP_DIR}" ]]; then
  echo "Error: smplxpp directory not found at ${SMPLXPP_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Error: patch file not found at ${PATCH_FILE}" >&2
  exit 1
fi

if git -C "${SMPLXPP_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Applying patch: ${PATCH_FILE}"
  git -C "${SMPLXPP_DIR}" apply "${PATCH_FILE}"
else
  echo "Patch already applied or cannot be applied cleanly; continuing with rebuild."
fi

cd "${SMPLXPP_DIR}"
rm -rf build dist *.egg-info

echo "Rebuilding smplxpp with CUDA from ${CUDA_HOME}"
SMPLXPP_USE_CUDA=ON \
CUDA_HOME="${CUDA_HOME}" \
CUDACXX="${CUDACXX}" \
CC="${CC}" \
CXX="${CXX}" \
CUDAHOSTCXX="${CUDAHOSTCXX}" \
PATH="${CUDA_HOME}/bin:${PATH}" \
LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" \
"${CONDA_BIN}" run -n "${CONDA_ENV}" \
python -m pip install -v --no-build-isolation --force-reinstall .

echo "Verifying smplxpp CUDA runtime flag"
"${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import smplxpp
cuda_flag = bool(getattr(smplxpp, "cuda", False))
print("smplxpp.cuda =", cuda_flag)
if not cuda_flag:
    raise SystemExit("smplxpp CUDA flag is False after rebuild")
PY

echo "Done. smplxpp CUDA rebuild succeeded."
