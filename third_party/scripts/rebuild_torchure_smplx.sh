#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TORCHURE_DIR="${ROOT_DIR}/third_party/torchure_smplx"
PATCH_FILE="${ROOT_DIR}/third_party/patches/torchure_smplx_dtype_benchmark.patch"
BUILD_DIR="${BUILD_DIR:-${TORCHURE_DIR}/build}"

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
CONDA_ENV="${CONDA_ENV:-body}"
BUILD_TESTS="${BUILD_TESTS:-OFF}"
USE_OPEN3D="${USE_OPEN3D:-OFF}"
JOBS="${JOBS:-$(nproc)}"

if [[ -z "${CONDA_BIN}" ]]; then
  echo "Error: conda executable not found in PATH. Set CONDA_BIN explicitly." >&2
  exit 1
fi

if [[ ! -d "${TORCHURE_DIR}" ]]; then
  echo "Error: torchure_smplx directory not found at ${TORCHURE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Error: patch file not found at ${PATCH_FILE}" >&2
  exit 1
fi

if git -C "${TORCHURE_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Applying patch: ${PATCH_FILE}"
  git -C "${TORCHURE_DIR}" apply "${PATCH_FILE}"
else
  echo "Patch already applied or cannot be applied cleanly; continuing with build."
fi

if [[ -z "${CMAKE_PREFIX_PATH:-}" ]]; then
  echo "Resolving CMAKE_PREFIX_PATH from torch in conda env: ${CONDA_ENV}"
  CMAKE_PREFIX_PATH="$("${CONDA_BIN}" run -n "${CONDA_ENV}" python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
fi

echo "Using CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"

echo "Configuring torchure_smplx"
cmake -S "${TORCHURE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DBUILD_TESTS="${BUILD_TESTS}" \
  -DUSE_OPEN3D="${USE_OPEN3D}"

echo "Building benchmark target"
cmake --build "${BUILD_DIR}" --target benchmark -j "${JOBS}"

echo "Done. Benchmark binary at ${BUILD_DIR}/benchmark"
