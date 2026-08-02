#!/usr/bin/env bash
# Sequence-mode demo: poses a whole SOMA clip and animates it in Open3D.
# Extra arguments are forwarded to pose_sequence.py.
#
# JAX_PLATFORMS defaults to cpu because the JAX Metal backend is unstable on
# Apple Silicon and this demo is interactive rather than throughput-bound.
# Override it (JAX_PLATFORMS=cuda bash examples/pose_sequence.sh) to use a GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} python "$REPO/examples/pose_sequence.py" \
  --sequence "$REPO/datasets/SOMA/soma_subject1/walk_001_stageii.npz" \
  --model "$REPO/data/smplx/SMPLX_NEUTRAL.npz" \
  --mode sequence \
  --frame-stride 2 \
  --fps 60 \
  --source-up-axis auto \
  --camera-view front \
  --camera-zoom 1 \
  --no-body-center \
  "$@"
