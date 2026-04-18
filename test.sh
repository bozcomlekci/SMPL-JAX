#!/usr/bin/env bash
set -euo pipefail

JAX_PLATFORMS=cpu python test.py \
  --sequence datasets/SOMA/soma_subject1/walk_001_stageii.npz \
  --model smpl_models/smplx/SMPLX_NEUTRAL.npz \
  --mode sequence \
  --frame-stride 2 \
  --fps 60 \
  --source-up-axis auto \
  --camera-view front \
  --camera-zoom 1 \
  --no-body-center \
  "$@"