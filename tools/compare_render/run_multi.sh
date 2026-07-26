#!/usr/bin/env bash
# Multi-method SMPL-X comparison GIF: SMPL-JAX vs vchoutas/smplx vs sxyu/smplxpp.
# Same model file, same SOMA-dataset motion, same settings; each column's frame
# cadence is slowed in proportion to its measured throughput (the fastest plays
# every frame, slower methods advance fewer distinct poses per wall-clock second).
#
# torch/smplxpp need CUDA 13 NVRTC; JAX ships CUDA 12 — keep them on separate
# LD_LIBRARY_PATHs and run as subprocesses. Each method poses BENCH_BATCH
# (default 2048) with the identical timing protocol under its framework default.
set -euo pipefail
REPO=/home/bozcomlekci/Desktop/projects/SMPL-JAX
PY=/home/bozcomlekci/miniforge3/envs/body/bin/python
CR=$REPO/tools/compare_render
TORCH_CUDA_LIBS=/home/bozcomlekci/miniforge3/envs/body/lib/python3.10/site-packages/nvidia/cu13/lib
OUT=${OUT:-/tmp/smpl_compare_multi}
GIF=${GIF:-$REPO/assets/teaser_multi.gif}
BENCH_BATCH=${BENCH_BATCH:-2048}
WARMUP=${WARMUP:-10}
REPEATS=${REPEATS:-50}
MATMUL_PRECISION=${MATMUL_PRECISION:-tf32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$OUT"

SEQ=${SEQ:-$REPO/datasets/SOMA/soma_subject1/dance_001_stageii.npz}

echo "==> generate shared motion (clip: $(basename "$SEQ"))"
env -u LD_LIBRARY_PATH "$PY" "$CR/gen_motion.py" \
    --seq "$SEQ" --seconds "${SECONDS_WIN:-6}" --play-fps "${PLAY_FPS:-25}" \
    --out "$OUT/motion.npz"

echo "==> pose with SMPL-JAX  [batch $BENCH_BATCH, $MATMUL_PRECISION]"
env -u LD_LIBRARY_PATH "$PY" "$CR/pose_smpljax.py" \
    --motion "$OUT/motion.npz" --out "$OUT/smpljax.npz" \
    --bench-batch "$BENCH_BATCH" --warmup "$WARMUP" --repeats "$REPEATS" \
    --matmul-precision "$MATMUL_PRECISION"

echo "==> pose with vchoutas/smplx (PyTorch)  [batch $BENCH_BATCH]"
LD_LIBRARY_PATH="$TORCH_CUDA_LIBS" "$PY" "$CR/pose_smplx.py" \
    --motion "$OUT/motion.npz" --out "$OUT/smplx.npz" \
    --bench-batch "$BENCH_BATCH" --warmup "$WARMUP" --repeats "$REPEATS"

echo "==> pose with sxyu/smplxpp (C++/CUDA)  [batch $BENCH_BATCH]"
LD_LIBRARY_PATH="$TORCH_CUDA_LIBS" "$PY" "$CR/pose_smplxpp.py" \
    --motion "$OUT/motion.npz" --out "$OUT/smplxpp.npz" \
    --bench-batch "$BENCH_BATCH" --warmup "$WARMUP" --repeats "$REPEATS"

echo "==> render multi-method comparison GIF"
env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl "$PY" "$CR/render_compare_multi.py" \
    --inputs "$OUT/smpljax.npz" "$OUT/smplx.npz" "$OUT/smplxpp.npz" \
    --gif "$GIF"
echo "done -> $GIF"
