#!/usr/bin/env bash
# SMPL-X (PyTorch) vs SMPL-JAX comparison GIF. Same model file, same motion,
# same settings (full forward, num_betas 10, expr 10, flat hands).
#
# Fair comparison: both sides pose the identical batch (BENCH_BATCH, default
# 2048) with the identical timing protocol (WARMUP untimed + median of REPEATS)
# under each framework's DEFAULT allocator and DEFAULT matmul precision —
# "framework defaults vs framework defaults". The reference smplx package is
# run untouched (its default = full-fp32 GEMMs); SMPL-JAX runs its default
# TF32 tensor-core mode (~1 mm max vertex delta vs the fp64 reference), and
# the GIF banner discloses both precisions. For matched full-fp32 arithmetic
# (sub-um agreement, lower JAX throughput) set MATMUL_PRECISION=fp32.
#
# Paths and the interpreter are resolved in _env.sh; see it for the knobs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

OUT=${OUT:-/tmp/smpl_compare}
GIF=${GIF:-$REPO/assets/teaser.gif}
mkdir -p "$OUT"

echo "==> generate shared motion (clip: $(basename "$SEQ"))"
env -u LD_LIBRARY_PATH "$PY" "$CR/gen_motion.py" \
    --seq "$SEQ" --seconds "${SECONDS_WIN:-6}" --play-fps "${PLAY_FPS:-25}" \
    --out "$OUT/motion.npz"

echo "==> pose with SMPL-X (PyTorch)  [batch $BENCH_BATCH, warmup $WARMUP, median of $REPEATS]"
LD_LIBRARY_PATH="$TORCH_CUDA_LIBS" "$PY" "$CR/pose_smplx.py" \
    --motion "$OUT/motion.npz" --out "$OUT/smplx.npz" \
    --bench-batch "$BENCH_BATCH" --warmup "$WARMUP" --repeats "$REPEATS"

echo "==> pose with SMPL-JAX  [batch $BENCH_BATCH, warmup $WARMUP, median of $REPEATS, $MATMUL_PRECISION]"
env -u LD_LIBRARY_PATH "$PY" "$CR/pose_smpljax.py" \
    --motion "$OUT/motion.npz" --out "$OUT/smpljax.npz" \
    --bench-batch "$BENCH_BATCH" --warmup "$WARMUP" --repeats "$REPEATS" \
    --matmul-precision "$MATMUL_PRECISION"

echo "==> render comparison GIF"
env -u LD_LIBRARY_PATH PYOPENGL_PLATFORM=egl "$PY" "$CR/render_compare.py" \
    --smplx "$OUT/smplx.npz" --smpljax "$OUT/smpljax.npz" \
    --gif "$GIF"
echo "done -> $GIF"
