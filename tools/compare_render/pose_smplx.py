"""Pose the body over the shared motion using the reference SMPL-X (PyTorch).

Two jobs:
  1. RENDER: full forward over the clip's frames -> posed vertex + joint
     sequences for the GIF.
  2. SPEED: the reported relative speed is the LARGE-BATCH (default 2048) full
     forward throughput — shape/expression blend shapes + FK + pose correctives
     + LBS — the honest long-term-scaling regime (batch-1 is dominated by
     kernel-launch latency on both sides). Settings mirror
     benchmarks/benchmark_runtime.py: ``use_pca=False``, ``flat_hand_mean=True``,
     num_betas=10, num_expression_coeffs=10.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--model-dir", default=str(REPO / "data"))
    p.add_argument("--bench-batch", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10,
                   help="untimed warmup iterations (excludes autotune/allocation)")
    p.add_argument("--repeats", type=int, default=50,
                   help="timed iterations; the median is reported")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    import smplx

    m = np.load(args.motion, allow_pickle=True)
    T = m["trans"].shape[0]
    num_betas = int(m["betas"].shape[0])
    num_expr = 10
    dev = "cuda:0"

    def make_model(B):
        return smplx.create(
            model_path=args.model_dir, model_type="smplx", gender="neutral",
            ext="npz", use_pca=False, num_betas=num_betas,
            num_expression_coeffs=num_expr, flat_hand_mean=True, batch_size=B,
        ).to(dev).eval()

    def tt(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(dev)

    # ---------- 1) render: pose the clip (single full-sequence forward) ----------
    model = make_model(T)
    betas_t = tt(np.broadcast_to(m["betas"][None], (T, num_betas)).copy())
    with torch.inference_mode():
        out = model(
            betas=betas_t, body_pose=tt(m["body_pose"]),
            global_orient=tt(m["global_orient"]), transl=tt(m["trans"]),
            left_hand_pose=tt(m["left_hand_pose"]), right_hand_pose=tt(m["right_hand_pose"]),
            jaw_pose=tt(m["jaw_pose"]), leye_pose=tt(m["leye_pose"]),
            reye_pose=tt(m["reye_pose"]),
            expression=torch.zeros((T, num_expr), dtype=torch.float32, device=dev),
            return_verts=True,
        )
    verts = out.vertices.detach().cpu().numpy()
    J = 55  # skeleton joints; smplx appends extra landmark joints after these
    joints = out.joints[:, :J].detach().cpu().numpy()

    # ---------- 2) speed: large-batch full forward throughput ----------
    B = args.bench_batch
    bench = make_model(B)
    rep = lambda x: tt(np.tile(x, (int(np.ceil(B / T)), 1))[:B])
    kw = dict(
        betas=tt(np.broadcast_to(m["betas"][None], (B, num_betas)).copy()),
        body_pose=rep(m["body_pose"]), global_orient=rep(m["global_orient"]),
        transl=rep(m["trans"]), left_hand_pose=rep(m["left_hand_pose"]),
        right_hand_pose=rep(m["right_hand_pose"]), jaw_pose=rep(m["jaw_pose"]),
        leye_pose=rep(m["leye_pose"]), reye_pose=rep(m["reye_pose"]),
        expression=torch.zeros((B, num_expr), dtype=torch.float32, device=dev),
        return_verts=True,
    )

    def full_forward():
        return bench(**kw)

    # Fair protocol (identical on the JAX side): `warmup` untimed iterations to
    # exclude cuDNN autotune + allocation, then median of `repeats` timed
    # iterations. torch runs under its default caching allocator (no tuning).
    for _ in range(args.warmup):
        with torch.inference_mode():
            full_forward()
    torch.cuda.synchronize()
    ts = np.empty(args.repeats)
    for i in range(args.repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            full_forward()
        torch.cuda.synchronize(); ts[i] = time.perf_counter() - t0
    total = float(np.median(ts))
    fps = B / total

    faces = model.faces.astype(np.int32)
    parents = model.parents.detach().cpu().numpy().astype(int)[:J].copy()
    parents[0] = 0
    # torch's own default: allow_tf32=False -> full-fp32 GEMMs. The reference
    # implementation is run untouched, exactly as shipped.
    np.savez(args.out, verts=verts, faces=faces, fps=fps, batch=B, bench_total_s=total,
             joints=joints, parents=parents,
             duration_s=float(m["duration_s"]), play_fps=float(m["play_fps"]),
             label="SMPL-X (PyTorch)", precision="FP32")
    print(f"[smplx_torch] render T={T}  |  full-forward B={B} "
          f"(warmup {args.warmup}, median of {args.repeats}): {total*1e3:.2f} ms "
          f"=> {fps:.0f} meshes/s  wrote {args.out}")


if __name__ == "__main__":
    main()
