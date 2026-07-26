"""Pose the body over the shared motion using sxyu/smplxpp (C++/CUDA binding).

Same two jobs as the other posers:
  1. RENDER: forward each clip frame through a single BodyX -> posed vertex +
     joint sequences for the GIF.
  2. SPEED: large-batch (default 2048) throughput via FusedBatchForwardX. Each
     timed iteration materializes the result to host (np.asarray) so the async
     CUDA work is actually synchronized — the repo benchmark's smplxpp path
     leaves sync_once a no-op and under-measures it, so we do NOT reuse that
     number here.

smplxpp's 165-dim pose is [global(3), body(63), jaw(3), leye(3), reye(3),
hands(90)] — identical joint order to the torch/JAX SMPL-X posers, so all three
columns show the same motion.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _pose165(m):
    """Assemble smplxpp's (T,165) pose from the shared motion arrays."""
    return np.concatenate([
        m["global_orient"], m["body_pose"],
        m["jaw_pose"], m["leye_pose"], m["reye_pose"],
        m["left_hand_pose"], m["right_hand_pose"],
    ], axis=1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--model", default=str(REPO / "data" / "smplx" / "SMPLX_NEUTRAL.npz"))
    p.add_argument("--bench-batch", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import smplxpp

    m = np.load(args.motion, allow_pickle=True)
    T = m["trans"].shape[0]
    pose = _pose165(m)
    trans = np.asarray(m["trans"], np.float32)

    model = smplxpp.ModelX(args.model, "", smplxpp.Gender.neutral)
    n_shape = int(smplxpp.ModelX.n_shape_blends)
    n_joints = int(np.asarray(model.n_joints))
    betas = np.zeros((n_shape,), np.float32)
    bsrc = np.asarray(m["betas"], np.float32)
    betas[:min(n_shape, bsrc.shape[0])] = bsrc[:n_shape]
    parents = np.asarray([int(model.parent(i)) for i in range(n_joints)], np.int64)

    force_cpu = not bool(getattr(smplxpp, "cuda", False))

    # ---------- 1) render: forward each clip frame ----------
    body = smplxpp.BodyX(model)
    body.betas = betas
    verts, joints = [], []
    for i in range(T):
        body.trans = trans[i]
        body.pose = pose[i]
        body.update(force_cpu=force_cpu)
        verts.append(np.asarray(body.verts).copy())
        joints.append(np.asarray(body.joints).copy())
    verts = np.stack(verts).astype(np.float32)      # (T, V, 3)
    joints = np.stack(joints).astype(np.float32)     # (T, J, 3)

    # ---------- 2) speed: large-batch fused forward throughput ----------
    B = args.bench_batch
    fused_cls = getattr(smplxpp, "FusedBatchForwardX", None)
    if fused_cls is None:
        raise RuntimeError("this smplxpp build lacks FusedBatchForwardX")
    tile = lambda x: np.tile(x, (int(np.ceil(B / T)), 1))[:B].astype(np.float32)
    trans_b, pose_b = tile(trans), tile(pose)
    fused = fused_cls(model, B)
    fused.set_betas(betas)

    def full_forward():
        # np.asarray on the returned device tensor forces a host copy => sync.
        return np.asarray(fused.forward_last(trans_b, pose_b, force_cpu=force_cpu))

    for _ in range(args.warmup):
        full_forward()
    ts = np.empty(args.repeats)
    for i in range(args.repeats):
        t0 = time.perf_counter()
        full_forward()
        ts[i] = time.perf_counter() - t0
    total = float(np.median(ts))
    fps = B / total

    faces = np.asarray(model.faces).astype(np.int32)
    np.savez(args.out, verts=verts, faces=faces, fps=fps, batch=B, bench_total_s=total,
             joints=joints, parents=parents,
             duration_s=float(m["duration_s"]), play_fps=float(m["play_fps"]),
             label="smplxpp (C++/CUDA)", precision="FP32")
    print(f"[smplxpp] render T={T}  |  fused-forward B={B} "
          f"(warmup {args.warmup}, median of {args.repeats}): {total*1e3:.2f} ms "
          f"=> {fps:.0f} meshes/s  wrote {args.out}")


if __name__ == "__main__":
    main()
