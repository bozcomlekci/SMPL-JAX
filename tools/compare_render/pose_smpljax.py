"""Pose the body over the shared motion using SMPL-JAX.

Two jobs:
  1. RENDER: jitted full forward over the clip's frames -> posed vertex +
     joint sequences for the GIF (same model file, same float32 parameters as
     the torch side).

Matmul precision (--matmul-precision)
-------------------------------------
  tf32 (default) — JAX's out-of-the-box mode: fp32 matmuls run on TF32 tensor
     cores (~10-bit mantissa). This is how users get SMPL-JAX, so it is what
     the teaser advertises ("framework defaults vs framework defaults"; the
     reference smplx package likewise keeps ITS default, allow_tf32=False).
     Posed vertices agree with the fp64 reference to ~1 mm max / 0.1 mm mean.
  fp32 — pins JAX_DEFAULT_MATMUL_PRECISION=highest so both sides do identical
     full-fp32 arithmetic; agreement becomes sub-um, throughput drops
     (155k -> 84k meshes/s at batch 2048 on an RTX 5080). Use this for
     matched-arithmetic numbers or numerical-faithfulness checks.
  2. SPEED: the reported relative speed is the LARGE-BATCH (default 2048) full
     forward throughput — shape/expression blend shapes + FK (lax.scan) + pose
     correctives + LBS, one jit-compiled XLA program. Warmup runs exclude the
     compile time from the timing, mirroring the torch side's warmup.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--model", default=str(REPO / "data" / "smplx" / "SMPLX_NEUTRAL.npz"))
    p.add_argument("--bench-batch", type=int, default=2048,
                   help="batch size for the reported large-batch throughput")
    p.add_argument("--warmup", type=int, default=10,
                   help="untimed warmup iterations (excludes JIT compile/allocation)")
    p.add_argument("--repeats", type=int, default=50,
                   help="timed iterations; the median is reported")
    p.add_argument("--matmul-precision", choices=["tf32", "fp32"], default="tf32",
                   help="tf32 = JAX default (tensor cores); fp32 = match torch's "
                        "full-fp32 GEMMs exactly (see module docstring)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.matmul_precision == "fp32":
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"  # before jax import
    import jax
    import jax.numpy as jnp
    from smpl_jax import SMPLXModel, SMPLXParams

    m = np.load(args.motion, allow_pickle=True)
    T = m["trans"].shape[0]
    num_betas = int(m["betas"].shape[0])
    num_expr = 10

    # npz loading mirrors benchmarks/benchmark_runtime.py::_load_jax_smplx_model
    data = np.load(args.model, allow_pickle=True)
    v_template = np.asarray(data["v_template"], np.float32)
    shapedirs = np.asarray(data["shapedirs"], np.float32)
    posedirs_raw = np.asarray(data["posedirs"], np.float32)
    if posedirs_raw.ndim == 3:
        posedirs = posedirs_raw.reshape(v_template.shape[0] * 3, -1)
    elif posedirs_raw.ndim == 2 and posedirs_raw.shape[0] == v_template.shape[0] * 3:
        posedirs = posedirs_raw
    else:
        posedirs = posedirs_raw.T
    exprdirs = (np.asarray(data["expr_dirs"], np.float32) if "expr_dirs" in data.files
                else shapedirs[..., 300:])
    parents = np.asarray(data["kintree_table"], np.int32)[0].copy(); parents[0] = -1
    faces = np.asarray(data["f"], np.int32)
    model = SMPLXModel(
        v_template=v_template, shapedirs=shapedirs, exprdirs=exprdirs,
        posedirs=posedirs, J_regressor=np.asarray(data["J_regressor"], np.float32),
        parents=parents, weights=np.asarray(data["weights"], np.float32),
        faces=faces, num_betas=num_betas, num_expression_coeffs=num_expr,
    )
    forward = jax.jit(model.forward)

    def params_for(B, rep):
        return SMPLXParams(
            betas=jnp.asarray(np.broadcast_to(m["betas"][None], (B, num_betas)).copy()),
            body_pose=jnp.asarray(rep(m["body_pose"])),
            global_orient=jnp.asarray(rep(m["global_orient"])),
            transl=jnp.asarray(rep(m["trans"])),
            expression=jnp.zeros((B, num_expr), jnp.float32),
            jaw_pose=jnp.asarray(rep(m["jaw_pose"])),
            leye_pose=jnp.asarray(rep(m["leye_pose"])),
            reye_pose=jnp.asarray(rep(m["reye_pose"])),
            left_hand_pose=jnp.asarray(rep(m["left_hand_pose"])),
            right_hand_pose=jnp.asarray(rep(m["right_hand_pose"])),
        )

    # ---------- 1) render: pose the clip (single full-sequence forward) ----------
    out = forward(params_for(T, lambda x: x))
    verts = np.asarray(out.vertices)
    joints = np.asarray(out.joints)

    # ---------- 2) speed: large-batch full forward throughput ----------
    B = args.bench_batch
    bench_params = params_for(B, lambda x: np.tile(x, (int(np.ceil(B / T)), 1))[:B])
    # Fair protocol (identical on the torch side): `warmup` untimed iterations
    # to exclude JIT compile + allocation, then median of `repeats` timed
    # iterations. JAX runs under its default caching allocator (preallocate off
    # only so it can share the GPU with the torch subprocess) — no per-side
    # tuning flags that would advantage either framework.
    for _ in range(args.warmup):
        forward(bench_params).vertices.block_until_ready()
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        forward(bench_params).vertices.block_until_ready()
        ts.append(time.perf_counter() - t0)
    total = float(np.median(ts))
    fps = B / total

    parents_out = parents.copy(); parents_out[0] = 0
    np.savez(args.out, verts=verts, faces=faces, fps=fps,
             batch=B, bench_total_s=total, joints=joints, parents=parents_out,
             duration_s=float(m["duration_s"]), play_fps=float(m["play_fps"]),
             label="SMPL-JAX (JAX)",
             precision="TF32" if args.matmul_precision == "tf32" else "FP32")
    print(f"[smpl_jax] render T={T}  |  full-forward B={B} "
          f"(warmup {args.warmup}, median of {args.repeats}): {total*1e3:.2f} ms "
          f"=> {fps:.0f} meshes/s  wrote {args.out}")


if __name__ == "__main__":
    main()
