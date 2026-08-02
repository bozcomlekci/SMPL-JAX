# Comparison render pipeline

Generates the side-by-side teaser GIFs in `assets/`. Every column poses the
*identical* rig over the *identical* motion with matched settings, so the meshes
agree to sub-millimetre — the only visible difference is how many distinct poses
each implementation can afford in a fixed wall-clock budget.

## Pipeline

```
gen_motion.py        SOMA *_stageii.npz  →  shared float32 parameter sequence
      │
      ├── pose_smpljax.py     this repo          ┐
      ├── pose_smplx.py       vchoutas/smplx     ├─ posed verts + joints + median throughput
      └── pose_smplxpp.py     sxyu/smplxpp       ┘
                    │
      render_compare.py        2 columns  →  assets/teaser.gif
      render_compare_multi.py  N columns  →  assets/teaser_multi.gif
```

`render_utils.py` holds the offscreen pyrender setup (ground plane, projected
shadow, multi-directional lighting, optional skeleton overlay) shared by both
renderers.

## Running

```bash
bash tools/compare_render/run.sh          # SMPL-JAX vs PyTorch smplx  → assets/teaser.gif
bash tools/compare_render/run_multi.sh    # + sxyu/smplxpp             → assets/teaser_multi.gif
```

Both drivers source [`_env.sh`](_env.sh), which resolves the repo root from the
script location and auto-detects the CUDA libraries torch needs. Nothing is
hardcoded to a particular machine; override any of these:

| Variable | Default | Meaning |
| -------- | ------- | ------- |
| `PY` | `python3` on `PATH` | Interpreter used for every stage |
| `TORCH_CUDA_LIBS` | auto-detected from site-packages | Dir with the CUDA 13 NVRTC libs torch/smplxpp need |
| `SEQ` | a SOMA `dance_001` clip | Input mocap clip |
| `OUT` | `/tmp/smpl_compare[_multi]` | Scratch dir for intermediate `.npz` |
| `GIF` | `assets/teaser[_multi].gif` | Output path |
| `BENCH_BATCH` | `2048` | Batch size used for the throughput measurement |
| `WARMUP` / `REPEATS` | `10` / `50` | Untimed warmup, then median of N timed runs |
| `MATMUL_PRECISION` | `tf32` | `tf32` (JAX default) or `fp32` (matched arithmetic) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |

torch and smplxpp need CUDA 13 NVRTC while JAX ships CUDA 12, so each stage runs
as a separate subprocess with its own `LD_LIBRARY_PATH` — that is why the driver
is a shell script rather than one Python program.

## Timing semantics

The reported speed is large-batch (`BENCH_BATCH`, default 2048) full-forward
throughput, not batch-1 latency, which on every implementation is dominated by
kernel-launch overhead. Warmup is untimed so JIT compilation is excluded, as it
is amortised in real use.

The GIF's playback encodes the measured ratio: all columns run for the same
wall-clock duration, the fastest advances a distinct pose every slot, and slower
methods advance proportionally fewer poses (sample-and-hold).

## Precision disclosure

`MATMUL_PRECISION=tf32` is the default because it is what users get
out of the box from JAX, and the reference `smplx` package likewise keeps *its*
default (`allow_tf32=False`, full-fp32 GEMMs) — "framework defaults vs framework
defaults". Under TF32 the posed vertices agree with the fp64 reference to ~1 mm
max / 0.1 mm mean, and the GIF banner states both precisions. For matched
full-fp32 arithmetic (sub-micrometre agreement, lower JAX throughput), set
`MATMUL_PRECISION=fp32`.

## Requirements

```bash
pip install -e ".[render]"      # pyrender, trimesh, Pillow, PyOpenGL
```

Plus EGL for offscreen GL, and whichever baselines you want as columns
(`pip install -e ".[reference]"` for the PyTorch `smplx` column). `run_multi.sh`
additionally needs `smplxpp` built with CUDA — see
[`third_party/scripts/README.md`](../../third_party/scripts/README.md).

The GIFs are assembled with Pillow (`Image.save(save_all=True, …)`), so no
separate video encoder is needed.
