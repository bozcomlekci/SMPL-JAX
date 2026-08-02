# Environments

Conda environment files that reproduce the two published benchmark result sets.
**You do not need these to use the library** — for that, `pip install -e .` plus
whichever [extras](../pyproject.toml) your task needs is enough. Reach for these
only to reproduce measured numbers exactly, or to get a known-good starting
point on a fresh machine.

| File | Platform | Reproduces |
| ---- | -------- | ---------- |
| [`linux-cuda.yml`](linux-cuda.yml) | Linux + NVIDIA CUDA | [`benchmarks/results/rtx5080/`](../benchmarks/results/rtx5080/) |
| [`macos-arm64.yml`](macos-arm64.yml) | macOS / Apple Silicon, `jax-metal` | [`benchmarks/results/cpu/benchmark_m4_*.json`](../benchmarks/results/cpu/) |

```bash
conda env create -f envs/linux-cuda.yml   # or envs/macos-arm64.yml
conda activate smpl-jax-cuda              # or smpl-jax-macos
pip install -e .
```

## Why there are two CUDA majors on Linux

`linux-cuda.yml` ends up with **both** CUDA 12 and CUDA 13 runtime wheels
installed, which looks like a mistake and is not:

- **JAX** loads its GPU backend through `jax-cuda12-*` plugin wheels → CUDA 12.
- **torch** 2.11 ships its own `nvidia-*` runtime wheels → CUDA 13, and needs
  CUDA 13 NVRTC.

They coexist in site-packages but cannot both be first on `LD_LIBRARY_PATH`.
That is why [`tools/compare_render/_env.sh`](../tools/compare_render/_env.sh)
runs every stage of the render pipeline as a separate subprocess, each with its
own `LD_LIBRARY_PATH`, instead of importing both frameworks into one process.
`_env.sh` auto-detects the highest-numbered `nvidia/cu*/lib` directory; override
`TORCH_CUDA_LIBS` if your layout differs.

If you only want SMPL-JAX on GPU and none of the baselines, skip this file
entirely and install a CUDA-enabled `jaxlib` yourself:

```bash
pip install -e ".[dev]" && pip install "jax[cuda12]"
```

## Notes on the pins

`linux-cuda.yml` is hand-maintained, not `conda env export`ed. It lists only the
packages this project uses, pinned to the versions the published runs were
measured with. The CUDA runtime wheels are left unpinned so torch and the JAX
plugin can each resolve the ones they need.

`macos-arm64.yml` is a full snapshot, so its conda build strings are `osx-arm64`
specific and will not solve on Linux.

Neither file can pin the C++ baselines — `sxyu/smplxpp` and
`Hydran00/torchure_smplx` are git submodules compiled against your local CUDA
toolchain. Build them with
[`third_party/scripts/`](../third_party/scripts/README.md) after the environment
is created.
