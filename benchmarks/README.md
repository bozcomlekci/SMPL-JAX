# Benchmarks

Throughput comparison of SMPL-JAX against the reference PyTorch and C++/CUDA
implementations, on identical mocap input with a shared timing protocol.

## Contents

| Path | What it is |
| ---- | ---------- |
| `benchmark_runtime.py` | The harness. Poses one clip through every available implementation and writes a result JSON. |
| `plot_figures.py` | Builds the static figures in `figures/` that the top-level README embeds. |
| `plot_style.py` | Shared palette, naming maps, and mark specs for those figures. |
| `plot_benchmark_results.py` | Builds the interactive Plotly dashboard (`results/benchmark_dashboard.html`). |
| `run_jax_guarded.sh` | Serial JAX-only runner with GPU/host memory caps, for large sweeps that would otherwise OOM the machine. |
| `results/` | Committed result JSON, split by device (`cpu/`, `rtx5080/`). |
| `figures/` | Generated plots referenced by the top-level README. |
| `logs/` | Run logs and `nvidia-smi` traces (gitignored). |

## Implementations compared

| Key | Implementation |
| --- | -------------- |
| `smpl_jax_smplx`, `smpl_jax_smpl` | this repo |
| `smplx_torch`, `smplx_torch_smpl` | [vchoutas/smplx](https://github.com/vchoutas/smplx) (PyTorch) |
| `smplxpp_python_smplx`, `smplxpp_python` | [sxyu/smplxpp](https://github.com/sxyu/smplxpp) (C++/CUDA) |
| `smplpytorch_torch` | [gulvarol/smplpytorch](https://github.com/gulvarol/smplpytorch) |
| `torchure_*` | [Hydran00/torchure_smplx](https://github.com/Hydran00/torchure_smplx) (C++) |

The C++ baselines are git submodules under `third_party/` and need building
first — see [`third_party/scripts/README.md`](../third_party/scripts/README.md).
Methods that are not importable are skipped rather than failing the run.

## Running

Requires model weights in `data/` and a mocap clip (see the top-level README),
plus `pip install -e ".[reference]"` for the PyTorch baselines.

The published `results/rtx5080/` numbers were measured in
[`envs/linux-cuda.yml`](../envs/linux-cuda.yml); `results/cpu/benchmark_m4_*.json`
in [`envs/macos-arm64.yml`](../envs/macos-arm64.yml). Use those to reproduce them
exactly.

```bash
# JAX vs PyTorch only, batch-size sweep, results to JSON
python benchmarks/benchmark_runtime.py \
  --method-filter jax_smplx,jax_smpl,smplx_torch \
  --batch-size-sweep \
  --batch-size-sweep-sizes 1,8,32,128,512,2048 \
  --json-out benchmarks/results/my_run.json

# Everything the machine can import
python benchmarks/benchmark_runtime.py --json-out benchmarks/results/my_run.json

python benchmarks/benchmark_runtime.py --help   # full option list
```

For long GPU sweeps, `run_jax_guarded.sh` wraps the harness in a lock file, a
`systemd-run` cgroup with memory caps, and background `nvidia-smi`/`vmstat`
logging, so a runaway allocation cannot take down the desktop session:

```bash
GPU_INDEX=0 MEM_FRACTION=0.75 bash benchmarks/run_jax_guarded.sh
```

## Plotting

```bash
pip install -e ".[plots]"

python benchmarks/plot_figures.py                      # → figures/*.png, *_dark.png, *.pdf
python benchmarks/plot_figures.py --results <file>     # plot a different run
python benchmarks/plot_figures.py --figure runtime-bar # just one figure

python benchmarks/plot_benchmark_results.py            # → results/benchmark_dashboard.html
```

Plotting reads committed result JSON, so it needs neither model weights nor a
GPU. `plot_figures.py` writes a light and a dark
variant of each figure (the README serves the right one per viewer theme) plus a
vector PDF, and it defaults to `results/rtx5080/all_methods_float32_fresh.json` —
the run the README quotes. Pass `--results` to plot any other file.

Two guards run before anything is drawn, because both failure modes flatter the
JAX numbers if left alone:

- **Model family comes from the `benchmark_family` field, never from the
  implementation name.** `smplxpp_python` and `torchure_smplx_cpp` both contain
  the substring "smplx" but are SMPL benchmarks; classifying them by name files
  them under SMPL-X.
- **Timings that fall as batch size rises are dropped, and reported on stderr.**
  Total work grows with the batch, so runtime cannot shrink. When it does, the
  timer closed before the device finished — the no-op `sync_once` path in the
  smplxpp binding measures kernel launch instead of completion. In the current
  data this removes smplxpp's batch-8,192 points, which would otherwise show it
  at ~800k FPS, a 5× lead over everything else.

Colours are assigned to implementations in a fixed order and never recycled, so
a library keeps its colour across every figure. Both themes were validated for
adjacent-pair colour-vision-deficiency separation and contrast against their own
surface; every bar carries a value label so identity and magnitude never rest on
fill colour alone.

## Fairness notes

The harness is deliberately strict about comparing like with like, because the
easy mistakes here all flatter the JAX side:

- **Same work.** Every method poses the same number of frames from the same
  clip, with the same `num_betas` / `num_expression`. `--max-frames` with
  `--tile-sequence-to-max-frames` tiles the clip rather than letting methods run
  different lengths; `--allow-mixed-sequence-lengths` must be passed explicitly
  to opt out.
- **Same protocol.** `--warmup` iterations are untimed (JIT compilation is
  excluded from the measurement, as it is amortised in real use), then the
  median of `--repeats` timed runs is reported.
- **Processing mode.** `--enforce-processing-mode-fairness` rejects runs that
  mix batched and per-frame implementations without an explicit override.
- **Precision.** JAX defaults to TF32 matmuls on GPU while the PyTorch reference
  defaults to full fp32. Results labelled `smplx_fair_fp32*` pin both sides to
  fp32; `smplx_defaults_*` compare framework defaults. Check which file you are
  reading before quoting a speedup.
- **Isolation.** `--jax-isolated-subprocess` / `--smplxpp-isolated-subprocess`
  run each framework in its own process so allocator behaviour and CUDA library
  versions do not leak across methods.

`results/rtx5080/` contains both the defaults-vs-defaults and the matched-fp32
runs so the two framings can be compared directly.
