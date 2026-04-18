# Third-party patch bundle

## smplxpp_cuda_build_fix.patch

Purpose:
- Avoid nvcc crashes during smplxpp CUDA builds by limiting CUDA translation units to CUDA method instantiations.
- Keep `set_zero` / `set_random` host-only for Body parameter helpers.
- Add `SMPLXPP_USE_CUDA` -> `SMPLX_USE_CUDA` wiring in `setup.py` for explicit CUDA on/off control.
- Add fused tensor batch-forward bindings in `pybind.cpp` (`FusedBatchForwardS` / `FusedBatchForwardX`).
- Add stream-aware CUDA update plumbing and per-slot CUDA stream dispatch for fused batch inference.

Apply:

```bash
cd third_party/smplxpp
git apply ../patches/smplxpp_cuda_build_fix.patch
```

If the patch was already applied, `git apply` may fail. In that case, confirm current state with:

```bash
git diff -- include/smplx/smplx.hpp pybind.cpp setup.py src/body.cpp src/cuda/body.cu src/cuda/model.cu
```

## torchure_smplx_dtype_benchmark.patch

Purpose:
- Add explicit dtype selection (`--dtype float32|float64`) in torchure benchmark CLI.
- Wire dtype through model loading/options so tensor dtype is honored end-to-end.
- Keep benchmark output structured for runtime parser compatibility.

Apply:

```bash
cd third_party/torchure_smplx
git apply ../patches/torchure_smplx_dtype_benchmark.patch
```

If the patch was already applied, `git apply` may fail. In that case, confirm current state with:

```bash
git diff -- include/smplx/smplx.hpp samples/benchmark.cpp src/smplx/smplx.cpp
```
