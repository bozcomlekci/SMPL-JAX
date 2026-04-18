# third_party scripts

This folder contains helper scripts for third-party dependency maintenance.

## Available scripts

- rebuild_smplxpp_cuda.sh
  - Applies third_party/patches/smplxpp_cuda_build_fix.patch if needed.
  - Rebuilds and reinstalls smplxpp with CUDA in the current machine setup.
  - Verifies runtime CUDA availability via smplxpp.cuda.

- rebuild_torchure_smplx.sh
  - Applies third_party/patches/torchure_smplx_dtype_benchmark.patch if needed.
  - Configures and builds torchure_smplx benchmark target using Torch CMake prefix from a conda env.
  - Produces benchmark binary at third_party/torchure_smplx/build/benchmark.

## Usage

From the repository root:

```bash
bash third_party/scripts/rebuild_smplxpp_cuda.sh
bash third_party/scripts/rebuild_torchure_smplx.sh
```

Optional environment overrides:

- CUDA_HOME (default: /usr/local/cuda-12.4)
- CC (default: /usr/bin/gcc-13)
- CXX (default: /usr/bin/g++-13)
- CUDAHOSTCXX (default: CXX)
- CUDACXX (default: CUDA_HOME/bin/nvcc)
- CONDA_BIN (default: auto-detected from PATH via `command -v conda`)
- CONDA_ENV (default: body)

Additional overrides for rebuild_torchure_smplx.sh:

- CMAKE_PREFIX_PATH (auto-resolved from torch if unset)
- BUILD_DIR (default: third_party/torchure_smplx/build)
- BUILD_TESTS (default: OFF)
- USE_OPEN3D (default: OFF)
- JOBS (default: nproc)
