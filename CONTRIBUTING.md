# Contributing to SMPL-JAX

Issues and pull requests are welcome.

## Development setup

```bash
git clone --recurse-submodules https://github.com/bozcomlekci/SMPL-JAX.git
cd SMPL-JAX
pip install -e ".[dev]"
```

Extras are scoped per task, so install only what you are working on:

| Extra | For |
| ----- | --- |
| `.[dev]` | the test suite |
| `.[reference]` | reference-parity tests, cross-implementation benchmarks |
| `.[plots]` | `benchmarks/` figures and dashboard |
| `.[examples]` | `examples/pose_sequence.py` |
| `.[render]` | `tools/compare_render/` GIF pipeline |
| `.[all]` | everything above |

Submodules under `third_party/` are only needed to run the cross-implementation
benchmarks and comparison renders; the library and its test suite do not
require them. To reproduce published benchmark numbers exactly, use the pinned
environments in [`envs/`](envs/README.md).

## Running the tests

```bash
pytest
```

The suite splits into two groups:

- **Synthetic tests** (`test_rotations.py`, `test_blend_shapes.py`,
  `test_kinematics.py`, `test_forward.py`, `test_inverse_lbs.py`,
  `test_model_io.py`) build small random models in `conftest.py` and need no
  downloaded weights. These always run.
- **Reference-parity tests** (`test_smplx_reference_parity.py`) run the upstream
  PyTorch `smplx` package side by side with the JAX port. They skip cleanly
  when `torch` / `smplx` are missing, and the end-to-end cases additionally skip
  when the model weights are absent from `data/`.

To exercise the full comparison, install the reference implementations and place
the model weights as described in the README:

```bash
pip install -e ".[reference]"
pytest tests/test_smplx_reference_parity.py
```

### A note on matmul precision

On GPU, JAX defaults to TF32 tensor-core matmuls, which carry roughly 1e-3
error at body scale — far above the 1e-5 tolerance the parity tests assert
against the float64 reference. `tests/conftest.py` therefore pins
`jax_default_matmul_precision="highest"` for the whole suite. If you add a test
that compares against a high-precision reference, keep that pin in place rather
than loosening the tolerance.

## Conventions

- Every operation in `smpl_jax/` must stay compatible with `jax.jit`,
  `jax.vmap`, and `jax.grad`. No Python-level loops over batch or joint axes,
  and no `.item()` / host-side branching in the forward path.
- Arrays are `float32`; indices are `int32`.
- Normalise with `safe_normalize`, never `jnp.linalg.norm`, so gradients stay
  finite at zero.
- New public symbols go in `smpl_jax/__init__.py`'s `__all__` and get a row in
  the `ARCHITECTURE.md` module reference.

`ARCHITECTURE.md` documents the forward-pass pipeline, the module layout, and
how to add a new model variant or rotation representation — read it before
making structural changes.

## Pull requests

Confirm `pytest` passes before opening a PR, and say in the description which
hardware and JAX backend you ran on.
