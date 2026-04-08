# SMPL-JAX

**Fully differentiable, JIT-compiled implementations of SMPL and SMPL-X in JAX.**

SMPL-JAX provides a clean, hardware-accelerated JAX port of the [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/) parametric human body models. Every operation — shape blend shapes, forward kinematics, linear blend skinning, and pose inversion — is compatible with `jax.jit`, `jax.vmap`, and `jax.grad`, enabling large-scale batched fitting, differentiable optimization, and humanoid robotics pipelines.

This library is the foundational building block for [SOMA-JAX](https://github.com/batuhanozcomekcii/SOMA-JAX), a full JAX-native alternative to NVIDIA SOMA-X.

---

## Features

- **SMPL-X forward pass** — shape/expression blend shapes, FK via `lax.scan`, LBS skinning
- **SMPL forward pass** — lightweight 6,890-vertex model sharing the same FK/LBS core
- **Pose representations** — axis-angle ↔ rotation matrix ↔ 6D continuous (Gram-Schmidt), all differentiable
- **Inverse-LBS** — analytical pose abstraction (Newton-Schulz orthogonalization) + Adam-based autograd refinement via `optax`
- **Fully batched** — `vmap` over arbitrary batch dimensions with no Python loops
- **Pure JAX** — no PyTorch, no CUDA extensions; runs on GPU, CPU, and TPU

---

## Supported Models

| Model | Vertices | Shape Components | Expression | Hands |
|-------|----------|-----------------|------------|-------|
| SMPL | 6,890 | 10 | ✗ | ✗ |
| SMPL-X | 10,475 | 300 | 50 | ✓ (15×2) |

---

## Installation

```bash
git clone https://github.com/batuhanozcomekcii/SMPL-JAX.git
cd SMPL-JAX
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, JAX ≥ 0.4.30 (GPU: install `jaxlib` with CUDA 12 support), `optax`, `flax`.

Download model weights from the [SMPL-X project page](https://smpl-x.is.tue.mpg.de/) and place them in `data/`:

```
data/
  smplx/
    SMPLX_NEUTRAL.pkl
    SMPLX_MALE.pkl
    SMPLX_FEMALE.pkl
  smpl/
    SMPL_NEUTRAL.pkl
```

---

## Quickstart

```python
import jax
import jax.numpy as jnp
from smpl_jax import SMPLXModel, SMPLXParams

# Load model
model = SMPLXModel.load("data/smplx/SMPLX_NEUTRAL.pkl")

# Define parameters (batch size 8)
params = SMPLXParams(
    betas=jnp.zeros((8, 10)),
    body_pose=jnp.zeros((8, 63)),    # 21 joints × 3 axis-angle
    global_orient=jnp.zeros((8, 3)),
    transl=jnp.zeros((8, 3)),
    expression=jnp.zeros((8, 10)),
    left_hand_pose=jnp.zeros((8, 45)),
    right_hand_pose=jnp.zeros((8, 45)),
)

# JIT-compiled forward pass
forward = jax.jit(model)
output = forward(params)

print(output.vertices.shape)  # (8, 10475, 3)
print(output.joints.shape)    # (8, 144, 3)
```

### Batched fitting with `vmap`

```python
# Fit pose for 1024 subjects in parallel
batched_forward = jax.vmap(model)
output = batched_forward(large_batch_params)  # (1024, 10475, 3)
```

### Differentiable optimization

```python
import optax

def loss_fn(theta, target_joints):
    params = SMPLXParams(body_pose=theta, ...)
    out = model(params)
    return jnp.mean((out.joints - target_joints) ** 2)

grad_fn = jax.jit(jax.value_and_grad(loss_fn))
optimizer = optax.adam(1e-3)
```

---

## Architecture

```
SMPLXModel.forward(params)
│
├── shape_blend_shapes(betas, shapedirs)          → v_shaped
├── expression_blend_shapes(expression, expr_dirs) → v_shaped
├── lbs_joints(v_shaped, J_regressor)              → joints (bind pose)
├── axis_angle_to_rotmat(body_pose)                → rotmats (B, J, 3, 3)
├── fk_forward(rotmats, joints, parents)           → global_transforms  [lax.scan]
├── pose_blend_shapes(rotmats, posedirs)           → pose_correctives
└── lbs(v_shaped, pose_correctives,
        global_transforms, lbs_weights)            → vertices (B, N, 3)
```

### Pose Inversion

```
inverse_lbs(posed_verts, model)
│
├── skeleton_transfer(posed_verts)    → T_init  [Kabsch + Newton-Schulz]
└── autograd_refine(T_init, ...)      → rotmats [Adam via optax, lax.fori_loop]
```

---

## Benchmarks

Measured on RTX 5080, batch size 128, FP32.

| Operation | PyTorch smplx | SMPL-JAX (jit+vmap) | Speedup |
|-----------|--------------|---------------------|---------|
| SMPL-X forward | — | — | TBD |
| Batched LBS | — | — | TBD |
| Pose inversion (analytical) | — | — | TBD |

> Benchmarks will be populated after Phase 1–2 implementation. Target: match or exceed SOMA-X Warp analytical solver at 882 FPS on A100.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | SMPL-X forward pass (FK, LBS, blend shapes) | 🔲 In progress |
| 2 | SMPL forward pass | 🔲 Planned |
| 3 | Pose representations (AA ↔ rotmat ↔ 6D) | 🔲 Planned |
| 4 | Inverse-LBS (analytical + autograd) | 🔲 Planned |
| 5 | Unit tests vs PyTorch smplx reference | 🔲 Planned |

---

## Relation to SOMA-JAX

SMPL-JAX exposes only the per-model forward pass and pose inversion. **SOMA-JAX** builds on top to provide:
- 3D barycentric mesh topology abstraction (multi-model unification)
- RBF skeletal abstraction
- Pose-dependent correctives MLP (Flax)
- Multi-backend registry (SMPL, SMPL-X, MHR, Anny)
- SOMA-X compatible `SOMALayer` API

See [SOMA-JAX](https://github.com/batuhanozcomekcii/SOMA-JAX) for the full pipeline.

---

## Contributing

PRs and issues welcome. Please run `pytest tests/` and confirm all reference tests pass before opening a PR.

---

## License

Model weights are subject to their respective licenses from MPI-IS. Code in this repository is MIT licensed.

---

## Citation

If you use SMPL-JAX in your research, please also cite the original SMPL-X work:

```bibtex
@inproceedings{SMPL-X:2019,
  title     = {Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author    = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and
               Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {CVPR},
  year      = {2019}
}
```
