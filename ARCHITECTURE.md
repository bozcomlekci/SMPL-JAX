# SMPL-JAX Architecture

A fully differentiable, JIT-compiled implementation of the SMPL and SMPL-X
parametric human body models in JAX.  Every operation in the forward pass —
including shape blend shapes, pose correctives, forward kinematics, and linear
blend skinning — supports `jax.jit`, `jax.vmap`, and `jax.grad`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project layout](#2-project-layout)
3. [Forward-pass pipeline](#3-forward-pass-pipeline)
4. [Module reference](#4-module-reference)
   - [types.py](#typespycontainers)
   - [model_io.py](#model_iopyloading-pkl-files)
   - [_base.py](#_basepyshared-lbs-backbone)
   - [smpl.py / smplx.py](#smplpy--smplxpymodel-classes)
   - [rotations.py](#rotationspyrotation-representations)
   - [blend_shapes.py](#blend_shapespyblend-shapes)
   - [kinematics.py](#kinematicspyforward-kinematics)
   - [lbs.py](#lbspylinear-blend-skinning)
   - [inverse_lbs.py](#inverse_lbspyinverse-lbs)
5. [Design decisions](#5-design-decisions)
   - [Speed](#speed)
   - [Differentiability](#differentiability)
   - [Modularity](#modularity)
6. [Public API](#6-public-api)
7. [Usage examples](#7-usage-examples)
8. [Extending the library](#8-extending-the-library)

---

## 1. Overview

SMPL and SMPL-X are parametric 3D human body models defined by:

| Parameter | SMPL | SMPL-X |
|-----------|------|--------|
| Shape coefficients `betas` | (B, 10) | (B, 10) |
| Global orientation | (B, 3) | (B, 3) |
| Body pose | (B, 69) = 23 joints × 3 | (B, 63) = 21 joints × 3 |
| Expression | — | (B, 10) |
| Left-hand pose | — | (B, 45) = 15 joints × 3 |
| Right-hand pose | — | (B, 45) = 15 joints × 3 |
| Translation | (B, 3) | (B, 3) |

Outputs are posed mesh vertices and joint positions:

| Output | SMPL | SMPL-X |
|--------|------|--------|
| Vertices | (B, 6890, 3) | (B, 10475, 3) |
| Joints | (B, 24, 3) | (B, 55, 3) |

All pose parameters are axis-angle vectors; magnitude equals the rotation angle
in radians.

---

## 2. Project layout

```
smpl_jax/
├── __init__.py          public API surface
├── types.py             SMPLParams, SMPLXParams, SMPLOutput, SMPLXOutput
├── model_io.py          .pkl loader (normalises all known file variants)
├── _base.py             _SMPLBase — shared 7-step LBS forward pass
├── smpl.py              SMPLModel  (inherits _SMPLBase)
├── smplx.py             SMPLXModel (inherits _SMPLBase)
├── rotations.py         axis-angle ↔ rotmat ↔ 6D, safe_normalize
├── blend_shapes.py      shape / expression / pose blend shapes
├── kinematics.py        FK via jax.lax.scan
└── lbs.py               lbs_transforms, lbs (compact 3×4 format)

tests/
├── conftest.py          synthetic model fixtures (no .pkl files needed)
├── test_rotations.py
├── test_blend_shapes.py
├── test_kinematics.py
├── test_forward.py
└── test_inverse_lbs.py
```

---

## 3. Forward-pass pipeline

Both `SMPLModel` and `SMPLXModel` execute the same seven-step pipeline, defined
once in `_base.py`.  SMPL-X differences are injected through two hooks.

```
params (betas, body_pose, global_orient, transl, [expression, hand_poses])
   │
   ▼  ──────────────────────────────────────────────────────── step 1
shape_blend_shapes(v_template, shapedirs, betas)           ← SMPLModel
   + expression_blend_shapes(v_shaped, exprdirs, expression) ← SMPLXModel only
   │  v_shaped  (B, V, 3)
   │
   ▼  ──────────────────────────────────────────────────────── step 2
einsum("jv,bvd->bjd", J_regressor, v_shaped)
   │  joints  (B, J, 3)   bind-pose joint positions
   │
   ▼  ──────────────────────────────────────────────────────── step 3
_build_rotmats(params, B)                                  ← per-subclass
   │  rotmats  (B, J, 3, 3)
   │
   ▼  ──────────────────────────────────────────────────────── step 4
fk_forward_batched(rotmats, joints, parents)
   │  G  (B, J, 4, 4)   global rigid-body transforms in SE(3)
   │
   ▼  ──────────────────────────────────────────────────────── step 5
pose_blend_shapes(rotmats, posedirs)
   │  pose_corr  (B, V, 3)   corrective displacements
   │
   ▼  ──────────────────────────────────────────────────────── step 6
M = lbs_transforms(G, joints)          (B, J, 3, 4) compact [R|t]
vertices = lbs(v_shaped, pose_corr, M, weights)
   │  vertices  (B, V, 3)
   │
   ▼  ──────────────────────────────────────────────────────── step 7
vertices    += transl[:, None, :]
posed_joints = G[..., :3, 3] + transl[:, None, :]
   │
   ▼
SMPLOutput / SMPLXOutput (vertices, joints)
```

---

## 4. Module reference

### `types.py` — Containers

All containers are `NamedTuple` subclasses, making them valid JAX pytrees
automatically.  They pass through `jax.jit`, `jax.vmap`, and `jax.grad`
without any registration boilerplate.

```python
class SMPLParams(NamedTuple):
    betas:         jnp.ndarray  # (B, num_betas)
    body_pose:     jnp.ndarray  # (B, 69)  — 23 joints × 3 axis-angle
    global_orient: jnp.ndarray  # (B, 3)
    transl:        jnp.ndarray  # (B, 3)

class SMPLXParams(NamedTuple):
    betas:           jnp.ndarray  # (B, num_betas)
    body_pose:       jnp.ndarray  # (B, 63)  — 21 joints × 3 axis-angle
    global_orient:   jnp.ndarray  # (B, 3)
    transl:          jnp.ndarray  # (B, 3)
    expression:      jnp.ndarray  # (B, num_expression_coeffs)
    left_hand_pose:  jnp.ndarray  # (B, 45) — 15 joints × 3
    right_hand_pose: jnp.ndarray  # (B, 45) — 15 joints × 3
```

---

### `model_io.py` — Loading `.pkl` files

`load_model_data(path)` reads a SMPL or SMPL-X `.pkl` file and returns a
normalised `dict` of `numpy` arrays regardless of which model version produced
the file.

Handled variations:

| Field | Known layouts | Normalised to |
|-------|---------------|---------------|
| `shapedirs` | `(V*3, K)` or `(V, 3, K)` | `(V, 3, K)` |
| `posedirs` | `(V, 3, P)`, `(V*3, P)`, or `(P, V*3)` | `(V*3, P)` |
| `J_regressor` | dense `(J, V)` or `scipy.sparse` | dense `(J, V)` float32 |
| `exprdirs` | key `"expr_dirs"` or `"exprdirs"` | `(V, 3, E)` or `None` |

All float arrays are cast to `float32`; index arrays to `int32`.

---

### `_base.py` — Shared LBS backbone

`_SMPLBase` defines the complete seven-step forward pass and the common
constructor.  Concrete models only need to supply:

| Hook | Purpose |
|------|---------|
| `_output_cls` | class attribute — `SMPLOutput` or `SMPLXOutput` |
| `_vertex_blend_shapes(params)` | returns shaped vertices `(B, V, 3)`; base adds expression in SMPL-X |
| `_build_rotmats(params, B)` | returns `(B, J, 3, 3)` rotation matrices |

**Unbatched input handling** — the forward pass detects single-sample inputs
(`params.betas.ndim == 1`) and wraps them in a batch-1 dimension, enabling
both `model(params)` for a single sample and `jax.vmap(model)(params)` for an
outer batch, without duplication.

---

### `smpl.py` / `smplx.py` — Model classes

Both classes inherit `_SMPLBase` and contribute only their per-model logic:

**`SMPLModel._build_rotmats`** — concatenates root and 23 body joint rotations:
```
axis_angle → (B, 1, 3, 3)   root
axis_angle → (B, 23, 3, 3)  body
  concat   → (B, 24, 3, 3)
```

**`SMPLXModel._vertex_blend_shapes`** — adds expression on top of the base
shape blend shapes.

**`SMPLXModel._build_rotmats`** — assembles the 55-joint rotation tensor:
```
axis_angle → (B,  1, 3, 3)  root
axis_angle → (B, 21, 3, 3)  body
identity   → (B,  3, 3, 3)  face joints (jaw, left-eye, right-eye)
axis_angle → (B, 15, 3, 3)  left hand
axis_angle → (B, 15, 3, 3)  right hand
  concat   → (B, 55, 3, 3)
```

Face joints are always identity; they are not exposed as input parameters in
the base SMPL-X model.

---

### `rotations.py` — Rotation representations

All functions are `jax.jit`, `jax.vmap`, and `jax.grad` compatible.

#### `safe_normalize(x, axis=-1, eps=1e-12)`

```
x / sqrt(sum(x²) + eps)
```

Gradient-safe L2 normalisation.  `jnp.linalg.norm` has an undefined gradient
at zero (`0/0`); this formulation keeps the gradient finite everywhere by
shifting the squared norm before the sqrt.  Used throughout the library
wherever a vector is normalised.

#### `axis_angle_to_rotmat(aa)` — `(..., 3) → (..., 3, 3)`

Rodrigues' rotation formula:
```
R = cos(θ)·I + (1−cos(θ))·(n⊗n) + sin(θ)·K(n)
```
where `θ = ‖aa‖`, `n = aa/θ`, and `K(n)` is the skew-symmetric matrix for
the cross-product with `n`.

Gradient stability: the norm is computed as `sqrt(‖aa‖² + ε)` rather than
`‖aa‖`, avoiding `0/0` at `aa = 0`.

#### `rotmat_to_axis_angle(R)` — `(..., 3, 3) → (..., 3)`

Extracts axis-angle from a rotation matrix via the trace formula.  Handles
near-identity matrices via a zero-threshold guard.

#### `rotmat_to_6d(R)` — `(..., 3, 3) → (..., 6)`

Returns the first two columns of `R` (Zhou et al., 2019).  The six values
uniquely determine `R` up to SO(3) and form a topologically correct continuous
parameterisation — unlike axis-angle or quaternions, which have discontinuities.

#### `rotation_6d_to_rotmat(r6d)` — `(..., 6) → (..., 3, 3)`

Recovers the full rotation matrix via Gram-Schmidt orthogonalisation:
```
b1 = safe_normalize(a1)
b2 = safe_normalize(a2 − (b1·a2)·b1)
b3 = b1 × b2
R  = [b1 | b2 | b3]
```

---

### `blend_shapes.py` — Blend shapes

#### `shape_blend_shapes(v_template, shapedirs, betas)` → `(B, V, 3)`

```
v_shaped = v_template + einsum("vcp,bp->bvc", shapedirs, betas)
```

Linear combination of shape blend shape directions, added to the template mesh.

#### `expression_blend_shapes(v_shaped, exprdirs, expression)` → `(B, V, 3)`

Same operation as shape blend shapes, applied on top of the shaped vertices.
SMPL-X only.

#### `pose_blend_shapes(rotmats, posedirs)` → `(B, V, 3)`

Pose corrective blend shapes driven by the difference between the current joint
rotations and the identity:
```
pose_feat = (R_i − I)_{i=1..J−1}  flattened to (B, P)
correctives = pose_feat @ posedirs.T  →  (B, V*3)  →  (B, V, 3)
```

`posedirs` has `P ≤ (J−1)×9` columns; the feature vector is trimmed to match.

---

### `kinematics.py` — Forward kinematics

#### `fk_forward(rotmats, joints, parents)` → `(J, 4, 4)`

Computes global rigid-body transforms for all joints via `jax.lax.scan` over
the (topologically ordered) joint list.

**Key implementation choices:**

*Compact 3×4 scan carry* — transforms are accumulated as `(J, 3, 4) = [R|t]`
(the bottom `[0,0,0,1]` row is implicit).  This is 25% smaller than a `(J, 4,
4)` carry and replaces a `4×4 @ 4×4` multiply (64 ops) with a `3×3 @ 3×3 +
3-vec` composition (≈15 ops) per step.

*Dynamic gather inside scan* — `G_all[safe_parents[i]]` dynamically indexes
into the running carry array.  This is a standard JAX scatter/gather pattern
that XLA compiles efficiently.

*Output format* — the `(J, 3, 4)` result is padded to `(J, 4, 4)` at the end
for API compatibility; the bottom row cost is paid once rather than J times.

```python
# Each scan step (single joint i):
R_g = where(is_root, R_l,  R_parent @ R_local)
t_g = where(is_root, t_l,  R_parent @ t_local + t_parent)
```

#### `fk_forward_batched(rotmats, joints, parents)` → `(B, J, 4, 4)`

`jax.vmap` wrapper over `fk_forward`.

---

### `lbs.py` — Linear blend skinning

#### `lbs_transforms(G, joints)` → `(B, J, 3, 4)`

Computes per-joint LBS transforms relative to the bind pose:
```
M_i = G_i @ G_i_bind⁻¹  =  [R_i | t_i − R_i @ j_i]
```

Returns compact `(B, J, 3, 4)` = `[R | t_rel]`; the implicit bottom row
`[0,0,0,1]` is never materialised.

#### `lbs(v_shaped, pose_correctives, M, weights)` → `(B, V, 3)`

Applies linear blend skinning without homogeneous coordinates:
```
v = v_shaped + pose_correctives
R_v = einsum("vj,bjrc->bvrc", weights, M[..., :3])   # (B, V, 3, 3)
t_v = einsum("vj,bjr->bvr",  weights, M[...,  3])    # (B, V, 3)
output = einsum("bvrc,bvc->bvr", R_v, v) + t_v
```

This avoids the `(B, V, 4, 4)` intermediate that the naïve homogeneous
formulation would create (for V = 10475, B = 8 that is 3.5 M extra floats).

---

### `inverse_lbs.py` — Inverse LBS

Recovers per-joint rotation matrices from a posed mesh in two stages.
Operates on a **single sample**; use `jax.vmap` for batching.

#### Stage 1 — Analytical init: `analytical_init(posed_verts, v_template, weights)`

For each joint `j`:
1. Weight the vertex cloud by `weights[:, j]` (Kabsch weighting).
2. Centre both the posed and bind-pose clouds.
3. Compute the weighted cross-covariance matrix `H`.
4. Extract the optimal rotation via SVD: `R = Vt.T @ diag(1,1,det) @ U.T`.
5. Polish with Newton-Schulz iteration until `R` is orthogonal.

All per-joint computations are vectorised with `jax.vmap`.

#### Newton-Schulz: `_newton_schulz(A, num_iter=10)`

Finds the nearest orthogonal matrix via the iteration:
```
X₀ = A / ‖A‖_F
Xₖ₊₁ = 1.5·Xₖ − 0.5·Xₖ·(Xₖᵀ·Xₖ)
```

Run inside `jax.lax.scan` for JIT compatibility.  `safe_normalize` provides
gradient-safe Frobenius-norm normalisation for the initial scaling.

#### Stage 2 — Autograd refinement: `autograd_refine(...)`

Minimises vertex reconstruction error in the 6D rotation space:
```
r6d* = argmin_r6d  ‖LBS(rotation_6d_to_rotmat(r6d)) − posed_verts‖²
```

Optimised with Adam (`optax.adam`) inside `jax.lax.scan` so the entire
optimisation loop is a single JIT-compiled operation.

#### `inverse_lbs(posed_verts, model, num_refine_iters=50, lr=1e-3)`

Public entry point combining both stages.

---

## 5. Design decisions

### Speed

| Decision | Rationale |
|----------|-----------|
| `float32` enforced at model construction | Prevents silent float64 promotion, keeps memory bandwidth low |
| FK scan carry: `(J, 3, 4)` not `(J, 4, 4)` | 25% smaller carry; 4× fewer FLOPs per composition step |
| `lbs_transforms` returns `(B, J, 3, 4)` | Bottom row `[0,0,0,1]` never materialised |
| `lbs` blends R and t separately | No `(B, V, 4, 4)` intermediate; no homogeneous coordinate allocation |
| All loops inside `jax.lax.scan` | Single JIT-compiled kernel; no Python overhead at run-time |
| `jax.vmap` for per-joint operations | Maps over vectorised hardware instructions |
| Model arrays stored as `jnp.array` | JIT closures capture them as compile-time constants |

### Differentiability

| Issue | Fix |
|-------|-----|
| `jnp.linalg.norm` gradient is `0/0` at zero | Replaced by `sqrt(sum(x²) + ε)` via `safe_normalize` |
| `jnp.where` evaluates both branches | Accepted; both branches are valid computations |
| `jax.lax.scan` inside Adam loop | Entire optimisation is one differentiable graph |
| Newton-Schulz orthogonalisation | Differentiable — no `jnp.linalg.qr` or SVD in the loop |
| Kabsch SVD in stage 1 | JAX's `jnp.linalg.svd` is differentiable |

### Modularity

The `_SMPLBase` base class encodes the seven-step pipeline once.  Concrete
models provide three things:

1. **`_output_cls`** — the NamedTuple type returned by `forward`.
2. **`_vertex_blend_shapes(params)`** — SMPL returns shaped vertices; SMPL-X
   adds expression on top via `super()`.
3. **`_build_rotmats(params, B)`** — encapsulates the joint count and parameter
   names of each model variant.

Adding a new model variant (e.g. SMPL+H, FLAME) requires only a new subclass
with these three pieces; the entire LBS backbone, I/O handling, and JIT
compatibility are inherited automatically.

---

## 6. Public API

```python
from smpl_jax import (
    # Models
    SMPLModel,
    SMPLXModel,

    # Parameter containers (JAX pytrees)
    SMPLParams,
    SMPLXParams,

    # Output containers (JAX pytrees)
    SMPLOutput,
    SMPLXOutput,

    # Inverse kinematics
    inverse_lbs,

    # Rotation utilities
    axis_angle_to_rotmat,
    rotmat_to_axis_angle,
    rotmat_to_6d,
    rotation_6d_to_rotmat,
    safe_normalize,
)
```

---

## 7. Usage examples

### Basic forward pass

```python
import jax
import jax.numpy as jnp
from smpl_jax import SMPLXModel, SMPLXParams

model = SMPLXModel.load("data/smplx/SMPLX_NEUTRAL.pkl")

params = SMPLXParams(
    betas=jnp.zeros((4, 10)),
    body_pose=jnp.zeros((4, 63)),
    global_orient=jnp.zeros((4, 3)),
    transl=jnp.zeros((4, 3)),
    expression=jnp.zeros((4, 10)),
    left_hand_pose=jnp.zeros((4, 45)),
    right_hand_pose=jnp.zeros((4, 45)),
)

forward = jax.jit(model)
out = forward(params)
# out.vertices.shape → (4, 10475, 3)
# out.joints.shape   → (4, 55, 3)
```

### JIT + gradient through pose

```python
@jax.jit
def pose_loss(body_pose):
    p = params._replace(body_pose=body_pose)
    v = model(p).vertices
    return jnp.mean(v[..., 1] ** 2)   # minimise y-coordinate

grad = jax.grad(pose_loss)(params.body_pose)
```

### Batching with vmap

```python
# vmap over an outer dataset dimension
dataset_params = SMPLXParams(
    betas=jnp.zeros((100, 4, 10)),     # 100 sequences × batch 4
    ...
)
per_sequence = jax.vmap(jax.jit(model))(dataset_params)
# per_sequence.vertices.shape → (100, 4, 10475, 3)
```

### Single-sample call (no batch dimension)

```python
single = SMPLXParams(
    betas=jnp.zeros(10),
    body_pose=jnp.zeros(63),
    global_orient=jnp.zeros(3),
    transl=jnp.zeros(3),
    expression=jnp.zeros(10),
    left_hand_pose=jnp.zeros(45),
    right_hand_pose=jnp.zeros(45),
)
out = model(single)
# out.vertices.shape → (10475, 3)   ← batch dim removed automatically
```

### Inverse LBS

```python
from smpl_jax import SMPLModel, inverse_lbs

smpl = SMPLModel.load("data/smpl/SMPL_NEUTRAL.pkl")
posed_verts = ...   # (6890, 3)

R_joints = inverse_lbs(posed_verts, smpl, num_refine_iters=100, lr=1e-3)
# R_joints.shape → (24, 3, 3)

# Batched inversion
batch_verts = ...   # (B, 6890, 3)
R_batch = jax.vmap(inverse_lbs, in_axes=(0, None))(batch_verts, smpl)
```

### Rotation representation conversion

```python
from smpl_jax import (
    axis_angle_to_rotmat,
    rotmat_to_6d,
    rotation_6d_to_rotmat,
    safe_normalize,
)

aa = jnp.array([0.0, 0.0, jnp.pi / 2])
R  = axis_angle_to_rotmat(aa)          # (3, 3)
r6 = rotmat_to_6d(R)                   # (6,)  — continuous representation
R2 = rotation_6d_to_rotmat(r6)         # (3, 3)

# Custom loss using safe_normalize
def orientation_loss(aa):
    R = jax.vmap(axis_angle_to_rotmat)(aa)    # (B, 3, 3)
    up = R[:, :, 1]                           # (B, 3)  y-column
    target = safe_normalize(jnp.array([[0., 1., 0.]]))
    return jnp.mean((up - target) ** 2)

grad = jax.grad(orientation_loss)(aa_batch)
```

---

## 8. Extending the library

### Adding a new model variant (e.g. SMPL+H, FLAME)

1. Create a new parameter container in `types.py`:
   ```python
   class MyModelParams(NamedTuple):
       betas: jnp.ndarray
       ...
   ```

2. Create a new output container:
   ```python
   class MyModelOutput(NamedTuple):
       vertices: jnp.ndarray
       joints: jnp.ndarray
   ```

3. Subclass `_SMPLBase`:
   ```python
   from smpl_jax._base import _SMPLBase

   class MyModel(_SMPLBase):
       _output_cls = MyModelOutput

       @classmethod
       def load(cls, path, **kwargs):
           data = load_model_data(path)
           return cls(...)

       def _build_rotmats(self, params, B):
           # assemble (B, J, 3, 3) from params
           ...
   ```

4. Optionally override `_vertex_blend_shapes` if the model has additional
   per-vertex blend shapes beyond shape/expression.

The full LBS pipeline, JIT compatibility, unbatched-input handling, and
differentiability are all inherited.

### Adding a new rotation representation

Add a conversion function to `rotations.py`.  Follow the pattern:
- Accept `(..., N)` input (arbitrary leading batch dimensions).
- Return `(..., M)` output.
- Use `safe_normalize` wherever L2 normalisation is needed to preserve gradient
  flow through `jax.grad`.
- Add the function to `__all__` in `__init__.py`.
