"""
Inverse LBS: recover per-joint rotation matrices from posed vertices.

Two-stage pipeline (single sample; use jax.vmap for batching):

  Stage 1 – Analytical initialisation
      For each joint, fit a rotation via weighted Kabsch alignment of the
      influenced vertex cloud, then orthogonalise with Newton-Schulz.

  Stage 2 – Autograd refinement
      Represent rotations in the 6D continuous space (Zhou et al., 2019)
      and minimise vertex reconstruction error with Adam (optax), all inside
      a jax.lax.scan loop for full JIT compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from .rotations import rotation_6d_to_rotmat, rotmat_to_6d
from .kinematics import fk_forward
from .lbs import lbs_transforms, lbs


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation
# ---------------------------------------------------------------------------

def _newton_schulz(A: jnp.ndarray, num_iter: int = 10) -> jnp.ndarray:
    """Find the nearest orthogonal matrix to A via Newton-Schulz iteration.

    Iteration: X_{k+1} = X_k @ (1.5·I − 0.5·X_k.T @ X_k)

    Args:
        A: (3, 3) matrix to orthogonalise.
        num_iter: number of iterations (10 is usually sufficient).

    Returns:
        (3, 3) orthogonal matrix (det ≈ ±1).
    """
    # sqrt(sum²+ε) keeps the gradient finite everywhere (linalg.norm has 0/0 at zero)
    X = A / jnp.sqrt(jnp.sum(A * A, axis=(-2, -1), keepdims=True) + 1e-12)
    I = jnp.eye(3, dtype=A.dtype)

    def step(X: jnp.ndarray, _: None):
        return 1.5 * X - 0.5 * (X @ (X.T @ X)), None

    X, _ = jax.lax.scan(step, X, None, length=num_iter)
    return X


# ---------------------------------------------------------------------------
# Stage 1: analytical initialisation via per-joint Kabsch
# ---------------------------------------------------------------------------

def analytical_init(
    posed_verts: jnp.ndarray,
    v_template: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate per-joint rotations analytically.

    For each joint j:
      1. Weight the vertex clouds by LBS influence w_j.
      2. Align the posed cloud to the bind-pose cloud with Kabsch (SVD).
      3. Polish the result with Newton-Schulz.

    Args:
        posed_verts: (V, 3) target posed vertices (single sample).
        v_template:  (V, 3) bind-pose template vertices.
        weights:     (V, J) skinning weights.

    Returns:
        (J, 3, 3) initial rotation estimate.
    """
    J = weights.shape[1]

    def per_joint(j: jnp.ndarray) -> jnp.ndarray:
        w = weights[:, j]                                          # (V,)
        w_norm = w / (w.sum() + 1e-8)

        c_bind  = jnp.einsum("v,vd->d", w_norm, v_template)       # (3,)
        c_posed = jnp.einsum("v,vd->d", w_norm, posed_verts)      # (3,)

        P = posed_verts - c_posed                                  # (V, 3)
        Q = v_template - c_bind                                    # (V, 3)

        # Weighted cross-covariance matrix
        H = jnp.einsum("v,vr,vc->rc", w_norm, Q, P)               # (3, 3)

        U, _, Vt = jnp.linalg.svd(H)
        d = jnp.linalg.det(Vt.T @ U.T)
        D = jnp.eye(3, dtype=H.dtype).at[2, 2].set(d)
        return Vt.T @ D @ U.T

    R_init = jax.vmap(per_joint)(jnp.arange(J))                   # (J, 3, 3)
    R_init = jax.vmap(_newton_schulz)(R_init)                     # (J, 3, 3)
    return R_init


# ---------------------------------------------------------------------------
# Stage 2: autograd refinement with Adam (inside lax.scan)
# ---------------------------------------------------------------------------

def autograd_refine(
    R_init: jnp.ndarray,
    posed_verts: jnp.ndarray,
    v_template: jnp.ndarray,
    weights: jnp.ndarray,
    joints: jnp.ndarray,
    parents: jnp.ndarray,
    num_iters: int = 50,
    lr: float = 1e-3,
) -> jnp.ndarray:
    """Refine rotation matrices via Adam in the 6D representation.

    Args:
        R_init:      (J, 3, 3) initial rotation matrices.
        posed_verts: (V, 3)    target posed vertices.
        v_template:  (V, 3)    bind-pose template vertices.
        weights:     (V, J)    skinning weights.
        joints:      (J, 3)    bind-pose joint positions.
        parents:     (J,)      parent indices.
        num_iters:   Adam iterations.
        lr:          learning rate.

    Returns:
        (J, 3, 3) refined rotation matrices.
    """
    r6d_init = jax.vmap(rotmat_to_6d)(R_init)                     # (J, 6)

    def forward_from_6d(r6d: jnp.ndarray) -> jnp.ndarray:
        R = jax.vmap(rotation_6d_to_rotmat)(r6d)                  # (J, 3, 3)
        G = fk_forward(R, joints, parents)                         # (J, 4, 4)
        M = lbs_transforms(G[None], joints[None])                  # (1, J, 3, 4)
        v_posed = lbs(
            v_template[None],
            jnp.zeros_like(v_template[None]),
            M,
            weights,
        )[0]                                                        # (V, 3)
        return v_posed

    def loss_fn(r6d: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((forward_from_6d(r6d) - posed_verts) ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(r6d_init)

    def opt_step(carry, _):
        r6d, state = carry
        loss, grads = jax.value_and_grad(loss_fn)(r6d)
        updates, state = optimizer.update(grads, state)
        r6d = optax.apply_updates(r6d, updates)
        return (r6d, state), loss

    (r6d_final, _), _ = jax.lax.scan(
        opt_step, (r6d_init, opt_state), None, length=num_iters
    )
    return jax.vmap(rotation_6d_to_rotmat)(r6d_final)             # (J, 3, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inverse_lbs(
    posed_verts: jnp.ndarray,
    model,
    num_refine_iters: int = 50,
    lr: float = 1e-3,
) -> jnp.ndarray:
    """Full inverse-LBS pipeline for a single posed mesh.

    Args:
        posed_verts:      (V, 3) posed vertex positions.
        model:            SMPLModel or SMPLXModel instance.
        num_refine_iters: Adam iterations for stage-2 refinement.
        lr:               Adam learning rate.

    Returns:
        (J, 3, 3) recovered per-joint rotation matrices.

    Note:
        Use jax.vmap(inverse_lbs, in_axes=(0, None))(batch_verts, model)
        for batched inversion.
    """
    bind_joints = model.J_regressor @ model.v_template             # (J, 3)

    R_init = analytical_init(
        posed_verts, model.v_template, model.weights
    )

    R_refined = autograd_refine(
        R_init,
        posed_verts,
        model.v_template,
        model.weights,
        bind_joints,
        model.parents,
        num_iters=num_refine_iters,
        lr=lr,
    )
    return R_refined
