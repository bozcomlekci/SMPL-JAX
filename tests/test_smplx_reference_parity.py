"""
Numerical parity against the reference PyTorch ``smplx`` package.

The rest of the suite exercises SMPL-JAX on synthetic fixtures; this file runs
the *actual* upstream implementation (`smplx.lbs`, `smplx.SMPL`, `smplx.SMPLX`)
side by side with the JAX port on identical inputs, primitive by primitive and
then end to end.

Skips cleanly when ``smplx`` / ``torch`` are not installed, and the end-to-end
model tests additionally skip when the model weights are absent from ``data/``.

Tolerance: SMPL-JAX pins float32 internally while the reference runs float64
here, so agreement is bounded by float32 round-off (~1e-6 at body scale).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
smplx = pytest.importorskip("smplx")
RL = pytest.importorskip("smplx.lbs")

from smpl_jax import rotations as JRot          # noqa: E402
from smpl_jax.blend_shapes import pose_blend_shapes, shape_blend_shapes  # noqa: E402
from smpl_jax.kinematics import fk_forward_batched  # noqa: E402
from smpl_jax.lbs import lbs as jax_lbs         # noqa: E402
from smpl_jax.lbs import lbs_transforms         # noqa: E402
from smpl_jax.smpl import SMPLModel             # noqa: E402
from smpl_jax.smplx import SMPLXModel           # noqa: E402
from smpl_jax.types import SMPLParams, SMPLXParams  # noqa: E402

TOL = 1e-5

DATA = Path(__file__).resolve().parents[1] / "data"
SMPL_NPZ = DATA / "smpl" / "SMPL_NEUTRAL.npz"
SMPLX_NPZ = DATA / "smplx" / "SMPLX_NEUTRAL.npz"

V, J, B, K = 60, 12, 3, 10


def tt(x):
    return torch.tensor(np.asarray(x, np.float64), dtype=torch.float64)


def nn(x):
    return np.asarray(x, np.float64)


def assert_close(ref, got, tol=TOL, msg=""):
    ref, got = nn(ref), nn(got)
    assert ref.shape == got.shape, f"{msg} shape {ref.shape} vs {got.shape}"
    err = np.abs(ref - got).max()
    assert err <= tol, f"{msg} max abs diff {err:.3e} > {tol:.0e}"


def create_reference(**kwargs):
    """Build an upstream model, skipping if the reference can't read its own file.

    Some SMPL ``.npz`` / ``.pkl`` releases store chumpy objects, which fail to
    unpickle on numpy >= 1.24 (``from numpy import bool``). That is a defect in
    the environment's chumpy, not in SMPL-JAX, so the parity test has nothing to
    compare against and skips instead of failing.
    """
    try:
        return smplx.create(str(DATA), gender="neutral", **kwargs)
    except ImportError as e:  # chumpy / numpy incompatibility
        pytest.skip(f"reference smplx cannot load this model file: {e}")


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def model_arrays(rng):
    """A synthetic SMPL-shaped model + a random pose, shared across the tests."""
    parents = np.array([-1] + [max(0, i - 1) for i in range(1, J)], dtype=np.int64)
    Jr = np.abs(rng.standard_normal((J, V)))
    Jr /= Jr.sum(1, keepdims=True)
    w = np.abs(rng.standard_normal((V, J)))
    w /= w.sum(1, keepdims=True)
    return dict(
        parents=parents,
        v_template=rng.standard_normal((V, 3)),
        shapedirs=rng.standard_normal((V, 3, K)),
        posedirs=rng.standard_normal(((J - 1) * 9, V * 3)),   # reference layout (P, V*3)
        J_regressor=Jr,
        weights=w,
        betas=rng.standard_normal((B, K)),
        pose_aa=np.concatenate(
            [np.zeros((B, 3)), rng.standard_normal((B, (J - 1) * 3)) * 0.4], axis=1
        ),
    )


# --------------------------------------------------------------- primitives --
class TestPrimitives:
    @pytest.mark.parametrize("tag", ["generic", "zero", "tiny", "near_pi"])
    def test_batch_rodrigues(self, tag, rng):
        aa = {
            "generic": rng.standard_normal((10, 3)) * 0.8,
            "zero": np.zeros((3, 3)),
            "tiny": np.full((3, 3), 1e-7),
            "near_pi": np.eye(3) * (np.pi - 1e-4),
        }[tag]
        assert_close(RL.batch_rodrigues(tt(aa)),
                     JRot.axis_angle_to_rotmat(jnp.asarray(aa)), msg=tag)

    def test_blend_shapes(self, model_arrays):
        m = model_arrays
        assert_close(
            RL.blend_shapes(tt(m["betas"]), tt(m["shapedirs"])),
            shape_blend_shapes(jnp.zeros((V, 3)), jnp.asarray(m["shapedirs"]),
                               jnp.asarray(m["betas"])),
        )

    def test_vertices2joints(self, model_arrays, rng):
        m = model_arrays
        verts = rng.standard_normal((B, V, 3))
        assert_close(
            RL.vertices2joints(tt(m["J_regressor"]), tt(verts)),
            jnp.einsum("jv,bvd->bjd", jnp.asarray(m["J_regressor"]), jnp.asarray(verts)),
        )

    def test_pose_blend_shapes(self, model_arrays, rng):
        """Reference posedirs are (P, V*3); SMPL-JAX stores the transpose."""
        m = model_arrays
        rot = nn(RL.batch_rodrigues(tt(rng.standard_normal((B * J, 3)) * 0.4))
                 ).reshape(B, J, 3, 3)
        ref = torch.matmul(tt((rot[:, 1:] - np.eye(3)).reshape(B, -1)),
                           tt(m["posedirs"])).reshape(B, V, 3)
        assert_close(ref, pose_blend_shapes(jnp.asarray(rot),
                                            jnp.asarray(m["posedirs"].T)))

    def test_batch_rigid_transform(self, model_arrays, rng):
        """FK globals and the bind-relative transforms LBS actually blends."""
        m = model_arrays
        rot = nn(RL.batch_rodrigues(tt(rng.standard_normal((B * J, 3)) * 0.4))
                 ).reshape(B, J, 3, 3)
        joints = rng.standard_normal((B, J, 3))
        parents_t = torch.tensor(m["parents"].copy())
        parents_t[0] = 0                      # upstream indexes parents[1:] only
        J_ref, A_ref = RL.batch_rigid_transform(tt(rot), tt(joints), parents_t)

        G = fk_forward_batched(jnp.asarray(rot), jnp.asarray(joints),
                               jnp.asarray(m["parents"]))
        assert_close(J_ref, G[..., :3, 3], msg="posed joints")
        M = lbs_transforms(G, jnp.asarray(joints))
        assert_close(nn(A_ref)[..., :3, :3], M[..., :3], msg="rel R")
        assert_close(nn(A_ref)[..., :3, 3], M[..., 3], msg="rel t")

    def test_full_lbs(self, model_arrays):
        """The whole `smplx.lbs.lbs` pipeline vs SMPL-JAX's step-by-step form."""
        m = model_arrays
        parents_t = torch.tensor(m["parents"].copy())
        parents_t[0] = 0
        verts_ref, joints_ref = RL.lbs(
            tt(m["betas"]), tt(m["pose_aa"]), tt(m["v_template"]), tt(m["shapedirs"]),
            tt(m["posedirs"]), tt(m["J_regressor"]), parents_t, tt(m["weights"]),
        )
        rot = JRot.axis_angle_to_rotmat(jnp.asarray(m["pose_aa"].reshape(B, J, 3)))
        v_shaped = shape_blend_shapes(jnp.asarray(m["v_template"]),
                                      jnp.asarray(m["shapedirs"]), jnp.asarray(m["betas"]))
        joints = jnp.einsum("jv,bvd->bjd", jnp.asarray(m["J_regressor"]), v_shaped)
        G = fk_forward_batched(rot, joints, jnp.asarray(m["parents"]))
        pose_corr = pose_blend_shapes(rot, jnp.asarray(m["posedirs"].T))
        verts = jax_lbs(v_shaped, pose_corr, lbs_transforms(G, joints),
                        jnp.asarray(m["weights"]))
        assert_close(verts_ref, verts, msg="vertices")
        assert_close(joints_ref, G[..., :3, 3], msg="joints")


# ------------------------------------------------------------- end to end ----
@pytest.mark.skipif(not SMPL_NPZ.exists(), reason=f"missing {SMPL_NPZ}")
class TestSMPLEndToEnd:
    def test_forward_matches_reference(self, rng):
        ref = create_reference(model_type="smpl", ext="npz", num_betas=10,
                               batch_size=2)
        model = SMPLModel.load(str(SMPL_NPZ), num_betas=10)
        kw = dict(
            betas=(rng.standard_normal((2, 10)) * 0.5).astype(np.float32),
            body_pose=(rng.standard_normal((2, 69)) * 0.3).astype(np.float32),
            global_orient=(rng.standard_normal((2, 3)) * 0.3).astype(np.float32),
            transl=(rng.standard_normal((2, 3)) * 0.1).astype(np.float32),
        )
        out_ref = ref(**{k: torch.tensor(v) for k, v in kw.items()})
        out = model(SMPLParams(**{k: jnp.asarray(v) for k, v in kw.items()}))
        assert_close(out_ref.vertices.detach().numpy(), out.vertices,
                     tol=1e-6, msg="vertices")
        assert_close(out_ref.joints.detach().numpy()[:, :24], out.joints,
                     tol=1e-6, msg="joints")

    def test_real_weights_against_reference_lbs(self, rng):
        """Real SMPL weights through upstream ``smplx.lbs.lbs`` directly.

        Bypasses ``smplx.create`` so this still runs where the environment's
        chumpy cannot unpickle the model file, while keeping both the real
        weights and the real reference kernel in the comparison.
        """
        from smpl_jax.model_io import load_model_data
        d = load_model_data(str(SMPL_NPZ))
        n_body = d["weights"].shape[1] - 1
        betas = (rng.standard_normal((2, 10)) * 0.5).astype(np.float64)
        pose = np.concatenate(
            [np.zeros((2, 3)), rng.standard_normal((2, n_body * 3)) * 0.3], axis=1,
        )
        parents_t = torch.tensor(np.asarray(d["parents"], np.int64).copy())
        parents_t[0] = 0

        verts_ref, joints_ref = RL.lbs(
            tt(betas), tt(pose), tt(d["v_template"]),
            tt(d["shapedirs"][..., :10]), tt(d["posedirs"].T),
            tt(d["J_regressor"][: n_body + 1]), parents_t, tt(d["weights"]),
        )
        model = SMPLModel.load(str(SMPL_NPZ), num_betas=10)
        out = model(SMPLParams(
            betas=jnp.asarray(betas), body_pose=jnp.asarray(pose[:, 3:]),
            global_orient=jnp.asarray(pose[:, :3]), transl=jnp.zeros((2, 3)),
        ))
        assert_close(verts_ref, out.vertices, tol=1e-5, msg="vertices")
        assert_close(joints_ref, out.joints, tol=1e-5, msg="joints")


@pytest.mark.skipif(not SMPLX_NPZ.exists(), reason=f"missing {SMPLX_NPZ}")
class TestSMPLXEndToEnd:
    @pytest.mark.parametrize("flat_hand_mean", [True, False])
    def test_forward_matches_reference(self, rng, flat_hand_mean):
        """Both hand conventions must reproduce the reference exactly."""
        ref = create_reference(model_type="smplx", ext="npz", use_pca=False,
                               flat_hand_mean=flat_hand_mean, num_betas=10,
                               num_expression_coeffs=10, batch_size=2)
        model = SMPLXModel.load(str(SMPLX_NPZ), num_betas=10, num_expression_coeffs=10,
                                flat_hand_mean=flat_hand_mean)
        kw = dict(
            betas=(rng.standard_normal((2, 10)) * 0.5).astype(np.float32),
            body_pose=(rng.standard_normal((2, 63)) * 0.3).astype(np.float32),
            global_orient=(rng.standard_normal((2, 3)) * 0.3).astype(np.float32),
            transl=(rng.standard_normal((2, 3)) * 0.1).astype(np.float32),
            expression=(rng.standard_normal((2, 10)) * 0.5).astype(np.float32),
            jaw_pose=(rng.standard_normal((2, 3)) * 0.1).astype(np.float32),
            leye_pose=np.zeros((2, 3), np.float32),
            reye_pose=np.zeros((2, 3), np.float32),
            left_hand_pose=(rng.standard_normal((2, 45)) * 0.3).astype(np.float32),
            right_hand_pose=(rng.standard_normal((2, 45)) * 0.3).astype(np.float32),
        )
        out_ref = ref(**{k: torch.tensor(v) for k, v in kw.items()}, return_verts=True)
        out = model(SMPLXParams(**{k: jnp.asarray(v) for k, v in kw.items()}))
        assert_close(out_ref.vertices.detach().numpy(), out.vertices,
                     tol=1e-6, msg="vertices")
        assert_close(out_ref.joints.detach().numpy()[:, :55], out.joints,
                     tol=1e-6, msg="joints")

    def test_shape_and_expression_bases_match_reference(self):
        """The packed 400-column basis must be split the same way upstream does."""
        ref = create_reference(model_type="smplx", ext="npz", use_pca=False,
                               num_betas=10, num_expression_coeffs=10)
        model = SMPLXModel.load(str(SMPLX_NPZ), num_betas=10, num_expression_coeffs=10)
        assert_close(ref.shapedirs.detach().numpy(), model.shapedirs, tol=1e-6)
        assert_close(ref.expr_dirs.detach().numpy(), model.exprdirs, tol=1e-6)
