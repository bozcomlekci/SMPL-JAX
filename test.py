from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from smpl_jax import SMPLXModel, SMPLXParams


def _load_smplx_model(model_path: Path, num_betas: int, num_expression_coeffs: int) -> SMPLXModel:
    if model_path.suffix == ".pkl":
        return SMPLXModel.load(
            str(model_path),
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
        )

    if model_path.suffix != ".npz":
        raise ValueError(f"Unsupported model format: {model_path.suffix}. Use .pkl or .npz.")

    data = np.load(model_path, allow_pickle=True)

    v_template = np.asarray(data["v_template"], dtype=np.float32)
    shapedirs = np.asarray(data["shapedirs"], dtype=np.float32)
    posedirs_raw = np.asarray(data["posedirs"], dtype=np.float32)
    J_regressor = np.asarray(data["J_regressor"], dtype=np.float32)
    kintree_table = np.asarray(data["kintree_table"], dtype=np.int32)
    weights = np.asarray(data["weights"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int32)

    if posedirs_raw.ndim == 3:
        posedirs = posedirs_raw.reshape(v_template.shape[0] * 3, -1)
    elif posedirs_raw.ndim == 2 and posedirs_raw.shape[0] == v_template.shape[0] * 3:
        posedirs = posedirs_raw
    else:
        posedirs = posedirs_raw.T

    if "expr_dirs" in data.files:
        exprdirs = np.asarray(data["expr_dirs"], dtype=np.float32)
    elif "exprdirs" in data.files:
        exprdirs = np.asarray(data["exprdirs"], dtype=np.float32)
    else:
        tail_start = 300
        if shapedirs.shape[-1] <= tail_start:
            raise ValueError(
                "Could not infer expression blend shapes from model .npz. "
                "Expected expr_dirs/exprdirs or shapedirs with >= 301 components."
            )
        exprdirs = shapedirs[..., tail_start:]

    parents = kintree_table[0].copy()
    parents[0] = -1

    return SMPLXModel(
        v_template=v_template,
        shapedirs=shapedirs,
        exprdirs=exprdirs,
        posedirs=posedirs,
        J_regressor=J_regressor,
        parents=parents,
        weights=weights,
        faces=faces,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
    )


def _build_params_from_sequence(
    sequence_data: np.lib.npyio.NpzFile,
    frame_idx: int,
    num_betas: int,
    num_expression_coeffs: int,
) -> tuple[SMPLXParams, SMPLXParams]:
    trans = np.asarray(sequence_data["trans"], dtype=np.float32)
    root_orient = np.asarray(sequence_data["root_orient"], dtype=np.float32)
    pose_body = np.asarray(sequence_data["pose_body"], dtype=np.float32)
    pose_hand = np.asarray(sequence_data["pose_hand"], dtype=np.float32)
    betas_all = np.asarray(sequence_data["betas"], dtype=np.float32)

    n_frames = trans.shape[0]
    if frame_idx < 0:
        frame_idx = n_frames + frame_idx
    if frame_idx < 0 or frame_idx >= n_frames:
        raise IndexError(f"frame_idx={frame_idx} out of range for {n_frames} frames")

    left_hand = pose_hand[frame_idx, :45]
    right_hand = pose_hand[frame_idx, 45:90]
    betas = betas_all[:num_betas]

    rest_params = SMPLXParams(
        betas=jnp.asarray(betas[None, :], dtype=jnp.float32),
        body_pose=jnp.zeros((1, 63), dtype=jnp.float32),
        global_orient=jnp.zeros((1, 3), dtype=jnp.float32),
        transl=jnp.zeros((1, 3), dtype=jnp.float32),
        expression=jnp.zeros((1, num_expression_coeffs), dtype=jnp.float32),
        jaw_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        leye_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        reye_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        left_hand_pose=jnp.zeros((1, 45), dtype=jnp.float32),
        right_hand_pose=jnp.zeros((1, 45), dtype=jnp.float32),
    )

    posed_params = SMPLXParams(
        betas=jnp.asarray(betas[None, :], dtype=jnp.float32),
        body_pose=jnp.asarray(pose_body[frame_idx][None, :], dtype=jnp.float32),
        global_orient=jnp.asarray(root_orient[frame_idx][None, :], dtype=jnp.float32),
        transl=jnp.asarray(trans[frame_idx][None, :], dtype=jnp.float32),
        expression=jnp.zeros((1, num_expression_coeffs), dtype=jnp.float32),
        jaw_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        leye_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        reye_pose=jnp.zeros((1, 3), dtype=jnp.float32),
        left_hand_pose=jnp.asarray(left_hand[None, :], dtype=jnp.float32),
        right_hand_pose=jnp.asarray(right_hand[None, :], dtype=jnp.float32),
    )
    return rest_params, posed_params


def _build_full_sequence_params(
    sequence_data: np.lib.npyio.NpzFile,
    num_betas: int,
    num_expression_coeffs: int,
) -> SMPLXParams:
    trans = np.asarray(sequence_data["trans"], dtype=np.float32)
    root_orient = np.asarray(sequence_data["root_orient"], dtype=np.float32)
    pose_body = np.asarray(sequence_data["pose_body"], dtype=np.float32)
    pose_hand = np.asarray(sequence_data["pose_hand"], dtype=np.float32)
    betas_all = np.asarray(sequence_data["betas"], dtype=np.float32)

    n_frames = trans.shape[0]
    betas = betas_all[:num_betas]
    betas_batch = np.broadcast_to(betas[None, :], (n_frames, num_betas)).copy()

    return SMPLXParams(
        betas=jnp.asarray(betas_batch, dtype=jnp.float32),
        body_pose=jnp.asarray(pose_body, dtype=jnp.float32),
        global_orient=jnp.asarray(root_orient, dtype=jnp.float32),
        transl=jnp.asarray(trans, dtype=jnp.float32),
        expression=jnp.zeros((n_frames, num_expression_coeffs), dtype=jnp.float32),
        jaw_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        leye_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        reye_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        left_hand_pose=jnp.asarray(pose_hand[:, :45], dtype=jnp.float32),
        right_hand_pose=jnp.asarray(pose_hand[:, 45:90], dtype=jnp.float32),
    )


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _visualize(
    before_vertices: np.ndarray,
    after_vertices: np.ndarray,
    after_joints: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot_trisurf(
        before_vertices[:, 0],
        before_vertices[:, 1],
        before_vertices[:, 2],
        triangles=faces,
        linewidth=0.02,
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )
    ax1.set_title("Before pose (rest shape)")
    _set_axes_equal(ax1, before_vertices)

    ax2.plot_trisurf(
        after_vertices[:, 0],
        after_vertices[:, 1],
        after_vertices[:, 2],
        triangles=faces,
        linewidth=0.02,
        edgecolor="none",
        antialiased=True,
        alpha=0.95,
    )
    ax2.scatter(after_joints[:, 0], after_joints[:, 1], after_joints[:, 2], s=8, c="crimson", label="Joints")
    ax2.set_title("After pose (sequence frame)")
    _set_axes_equal(ax2, after_vertices)
    ax2.legend(loc="upper right")

    for ax in (ax1, ax2):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.show()


def _animate_mesh_sequence_open3d(
    vertices_seq: np.ndarray,
    faces: np.ndarray,
    joints_seq: np.ndarray | None = None,
    frame_stride: int = 1,
    fps: float = 60.0,
    source_up_axis: str = "auto",
    camera_view: str = "front",
    body_centered_view: bool = False,
    camera_zoom: float = -1.0,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "open3d is required for fast sequence visualization. Install with: pip install open3d"
        ) from exc

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if source_up_axis not in {"auto", "y-up", "z-up"}:
        raise ValueError("source_up_axis must be 'auto', 'y-up', or 'z-up'")
    if camera_view not in {"front", "side", "top"}:
        raise ValueError("camera_view must be 'front', 'side', or 'top'")
    if camera_zoom != -1.0 and not (0.000001 <= camera_zoom <= 2.0):
        raise ValueError("camera_zoom must be in [0.02, 2.0], or -1 for auto")

    raw_vertices = np.array(vertices_seq, dtype=np.float64, order="C", copy=True)
    raw_joints = None if joints_seq is None else np.array(joints_seq, dtype=np.float64, order="C", copy=True)

    if source_up_axis == "auto":
        # Prefer trajectory-based detection: up-axis usually has smaller temporal motion.
        if raw_joints is not None and raw_joints.shape[1] > 0:
            pelvis_traj = raw_joints[:, 0, :]
        else:
            pelvis_traj = raw_vertices.mean(axis=1)

        y_std = float(np.std(pelvis_traj[:, 1]))
        z_std = float(np.std(pelvis_traj[:, 2]))
        source_up_axis = "y-up" if y_std < z_std else "z-up"

    def _convert_points(points: np.ndarray) -> np.ndarray:
        points = np.array(points, dtype=np.float64, order="C", copy=True)
        if source_up_axis == "z-up":
            return points
        converted = np.empty_like(points)
        converted[..., 0] = points[..., 0]
        converted[..., 1] = points[..., 2]
        converted[..., 2] = -points[..., 1]
        return converted

    vertices_seq = _convert_points(raw_vertices)
    if raw_joints is not None:
        joints_seq = _convert_points(raw_joints)

    if body_centered_view:
        if joints_seq is not None:
            centers = joints_seq[:, 0:1, :]
        else:
            centers = vertices_seq.mean(axis=1, keepdims=True)
        vertices_seq = vertices_seq - centers
        if joints_seq is not None:
            joints_seq = joints_seq - centers

    faces = np.array(faces, dtype=np.int32, order="C", copy=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SMPL-X Sequence Animation", width=1280, height=800)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices_seq[0], copy=True, order="C"))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.77, 0.88])
    vis.add_geometry(mesh)

    joints_cloud = None
    if joints_seq is not None:
        joints_cloud = o3d.geometry.PointCloud()
        joints_cloud.points = o3d.utility.Vector3dVector(np.array(joints_seq[0], dtype=np.float64, copy=True))
        joints_cloud.paint_uniform_color([0.9, 0.1, 0.1])
        vis.add_geometry(joints_cloud)

    if joints_seq is not None:
        center = joints_seq[:, 0, :].mean(axis=0)
        center[2] += 0.1
    else:
        center = vertices_seq.reshape(-1, 3).mean(axis=0)

    scene_min = vertices_seq.reshape(-1, 3).min(axis=0)
    scene_max = vertices_seq.reshape(-1, 3).max(axis=0)
    extent = np.max(scene_max - scene_min)
    frame_size = max(0.1, 0.2 * float(extent))
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size))

    view_control = vis.get_view_control()
    view_control.set_lookat(center.tolist())
    if camera_view == "front":
        view_control.set_up([0.0, 0.0, 1.0])
        if joints_seq is not None and joints_seq.shape[1] >= 3:
            # Prefer face references at frame 0 for true frontal camera direction.
            if joints_seq.shape[1] >= 25:
                left_ref = joints_seq[0, 23, :]   # left eye
                right_ref = joints_seq[0, 24, :]  # right eye
            elif joints_seq.shape[1] >= 18:
                left_ref = joints_seq[0, 17, :]
                right_ref = joints_seq[0, 16, :]
            else:
                left_ref = joints_seq[0, 1, :]
                right_ref = joints_seq[0, 2, :]

            left_right = left_ref - right_ref
            left_right[2] = 0.0
            lr_norm = np.linalg.norm(left_right)
            if lr_norm > 1e-8:
                left_right = left_right / lr_norm
                human_forward = np.array([-left_right[1], left_right[0], 0.0], dtype=np.float64)
                hf_norm = np.linalg.norm(human_forward)
                if hf_norm > 1e-8:
                    human_forward = human_forward / hf_norm
                    camera_front = -human_forward
                    view_control.set_front(camera_front.tolist())
                else:
                    view_control.set_front([0.0, -1.0, 0.0])
            else:
                view_control.set_front([0.0, -1.0, 0.0])
        else:
            view_control.set_front([0.0, -1.0, 0.0])
    elif camera_view == "side":
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_front([1.0, 0.0, 0.0])
    else:  # top
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_front([0.0, 0.0, -1.0])
    if camera_zoom == -1.0:
        auto_zoom = float(np.clip(0.35 / max(extent, 1e-6), 0.03, 0.12))
        view_control.set_zoom(auto_zoom)
    else:
        view_control.set_zoom(camera_zoom)

    target_dt = 1.0 / fps
    frame_indices = list(range(0, vertices_seq.shape[0], frame_stride))

    try:
        while True:
            for frame_idx in frame_indices:
                if not vis.poll_events():
                    return

                t0 = time.perf_counter()

                mesh.vertices = o3d.utility.Vector3dVector(
                    np.array(vertices_seq[frame_idx], copy=True, order="C")
                )
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)

                if joints_cloud is not None:
                    joints_cloud.points = o3d.utility.Vector3dVector(
                        np.asarray(joints_seq[frame_idx], dtype=np.float64)
                    )
                    vis.update_geometry(joints_cloud)

                vis.update_renderer()

                elapsed = time.perf_counter() - t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end SMPL-X posing from an example SOMA datum")
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("datasets/SOMA/soma_subject1/walk_001_stageii.npz"),
        help="Path to a SOMA stage-II sequence .npz",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("smpl_models/smplx/SMPLX_NEUTRAL.npz"),
        help="Path to SMPL-X model file (.npz or .pkl)",
    )
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize (supports negative index)")
    parser.add_argument(
        "--mode",
        choices=["frame", "sequence"],
        default="sequence",
        help="Visualization mode: single frame image or full sequence animation",
    )
    parser.add_argument("--frame-stride", type=int, default=2, help="Stride for sequence animation frames")
    parser.add_argument("--fps", type=float, default=60.0, help="Playback FPS for sequence animation")
    parser.add_argument(
        "--source-up-axis",
        choices=["auto", "y-up", "z-up"],
        default="auto",
        help="Source coordinate up-axis convention for sequence mode",
    )
    parser.add_argument(
        "--camera-view",
        choices=["front", "side", "top"],
        default="front",
        help="Initial camera view preset for sequence mode",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=-1.0,
        help="Fixed camera zoom (Open3D), or -1 for automatic full-motion framing",
    )
    parser.add_argument(
        "--no-body-center",
        action="store_true",
        help="Disable body-centered visualization coordinates in sequence mode",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Run posing without opening the visualization window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on number of sequence frames to pose (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/smplx_e2e_before_after.png"),
        help="Output image path (used in frame mode)",
    )
    args = parser.parse_args()

    if not args.sequence.exists():
        raise FileNotFoundError(f"Sequence not found: {args.sequence}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    sequence_data = np.load(args.sequence, allow_pickle=True)
    required = ["trans", "root_orient", "pose_body", "pose_hand", "betas"]
    missing = [k for k in required if k not in sequence_data.files]
    if missing:
        raise KeyError(f"Sequence file missing required keys: {missing}")

    betas_len = int(np.asarray(sequence_data["betas"]).shape[0])
    num_betas = min(10, betas_len)
    num_expression_coeffs = 10

    model = _load_smplx_model(args.model, num_betas=num_betas, num_expression_coeffs=num_expression_coeffs)
    faces = np.asarray(model.faces)

    if args.mode == "frame":
        rest_params, posed_params = _build_params_from_sequence(
            sequence_data,
            frame_idx=args.frame,
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
        )

        out_before = model(rest_params)
        out_after = model(posed_params)

        before_vertices = np.asarray(out_before.vertices[0])
        after_vertices = np.asarray(out_after.vertices[0])
        after_joints = np.asarray(out_after.joints[0])

        _visualize(before_vertices, after_vertices, after_joints, faces, args.output)

        print(f"Sequence: {args.sequence}")
        print(f"Model:    {args.model}")
        print(f"Mode:     frame")
        print(f"Frame:    {args.frame}")
        print(f"Vertices: {after_vertices.shape[0]}")
        print(f"Joints:   {after_joints.shape[0]}")
        print(f"Saved:    {args.output}")
        return

    full_params = _build_full_sequence_params(
        sequence_data,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
    )

    if args.max_frames > 0:
        max_frames = min(args.max_frames, full_params.transl.shape[0])
        full_params = SMPLXParams(
            betas=full_params.betas[:max_frames],
            body_pose=full_params.body_pose[:max_frames],
            global_orient=full_params.global_orient[:max_frames],
            transl=full_params.transl[:max_frames],
            expression=full_params.expression[:max_frames],
            jaw_pose=full_params.jaw_pose[:max_frames],
            leye_pose=full_params.leye_pose[:max_frames],
            reye_pose=full_params.reye_pose[:max_frames],
            left_hand_pose=full_params.left_hand_pose[:max_frames],
            right_hand_pose=full_params.right_hand_pose[:max_frames],
        )

    out_seq = model(full_params)
    vertices_seq = np.asarray(out_seq.vertices)
    joints_seq = np.asarray(out_seq.joints)

    print(f"Sequence: {args.sequence}")
    print(f"Model:    {args.model}")
    print(f"Mode:     sequence")
    print(f"Frames:   {vertices_seq.shape[0]}")
    print(f"Vertices: {vertices_seq.shape[1]}")
    print(f"Joints:   {joints_seq.shape[1]}")
    if args.no_vis:
        print("Visualization skipped (--no-vis).")
        return

    print("Opening Open3D animation window (close window to stop)...")

    _animate_mesh_sequence_open3d(
        vertices_seq=vertices_seq,
        faces=faces,
        joints_seq=joints_seq,
        frame_stride=args.frame_stride,
        fps=args.fps,
        source_up_axis=args.source_up_axis,
        camera_view=args.camera_view,
        body_centered_view=not args.no_body_center,
        camera_zoom=args.camera_zoom,
    )


if __name__ == "__main__":
    main()