"""Offscreen pyrender helpers for the comparison GIF.

Self-contained port of the proven SOMA-JAX ``demo_soma_vis`` rendering path:
ground plane + projection shadow + multi-directional lighting + optional
skeleton overlay inside a translucent body, rendered with a cached EGL
offscreen renderer.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_BONE_COLOR = (0.18, 0.18, 0.22)
_JOINT_COLOR = (0.95, 0.85, 0.25)


def _build_skeleton_mesh(joints, parents, radius):
    """Joint spheres + bone cylinders as one trimesh.

    Per-joint radius scales with the bone-to-parent length so finger and face
    chains aren't drawn at body-bone thickness. Joints set to NaN are skipped.
    """
    import trimesh
    geoms = []
    valid = np.isfinite(joints).all(axis=1)
    min_r = max(radius * 0.25, 0.002)
    for j in range(len(joints)):
        if not valid[j]:
            continue
        p = int(parents[j])
        if 0 <= p < len(joints) and valid[p]:
            bone_len = float(np.linalg.norm(joints[j] - joints[p]))
            r = float(np.clip(bone_len * 0.18, min_r, radius))
        else:
            r = radius
        s = trimesh.creation.uv_sphere(radius=r, count=[8, 8])
        s.apply_translation(joints[j])
        s.visual.vertex_colors = np.tile(
            (np.array([*_JOINT_COLOR, 1.0]) * 255).astype(np.uint8), (len(s.vertices), 1))
        geoms.append(s)
        if p < 0 or not (0 <= p < len(joints)) or not valid[p] or j == p:
            continue
        a, b = joints[j], joints[p]
        if np.linalg.norm(a - b) < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=r * 0.5, segment=np.array([a, b]), sections=8)
        cyl.visual.vertex_colors = np.tile(
            (np.array([*_BONE_COLOR, 1.0]) * 255).astype(np.uint8), (len(cyl.vertices), 1))
        geoms.append(cyl)
    return trimesh.util.concatenate(geoms)


def _make_ground_and_shadow(vertices, faces, ground_y=0.0, ground_extent=20.0,
                            shadow_eps=0.003):
    """Ground-plane quad + a projection-shadow copy of the body.

    The shadow is the body mesh with y collapsed onto the plane (parallel
    light from overhead), drawn OPAQUE just above the plane: alpha-blending
    the many coplanar projected triangles flickers frame-to-frame, one solid
    grey silhouette is stable.
    """
    import trimesh
    import pyrender
    g = ground_extent
    ground_verts = np.array([
        [-g, ground_y, -g], [g, ground_y, -g],
        [ g, ground_y,  g], [-g, ground_y,  g],
    ], dtype=np.float32)
    # CCW from above so the surface normal points +Y (back-face culling).
    ground_faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.88, 0.88, 0.92, 1.0],
        metallicFactor=0.0, roughnessFactor=0.95, alphaMode="OPAQUE",
    )
    ground_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=ground_verts, faces=ground_faces, process=False),
        material=ground_mat, smooth=False,
    )
    shadow_verts = vertices.copy()
    shadow_verts[:, 1] = ground_y + shadow_eps
    shadow_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.32, 0.32, 0.35, 1.0],
        metallicFactor=0.0, roughnessFactor=1.0, alphaMode="OPAQUE",
        doubleSided=True,
    )
    shadow_mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(vertices=shadow_verts, faces=faces, process=False),
        material=shadow_mat, smooth=False,
    )
    return ground_mesh, shadow_mesh


def render_mesh_png(vertices, faces, output_path, width=512, height=512,
                    color=(0.7, 0.7, 0.85), joints=None, parents=None,
                    body_alpha=1.0, camera_pose=None, ground=False):
    """Render a mesh (optionally with a skeleton overlay) to an RGB array.

    ``camera_pose``: (4, 4) world transform of the camera; pass a shared pose
    across side-by-side columns to get a consistent floor line. ``ground``
    adds the plane + contact shadow.
    """
    import trimesh
    import pyrender
    show_skeleton = joints is not None and parents is not None
    alpha = body_alpha if show_skeleton else 1.0

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*color, alpha],
        metallicFactor=0.1, roughnessFactor=0.6,
        alphaMode="BLEND" if alpha < 1.0 else "OPAQUE",
    )
    mesh = pyrender.Mesh.from_trimesh(tm, material=material, smooth=True)
    scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35])

    if ground:
        ground_mesh, shadow_mesh = _make_ground_and_shadow(vertices, faces)
        scene.add(ground_mesh)
        scene.add(shadow_mesh)

    # Add skeleton first (opaque) so the translucent body blends over it.
    if show_skeleton:
        extent = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        skel = _build_skeleton_mesh(np.asarray(joints), np.asarray(parents),
                                    radius=extent * 0.005)
        scene.add(pyrender.Mesh.from_trimesh(skel, smooth=False))
    scene.add(mesh)

    if camera_pose is not None:
        cam_pose = np.asarray(camera_pose, dtype=np.float32)
    else:
        center = vertices.mean(axis=0)
        body_height = float(vertices[:, 1].max() - vertices[:, 1].min())
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = center + np.array([0.0, 0.0, body_height * 1.35 + 0.3])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(cam, pose=cam_pose)

    light_main = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light_main, pose=cam_pose)
    light_fill = pyrender.DirectionalLight(color=[0.8, 0.8, 1.0], intensity=2.0)
    lp_fill = np.eye(4)
    lp_fill[:3, :3] = np.array([[0.707, 0, 0.707], [0, 1, 0], [-0.707, 0, 0.707]])
    scene.add(light_fill, pose=lp_fill)
    light_back = pyrender.DirectionalLight(color=[1.0, 0.9, 0.9], intensity=2.0)
    lp_back = np.eye(4)
    lp_back[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    scene.add(light_back, pose=lp_back)

    # Cache the OffscreenRenderer per (w, h): creating + deleting an EGL
    # context every frame dominates GIF render time.
    cache = globals().setdefault("_OFFSCREEN_RENDERER_CACHE", {})
    key = (width, height)
    renderer = cache.get(key)
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(width, height)
        cache[key] = renderer
    color_img, _ = renderer.render(scene)

    if output_path is not None:
        from PIL import Image
        Image.fromarray(color_img).save(output_path)
    return color_img
