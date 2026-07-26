"""Render an N-method SMPL-X side-by-side comparison GIF (skeleton overlay).

Generalizes the two-column teaser (render_compare.py) to any number of
posers. Each ``--inputs`` npz is a ``pose_*.py`` output (verts / joints /
faces / median throughput / label). Every column poses the identical rig over
the identical motion, so the meshes agree — the only thing that differs is how
many distinct poses each method can afford in a fixed wall-clock budget.

Speed-based frame slowdown
--------------------------
All columns play for the same wall-clock duration (the same T display slots).
The FASTEST method (max meshes/s) advances every slot — smooth. Every slower
method j advances only ``n_j = round(T * fps_j / fps_max)`` distinct poses,
spread across the whole motion (sample-and-hold), so it looks proportionally
choppier: "in the same time, the faster method computes more frames." The live
per-column counter shows exactly how many frames each has produced so far.

The GIF ends on a banner ranking the methods by throughput.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from render_utils import render_mesh_png

# SMPL-X joints 22-24 (jaw, eyes) sit inside the head; NaN-mask for a clean rig.
_NONANATOMICAL_JOINTS = (22, 23, 24)

# SOMA mocap world is Z-up; rotate to the renderer's Y-up frame.
_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0]], dtype=np.float32)

# Distinct body colors per column (extend if you add more methods).
_COLORS = [
    (0.20, 0.72, 0.55),   # teal
    (0.85, 0.55, 0.30),   # orange
    (0.62, 0.44, 0.82),   # purple
    (0.30, 0.58, 0.85),   # blue
    (0.85, 0.45, 0.55),   # rose
]


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _label(img, line1, line2=None, color=(255, 255, 255), y=8):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
    f, fs = _font(17), _font(14)
    w1 = int(d.textlength(line1, font=f))
    w2 = int(d.textlength(line2, font=fs)) if line2 else 0
    bh = 24 + (18 if line2 else 0)
    d.rectangle([6, y - 2, 6 + max(w1, w2) + 10, y + bh], fill=(0, 0, 0))
    d.text((11, y), line1, fill=color, font=f)
    if line2:
        d.text((11, y + 22), line2, fill=(180, 220, 255), font=fs)
    return np.asarray(pil)


def _banner(img, title, lines):
    """Centered translucent banner: one big title + several info lines."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img.copy()); W, H = pil.size
    big, small = _font(46), _font(18)
    band_h = 78 + 26 * len(lines)
    top = H // 2 - band_h // 2
    band = Image.new("RGBA", (W, band_h), (0, 0, 0, 185))
    pil.paste(Image.alpha_composite(
        pil.crop((0, top, W, top + band_h)).convert("RGBA"), band), (0, top))
    d = ImageDraw.Draw(pil)
    bb = d.textbbox((0, 0), title, font=big)
    d.text(((W - (bb[2] - bb[0])) / 2, top + 16), title, fill=(90, 230, 170), font=big)
    for i, ln in enumerate(lines):
        bb2 = d.textbbox((0, 0), ln, font=small)
        d.text(((W - (bb2[2] - bb2[0])) / 2, top + 74 + 26 * i), ln, fill=(235, 235, 235), font=small)
    return np.asarray(pil)


def _shared_camera(view_seqs, width, height):
    """Frame the union of XYZ extents across all frames + all columns."""
    fov_y = np.pi / 3.0
    aspect = float(width) / float(height)
    all_min = np.array([np.inf] * 3); all_max = np.array([-np.inf] * 3)
    for vs in view_seqs:
        all_min = np.minimum(all_min, vs.reshape(-1, 3).min(0))
        all_max = np.maximum(all_max, vs.reshape(-1, 3).max(0))
    center = (all_min + all_max) * 0.5
    span = all_max - all_min
    margin = 1.20
    dist_v = (span[1] * 0.5 * margin) / np.tan(fov_y * 0.5)
    dist_h = (span[0] * 0.5 * margin) / np.tan(fov_y * 0.5) / aspect
    cam_distance = max(dist_v, dist_h) + span[2] * 0.5
    cam = np.eye(4, dtype=np.float32)
    cam[:3, 3] = center.astype(np.float32) + np.array([0.0, 0.0, cam_distance], np.float32)
    return cam, span


def _human_fps(x):
    return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="pose_*.py output npz files")
    p.add_argument("--gif", required=True)
    p.add_argument("--width", type=int, default=380)
    p.add_argument("--height", type=int, default=460)
    args = p.parse_args()

    from PIL import Image

    cols = []
    for path in args.inputs:
        d = np.load(path, allow_pickle=True)
        v = d["verts"].astype(np.float32) @ _ZUP_TO_YUP.T
        j = d["joints"].astype(np.float32) @ _ZUP_TO_YUP.T
        j[:, _NONANATOMICAL_JOINTS] = np.nan
        cols.append(dict(
            v=v, j=j, faces=d["faces"], parents=d["parents"].astype(int),
            fps=float(d["fps"]), label=str(d["label"]),
            precision=str(d["precision"]) if "precision" in d else "",
            batch=int(d["batch"]), duration_s=float(d["duration_s"]),
        ))

    T = min(len(c["v"]) for c in cols)
    for c in cols:
        c["v"], c["j"] = c["v"][:T], c["j"][:T]
        # Ground-lock each column: drop its lowest point to Y=0; shift joints too.
        floor = c["v"][..., 1].min()
        c["v"][..., 1] -= floor
        c["j"][..., 1] -= floor

    # Cross-method faithfulness sanity (against the first column).
    ref = cols[0]
    for c in cols[1:]:
        if c["v"].shape == ref["v"].shape:
            dmm = np.linalg.norm(c["v"] - ref["v"], axis=-1)
            print(f"[mesh agreement] {c['label']} vs {ref['label']}: "
                  f"max={dmm.max()*1000:.3f} mm  mean={dmm.mean()*1000:.4f} mm")

    cam, span = _shared_camera([c["v"] for c in cols], args.width, args.height)
    print(f"[camera] motion XYZ span=({span[0]:.2f},{span[1]:.2f},{span[2]:.2f})m")

    # --- speed-based frame budget -------------------------------------------
    fps_max = max(c["fps"] for c in cols)
    for c in cols:
        c["n"] = T if c["fps"] >= fps_max else max(2, int(round(T * c["fps"] / fps_max)))
    print("[equal-time] frames rendered over %d slots: " % T +
          "  ".join(f"{c['label'].split()[0]}={c['n']}" for c in cols))

    def budget(t, n):
        g = min(n - 1, int(t * n / T))
        idx = int(round(g * (T - 1) / (n - 1))) if n > 1 else 0
        return idx, g + 1

    W, H = args.width, args.height
    frames = []
    for t in range(T):
        tiles = []
        for k, c in enumerate(cols):
            idx, count = budget(t, c["n"])
            img = render_mesh_png(c["v"][idx], c["faces"], None, W, H,
                                  color=_COLORS[k % len(_COLORS)],
                                  joints=c["j"][idx], parents=c["parents"],
                                  body_alpha=0.6, camera_pose=cam, ground=True)
            tiles.append(_label(img, c["label"], f"{count} frames"))
        frames.append(np.concatenate(tiles, axis=1))

    # Final banner (held ~2.5 s): rank by throughput.
    order = sorted(cols, key=lambda c: -c["fps"])
    fastest = order[0]
    speed_line = "   ·   ".join(
        f"{c['label'].split()[0]} {_human_fps(c['fps'])}/s ({c['fps']/order[-1]['fps']:.1f}x)"
        for c in order)
    frame_line = "equal wall-clock  ·  distinct poses: " + "  ·  ".join(
        f"{c['label'].split()[0]} {c['n']}" for c in order)
    banner = _banner(frames[-1].copy(),
                     f"{fastest['label'].split()[0]}  fastest",
                     [speed_line, frame_line,
                      f"full forward, batch {fastest['batch']}  ·  "
                      f"meshes/s vs slowest  ·  {T} realtime frames"])

    duration_s = cols[0]["duration_s"]
    play_fps = T / duration_s
    hold = int(round(2.5 * play_fps))
    frames.extend([banner] * hold)

    dur = int(round(1000.0 * duration_s / T))          # ms/frame = realtime
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(x) for x in frames]
    ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=dur, loop=0)
    Image.fromarray(banner).save(str(Path(args.gif).with_suffix(".png")))
    print(f"wrote {args.gif}  ({len(frames)} frames, {play_fps:.0f} fps realtime)  "
          f"fastest={fastest['label']}")


if __name__ == "__main__":
    main()
