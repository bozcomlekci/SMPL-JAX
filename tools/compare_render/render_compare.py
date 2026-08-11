"""Render the SMPL-X vs SMPL-JAX side-by-side comparison GIF.

Inputs are the two ``pose_*.py`` outputs (verts / joints / faces / median
throughput). Both columns run the identical SMPL-X model over the identical
SOMA-dataset motion with identical settings, so the vertices agree to sub-mm —
the ONLY thing that differs is how many frames each pipeline can afford in a
fixed time budget.

Equal-time visualization
------------------------
Both columns play for the same wall-clock duration (the same number of GIF
slots). The FASTER pipeline advances every slot — it renders the whole motion
smoothly. The SLOWER pipeline is subsampled by the measured speed ratio: it
advances only every ``step`` slots (sample-and-hold), so within the same
duration it shows ``1/ratio`` as many distinct poses — visibly
sparser/choppier. Exactly "in the same time, the faster method computes
``ratio``x more frames."

The GIF ends on a centered "SMPL-JAX ~N.Nx faster" popup held for ~2 s.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from render_utils import render_mesh_png

# SMPL-X joints 22-24 (jaw, eyes) sit inside the head; NaN-mask them so the
# skeleton overlay draws cleanly (mirrors the SOMA comparison's joint mask).
_NONANATOMICAL_JOINTS = (22, 23, 24)

# SOMA mocap world is Z-up; rotate to the renderer's Y-up frame.
_ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0]], dtype=np.float32)


def _label(img, line1, line2=None, color=(255, 255, 255), y=8):
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        f = fs = ImageFont.load_default()
    w1 = int(d.textlength(line1, font=f))
    w2 = int(d.textlength(line2, font=fs)) if line2 else 0
    bh = 24 + (18 if line2 else 0)
    d.rectangle([6, y - 2, 6 + max(w1, w2) + 10, y + bh], fill=(0, 0, 0))
    d.text((11, y), line1, fill=color, font=f)
    if line2:
        d.text((11, y + 22), line2, fill=(180, 220, 255), font=fs)
    return np.asarray(pil)


def _centered_banner(img, lines, sub=None):
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img.copy()); d = ImageDraw.Draw(pil)
    W, H = pil.size
    try:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        big = small = ImageFont.load_default()
    band = Image.new("RGBA", (W, 150), (0, 0, 0, 175))
    pil.paste(Image.alpha_composite(pil.crop((0, H // 2 - 75, W, H // 2 + 75)).convert("RGBA"), band),
              (0, H // 2 - 75))
    d = ImageDraw.Draw(pil)
    bb = d.textbbox((0, 0), lines, font=big)
    d.text(((W - (bb[2] - bb[0])) / 2, H // 2 - 42), lines, fill=(90, 230, 170), font=big)
    if sub:
        bb2 = d.textbbox((0, 0), sub, font=small)
        d.text(((W - (bb2[2] - bb2[0])) / 2, H // 2 + 24), sub, fill=(235, 235, 235), font=small)
    return np.asarray(pil)


def _shared_camera(view_seqs, width, height):
    """Frame the union of XYZ extents across all frames + both columns,
    +20% margin, camera pulled back along +Z."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smplx", required=True)
    p.add_argument("--smpljax", required=True)
    p.add_argument("--gif", required=True)
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=460)
    args = p.parse_args()

    from PIL import Image

    A = np.load(args.smplx, allow_pickle=True); B = np.load(args.smpljax, allow_pickle=True)
    va = A["verts"].astype(np.float32) @ _ZUP_TO_YUP.T
    vb = B["verts"].astype(np.float32) @ _ZUP_TO_YUP.T
    ja = A["joints"].astype(np.float32) @ _ZUP_TO_YUP.T
    jb = B["joints"].astype(np.float32) @ _ZUP_TO_YUP.T
    fa, fb = A["faces"], B["faces"]
    T = min(len(va), len(vb))
    va, vb, ja, jb = va[:T], vb[:T], ja[:T], jb[:T]
    fps_a, fps_b = float(A["fps"]), float(B["fps"])
    lab_a, lab_b = str(A["label"]), str(B["label"])
    pa, pb = A["parents"].astype(int), B["parents"].astype(int)
    ja[:, _NONANATOMICAL_JOINTS] = np.nan
    jb[:, _NONANATOMICAL_JOINTS] = np.nan

    # Faithfulness sanity: how close are the two posed meshes? (same model+motion)
    if va.shape == vb.shape:
        dmm = np.linalg.norm(va - vb, axis=-1)
        print(f"[mesh agreement] SMPL-X vs SMPL-JAX posed verts: "
              f"max={dmm.max()*1000:.3f} mm  mean={dmm.mean()*1000:.4f} mm")

    # Ground-lock each column: drop its lowest point (over the whole clip) to
    # Y=0 so both share one floor; shift joints by the same offset.
    fa_floor = va[..., 1].min(); va[..., 1] -= fa_floor; ja[..., 1] -= fa_floor
    fb_floor = vb[..., 1].min(); vb[..., 1] -= fb_floor; jb[..., 1] -= fb_floor

    cam, span = _shared_camera([va, vb], args.width, args.height)
    print(f"[camera] motion XYZ span=({span[0]:.2f},{span[1]:.2f},{span[2]:.2f})m")

    # --- equal-time frame budget ---------------------------------------------
    # Both columns run for the same wall-clock (T display slots = the clip's
    # realtime duration). The FASTER pipeline renders all T frames; the SLOWER
    # renders only round(T / ratio) frames, spread across the whole motion
    # (sample-and-hold => choppier). The live counter shows exactly "total
    # frames" (fast) vs "total / relative-speed frames" (slow).
    ratio = max(fps_a, fps_b) / max(min(fps_a, fps_b), 1e-9)
    n_a = T if fps_a >= fps_b else max(2, int(round(T / ratio)))
    n_b = T if fps_b >= fps_a else max(2, int(round(T / ratio)))
    print(f"[equal-time] ratio {ratio:.2f}x  frames rendered: "
          f"SMPL-X {n_a}  SMPL-JAX {n_b}  over {T} realtime slots")

    def budget(t, n):
        """Motion-frame index + running count for a column that renders n frames
        (spread over the full motion) across the T display slots."""
        g = min(n - 1, int(t * n / T))
        idx = int(round(g * (T - 1) / (n - 1))) if n > 1 else 0
        return idx, g + 1

    W, H = args.width, args.height
    col_a = (0.85, 0.55, 0.30)   # SMPL-X orange
    col_b = (0.20, 0.72, 0.55)   # SMPL-JAX teal

    frames = []
    for t in range(T):
        ia_idx, ca = budget(t, n_a)
        ib_idx, cb = budget(t, n_b)
        ra = render_mesh_png(va[ia_idx], fa, None, W, H, color=col_a,
                             joints=ja[ia_idx], parents=pa, body_alpha=0.6,
                             camera_pose=cam, ground=True)
        rb = render_mesh_png(vb[ib_idx], fb, None, W, H, color=col_b,
                             joints=jb[ib_idx], parents=pb, body_alpha=0.6,
                             camera_pose=cam, ground=True)
        ia = _label(ra, lab_a, f"{ca} frames")
        ib = _label(rb, lab_b, f"{cb} frames")
        frames.append(np.concatenate([ia, ib], axis=1))

    # Final popup (held ~2 s): the relative speed increase, with the matmul
    # precision disclosed. When both sides ran the same precision — the fair
    # case the drivers now default to — say so once rather than printing it
    # twice, so a reader can see at a glance that the arithmetic matched.
    who = "SMPL-JAX" if fps_b >= fps_a else "SMPL-X"
    prec_a = str(A["precision"]) if "precision" in A else "?"
    prec_b = str(B["precision"]) if "precision" in B else "?"
    prec = (f"both {prec_a}" if prec_a == prec_b
            else f"JAX {prec_b} / torch {prec_a}")
    banner = _centered_banner(
        frames[-1].copy(), f"{who}  ~{ratio:.1f}x  faster",
        sub=f"same wall-clock  ·  {max(n_a, n_b)} vs {min(n_a, n_b)} frames  ·  "
            f"full forward, batch {int(A['batch'])}  ·  {prec}")
    duration_s = float(A["duration_s"])
    play_fps = T / duration_s                          # realtime playback
    hold = int(round(2.0 * play_fps))
    frames.extend([banner] * hold)

    dur = int(round(1000.0 * duration_s / T))          # ms/frame = realtime
    Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
    ims = [Image.fromarray(x) for x in frames]
    ims[0].save(args.gif, save_all=True, append_images=ims[1:], duration=dur, loop=0)
    Image.fromarray(banner).save(str(Path(args.gif).with_suffix(".png")))
    print(f"wrote {args.gif}  ({len(frames)} frames, {play_fps:.0f} fps realtime)  "
          f"{who} ~{ratio:.1f}x faster  ({max(n_a,n_b)} vs {min(n_a,n_b)} frames)")


if __name__ == "__main__":
    main()
