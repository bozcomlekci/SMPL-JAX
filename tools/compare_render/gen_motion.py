"""Build the shared parameter sequence for the SMPL-X vs SMPL-JAX comparison.

Source is a SOMA-dataset MoSh++ ``*_stageii.npz`` (SMPL-X parameters solved
from real mocap: ``root_orient`` / ``pose_body`` / ``pose_hand`` / ``trans`` /
``betas`` at 120 FPS). A realtime window of ``--seconds`` is extracted and
evenly resampled to ``--play-fps`` so GIF playback runs at true wall-clock
speed. Both posers consume the identical float32 arrays.

The SOMA mocap world is Z-up; the horizontal ground-plane axes are X,Y. By
default the horizontal root drift is frozen to frame 0 (an in-place
"treadmill") so the body stays centered/constant-size in a side-by-side speed
comparison, while the vertical (Z) bob is kept.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SEQ = REPO / "datasets" / "SOMA" / "soma_subject1" / "dance_001_stageii.npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default=str(DEFAULT_SEQ),
                   help="SOMA-dataset SMPL-X stageii npz clip")
    p.add_argument("--seconds", type=float, default=6.0,
                   help="length of the clip window to extract (realtime playback duration)")
    p.add_argument("--start-frac", type=float, default=0.10,
                   help="where to start the window in the clip (skip the settle-in intro)")
    p.add_argument("--play-fps", type=float, default=25.0,
                   help="realtime playback fps -> T = seconds*play_fps frames span the window")
    p.add_argument("--num-betas", type=int, default=10)
    p.add_argument("--keep-translation", action="store_true",
                   help="keep full root translation (default: freeze horizontal drift)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    seq = np.load(args.seq, allow_pickle=True)
    assert str(seq["surface_model_type"]) == "smplx", "expected an SMPL-X stageii clip"
    src_fps = float(seq["mocap_frame_rate"])
    N = seq["trans"].shape[0]
    src_dur = N / src_fps

    play_fps = float(args.play_fps)
    win = min(args.seconds, src_dur)
    T = max(2, int(round(win * play_fps)))
    f0 = int(args.start_frac * N)
    f1 = min(N - 1, f0 + int(round(win * src_fps)))
    idx = np.linspace(f0, f1, T).astype(int)

    trans = np.asarray(seq["trans"], np.float32)[idx]
    if not args.keep_translation:
        # Freeze horizontal (X,Y ground-plane; Z-up world) drift to frame 0,
        # keep vertical bob -> in-place motion at a constant, well-framed size.
        trans[:, 0] = trans[0, 0]
        trans[:, 1] = trans[0, 1]

    betas = np.asarray(seq["betas"], np.float32)[: args.num_betas]
    out = dict(
        betas=betas,
        global_orient=np.asarray(seq["root_orient"], np.float32)[idx],
        body_pose=np.asarray(seq["pose_body"], np.float32)[idx],
        left_hand_pose=np.asarray(seq["pose_hand"], np.float32)[idx, :45],
        right_hand_pose=np.asarray(seq["pose_hand"], np.float32)[idx, 45:90],
        jaw_pose=np.asarray(seq["pose_jaw"], np.float32)[idx],
        leye_pose=np.asarray(seq["pose_eye"], np.float32)[idx, :3],
        reye_pose=np.asarray(seq["pose_eye"], np.float32)[idx, 3:6],
        trans=trans,
        duration_s=np.float32(win),
        play_fps=np.float32(play_fps),
        source=str(Path(args.seq).name),
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"loaded {Path(args.seq).name}: {N}f @ {src_fps:.0f} FPS ({src_dur:.1f}s) "
          f"-> {win:.1f}s window @ {play_fps:.0f} FPS = {T} frames")
    print(f"wrote {args.out}: body_pose {out['body_pose'].shape}  trans {trans.shape}  "
          f"betas {betas.shape}")


if __name__ == "__main__":
    main()
