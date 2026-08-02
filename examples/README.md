# Examples

## `pose_sequence.py`

End-to-end SMPL-X posing from a real mocap clip: loads an AMASS / SOMA
`*_stageii.npz` sequence plus a SMPL-X model file, poses it, and visualises the
result.

```bash
# Whole sequence, interactive Open3D animation (needs open3d)
bash pose_sequence.sh

# Single frame, before/after PNG (needs matplotlib)
python pose_sequence.py --mode frame --frame 120
```

Defaults are `data/smplx/SMPLX_NEUTRAL.npz` and a SOMA walk clip, both resolved
relative to the repo root so the scripts run from any directory. Override with
`--model` and `--sequence`.

Install the visualisation dependencies with `pip install -e ".[examples]"`.

### Useful flags

| Flag | Meaning |
| ---- | ------- |
| `--mode {frame,sequence}` | Single-frame PNG or full-sequence animation |
| `--frame N` | Frame to render in frame mode (negative indexes from the end) |
| `--max-frames N` | Cap how many frames get posed (`0` = all) |
| `--frame-stride N` / `--fps F` | Animation playback cadence |
| `--camera-view {front,side,top}` | Initial camera preset |
| `--source-up-axis {auto,y-up,z-up}` | Coordinate convention of the clip; `auto` infers it from root-trajectory variance |
| `--no-vis` | Pose without opening a window (useful for timing or CI) |

`python pose_sequence.py --help` lists the rest.

**macOS / Apple Silicon:** the JAX Metal backend can be unstable — prepend
`JAX_PLATFORMS=cpu`. `pose_sequence.sh` already defaults to CPU.
