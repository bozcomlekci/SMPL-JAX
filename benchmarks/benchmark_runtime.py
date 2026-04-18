from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _ensure_numpy_chumpy_compat() -> None:
    import inspect as _inspect
    if not hasattr(_inspect, "getargspec"):
        _inspect.getargspec = _inspect.getfullargspec  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "complex"):
        np.complex = complex  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    if not hasattr(np, "str"):
        np.str = str  # type: ignore[attr-defined]
    if not hasattr(np, "unicode"):
        np.unicode = str  # type: ignore[attr-defined]


def _load_jax_smplx_model(model_path: Path, num_betas: int, num_expression_coeffs: int):
    from smpl_jax import SMPLXModel

    if model_path.suffix == ".pkl":
        return SMPLXModel.load(
            str(model_path),
            num_betas=num_betas,
            num_expression_coeffs=num_expression_coeffs,
        )

    if model_path.suffix != ".npz":
        raise ValueError(f"Unsupported SMPL-X model format for JAX: {model_path.suffix}")

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
        exprdirs = shapedirs[..., 300:]

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


def _resolve_torch_device(torch, preference: str):
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _cpu_device_name() -> str:
    cpu_name = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_name = line.split(":", 1)[1].strip()
                if cpu_name:
                    break

    if cpu_name:
        return cpu_name

    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
            )
            mac_name = completed.stdout.strip()
            if mac_name:
                return mac_name
        except Exception:
            pass

    fallback = platform.processor() or platform.machine()
    return fallback if fallback else "CPU"


@lru_cache(maxsize=1)
def _gpu_device_name_fallback() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if names:
            return names[0]
    except Exception:
        pass
    return "GPU"


_MIB_BYTES = 1024.0 * 1024.0


def _query_process_gpu_memory_mib(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    total_mib = 0.0
    found = False
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
        except ValueError:
            continue
        if row_pid != int(pid):
            continue
        value_text = parts[1].split()[0]
        try:
            used_mib = float(value_text)
        except ValueError:
            continue
        total_mib += used_mib
        found = True

    if found:
        return total_mib
    return None


def _query_process_gpu_memory_peak_window_mib(
    pid: int,
    duration_s: float = 0.03,
    poll_interval_s: float = 0.005,
) -> float | None:
    deadline = time.perf_counter() + max(0.0, float(duration_s))
    peak_mib: float | None = None
    while True:
        sample = _query_process_gpu_memory_mib(pid)
        if sample is not None:
            peak_mib = sample if peak_mib is None else max(peak_mib, sample)

        if time.perf_counter() >= deadline:
            break
        time.sleep(max(0.0, float(poll_interval_s)))
    return peak_mib


def _query_process_gpu_memory_mib_robust(pid: int) -> float | None:
    # Fast path first, then a short polling window to catch brief kernels that
    # may be missed by a single nvidia-smi snapshot.
    sample = _query_process_gpu_memory_mib(pid)
    if sample is not None and sample > 0.0:
        return sample

    peak_window = _query_process_gpu_memory_peak_window_mib(
        pid,
        duration_s=0.03,
        poll_interval_s=0.005,
    )
    if peak_window is not None:
        return peak_window
    return sample


def _gpu_memory_summary(samples_mib: list[float], source: str) -> dict[str, Any]:
    if not samples_mib:
        return {
            "gpu_memory_peak_mib": None,
            "gpu_memory_mean_mib": None,
            "gpu_memory_source": source,
        }

    arr = np.asarray(samples_mib, dtype=np.float64)
    return {
        "gpu_memory_peak_mib": float(np.max(arr)),
        "gpu_memory_mean_mib": float(np.mean(arr)),
        "gpu_memory_source": source,
    }


def _run_subprocess_with_optional_gpu_peak(
    cmd: list[str],
    env: dict[str, str],
    track_gpu_memory: bool,
) -> tuple[subprocess.CompletedProcess[str], float | None]:
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    peak_mib: float | None = None
    while True:
        if track_gpu_memory:
            sample = _query_process_gpu_memory_mib(proc.pid)
            if sample is not None:
                peak_mib = sample if peak_mib is None else max(peak_mib, sample)

        if proc.poll() is not None:
            break
        time.sleep(0.005)

    stdout, stderr = proc.communicate()
    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=int(proc.returncode if proc.returncode is not None else 1),
        stdout=stdout,
        stderr=stderr,
    )
    return completed, peak_mib


def _safe_pkg_version(package_name: str) -> str:
    try:
        return str(importlib_metadata.version(package_name))
    except Exception:
        return "unknown"


def _pkg_version_any(*package_names: str) -> str:
    for package_name in package_names:
        version = _safe_pkg_version(package_name)
        if version != "unknown":
            return version
    return "unknown"


def _runtime_stack(entries: dict[str, str]) -> str:
    ordered = [f"{name}={version}" for name, version in entries.items() if version]
    return ", ".join(ordered)


def _jax_device_label(device) -> str:
    return "gpu" if getattr(device, "platform", "").lower() in {"gpu", "cuda", "metal"} else "cpu"


def _jax_device_name(device) -> str:
    device_kind = str(getattr(device, "device_kind", "")).strip()
    if device_kind:
        kind_lower = device_kind.lower()
        if kind_lower in {"cpu"}:
            return _cpu_device_name()
        if kind_lower in {"gpu", "cuda"}:
            return _gpu_device_name_fallback()
        if kind_lower == "metal":
            return _cpu_device_name()
        return device_kind
    device_class = _jax_device_label(device)
    if device_class == "gpu":
        return _gpu_device_name_fallback()
    return _cpu_device_name()


def _torch_device_label(device) -> str:
    return "gpu" if str(getattr(device, "type", "")).lower() == "cuda" else "cpu"


def _torch_device_name(torch, device) -> str:
    if _torch_device_label(device) == "gpu":
        index = device.index if device.index is not None else 0
        try:
            return str(torch.cuda.get_device_name(index))
        except Exception:
            return _gpu_device_name_fallback()
    return _cpu_device_name()


def _method_enabled(method_filter: set[str], method: str) -> bool:
    return not method_filter or method in method_filter


def _summary(times_s: list[float], n_frames: int) -> dict[str, float]:
    arr = np.asarray(times_s, dtype=np.float64)
    mean_s = float(np.mean(arr))
    return {
        "mean_ms": mean_s * 1e3,
        "p50_ms": float(np.percentile(arr, 50)) * 1e3,
        "p95_ms": float(np.percentile(arr, 95)) * 1e3,
        "fps": float(n_frames / mean_s),
    }


def _time_repeats(
    run_once,
    sync_once,
    repeats: int,
    warmup: int,
    before_timed_iteration: Callable[[], None] | None = None,
    after_timed_iteration: Callable[[], float | None] | None = None,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        if before_timed_iteration is not None:
            before_timed_iteration()
        out = run_once()
        sync_once(out)
        if after_timed_iteration is not None:
            after_timed_iteration()

    times_s: list[float] = []
    gpu_memory_samples: list[float] = []
    for _ in range(repeats):
        if before_timed_iteration is not None:
            before_timed_iteration()
        t0 = time.perf_counter()
        out = run_once()
        sync_once(out)
        t1 = time.perf_counter()
        times_s.append(t1 - t0)
        if after_timed_iteration is not None:
            sample = after_timed_iteration()
            if sample is not None:
                gpu_memory_samples.append(float(sample))
    return times_s, gpu_memory_samples


def _validate_comparable_sequence_rows(results: list[dict[str, Any]], strict: bool) -> None:
    sequence_rows = [row for row in results if bool(row.get("uses_input_sequence", False))]
    if not sequence_rows:
        return

    groups = {
        (str(row.get("sequence")), int(row.get("frames", -1)))
        for row in sequence_rows
    }
    if len(groups) <= 1:
        return

    group_text = ", ".join([f"(sequence={seq}, frames={frames})" for seq, frames in sorted(groups)])
    message = (
        "Found multiple sequence/frame groups among comparable rows: "
        f"{group_text}. Use a single --sequence and consistent frame length "
        "(or pass --allow-mixed-sequence-lengths)."
    )

    if strict:
        raise RuntimeError(message)
    print(f"Warning: {message}")


def _validate_processing_mode_rows(results: list[dict[str, Any]], strict: bool) -> None:
    sequence_rows = [row for row in results if bool(row.get("uses_input_sequence", False))]
    if not sequence_rows:
        return

    grouped_modes: dict[tuple[str, str], set[str]] = {}
    for row in sequence_rows:
        family = str(row.get("benchmark_family", "unknown"))
        device_class = str(row.get("device_class", row.get("device", "unknown")))
        mode = str(row.get("processing_mode", "unknown"))
        key = (family, device_class)
        grouped_modes.setdefault(key, set()).add(mode)

    mixed = {key: sorted(modes) for key, modes in grouped_modes.items() if len(modes) > 1}
    if not mixed:
        return

    details = "; ".join(
        [f"family={family}, device={device}, modes={modes}" for (family, device), modes in sorted(mixed.items())]
    )
    message = (
        "Found mixed processing modes within at least one family/device group. "
        "This is not a strict apples-to-apples comparison. "
        f"Groups: {details}."
    )
    if strict:
        raise RuntimeError(message)
    print(f"Warning: {message}")


def _effective_sequence_batch_size(requested: int, n_frames: int) -> int:
    if requested <= 0:
        return max(1, int(n_frames))
    return max(1, min(int(requested), int(n_frames)))


def _slice_sequence(sequence: dict[str, np.ndarray], n_frames: int) -> dict[str, np.ndarray]:
    n = int(max(1, n_frames))
    source_n = int(sequence["trans"].shape[0])
    sliced: dict[str, np.ndarray] = {}
    for key, value in sequence.items():
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == source_n:
            if n <= source_n:
                sliced[key] = arr[:n].copy()
            else:
                reps = (n + source_n - 1) // source_n
                tile_shape = (reps,) + (1,) * (arr.ndim - 1)
                tiled = np.tile(arr, tile_shape)
                sliced[key] = tiled[:n].copy()
        else:
            sliced[key] = arr.copy()
    return sliced


def _parse_batch_sweep_sizes(spec: str, full_frames: int) -> list[int]:
    text = str(spec or "").strip()
    if not text:
        return []

    sizes: set[int] = set()
    tokens = [token.strip().lower() for token in text.split(",") if token.strip()]
    for token in tokens:
        if token in {"full", "max", "all"}:
            sizes.add(int(full_frames))
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid batch sweep token '{token}'. Use positive integers and/or 'full'."
            ) from exc
        if value <= 0:
            raise ValueError(f"Batch sweep size must be > 0, got {value}.")
        sizes.add(int(value))

    sizes.add(int(full_frames))
    return sorted(sizes)


def _load_sequence(
    sequence_path: Path,
    max_frames: int,
    tile_to_max_frames: bool = False,
) -> dict[str, np.ndarray]:
    seq = np.load(sequence_path, allow_pickle=True)
    required = ["trans", "root_orient", "pose_body", "pose_hand", "betas"]
    missing = [k for k in required if k not in seq.files]
    if missing:
        raise KeyError(f"Missing keys in sequence: {missing}")

    trans = np.asarray(seq["trans"], dtype=np.float32)
    root_orient = np.asarray(seq["root_orient"], dtype=np.float32)
    pose_body = np.asarray(seq["pose_body"], dtype=np.float32)
    pose_hand = np.asarray(seq["pose_hand"], dtype=np.float32)
    betas = np.asarray(seq["betas"], dtype=np.float32)

    sequence = {
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "pose_hand": pose_hand,
        "betas": betas,
    }

    if max_frames > 0:
        target_frames = int(max_frames)
        if tile_to_max_frames:
            return _slice_sequence(sequence, target_frames)
        return _slice_sequence(sequence, min(target_frames, int(trans.shape[0])))

    return sequence


def benchmark_jax_smplx(
    sequence: dict[str, np.ndarray],
    smplx_model_path: Path,
    num_betas: int,
    num_expression_coeffs: int,
    repeats: int,
    warmup: int,
    jax_platform: str,
    xla_gpu_autotune_level: int,
) -> dict[str, Any]:
    os.environ["JAX_PLATFORMS"] = "METAL" if jax_platform.lower() == "metal" else jax_platform
    os.environ["XLA_FLAGS"] = f"--xla_gpu_autotune_level={xla_gpu_autotune_level}"
    import jax
    import jax.numpy as jnp

    from smpl_jax import SMPLXModel, SMPLXParams

    n_frames = int(sequence["trans"].shape[0])
    betas = sequence["betas"][:num_betas]
    betas_batch = np.broadcast_to(betas[None, :], (n_frames, num_betas)).copy()

    params = SMPLXParams(
        betas=jnp.asarray(betas_batch, dtype=jnp.float32),
        body_pose=jnp.asarray(sequence["pose_body"], dtype=jnp.float32),
        global_orient=jnp.asarray(sequence["root_orient"], dtype=jnp.float32),
        transl=jnp.asarray(sequence["trans"], dtype=jnp.float32),
        expression=jnp.zeros((n_frames, num_expression_coeffs), dtype=jnp.float32),
        jaw_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        leye_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        reye_pose=jnp.zeros((n_frames, 3), dtype=jnp.float32),
        left_hand_pose=jnp.asarray(sequence["pose_hand"][:, :45], dtype=jnp.float32),
        right_hand_pose=jnp.asarray(sequence["pose_hand"][:, 45:90], dtype=jnp.float32),
    )

    model = _load_jax_smplx_model(
        model_path=smplx_model_path,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
    )
    forward = jax.jit(model)

    first = forward(params)
    first.vertices.block_until_ready()

    def run_once():
        return forward(params)

    def sync_once(out):
        out.vertices.block_until_ready()

    jax_device = jax.devices()[0]
    jax_device_class = _jax_device_label(jax_device)
    gpu_memory_source = "nvidia_smi_process" if jax_device_class == "gpu" else "not_applicable_cpu"
    after_memory_cb = (lambda: _query_process_gpu_memory_mib_robust(os.getpid())) if jax_device_class == "gpu" else None

    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        after_timed_iteration=after_memory_cb,
    )
    info = _summary(times_s, n_frames=n_frames)
    info.update({
        "implementation": "smpl_jax_smplx",
        "benchmark_family": "smplx",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "jax-xla",
        "impl_language": "python",
        "impl_dtype": "float32",
        "impl_autograd": "n/a (jit inference)",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": jax_device_class,
        "device": _jax_device_name(jax_device),
        "frames": n_frames,
        "vertices": int(first.vertices.shape[1]),
        "runtime_stack": _runtime_stack({
            "jax": str(getattr(jax, "__version__", "unknown")),
            "jaxlib": _pkg_version_any("jaxlib"),
            "smpl-jax": _pkg_version_any("smpl-jax", "smpl_jax"),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_jax_smpl(
    sequence: dict[str, np.ndarray],
    smpl_model_path: Path,
    num_betas: int,
    repeats: int,
    warmup: int,
    jax_platform: str,
    xla_gpu_autotune_level: int,
) -> dict[str, Any]:
    os.environ["JAX_PLATFORMS"] = "METAL" if jax_platform.lower() == "metal" else jax_platform
    os.environ["XLA_FLAGS"] = f"--xla_gpu_autotune_level={xla_gpu_autotune_level}"
    import jax
    import jax.numpy as jnp

    from smpl_jax import SMPLModel, SMPLParams

    _ensure_numpy_chumpy_compat()

    n_frames = int(sequence["trans"].shape[0])
    betas = sequence["betas"][:num_betas]
    betas_batch = np.broadcast_to(betas[None, :], (n_frames, num_betas)).copy()

    zeros6 = np.zeros((n_frames, 6), dtype=np.float32)
    body_pose69 = np.concatenate([sequence["pose_body"], zeros6], axis=1)

    params = SMPLParams(
        betas=jnp.asarray(betas_batch, dtype=jnp.float32),
        body_pose=jnp.asarray(body_pose69, dtype=jnp.float32),
        global_orient=jnp.asarray(sequence["root_orient"], dtype=jnp.float32),
        transl=jnp.asarray(sequence["trans"], dtype=jnp.float32),
    )

    model = SMPLModel.load(str(smpl_model_path), num_betas=num_betas)
    forward = jax.jit(model)

    first = forward(params)
    first.vertices.block_until_ready()

    def run_once():
        return forward(params)

    def sync_once(out):
        out.vertices.block_until_ready()

    jax_device = jax.devices()[0]
    jax_device_class = _jax_device_label(jax_device)
    gpu_memory_source = "nvidia_smi_process" if jax_device_class == "gpu" else "not_applicable_cpu"
    after_memory_cb = (lambda: _query_process_gpu_memory_mib_robust(os.getpid())) if jax_device_class == "gpu" else None

    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        after_timed_iteration=after_memory_cb,
    )
    info = _summary(times_s, n_frames=n_frames)
    info.update({
        "implementation": "smpl_jax_smpl",
        "benchmark_family": "smpl",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "jax-xla",
        "impl_language": "python",
        "impl_dtype": "float32",
        "impl_autograd": "n/a (jit inference)",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": jax_device_class,
        "device": _jax_device_name(jax_device),
        "frames": n_frames,
        "vertices": int(first.vertices.shape[1]),
        "note": "SMPL baseline; pose mapped from SMPL-X body sequence.",
        "runtime_stack": _runtime_stack({
            "jax": str(getattr(jax, "__version__", "unknown")),
            "jaxlib": _pkg_version_any("jaxlib"),
            "smpl-jax": _pkg_version_any("smpl-jax", "smpl_jax"),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_torch_smplx(
    sequence: dict[str, np.ndarray],
    smplx_model_dir: Path,
    smplx_ext: str,
    num_betas: int,
    num_expression_coeffs: int,
    repeats: int,
    warmup: int,
    torch_device_preference: str,
) -> dict[str, Any]:
    import torch
    import smplx

    n_frames = int(sequence["trans"].shape[0])
    device = _resolve_torch_device(torch, torch_device_preference)

    betas = sequence["betas"][:num_betas]
    betas_batch = np.broadcast_to(betas[None, :], (n_frames, num_betas)).copy()

    model = smplx.create(
        model_path=str(smplx_model_dir),
        model_type="smplx",
        gender="neutral",
        ext=smplx_ext,
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        flat_hand_mean=True,
        batch_size=n_frames,
    ).to(device)
    model.eval()

    betas_t = torch.from_numpy(betas_batch).to(device)
    body_pose_t = torch.from_numpy(sequence["pose_body"]).to(device)
    global_orient_t = torch.from_numpy(sequence["root_orient"]).to(device)
    transl_t = torch.from_numpy(sequence["trans"]).to(device)
    left_hand_t = torch.from_numpy(sequence["pose_hand"][:, :45]).to(device)
    right_hand_t = torch.from_numpy(sequence["pose_hand"][:, 45:90]).to(device)
    expression_t = torch.zeros((n_frames, num_expression_coeffs), dtype=torch.float32, device=device)

    @torch.inference_mode()
    def run_once():
        return model(
            betas=betas_t,
            body_pose=body_pose_t,
            global_orient=global_orient_t,
            transl=transl_t,
            left_hand_pose=left_hand_t,
            right_hand_pose=right_hand_t,
            expression=expression_t,
            return_verts=True,
        )

    def sync_once(_):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    first = run_once()
    sync_once(first)

    torch_device_class = _torch_device_label(device)
    gpu_memory_source = "not_applicable_cpu"
    before_memory_cb = None
    after_memory_cb = None
    if torch_device_class == "gpu":
        gpu_memory_source = "torch_cuda_allocator_peak"

        def before_memory_cb() -> None:
            torch.cuda.reset_peak_memory_stats(device)

        def after_memory_cb() -> float:
            return float(torch.cuda.max_memory_allocated(device)) / _MIB_BYTES

    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        before_timed_iteration=before_memory_cb,
        after_timed_iteration=after_memory_cb,
    )
    info = _summary(times_s, n_frames=n_frames)
    info.update({
        "implementation": "smplx_torch",
        "benchmark_family": "smplx",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "torch",
        "impl_language": "python",
        "impl_dtype": "float32",
        "impl_autograd": "torch.inference_mode",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": torch_device_class,
        "device": _torch_device_name(torch, device),
        "frames": n_frames,
        "vertices": int(first.vertices.shape[1]),
        "runtime_stack": _runtime_stack({
            "torch": str(getattr(torch, "__version__", _pkg_version_any("torch"))),
            "smplx": _pkg_version_any("smplx"),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_torch_smpl(
    sequence: dict[str, np.ndarray],
    smpl_model_dir: Path,
    smpl_ext: str,
    num_betas: int,
    repeats: int,
    warmup: int,
    torch_device_preference: str,
) -> dict[str, Any]:
    _ensure_numpy_chumpy_compat()

    import torch
    import smplx

    n_frames = int(sequence["trans"].shape[0])
    device = _resolve_torch_device(torch, torch_device_preference)

    betas = sequence["betas"][:num_betas]
    betas_batch = np.broadcast_to(betas[None, :], (n_frames, num_betas)).copy()

    zeros6 = np.zeros((n_frames, 6), dtype=np.float32)
    body_pose69 = np.concatenate([sequence["pose_body"], zeros6], axis=1)

    model = smplx.create(
        model_path=str(smpl_model_dir),
        model_type="smpl",
        gender="neutral",
        ext=smpl_ext,
        num_betas=num_betas,
        batch_size=n_frames,
    ).to(device)
    model.eval()

    betas_t = torch.from_numpy(betas_batch).to(device)
    body_pose_t = torch.from_numpy(body_pose69).to(device)
    global_orient_t = torch.from_numpy(sequence["root_orient"]).to(device)
    transl_t = torch.from_numpy(sequence["trans"]).to(device)

    @torch.inference_mode()
    def run_once():
        return model(
            betas=betas_t,
            body_pose=body_pose_t,
            global_orient=global_orient_t,
            transl=transl_t,
            return_verts=True,
        )

    def sync_once(_):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    first = run_once()
    sync_once(first)

    torch_device_class = _torch_device_label(device)
    gpu_memory_source = "not_applicable_cpu"
    before_memory_cb = None
    after_memory_cb = None
    if torch_device_class == "gpu":
        gpu_memory_source = "torch_cuda_allocator_peak"

        def before_memory_cb() -> None:
            torch.cuda.reset_peak_memory_stats(device)

        def after_memory_cb() -> float:
            return float(torch.cuda.max_memory_allocated(device)) / _MIB_BYTES

    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        before_timed_iteration=before_memory_cb,
        after_timed_iteration=after_memory_cb,
    )
    info = _summary(times_s, n_frames=n_frames)
    info.update({
        "implementation": "smplx_torch_smpl",
        "benchmark_family": "smpl",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "torch",
        "impl_language": "python",
        "impl_dtype": "float32",
        "impl_autograd": "torch.inference_mode",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": torch_device_class,
        "device": _torch_device_name(torch, device),
        "frames": n_frames,
        "vertices": int(first.vertices.shape[1]),
        "note": "SMPL baseline via vchoutas/smplx (model_type=smpl); pose mapped from SMPL-X body sequence.",
        "runtime_stack": _runtime_stack({
            "torch": str(getattr(torch, "__version__", _pkg_version_any("torch"))),
            "smplx": _pkg_version_any("smplx"),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_torch_smplpytorch(
    sequence: dict[str, np.ndarray],
    smpl_model_pkl: Path,
    repeats: int,
    warmup: int,
    torch_device_preference: str,
) -> dict[str, Any]:
    _ensure_numpy_chumpy_compat()

    import torch
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer

    n_frames = int(sequence["trans"].shape[0])
    device = _resolve_torch_device(torch, torch_device_preference)

    model_root = smpl_model_pkl.parent
    expected = model_root / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
    if not expected.exists() or not expected.is_file():
        if expected.exists() or expected.is_symlink():
            expected.unlink()
        os.symlink(smpl_model_pkl.resolve(), expected)

    layer = SMPL_Layer(gender="neutral", model_root=str(model_root)).to(device)
    layer.eval()

    # smplpytorch is SMPL (24 joints). We map SMPL-X body to SMPL-style axis-angle:
    # pose72 = [global_orient(3), body_pose(63), zeros(6)]
    zeros6 = np.zeros((n_frames, 6), dtype=np.float32)
    pose69 = np.concatenate([sequence["pose_body"], zeros6], axis=1)
    pose72 = np.concatenate([sequence["root_orient"], pose69], axis=1)

    n_betas_layer = int(layer.th_shapedirs.shape[-1])
    betas_src = np.asarray(sequence["betas"], dtype=np.float32)
    betas_one = np.zeros((n_betas_layer,), dtype=np.float32)
    copy_n = min(n_betas_layer, betas_src.shape[0])
    betas_one[:copy_n] = betas_src[:copy_n]
    betas_batch = np.broadcast_to(betas_one[None, :], (n_frames, n_betas_layer)).copy()

    pose_t = torch.from_numpy(pose72).to(device)
    betas_t = torch.from_numpy(betas_batch).to(device)
    trans_t = torch.from_numpy(sequence["trans"]).to(device)

    @torch.inference_mode()
    def run_once():
        return layer(pose_t, betas_t, trans_t)

    def sync_once(_):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    first = run_once()
    sync_once(first)

    verts = first[0] if isinstance(first, (tuple, list)) else first
    torch_device_class = _torch_device_label(device)
    gpu_memory_source = "not_applicable_cpu"
    before_memory_cb = None
    after_memory_cb = None
    if torch_device_class == "gpu":
        gpu_memory_source = "torch_cuda_allocator_peak"

        def before_memory_cb() -> None:
            torch.cuda.reset_peak_memory_stats(device)

        def after_memory_cb() -> float:
            return float(torch.cuda.max_memory_allocated(device)) / _MIB_BYTES

    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        before_timed_iteration=before_memory_cb,
        after_timed_iteration=after_memory_cb,
    )
    info = _summary(times_s, n_frames=n_frames)
    info.update({
        "implementation": "smplpytorch_torch",
        "benchmark_family": "smpl",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "torch",
        "impl_language": "python",
        "impl_dtype": "float32",
        "impl_autograd": "torch.inference_mode",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": torch_device_class,
        "device": _torch_device_name(torch, device),
        "frames": n_frames,
        "vertices": int(verts.shape[1]),
        "note": "SMPL baseline; pose mapped from SMPL-X body sequence.",
        "runtime_stack": _runtime_stack({
            "torch": str(getattr(torch, "__version__", _pkg_version_any("torch"))),
            "smplpytorch": _pkg_version_any("smplpytorch"),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_smplxpp_python(
    sequence: dict[str, np.ndarray],
    smpl_model_npz: Path,
    repeats: int,
    warmup: int,
    smplxpp_sequence_batch_size: int,
    smplxpp_device_preference: str,
) -> dict[str, Any]:
    import smplxpp

    if not smpl_model_npz.exists():
        raise FileNotFoundError(f"smplxpp SMPL model npz not found: {smpl_model_npz}")

    n_frames = int(sequence["trans"].shape[0])
    cuda_supported = bool(getattr(smplxpp, "cuda", False))
    if smplxpp_device_preference == "cuda":
        if not cuda_supported:
            raise RuntimeError("smplxpp CUDA backend is not available in this build")
        force_cpu = False
    elif smplxpp_device_preference == "cpu":
        force_cpu = True
    else:
        force_cpu = not cuda_supported

    model = smplxpp.ModelS(str(smpl_model_npz), "", smplxpp.Gender.neutral)
    batch_size = _effective_sequence_batch_size(smplxpp_sequence_batch_size, n_frames)
    fused_cls = getattr(smplxpp, "FusedBatchForwardS", None)
    use_fused_batch_forward = fused_cls is not None

    zeros6 = np.zeros((n_frames, 6), dtype=np.float32)
    pose69 = np.concatenate([sequence["pose_body"], zeros6], axis=1)
    pose72 = np.concatenate([sequence["root_orient"], pose69], axis=1).astype(np.float32, copy=False)
    trans = np.asarray(sequence["trans"], dtype=np.float32)

    n_shape = int(smplxpp.ModelS.n_shape_blends)
    betas_src = np.asarray(sequence["betas"], dtype=np.float32)
    betas = np.zeros((n_shape,), dtype=np.float32)
    copy_n = min(n_shape, betas_src.shape[0])
    betas[:copy_n] = betas_src[:copy_n]
    if use_fused_batch_forward:
        fused_runner = fused_cls(model, batch_size)
        fused_runner.set_betas(betas)
    else:
        bodies = [smplxpp.BodyS(model) for _ in range(batch_size)]
        for body in bodies:
            body.betas = betas

    def run_once():
        if use_fused_batch_forward:
            last = None
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                chunk_trans = trans[start:end]
                chunk_pose = pose72[start:end]
                if end == n_frames:
                    last = np.asarray(
                        fused_runner.forward_last(
                            chunk_trans,
                            chunk_pose,
                            force_cpu=force_cpu,
                        )
                    )
                else:
                    fused_runner.update(
                        chunk_trans,
                        chunk_pose,
                        force_cpu=force_cpu,
                    )
            return last

        last_slot = 0
        for start in range(0, n_frames, batch_size):
            chunk = min(batch_size, n_frames - start)
            for slot in range(chunk):
                i = start + slot
                body = bodies[slot]
                body.trans = trans[i]
                body.pose = pose72[i]
                body.update(force_cpu=force_cpu)
                last_slot = slot
        return np.asarray(bodies[last_slot].verts)

    def sync_once(_):
        return None

    first = run_once()
    gpu_memory_source = "nvidia_smi_process" if not force_cpu else "not_applicable_cpu"
    after_memory_cb = (lambda: _query_process_gpu_memory_mib_robust(os.getpid())) if not force_cpu else None
    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        after_timed_iteration=after_memory_cb,
    )

    info = _summary(times_s, n_frames=n_frames)
    device_class = "cpu" if force_cpu else "gpu"
    is_full_batch = batch_size == n_frames
    scope = "full_sequence_batch_forward" if is_full_batch else "full_sequence_frame_chunked_loop"
    mode = "batch_sequence_forward" if is_full_batch else "frame_loop_multi_body_batch"
    if use_fused_batch_forward:
        batch_impl = (
            "single_full_sequence_fused_batch_forward"
            if is_full_batch
            else "chunked_fused_batch_forward"
        )
        batching_note = (
            f"single full-sequence fused tensor batch forward (batch={batch_size})"
            if is_full_batch
            else f"chunked fused tensor batch forward (batch={batch_size})"
        )
        note_suffix = "using the fused smplxpp batch-forward C++ binding."
    else:
        batch_impl = "single_full_sequence_batch" if is_full_batch else "chunked_multi_body_loop"
        batching_note = (
            f"single full-sequence multi-body batch (batch={batch_size})"
            if is_full_batch
            else f"chunked multi-body loop (batch={batch_size})"
        )
        note_suffix = "smplxpp currently exposes single-body update only (no sequence batch-forward API)."
    info.update({
        "implementation": "smplxpp_python",
        "benchmark_family": "smpl",
        "benchmark_scope": scope,
        "processing_mode": mode,
        "impl_backend": "smplxpp-cpp",
        "impl_language": "python+cpp",
        "impl_dtype": "float32",
        "impl_autograd": "none (manual update)",
        "impl_sequence_strategy": batch_impl,
        "device_class": device_class,
        "device": _cpu_device_name() if device_class == "cpu" else _gpu_device_name_fallback(),
        "frames": n_frames,
        "sequence_batch_size": batch_size,
        "batch_implementation": batch_impl,
        "smplxpp_fused_batch_forward": bool(use_fused_batch_forward),
        "vertices": int(first.shape[0]),
        "note": (
            "SMPL baseline via sxyu/smplxpp Python binding; "
            f"{batching_note} over sequence. "
            f"{note_suffix}"
        ),
        "runtime_stack": _runtime_stack({
            "smplxpp": _pkg_version_any("smplxpp"),
            "numpy": str(np.__version__),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_smplxpp_python_smplx(
    sequence: dict[str, np.ndarray],
    smplx_model_npz: Path,
    repeats: int,
    warmup: int,
    smplxpp_sequence_batch_size: int,
    smplxpp_device_preference: str,
) -> dict[str, Any]:
    import smplxpp

    if not smplx_model_npz.exists():
        raise FileNotFoundError(f"smplxpp SMPL-X model npz not found: {smplx_model_npz}")

    n_frames = int(sequence["trans"].shape[0])
    cuda_supported = bool(getattr(smplxpp, "cuda", False))
    if smplxpp_device_preference == "cuda":
        if not cuda_supported:
            raise RuntimeError("smplxpp CUDA backend is not available in this build")
        force_cpu = False
    elif smplxpp_device_preference == "cpu":
        force_cpu = True
    else:
        force_cpu = not cuda_supported

    model = smplxpp.ModelX(str(smplx_model_npz), "", smplxpp.Gender.neutral)
    batch_size = _effective_sequence_batch_size(smplxpp_sequence_batch_size, n_frames)
    fused_cls = getattr(smplxpp, "FusedBatchForwardX", None)
    use_fused_batch_forward = fused_cls is not None

    trans = np.asarray(sequence["trans"], dtype=np.float32)
    jaw_eye_zeros = np.zeros((n_frames, 9), dtype=np.float32)
    # BodyX expects 55 axis-angle joints (165 dims):
    # [global(3), body(63), jaw+eyes(9), hands(90)]
    pose165 = np.concatenate(
        [
            np.asarray(sequence["root_orient"], dtype=np.float32),
            np.asarray(sequence["pose_body"], dtype=np.float32),
            jaw_eye_zeros,
            np.asarray(sequence["pose_hand"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    n_shape = int(smplxpp.ModelX.n_shape_blends)
    betas_src = np.asarray(sequence["betas"], dtype=np.float32)
    betas = np.zeros((n_shape,), dtype=np.float32)
    copy_n = min(n_shape, betas_src.shape[0])
    betas[:copy_n] = betas_src[:copy_n]
    if use_fused_batch_forward:
        fused_runner = fused_cls(model, batch_size)
        fused_runner.set_betas(betas)
    else:
        bodies = [smplxpp.BodyX(model) for _ in range(batch_size)]
        for body in bodies:
            body.betas = betas

    def run_once():
        if use_fused_batch_forward:
            last = None
            for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                chunk_trans = trans[start:end]
                chunk_pose = pose165[start:end]
                if end == n_frames:
                    last = np.asarray(
                        fused_runner.forward_last(
                            chunk_trans,
                            chunk_pose,
                            force_cpu=force_cpu,
                        )
                    )
                else:
                    fused_runner.update(
                        chunk_trans,
                        chunk_pose,
                        force_cpu=force_cpu,
                    )
            return last

        last_slot = 0
        for start in range(0, n_frames, batch_size):
            chunk = min(batch_size, n_frames - start)
            for slot in range(chunk):
                i = start + slot
                body = bodies[slot]
                body.trans = trans[i]
                body.pose = pose165[i]
                body.update(force_cpu=force_cpu)
                last_slot = slot
        return np.asarray(bodies[last_slot].verts)

    def sync_once(_):
        return None

    first = run_once()
    gpu_memory_source = "nvidia_smi_process" if not force_cpu else "not_applicable_cpu"
    after_memory_cb = (lambda: _query_process_gpu_memory_mib_robust(os.getpid())) if not force_cpu else None
    times_s, gpu_memory_samples = _time_repeats(
        run_once,
        sync_once,
        repeats=repeats,
        warmup=warmup,
        after_timed_iteration=after_memory_cb,
    )

    info = _summary(times_s, n_frames=n_frames)
    device_class = "cpu" if force_cpu else "gpu"
    is_full_batch = batch_size == n_frames
    scope = "full_sequence_batch_forward" if is_full_batch else "full_sequence_frame_chunked_loop"
    mode = "batch_sequence_forward" if is_full_batch else "frame_loop_multi_body_batch"
    if use_fused_batch_forward:
        batch_impl = (
            "single_full_sequence_fused_batch_forward"
            if is_full_batch
            else "chunked_fused_batch_forward"
        )
        batching_note = (
            f"single full-sequence fused tensor batch forward (batch={batch_size})"
            if is_full_batch
            else f"chunked fused tensor batch forward (batch={batch_size})"
        )
        note_suffix = "using the fused smplxpp batch-forward C++ binding."
    else:
        batch_impl = "single_full_sequence_batch" if is_full_batch else "chunked_multi_body_loop"
        batching_note = (
            f"single full-sequence multi-body batch (batch={batch_size})"
            if is_full_batch
            else f"chunked multi-body loop (batch={batch_size})"
        )
        note_suffix = (
            "smplxpp currently exposes single-body update only "
            "(no sequence batch-forward API)."
        )
    info.update({
        "implementation": "smplxpp_python_smplx",
        "benchmark_family": "smplx",
        "benchmark_scope": scope,
        "processing_mode": mode,
        "impl_backend": "smplxpp-cpp",
        "impl_language": "python+cpp",
        "impl_dtype": "float32",
        "impl_autograd": "none (manual update)",
        "impl_sequence_strategy": batch_impl,
        "device_class": device_class,
        "device": _cpu_device_name() if device_class == "cpu" else _gpu_device_name_fallback(),
        "frames": n_frames,
        "sequence_batch_size": batch_size,
        "batch_implementation": batch_impl,
        "smplxpp_fused_batch_forward": bool(use_fused_batch_forward),
        "vertices": int(first.shape[0]),
        "note": (
            "SMPL-X baseline via sxyu/smplxpp Python binding; "
            f"{batching_note} over sequence "
            f"(jaw/eye pose set to zero), {note_suffix}"
        ),
        "runtime_stack": _runtime_stack({
            "smplxpp": _pkg_version_any("smplxpp"),
            "numpy": str(np.__version__),
        }),
    })
    info.update(_gpu_memory_summary(gpu_memory_samples, source=gpu_memory_source))
    return info


def benchmark_torchure_smplx_external(
    benchmark_bin: Path,
    smpl_model_npz: Path,
    sequence_path: Path,
    repeats: int,
    warmup: int,
    max_frames: int,
    n_frames_ref: int,
    torchure_device_preference: str,
    torchure_dtype: str,
) -> dict[str, Any]:
    if not benchmark_bin.exists():
        raise FileNotFoundError(f"torchure benchmark binary not found: {benchmark_bin}")
    if not smpl_model_npz.exists():
        raise FileNotFoundError(f"torchure SMPL npz model not found: {smpl_model_npz}")

    if not sequence_path.exists():
        raise FileNotFoundError(f"torchure sequence npz not found: {sequence_path}")

    temp_sequence_npz: Path | None = None
    try:
        seq_npz = np.load(sequence_path, allow_pickle=True)
        required = ["trans", "root_orient", "pose_body", "betas"]
        missing = [k for k in required if k not in seq_npz.files]
        if missing:
            raise KeyError(f"Missing keys in sequence for torchure: {missing}")

        np_dtype = np.float32 if torchure_dtype == "float32" else np.float64
        trans = np.asarray(seq_npz["trans"], dtype=np_dtype)
        root_orient = np.asarray(seq_npz["root_orient"], dtype=np_dtype)
        pose_body = np.asarray(seq_npz["pose_body"], dtype=np_dtype)
        betas = np.asarray(seq_npz["betas"], dtype=np_dtype)

        if max_frames > 0:
            target = int(max_frames)
            source_n = int(trans.shape[0])
            if target <= source_n:
                trans = trans[:target]
                root_orient = root_orient[:target]
                pose_body = pose_body[:target]
            else:
                reps = (target + source_n - 1) // source_n
                trans = np.tile(trans, (reps, 1))[:target]
                root_orient = np.tile(root_orient, (reps, 1))[:target]
                pose_body = np.tile(pose_body, (reps, 1))[:target]

        # torchure's cnpy loader is most reliable with compact numeric-only npz files.
        with tempfile.NamedTemporaryFile(
            prefix="torchure_sequence_",
            suffix=".npz",
            delete=False,
        ) as tf:
            temp_sequence_npz = Path(tf.name)
        np.savez(
            temp_sequence_npz,
            trans=trans,
            root_orient=root_orient,
            pose_body=pose_body,
            betas=betas,
        )

        env = os.environ.copy()
        if torchure_device_preference == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        cmd = [
            str(benchmark_bin),
            str(smpl_model_npz),
            "--sequence",
            str(temp_sequence_npz),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--device",
            "auto" if torchure_device_preference == "auto" else torchure_device_preference,
            "--dtype",
            torchure_dtype,
        ]
        if max_frames > 0:
            cmd.extend(["--max-frames", str(max_frames)])

        completed, gpu_peak_mib = _run_subprocess_with_optional_gpu_peak(
            cmd,
            env,
            track_gpu_memory=torchure_device_preference != "cpu",
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        output = completed.stdout + "\n" + completed.stderr
    finally:
        if temp_sequence_npz is not None:
            temp_sequence_npz.unlink(missing_ok=True)

    mode_match = re.search(r"Benchmark mode:\s*(\w+)", output, flags=re.IGNORECASE)
    if not mode_match or mode_match.group(1).lower() != "sequence":
        raise RuntimeError(
            "torchure benchmark binary does not support sequence mode yet. "
            "Rebuild third_party/torchure_smplx after updating samples/benchmark.cpp. "
            f"Output was:\n{output}"
        )

    def _parse_float(label: str) -> float:
        m = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", output, flags=re.IGNORECASE)
        if m is None:
            raise RuntimeError(f"Could not parse '{label}' from torchure output.\n{output}")
        return float(m.group(1))

    def _parse_int(label: str, fallback: int) -> int:
        m = re.search(rf"{label}:\s*([0-9]+)", output, flags=re.IGNORECASE)
        if m is None:
            return int(fallback)
        return int(m.group(1))

    resolved_device_match = re.search(
        r"Resolved device:\s*(cuda|cpu)",
        output,
        flags=re.IGNORECASE,
    )
    resolved_device = (resolved_device_match.group(1).lower() if resolved_device_match else "cpu")
    device_label = "gpu" if resolved_device == "cuda" else "cpu"

    resolved_dtype_match = re.search(
        r"Resolved dtype:\s*(float32|float64)",
        output,
        flags=re.IGNORECASE,
    )
    resolved_dtype = (
        resolved_dtype_match.group(1).lower()
        if resolved_dtype_match
        else torchure_dtype.lower()
    )

    if torchure_device_preference == "cuda" and device_label != "gpu":
        raise RuntimeError("torchure CUDA requested but runtime resolved to CPU")
    if torchure_device_preference == "cpu" and device_label != "cpu":
        raise RuntimeError("torchure CPU requested but runtime resolved to CUDA")
    if resolved_dtype != torchure_dtype.lower():
        raise RuntimeError(
            f"torchure dtype mismatch: requested {torchure_dtype} but resolved {resolved_dtype}"
        )

    mean_ms = _parse_float("Mean runtime ms")
    p50_ms = _parse_float("P50 runtime ms")
    p95_ms = _parse_float("P95 runtime ms")
    fps = _parse_float("Sequence FPS")
    frames = _parse_int("Sequence frames", fallback=n_frames_ref)
    vertices = _parse_int("Vertices", fallback=6890)

    row = {
        "mean_ms": mean_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "fps": fps,
        "implementation": "torchure_smplx_cpp",
        "benchmark_family": "smpl",
        "benchmark_scope": "full_sequence_batch_forward",
        "processing_mode": "batch_sequence_forward",
        "impl_backend": "libtorch-cpp",
        "impl_language": "cpp",
        "impl_dtype": torchure_dtype,
        "impl_autograd": "torch::InferenceMode",
        "impl_sequence_strategy": "single full-sequence batch forward",
        "device_class": device_label,
        "device": _cpu_device_name() if device_label == "cpu" else _gpu_device_name_fallback(),
        "frames": frames,
        "vertices": vertices,
        "runtime_stack": _runtime_stack({
            "torch": _pkg_version_any("torch"),
            "torchure_smplx": "local-build",
        }),
        "note": (
            f"SMPL baseline via torchure_smplx C++ benchmark ({torchure_dtype}) on the provided sequence; "
            "SMPL pose mapped from SMPL-X body sequence."
        ),
    }
    gpu_memory_source = "nvidia_smi_process_subprocess" if device_label == "gpu" else "not_applicable_cpu"
    gpu_samples = [float(gpu_peak_mib)] if (device_label == "gpu" and gpu_peak_mib is not None) else []
    row.update(_gpu_memory_summary(gpu_samples, source=gpu_memory_source))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark runtime across SMPL-JAX, smplx, smplpytorch (SMPL-X and SMPL families)"
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("datasets/SOMA/soma_subject1/walk_001_stageii.npz"),
    )
    parser.add_argument(
        "--smplx-model-path",
        type=Path,
        default=Path("data/smplx/SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--smplx-model-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--smpl-model-pkl",
        type=Path,
        default=Path("data/smpl/SMPL_NEUTRAL.pkl"),
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument("--smplx-ext", choices=["pkl", "npz"], default="npz")
    parser.add_argument("--smpl-ext", choices=["pkl", "npz"], default="pkl")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--tile-sequence-to-max-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When used with --max-frames, tile sequence rows cyclically up to the requested "
            "frame count instead of clamping to source length."
        ),
    )
    parser.add_argument("--num-betas", type=int, default=10)
    parser.add_argument("--num-expression", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--jax-platform", choices=["cuda", "cpu", "metal", "METAL"], default="cuda")
    parser.add_argument(
        "--torch-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device preference for smplx/smplpytorch methods when not running both devices.",
    )
    parser.add_argument(
        "--smplxpp-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="smplxpp device preference when not running both devices.",
    )
    parser.add_argument(
        "--smplxpp-sequence-batch-size",
        type=int,
        default=0,
        help=(
            "Chunk size for smplxpp sequence evaluation. "
            "Set 0 (default) to use a single full-sequence batch. "
            "Positive values force chunked frame batches."
        ),
    )
    parser.add_argument(
        "--smplxpp-isolated-subprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run smplxpp benchmark calls in isolated subprocesses to avoid CUDA "
            "cross-backend side effects in mixed benchmark runs."
        ),
    )
    parser.add_argument(
        "--batch-size-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run additional batch-size scaling benchmarks for each enabled method. "
            "Use --no-batch-size-sweep to skip."
        ),
    )
    parser.add_argument(
        "--batch-size-sweep-sizes",
        type=str,
        default="1,8,32,128,512,full",
        help=(
            "Comma-separated batch sizes for scaling plots (e.g. 1,8,32,128,full). "
            "'full' maps to the full sequence length."
        ),
    )
    parser.add_argument(
        "--batch-size-sweep-repeats",
        type=int,
        default=3,
        help="Repeat count for each scaling benchmark point.",
    )
    parser.add_argument(
        "--batch-size-sweep-warmup",
        type=int,
        default=1,
        help="Warmup iterations for each scaling benchmark point.",
    )
    parser.add_argument(
        "--torchure-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="torchure external benchmark device preference when not running both devices.",
    )
    parser.add_argument(
        "--torchure-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="torchure external benchmark compute dtype (float32 default for fair cross-backend comparison).",
    )
    parser.add_argument(
        "--benchmark-both-devices",
        action="store_true",
        help="Benchmark CPU and GPU for methods where both are available.",
    )
    parser.add_argument(
        "--allow-nonsequence-benchmarks",
        action="store_true",
        help=(
            "Include backends that do not replay the provided motion sequence "
            "(legacy compatibility option)."
        ),
    )
    parser.add_argument(
        "--allow-mixed-sequence-lengths",
        action="store_true",
        help=(
            "Do not fail when sequence-backed rows contain multiple (sequence, frames) groups."
        ),
    )
    parser.add_argument(
        "--allow-mixed-processing-modes",
        action="store_true",
        help=(
            "Allow mixing processing modes within a family/device group. "
            "By default, strict mode is on for fair apples-to-apples comparisons."
        ),
    )
    parser.add_argument(
        "--enforce-processing-mode-fairness",
        action="store_true",
        help=(
            "Deprecated compatibility flag; strict processing-mode fairness is enabled by default."
        ),
    )
    parser.add_argument(
        "--method-filter",
        type=str,
        default="",
        help="Internal: comma-separated method keys to run (e.g. jax_smplx,jax_smpl).",
    )
    parser.add_argument("--xla-gpu-autotune-level", type=int, default=0)
    parser.add_argument(
        "--include-torchure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include torchure benchmark rows when available (use --no-include-torchure to skip).",
    )
    parser.add_argument(
        "--include-smplxpp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include smplxpp benchmark rows when available (use --no-include-smplxpp to skip).",
    )
    parser.add_argument(
        "--torchure-benchmark-bin",
        type=Path,
        default=Path("third_party/torchure_smplx/build/benchmark"),
    )
    parser.add_argument(
        "--torchure-smpl-model-npz",
        type=Path,
        default=Path("data/smpl/SMPL_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--smplxpp-smpl-model-npz",
        type=Path,
        default=Path("data/smpl/SMPL_NEUTRAL.npz"),
    )
    parser.add_argument(
        "--smplxpp-smplx-model-npz",
        type=Path,
        default=Path("data/smplx/SMPLX_NEUTRAL.npz"),
    )
    parser.add_argument("--json-out", type=Path, default=Path("benchmarks/results/benchmark_results.json"))
    args = parser.parse_args()
    method_filter = {m.strip() for m in args.method_filter.split(",") if m.strip()}

    sequence = _load_sequence(
        args.sequence,
        max_frames=args.max_frames,
        tile_to_max_frames=args.tile_sequence_to_max_frames,
    )
    full_frames = int(sequence["trans"].shape[0])
    sweep_sizes = _parse_batch_sweep_sizes(args.batch_size_sweep_sizes, full_frames) if args.batch_size_sweep else []

    results: list[dict[str, Any]] = []

    jax_platforms = [args.jax_platform]
    torch_devices = [args.torch_device]
    smplxpp_devices = [args.smplxpp_device]
    torchure_devices = [args.torchure_device]

    if args.benchmark_both_devices:
        jax_platforms = ["cuda", "cpu"]
        torch_devices = ["cuda", "cpu"]
        smplxpp_devices = ["cuda", "cpu"]
        torchure_devices = ["cuda", "cpu"]

    run_torchure = args.include_torchure and _method_enabled(method_filter, "torchure")
    run_jax_smplx = _method_enabled(method_filter, "jax_smplx")
    run_jax_smpl = _method_enabled(method_filter, "jax_smpl")
    run_torch_smplx = _method_enabled(method_filter, "torch_smplx")
    run_torch_smpl = _method_enabled(method_filter, "torch_smpl")
    run_smplxpp = args.include_smplxpp and _method_enabled(method_filter, "smplxpp")
    run_smplpytorch = _method_enabled(method_filter, "smplpytorch")

    def _run_jax_in_subprocess(
        platform: str,
        repeats: int,
        warmup: int,
        max_frames: int,
        method_filter_csv: str,
    ) -> list[dict[str, Any]]:
        with tempfile.NamedTemporaryFile(prefix=f"jax_{platform}_", suffix=".json", delete=False) as tf:
            tmp_json = Path(tf.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--sequence",
            str(args.sequence),
            "--smplx-model-path",
            str(args.smplx_model_path),
            "--smpl-model-pkl",
            str(args.smpl_model_pkl),
            "--num-betas",
            str(args.num_betas),
            "--num-expression",
            str(args.num_expression),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--jax-platform",
            platform,
            "--xla-gpu-autotune-level",
            str(args.xla_gpu_autotune_level),
            "--json-out",
            str(tmp_json),
            "--method-filter",
            method_filter_csv,
            "--no-include-smplxpp",
            "--no-include-torchure",
            "--no-batch-size-sweep",
        ]
        if max_frames > 0:
            cmd.extend(["--max-frames", str(max_frames)])
            cmd.append("--tile-sequence-to-max-frames")

        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"JAX subprocess failed for {platform}")

        rows = json.loads(tmp_json.read_text())
        try:
            tmp_json.unlink()
        except OSError:
            pass
        return rows

    def _run_smplxpp_in_subprocess(
        smplxpp_device: str,
        repeats: int,
        warmup: int,
        max_frames: int,
    ) -> list[dict[str, Any]]:
        with tempfile.NamedTemporaryFile(prefix=f"smplxpp_{smplxpp_device}_", suffix=".json", delete=False) as tf:
            tmp_json = Path(tf.name)

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--sequence",
            str(args.sequence),
            "--smplx-model-path",
            str(args.smplx_model_path),
            "--smplx-model-dir",
            str(args.smplx_model_dir),
            "--smpl-model-pkl",
            str(args.smpl_model_pkl),
            "--smpl-model-dir",
            str(args.smpl_model_dir),
            "--smplx-ext",
            str(args.smplx_ext),
            "--smpl-ext",
            str(args.smpl_ext),
            "--num-betas",
            str(args.num_betas),
            "--num-expression",
            str(args.num_expression),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--smplxpp-device",
            smplxpp_device,
            "--smplxpp-sequence-batch-size",
            str(args.smplxpp_sequence_batch_size),
            "--smplxpp-smpl-model-npz",
            str(args.smplxpp_smpl_model_npz),
            "--smplxpp-smplx-model-npz",
            str(args.smplxpp_smplx_model_npz),
            "--json-out",
            str(tmp_json),
            "--method-filter",
            "smplxpp",
            "--no-include-torchure",
            "--no-batch-size-sweep",
            "--no-smplxpp-isolated-subprocess",
        ]
        if max_frames > 0:
            cmd.extend(["--max-frames", str(max_frames)])
            cmd.append("--tile-sequence-to-max-frames")

        completed = subprocess.run(cmd, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"smplxpp subprocess failed for {smplxpp_device}")

        rows = json.loads(tmp_json.read_text())
        try:
            tmp_json.unlink()
        except OSError:
            pass
        return rows

    def _attach_profile_row(row: dict[str, Any], *, profile: str, batch_size: int) -> dict[str, Any]:
        row["run_profile"] = profile
        row["batch_size"] = int(batch_size)
        row["full_sequence_frames"] = int(full_frames)
        row.setdefault("gpu_memory_peak_mib", None)
        row.setdefault("gpu_memory_mean_mib", None)
        row.setdefault("gpu_memory_source", "unavailable")
        sequence_batch_size = row.get("sequence_batch_size")
        sequence_batch_missing = sequence_batch_size is None or sequence_batch_size == ""
        if not sequence_batch_missing and isinstance(sequence_batch_size, (float, np.floating)):
            sequence_batch_missing = bool(np.isnan(sequence_batch_size))
        if sequence_batch_missing:
            row["sequence_batch_size"] = int(batch_size)
        return row

    def _add_row(row: dict[str, Any], *, profile: str, batch_size: int) -> None:
        results.append(_attach_profile_row(row, profile=profile, batch_size=batch_size))

    def _run_suite(
        *,
        seq_data: dict[str, np.ndarray],
        repeats: int,
        warmup: int,
        profile: str,
    ) -> None:
        seq_frames = int(seq_data["trans"].shape[0])

        if run_torchure:
            for dev in torchure_devices:
                print(
                    "Running torchure_smplx C++ benchmark on sequence data "
                    f"({dev}, dtype={args.torchure_dtype}, batch={seq_frames}, profile={profile})..."
                )
                try:
                    row = benchmark_torchure_smplx_external(
                        benchmark_bin=args.torchure_benchmark_bin,
                        smpl_model_npz=args.torchure_smpl_model_npz,
                        sequence_path=args.sequence,
                        repeats=repeats,
                        warmup=warmup,
                        max_frames=seq_frames,
                        n_frames_ref=seq_frames,
                        torchure_device_preference=dev,
                        torchure_dtype=args.torchure_dtype,
                    )
                    row["frames"] = int(seq_frames)
                    _add_row(row, profile=profile, batch_size=seq_frames)
                except Exception as exc:
                    print(f"  skipped torchure ({dev}, batch={seq_frames}, profile={profile}): {exc}")

        if run_jax_smplx or run_jax_smpl:
            if args.benchmark_both_devices and len(jax_platforms) > 1:
                for jax_platform in jax_platforms:
                    print(f"Running JAX benchmarks in isolated subprocess ({jax_platform}, batch={seq_frames}, profile={profile})...")
                    try:
                        jax_rows = _run_jax_in_subprocess(
                            platform=jax_platform,
                            repeats=repeats,
                            warmup=warmup,
                            max_frames=seq_frames,
                            method_filter_csv="jax_smplx,jax_smpl",
                        )
                        for row in jax_rows:
                            row["frames"] = int(seq_frames)
                            _add_row(row, profile=profile, batch_size=seq_frames)
                    except Exception as exc:
                        print(f"  skipped JAX subprocess ({jax_platform}, batch={seq_frames}, profile={profile}): {exc}")
            else:
                for jax_platform in jax_platforms:
                    if run_jax_smplx:
                        print(f"Running SMPL-X: SMPL-JAX benchmark ({jax_platform}, batch={seq_frames}, profile={profile})...")
                        try:
                            row = benchmark_jax_smplx(
                                sequence=seq_data,
                                smplx_model_path=args.smplx_model_path,
                                num_betas=args.num_betas,
                                num_expression_coeffs=args.num_expression,
                                repeats=repeats,
                                warmup=warmup,
                                jax_platform=jax_platform,
                                xla_gpu_autotune_level=args.xla_gpu_autotune_level,
                            )
                            _add_row(row, profile=profile, batch_size=seq_frames)
                        except Exception as exc:
                            print(f"  skipped smpl_jax_smplx ({jax_platform}, batch={seq_frames}, profile={profile}): {exc}")

                    if run_jax_smpl:
                        print(f"Running SMPL: SMPL-JAX benchmark ({jax_platform}, batch={seq_frames}, profile={profile})...")
                        try:
                            row = benchmark_jax_smpl(
                                sequence=seq_data,
                                smpl_model_path=args.smpl_model_pkl,
                                num_betas=args.num_betas,
                                repeats=repeats,
                                warmup=warmup,
                                jax_platform=jax_platform,
                                xla_gpu_autotune_level=args.xla_gpu_autotune_level,
                            )
                            _add_row(row, profile=profile, batch_size=seq_frames)
                        except Exception as exc:
                            print(f"  skipped smpl_jax_smpl ({jax_platform}, batch={seq_frames}, profile={profile}): {exc}")

        if run_torch_smplx:
            for dev in torch_devices:
                print(f"Running SMPL-X: smplx (PyTorch) benchmark ({dev}, batch={seq_frames}, profile={profile})...")
                try:
                    row = benchmark_torch_smplx(
                        sequence=seq_data,
                        smplx_model_dir=args.smplx_model_dir,
                        smplx_ext=args.smplx_ext,
                        num_betas=args.num_betas,
                        num_expression_coeffs=args.num_expression,
                        repeats=repeats,
                        warmup=warmup,
                        torch_device_preference=dev,
                    )
                    _add_row(row, profile=profile, batch_size=seq_frames)
                except Exception as exc:
                    print(f"  skipped smplx_torch ({dev}, batch={seq_frames}, profile={profile}): {exc}")

        if run_torch_smpl:
            for dev in torch_devices:
                print(f"Running SMPL: smplx (PyTorch, model_type=smpl) benchmark ({dev}, batch={seq_frames}, profile={profile})...")
                try:
                    row = benchmark_torch_smpl(
                        sequence=seq_data,
                        smpl_model_dir=args.smpl_model_dir,
                        smpl_ext=args.smpl_ext,
                        num_betas=args.num_betas,
                        repeats=repeats,
                        warmup=warmup,
                        torch_device_preference=dev,
                    )
                    _add_row(row, profile=profile, batch_size=seq_frames)
                except Exception as exc:
                    print(f"  skipped smplx_torch_smpl ({dev}, batch={seq_frames}, profile={profile}): {exc}")

        if run_smplxpp:
            for dev in smplxpp_devices:
                if args.smplxpp_isolated_subprocess:
                    print(
                        "Running smplxpp benchmarks in isolated subprocess "
                        f"({dev}, batch={seq_frames}, profile={profile})..."
                    )
                    try:
                        smplxpp_rows = _run_smplxpp_in_subprocess(
                            smplxpp_device=dev,
                            repeats=repeats,
                            warmup=warmup,
                            max_frames=seq_frames,
                        )
                        for row in smplxpp_rows:
                            row["frames"] = int(seq_frames)
                            _add_row(row, profile=profile, batch_size=seq_frames)
                    except Exception as exc:
                        print(f"  skipped smplxpp subprocess ({dev}, batch={seq_frames}, profile={profile}): {exc}")
                else:
                    print(f"Running SMPL: smplxpp (Python binding) benchmark ({dev}, batch={seq_frames}, profile={profile})...")
                    try:
                        row = benchmark_smplxpp_python(
                            sequence=seq_data,
                            smpl_model_npz=args.smplxpp_smpl_model_npz,
                            repeats=repeats,
                            warmup=warmup,
                            smplxpp_sequence_batch_size=args.smplxpp_sequence_batch_size,
                            smplxpp_device_preference=dev,
                        )
                        _add_row(row, profile=profile, batch_size=seq_frames)
                    except Exception as exc:
                        print(f"  skipped smplxpp ({dev}, batch={seq_frames}, profile={profile}): {exc}")

                    print(f"Running SMPL-X: smplxpp (Python binding) benchmark ({dev}, batch={seq_frames}, profile={profile})...")
                    try:
                        row = benchmark_smplxpp_python_smplx(
                            sequence=seq_data,
                            smplx_model_npz=args.smplxpp_smplx_model_npz,
                            repeats=repeats,
                            warmup=warmup,
                            smplxpp_sequence_batch_size=args.smplxpp_sequence_batch_size,
                            smplxpp_device_preference=dev,
                        )
                        _add_row(row, profile=profile, batch_size=seq_frames)
                    except Exception as exc:
                        print(f"  skipped smplxpp_smplx ({dev}, batch={seq_frames}, profile={profile}): {exc}")

        if run_smplpytorch:
            for dev in torch_devices:
                print(f"Running smplpytorch benchmark ({dev}, batch={seq_frames}, profile={profile})...")
                try:
                    row = benchmark_torch_smplpytorch(
                        sequence=seq_data,
                        smpl_model_pkl=args.smpl_model_pkl,
                        repeats=repeats,
                        warmup=warmup,
                        torch_device_preference=dev,
                    )
                    _add_row(row, profile=profile, batch_size=seq_frames)
                except Exception as exc:
                    print(f"  skipped smplpytorch ({dev}, batch={seq_frames}, profile={profile}): {exc}")

    _run_suite(
        seq_data=sequence,
        repeats=args.repeats,
        warmup=args.warmup,
        profile="full_sequence",
    )

    if sweep_sizes:
        print(
            "\n=== Batch-size sweep ===\n"
            f"sizes={sweep_sizes}, repeats={args.batch_size_sweep_repeats}, warmup={args.batch_size_sweep_warmup}"
        )
        for sweep_size in sweep_sizes:
            sweep_sequence = _slice_sequence(sequence, sweep_size)
            _run_suite(
                seq_data=sweep_sequence,
                repeats=args.batch_size_sweep_repeats,
                warmup=args.batch_size_sweep_warmup,
                profile="batch_size_sweep",
            )

    # Attach sequence provenance for downstream filtering/reporting.
    # Rows that are microbenchmarks do not replay the motion sequence.
    nonsequence_scopes = {"single_forward_microbenchmark"}
    sequence_path_str = str(args.sequence)
    for row in results:
        uses_input_sequence = row.get("benchmark_scope") not in nonsequence_scopes
        row["uses_input_sequence"] = bool(uses_input_sequence)
        row["sequence"] = sequence_path_str if uses_input_sequence else None

    comparable_rows = [row for row in results if str(row.get("run_profile", "full_sequence")) == "full_sequence"]
    _validate_comparable_sequence_rows(
        comparable_rows,
        strict=not args.allow_mixed_sequence_lengths,
    )
    strict_processing_mode = not args.allow_mixed_processing_modes
    if args.enforce_processing_mode_fairness:
        strict_processing_mode = True
    _validate_processing_mode_rows(
        comparable_rows,
        strict=strict_processing_mode,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2))

    print("\n=== Benchmark results ===")
    for row in results:
        device_class = str(row.get("device_class", "unknown"))
        device_name = str(row.get("device", "unknown"))
        raw_batch_size = row.get("sequence_batch_size")
        profile = str(row.get("run_profile", "full_sequence"))
        batch_size = int(row.get("batch_size", row.get("frames", 0) or 0))
        gpu_mem_peak = row.get("gpu_memory_peak_mib")
        gpu_mem_text = "-"
        if gpu_mem_peak is not None:
            try:
                if not np.isnan(float(gpu_mem_peak)):
                    gpu_mem_text = f"{float(gpu_mem_peak):.1f} MiB"
            except Exception:
                gpu_mem_text = str(gpu_mem_peak)
        batch_size_text = "-"
        if raw_batch_size is not None:
            try:
                batch_size_text = str(int(raw_batch_size))
            except Exception:
                batch_size_text = str(raw_batch_size)
        print(
            f"{row['implementation']:18s} | family={row['benchmark_family']:5s} | "
            f"device={device_class:4s} | hw={device_name} | "
            f"profile={profile:16s} | "
            f"mode={str(row.get('processing_mode', 'unknown')):24s} | "
            f"batch={batch_size:4d} | slot={batch_size_text:>4s} | "
            f"gpu_mem={gpu_mem_text:>10s} | "
            f"frames={row['frames']:4d} | verts={row['vertices']:5d} | "
            f"mean={row['mean_ms']:.2f} ms | p50={row['p50_ms']:.2f} ms | p95={row['p95_ms']:.2f} ms"
        )
        if row.get("runtime_stack"):
            print(f"  runtime: {row['runtime_stack']}")
        if "note" in row:
            print(f"  note: {row['note']}")

    print(f"\nSaved JSON: {args.json_out}")


if __name__ == "__main__":
    main()
