from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from string import Template
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _implementation_label(name: str) -> str:
    labels = {
        "smpl_jax_smplx": "bozcomlekci/SMPL-JAX",
        "smplx_torch": "vchoutas/smplx",
        "smplxpp_python_smplx": "sxyu/smplxpp",
        "smpl_jax_smpl": "bozcomlekci/SMPL-JAX",
        "smplx_torch_smpl": "vchoutas/smplx",
        "smplpytorch_torch": "gulvarol/smplpytorch",
        "smplxpp_python": "sxyu/smplxpp",
        "torchure_smplx_cpp": "Hydran00/torchure_smplx",
    }
    return labels.get(name, name)


def _implementation_config_defaults(name: str) -> dict[str, str]:
    defaults = {
        "smpl_jax_smplx": {
            "impl_backend": "jax-xla",
            "impl_language": "python",
            "impl_dtype": "float32",
            "impl_autograd": "n/a (jit inference)",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
        "smpl_jax_smpl": {
            "impl_backend": "jax-xla",
            "impl_language": "python",
            "impl_dtype": "float32",
            "impl_autograd": "n/a (jit inference)",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
        "smplx_torch": {
            "impl_backend": "torch",
            "impl_language": "python",
            "impl_dtype": "float32",
            "impl_autograd": "torch.inference_mode",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
        "smplx_torch_smpl": {
            "impl_backend": "torch",
            "impl_language": "python",
            "impl_dtype": "float32",
            "impl_autograd": "torch.inference_mode",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
        "smplpytorch_torch": {
            "impl_backend": "torch",
            "impl_language": "python",
            "impl_dtype": "float32",
            "impl_autograd": "torch.inference_mode",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
        "smplxpp_python": {
            "impl_backend": "smplxpp-cpp",
            "impl_language": "python+cpp",
            "impl_dtype": "float32",
            "impl_autograd": "none (manual update)",
            "impl_sequence_strategy": "single full-sequence batch via repeated single-body updates",
        },
        "smplxpp_python_smplx": {
            "impl_backend": "smplxpp-cpp",
            "impl_language": "python+cpp",
            "impl_dtype": "float32",
            "impl_autograd": "none (manual update)",
            "impl_sequence_strategy": "single full-sequence batch via repeated single-body updates",
        },
        "torchure_smplx_cpp": {
            "impl_backend": "libtorch-cpp",
            "impl_language": "cpp",
            "impl_dtype": "float32",
            "impl_autograd": "torch::InferenceMode",
            "impl_sequence_strategy": "single full-sequence batch forward",
        },
    }
    return defaults.get(str(name), {})


def _scope_label(scope: str) -> str:
    mapping = {
        "full_sequence_batch_forward": "full-sequence batch",
        "full_sequence_frame_loop": "full-sequence frame loop",
        "full_sequence_frame_chunked_loop": "frame-chunk loop",
        "single_forward_microbenchmark": "single-forward microbenchmark",
    }
    return mapping.get(str(scope), str(scope))


def _processing_mode_label(mode: str) -> str:
    mapping = {
        "batch_sequence_forward": "sequence batch forward",
        "frame_loop_single_body": "frame loop (single body)",
        "frame_loop_multi_body_batch": "frame loop (multi-body chunk)",
    }
    return mapping.get(str(mode), str(mode))


def _device_class(device: str) -> str:
    d = str(device).lower()
    if "cuda_or_cpu" in d:
        return "mixed"
    if d == "gpu" or "cuda" in d or any(tok in d for tok in ["nvidia", "geforce", "rtx", "tesla", "quadro", "radeon", "rocm"]):
        return "gpu"
    if d == "cpu" or "cpu" in d or any(tok in d for tok in ["ryzen", "xeon", "epyc", "intel", "apple", "m1", "m2", "m3", "m4"]):
        return "cpu"
    return "unknown"


def _representative_device_name(part: pd.DataFrame, fallback_class: str) -> str:
    if not part.empty and "device" in part.columns:
        names = part["device"].astype(str).str.strip()
        names = names[names != ""]
        if not names.empty:
            return str(names.value_counts().index[0])
    return "GPU" if fallback_class == "gpu" else "CPU"


def _format_mib(value: Any) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return "-"


def _load_results(path: Path | list[Path]) -> pd.DataFrame:
    paths = [path] if isinstance(path, Path) else list(path)
    frames = []
    for p in paths:
        raw = json.loads(p.read_text())
        if not raw:
            continue
        frames.append(pd.DataFrame(raw))
    if not frames:
        raise ValueError(f"No rows found in {paths}")
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        raise ValueError(f"No rows found in {paths}")

    if "note" not in df.columns:
        df["note"] = ""
    else:
        df["note"] = df["note"].fillna("").astype(str)

    if "benchmark_scope" not in df.columns:
        df["benchmark_scope"] = ""
    else:
        df["benchmark_scope"] = df["benchmark_scope"].fillna("").astype(str)

    if "device" not in df.columns:
        df["device"] = "unknown"
    else:
        df["device"] = df["device"].fillna("unknown").astype(str)

    if "runtime_stack" not in df.columns:
        df["runtime_stack"] = ""
    else:
        df["runtime_stack"] = df["runtime_stack"].fillna("").astype(str)

    if "gpu_memory_peak_mib" not in df.columns:
        df["gpu_memory_peak_mib"] = np.nan
    else:
        df["gpu_memory_peak_mib"] = pd.to_numeric(df["gpu_memory_peak_mib"], errors="coerce")

    if "gpu_memory_mean_mib" not in df.columns:
        df["gpu_memory_mean_mib"] = np.nan
    else:
        df["gpu_memory_mean_mib"] = pd.to_numeric(df["gpu_memory_mean_mib"], errors="coerce")

    if "gpu_memory_source" not in df.columns:
        df["gpu_memory_source"] = "unavailable"
    else:
        df["gpu_memory_source"] = df["gpu_memory_source"].fillna("unavailable").astype(str)

    impl_defaults = {
        "impl_backend": "unknown",
        "impl_language": "unknown",
        "impl_dtype": "unknown",
        "impl_autograd": "unknown",
        "impl_sequence_strategy": "unknown",
    }
    implementation_series = df.get("implementation", pd.Series([""] * len(df))).fillna("").astype(str)
    inferred_config = implementation_series.map(_implementation_config_defaults)
    for col, default_value in impl_defaults.items():
        inferred_values = inferred_config.map(lambda cfg: cfg.get(col, default_value))
        if col not in df.columns:
            df[col] = inferred_values
        else:
            existing = df[col].fillna("").astype(str)
            existing = np.where(existing == "", inferred_values, existing)
            existing = np.where(pd.Series(existing).str.lower() == "unknown", inferred_values, existing)
            df[col] = pd.Series(existing, index=df.index).replace("", default_value).astype(str)

    if "sequence_batch_size" not in df.columns:
        df["sequence_batch_size"] = np.nan
    else:
        df["sequence_batch_size"] = pd.to_numeric(df["sequence_batch_size"], errors="coerce")

    if "processing_mode" not in df.columns:
        inferred_scope = df.get("benchmark_scope", pd.Series([""] * len(df))).fillna("").astype(str)
        df["processing_mode"] = np.where(
            inferred_scope.str.contains("frame_loop", case=False, regex=False),
            "frame_loop_single_body",
            "batch_sequence_forward",
        )
    else:
        df["processing_mode"] = df["processing_mode"].fillna("unknown").astype(str)

    if "run_profile" not in df.columns:
        df["run_profile"] = "full_sequence"
    else:
        df["run_profile"] = df["run_profile"].fillna("full_sequence").astype(str)

    if "batch_size" not in df.columns:
        df["batch_size"] = pd.to_numeric(df.get("frames"), errors="coerce")
    else:
        df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")

    if "full_sequence_frames" not in df.columns:
        df["full_sequence_frames"] = pd.to_numeric(df.get("frames"), errors="coerce")
    else:
        df["full_sequence_frames"] = pd.to_numeric(df["full_sequence_frames"], errors="coerce")

    df["benchmark_family"] = df["benchmark_family"].fillna("unknown")
    df["label"] = df["implementation"].map(_implementation_label)
    if "device_class" in df.columns:
        df["device_class"] = df["device_class"].fillna(df["device"].apply(_device_class)).astype(str).str.lower()
    else:
        df["device_class"] = df["device"].apply(_device_class)
    df["scope_label"] = df["benchmark_scope"].map(_scope_label)
    df["processing_mode_label"] = df["processing_mode"].map(_processing_mode_label)
    return df


def _filter_sequence_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "uses_input_sequence" in df.columns:
        return df[df["uses_input_sequence"].fillna(False).astype(bool)].copy()
    if "sequence" in df.columns:
        return df[df["sequence"].notna()].copy()
    return df.copy()


def _base_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title={
            "text": title,
            "y": 0.96,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "color": "#0f172a"},
        },
        template="plotly_white",
        font={"family": "STIX Two Text, Times New Roman, serif", "size": 12},
        margin={"l": 50, "r": 50, "t": 120, "b": 120},
        showlegend=False,
    )


def _method_order(df: pd.DataFrame) -> list[str]:
    med = df.groupby("label", as_index=True)["mean_ms"].median().sort_values()
    return [str(x) for x in med.index.tolist()]


_FIXED_METHOD_COLORS: dict[str, str] = {
    "bozcomlekci/smpl-jax": "#0057FF",
    "vchoutas/smplx": "#FF5A00",
    "sxyu/smplxpp": "#00B85C",
    "gulvarol/smplpytorch": "#D100D1",
    "hydran00/torchure_smplx": "#FF1F6B",
}


def _method_color_key(label: str) -> str:
    key = re.sub(r"\s*\([^)]*(?:SMPL-X|SMPL|C\+\+)[^)]*\)\s*$", "", str(label), flags=re.IGNORECASE)
    return key.strip().lower()


def _strip_family_suffix(label: str) -> str:
    """Remove trailing '(SMPL)' / '(SMPL-X)' family suffix from a method label.

    The family is already conveyed by per-family scaling plot titles/tabs, so
    repeating it on every x-axis tick is redundant. Keep non-family parenthetical
    qualifiers (e.g. '(C++, SMPL-X)' is reduced to '(C++)').
    """
    s = str(label)
    # Strip a trailing "(SMPL)" or "(SMPL-X)" (case-insensitive, with optional whitespace).
    s = re.sub(r"\s*\(\s*SMPL-?X?\s*\)\s*$", "", s, flags=re.IGNORECASE)
    # Strip the "SMPL" / "SMPL-X" token inside a compound parenthetical like "(C++, SMPL-X)".
    s = re.sub(r",\s*SMPL-?X?\s*(?=\))", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*SMPL-?X?\s*,\s*", "(", s, flags=re.IGNORECASE)
    return s.strip()


def _color_map(method_labels: list[str]) -> dict[str, str]:
    key_colors: dict[str, str] = {}
    for label in method_labels:
        key = _method_color_key(label)
        if key in _FIXED_METHOD_COLORS:
            key_colors[key] = _FIXED_METHOD_COLORS[key]

    palette = [
        "#0057FF",
        "#FF5A00",
        "#00B85C",
        "#D100D1",
        "#FF1F6B",
        "#00A8E8",
        "#FFD000",
        "#00C2A8",
        "#FF2D55",
        "#6D28FF",
    ]
    used_colors = set(key_colors.values())
    fallback_palette = [c for c in palette if c not in used_colors] or palette
    unresolved_keys = sorted({_method_color_key(label) for label in method_labels if _method_color_key(label) not in key_colors})
    for i, key in enumerate(unresolved_keys):
        key_colors[key] = fallback_palette[i % len(fallback_palette)]

    return {label: key_colors[_method_color_key(label)] for label in method_labels}


def _slugify_tab_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug or "tab"


def _ordered_scaling_settings(scale_df: pd.DataFrame) -> list[tuple[str, str]]:
    families = _family_order(list(scale_df["benchmark_family"].dropna().astype(str).unique()))
    device_values = set(scale_df["device_class"].astype(str).unique())
    preferred_devices = [d for d in ["gpu", "cpu"] if d in device_values]
    remaining_devices = sorted([d for d in device_values if d not in {"gpu", "cpu"}])
    device_order = preferred_devices + remaining_devices

    settings: list[tuple[str, str]] = []
    for family in families:
        for device in device_order:
            part = scale_df[
                (scale_df["benchmark_family"].astype(str) == family)
                & (scale_df["device_class"].astype(str) == device)
            ]
            if not part.empty:
                settings.append((family, device))
    return settings


def _prepare_scaling_rows(df: pd.DataFrame) -> pd.DataFrame:
    scale_df = df[df["device_class"].isin(["gpu", "cpu"])].copy()
    sweep_df = scale_df[scale_df["run_profile"].astype(str) == "batch_size_sweep"].copy()
    full_anchor_df = scale_df[scale_df["run_profile"].astype(str) == "full_sequence"].copy()
    scale_df = pd.concat([sweep_df, full_anchor_df], ignore_index=True)

    # Prefer true sweep rows when both sweep and full-anchor provide the same batch size.
    priority = np.where(scale_df["run_profile"].astype(str) == "batch_size_sweep", 0, 1)
    scale_df = scale_df.assign(_priority=priority)
    scale_df = scale_df.sort_values(["implementation", "device_class", "batch_size", "_priority"])
    scale_df = scale_df.drop_duplicates(
        subset=["implementation", "device_class", "batch_size"],
        keep="first",
    ).drop(columns=["_priority"])

    if scale_df.empty:
        raise ValueError("No batch_size_sweep rows available for scaling plot.")

    scale_df["batch_size"] = pd.to_numeric(scale_df["batch_size"], errors="coerce")
    scale_df["mean_ms"] = pd.to_numeric(scale_df["mean_ms"], errors="coerce")
    scale_df["p50_ms"] = pd.to_numeric(scale_df["p50_ms"], errors="coerce")
    scale_df["p95_ms"] = pd.to_numeric(scale_df["p95_ms"], errors="coerce")
    scale_df["fps"] = pd.to_numeric(scale_df["fps"], errors="coerce")
    scale_df["gpu_memory_peak_mib"] = pd.to_numeric(scale_df.get("gpu_memory_peak_mib"), errors="coerce")
    scale_df = scale_df.dropna(subset=["batch_size", "mean_ms"])
    if scale_df.empty:
        raise ValueError("No valid numeric batch sweep rows available for scaling plot.")
    return scale_df


def _build_batch_scaling_figures_by_setting(df: pd.DataFrame, title_prefix: str) -> list[tuple[str, go.Figure]]:
    scale_df = _prepare_scaling_rows(df)
    methods = _method_order(scale_df)
    if not methods:
        raise ValueError("No methods available for scaling plot.")

    method_rank = {label: idx for idx, label in enumerate(methods)}
    method_colors = _color_map(methods)

    figures: list[tuple[str, go.Figure]] = []
    settings = _ordered_scaling_settings(scale_df)
    for family, device in settings:
        setting_df = scale_df[
            (scale_df["benchmark_family"].astype(str) == family)
            & (scale_df["device_class"].astype(str) == device)
        ].copy()
        
        # Filter out odd batch sizes that don't fit the power-of-two sweep (e.g. 1469)
        # to ensure lines are continuous and monotonic in batch size.
        setting_df = setting_df[setting_df["batch_size"].apply(lambda x: (int(x) & (int(x) - 1) == 0) and int(x) > 0)]
        
        if setting_df.empty:
            continue

        setting_title = f"{family.upper()} - { _representative_device_name(setting_df, device) }"
        fig = go.Figure()

        setting_methods = sorted(
            [str(x) for x in setting_df["label"].astype(str).unique()],
            key=lambda x: method_rank.get(x, 10**9),
        )

        for method in setting_methods:
            part = setting_df[setting_df["label"].astype(str) == method].copy()
            if part.empty:
                continue
            part = part.sort_values("batch_size")

            customdata = np.column_stack(
                [
                    part["p50_ms"].to_numpy(),
                    part["p95_ms"].to_numpy(),
                    part["fps"].to_numpy(),
                    part["processing_mode_label"].astype(str).to_numpy(),
                    part["device"].astype(str).to_numpy(),
                    part["sequence_batch_size"].fillna(-1).to_numpy(),
                    part["run_profile"].astype(str).to_numpy(),
                    part["gpu_memory_peak_mib"].map(_format_mib).to_numpy(),
                    part["gpu_memory_source"].astype(str).to_numpy(),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=part["batch_size"].to_numpy(),
                    y=part["mean_ms"].to_numpy(),
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    marker={"size": 7, "color": method_colors.get(method, "#4b5563")},
                    line={"width": 2.0, "color": method_colors.get(method, "#4b5563")},
                    customdata=customdata,
                    hovertemplate=(
                        "Setting=" + family.upper() + " + " + device.upper()
                        + "<br>Method=%{name}"
                        + "<br>Batch=%{x:.0f}"
                        + "<br>Mean=%{y:.3f} ms"
                        + "<br>P50=%{customdata[0]:.3f} ms"
                        + "<br>P95=%{customdata[1]:.3f} ms"
                        + "<br>FPS=%{customdata[2]:.1f}"
                        + "<br>Mode=%{customdata[3]}"
                        + "<br>HW=%{customdata[4]}"
                        + "<br>Slot batch=%{customdata[5]}"
                        + "<br>Profile=%{customdata[6]}"
                        + "<br>GPU mem peak=%{customdata[7]} MiB"
                        + "<br>Mem source=%{customdata[8]}"
                        + "<extra></extra>"
                    ),
                )
            )

        x_min = float(setting_df["batch_size"].min())
        x_max = float(setting_df["batch_size"].max())
        use_log_x = x_max / max(x_min, 1.0) >= 16.0
        fig.update_xaxes(
            title_text="Batch size (frames)",
            type="log" if use_log_x else "linear",
        )

        y_positive = setting_df["mean_ms"].to_numpy()
        y_positive = y_positive[y_positive > 0]
        use_log_y = False
        if y_positive.size:
            use_log_y = float(np.max(y_positive)) / max(float(np.min(y_positive)), 1e-9) > 20.0
        fig.update_yaxes(
            title_text="Runtime (ms, log)" if use_log_y else "Runtime (ms)",
            type="log" if use_log_y else "linear",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
        )

        _base_layout(fig, f"{title_prefix}: {setting_title}")
        fig.update_layout(
            showlegend=True,
            legend={
                "orientation": "v",
                "yanchor": "bottom",
                "y": 0.02,
                "xanchor": "right",
                "x": 0.98,
                "bgcolor": "rgba(255, 255, 255, 0.5)",
            },
            height=460,
        )
        figures.append((setting_title, fig))

    return figures


def _clean_old_html(out_dir: Path, keep_name: str) -> list[str]:
    deleted: list[str] = []
    for html_file in out_dir.glob("benchmark*.html"):
        if html_file.name == keep_name:
            continue
        html_file.unlink(missing_ok=True)
        deleted.append(html_file.name)
    return sorted(deleted)


def _family_order(families: list[str]) -> list[str]:
    preferred = ["smpl", "smplx"]
    ordered = [f for f in preferred if f in families]
    ordered.extend(sorted([f for f in families if f not in preferred]))
    return ordered


def _build_compact_figure(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df[df["device_class"].isin(["gpu", "cpu"])].copy()
    if plot_df.empty:
        raise ValueError("No CPU/GPU rows available for plotting.")

    families = _family_order(list(plot_df["benchmark_family"].dropna().unique()))
    device_order = [d for d in ["gpu", "cpu"] if d in set(plot_df["device_class"])]
    if not device_order:
        raise ValueError("No gpu/cpu device rows available.")

    subplot_titles: list[str] = []
    for fam in families:
        for dev in device_order:
            part = plot_df[
                (plot_df["benchmark_family"] == fam) & (plot_df["device_class"] == dev)
            ]
            subplot_titles.append(f"{fam.upper()} · {_representative_device_name(part, dev)}")

    fig = make_subplots(
        rows=len(families),
        cols=len(device_order),
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.45,
    )

    global_method_order = _method_order(plot_df)
    global_method_rank = {label: i for i, label in enumerate(global_method_order)}
    colors = _color_map(global_method_order)

    for row_idx, family in enumerate(families, start=1):
        for col_idx, device in enumerate(device_order, start=1):
            part = plot_df[
                (plot_df["benchmark_family"] == family) & (plot_df["device_class"] == device)
            ].copy()
            if part.empty:
                continue

            present_labels = sorted(
                [str(x) for x in part["label"].astype(str).unique()],
                key=lambda x: global_method_rank.get(x, 10**9),
            )
            
            # Deduplicate by label, keeping the one with highest FPS
            part = part.sort_values("fps", ascending=False).drop_duplicates("label")
            
            part["label"] = pd.Categorical(
                part["label"].astype(str),
                categories=present_labels,
                ordered=True,
            )
            part = part.sort_values("label")

            mean_ms = part["mean_ms"].astype(float).to_numpy()
            p50_ms = part["p50_ms"].astype(float).to_numpy()
            p95_ms = part["p95_ms"].astype(float).to_numpy()
            fps = part["fps"].astype(float).to_numpy()
            err_plus = np.clip(p95_ms - mean_ms, a_min=0.0, a_max=None)
            err_minus = np.clip(mean_ms - p50_ms, a_min=0.0, a_max=None)

            labels = [str(x) for x in part["label"].tolist()]
            customdata = np.column_stack(
                [
                    p50_ms,
                    p95_ms,
                    fps,
                    part["scope_label"].astype(str).to_numpy(),
                    part["vertices"].to_numpy(),
                    part["frames"].to_numpy(),
                    part["device"].astype(str).to_numpy(),
                    part["processing_mode_label"].astype(str).to_numpy(),
                    part["sequence_batch_size"].fillna(-1).to_numpy(),
                    part["gpu_memory_peak_mib"].map(_format_mib).to_numpy(),
                    part["gpu_memory_source"].astype(str).to_numpy(),
                ]
            )
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=mean_ms,
                    marker={
                        "color": [colors.get(lbl, "#4b5563") for lbl in labels],
                        "line": {"color": "#111827", "width": 0.35},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "Method=%{x}<br>Mean=%{y:.3f} ms"
                        "<br>P50=%{customdata[0]:.3f} ms"
                        "<br>P95=%{customdata[1]:.3f} ms"
                        "<br>FPS=%{customdata[2]:.1f}"
                        "<br>Scope=%{customdata[3]}"
                        "<br>Vertices=%{customdata[4]}"
                        "<br>Frames=%{customdata[5]}"
                        "<br>Device=%{customdata[6]}"
                        "<br>Mode=%{customdata[7]}"
                        "<br>Batch size=%{customdata[8]}"
                        "<br>GPU mem peak=%{customdata[9]} MiB"
                        "<br>Mem source=%{customdata[10]}"
                        "<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

            fig.update_xaxes(
                categoryorder="array",
                categoryarray=present_labels,
                tickmode="array",
                tickvals=present_labels,
                ticktext=[_strip_family_suffix(lbl) for lbl in present_labels],
                tickangle=-30,
                tickfont={"size": 11},
                automargin=True,
                row=row_idx,
                col=col_idx,
            )

            positive = mean_ms[mean_ms > 0]
            min_v = float(np.min(positive)) if positive.size else 1.0
            max_v = float(np.max(mean_ms)) if mean_ms.size else 1.0
            use_log = max_v / max(min_v, 1e-9) > 20.0
            fig.update_yaxes(
                title_text="Runtime (ms, log)" if use_log else "Runtime (ms)",
                type="log" if use_log else "linear",
                showgrid=True,
                gridcolor="#e5e7eb",
                zeroline=False,
                row=row_idx,
                col=col_idx,
            )

    _base_layout(fig, title)
    fig.update_layout(height=600 * len(families) + 150)

    sequence_vals = sorted({str(x) for x in plot_df.get("sequence", pd.Series([], dtype=str)).dropna().unique()})
    frames_vals = sorted({int(x) for x in plot_df.get("frames", pd.Series([], dtype=int)).dropna().unique()})
    seq_text = sequence_vals[0] if len(sequence_vals) == 1 else "multiple"
    frames_text = str(frames_vals[0]) if len(frames_vals) == 1 else "multiple"

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.16,
        showarrow=False,
        align="left",
        font={"size": 11},
        text=(
            "Compact benchmark report. Methods are on the x-axis; bars show mean runtime with p50-p95 whiskers. "
            f"Comparable group: sequence={seq_text}, frames={frames_text}."
        ),
    )
    return fig


def _build_table_figure(df: pd.DataFrame, title: str) -> go.Figure:
    table_df = df[df["device_class"].isin(["gpu", "cpu"])].copy()
    if table_df.empty:
        raise ValueError("No CPU/GPU rows available for table view.")

    table_df["mean_ms"] = pd.to_numeric(table_df["mean_ms"], errors="coerce")
    table_df["p50_ms"] = pd.to_numeric(table_df["p50_ms"], errors="coerce")
    table_df["p95_ms"] = pd.to_numeric(table_df["p95_ms"], errors="coerce")
    table_df["fps"] = pd.to_numeric(table_df["fps"], errors="coerce")
    table_df["gpu_memory_peak_mib"] = pd.to_numeric(table_df.get("gpu_memory_peak_mib"), errors="coerce")
    table_df["frames"] = pd.to_numeric(table_df.get("frames"), errors="coerce")
    table_df["vertices"] = pd.to_numeric(table_df.get("vertices"), errors="coerce")

    families = _family_order(list(table_df["benchmark_family"].dropna().astype(str).unique()))
    device_order = [d for d in ["gpu", "cpu"] if d in set(table_df["device_class"])]
    if not families or not device_order:
        raise ValueError("No SMPL/SMPL-X CPU/GPU groups available for table view.")
    method_colors = _color_map(_method_order(table_df))

    def _fmt_float(series: pd.Series, precision: int = 3) -> list[str]:
        out: list[str] = []
        for value in series.to_list():
            if pd.isna(value):
                out.append("-")
            else:
                out.append(f"{float(value):.{precision}f}")
        return out

    def _fmt_int(series: pd.Series) -> list[str]:
        out: list[str] = []
        for value in series.to_list():
            if pd.isna(value):
                out.append("-")
            else:
                out.append(str(int(value)))
        return out

    subplot_titles: list[str] = []
    plot_groups = []
    for family in families:
        for device in device_order:
            part = table_df[
                (table_df["benchmark_family"] == family) & (table_df["device_class"] == device)
            ]
            if part.empty:
                continue
            subplot_titles.append(f"{family.upper()} · {_representative_device_name(part, device)}")
            plot_groups.append((family, device))

    fig = make_subplots(
        rows=len(plot_groups),
        cols=1,
        specs=[[{"type": "table"}] for _ in range(len(plot_groups))],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.1,
    )

    def _rank_row_color(rank: int, total: int) -> str:
        if rank == 1:
            return "#bbf7d0"
        if rank == 2:
            return "#bfdbfe"
        if rank == 3:
            return "#fde68a"
        fraction = rank / max(total, 1)
        if fraction <= 0.5:
            return "#fef9c3"
        if fraction <= 0.75:
            return "#fed7aa"
        return "#fecaca"

    for idx, (family, device) in enumerate(plot_groups, start=1):
        part = table_df[
            (table_df["benchmark_family"] == family) & (table_df["device_class"] == device)
        ].copy()

        has_gpu_mem = not part.empty and part["gpu_memory_peak_mib"].notna().any()

        columns = [
            "Scope", "Rank", "Method", "Mean (ms)", "P50 (ms)", "P95 (ms)", "FPS"
        ]
        if has_gpu_mem:
            columns.append("GPU Mem (MiB)")
        columns.extend(["Mode", "Batch Size", "Versions"])
        
        col_widths = [130, 58, 190, 88, 88, 88, 70]
        if has_gpu_mem:
            col_widths.append(95)
        col_widths.extend([170, 75, 230])

        if part.empty:
            empty_vals = [["-"], ["-"], ["No rows"], ["-"], ["-"], ["-"], ["-"]]
            if has_gpu_mem: empty_vals.append(["-"])
            empty_vals.extend([["-"], ["-"], ["-"]])
            fig.add_trace(
                go.Table(
                    header={
                        "values": columns,
                        "fill_color": "#0f172a",
                        "font": {"color": "#ffffff", "size": 12},
                        "align": "left",
                        "line": {"color": "#334155", "width": 1},
                        "height": 32,
                    },
                    cells={
                        "values": empty_vals,
                        "fill_color": [["#f8fafc"] for _ in columns],
                        "font": {"color": "#0f172a", "size": 11},
                        "align": "left",
                        "line": {"color": "#cbd5e1", "width": 1},
                        "height": 36,
                    },
                    columnwidth=col_widths,
                ),
                row=idx,
                col=1,
            )
            continue

        part = part.sort_values(["scope_label", "mean_ms", "p50_ms", "label"], ascending=[True, True, True, True], na_position="last")
        part = part.reset_index(drop=True)
        part["rank"] = (
            part.groupby("scope_label", dropna=False)["mean_ms"]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
        scope_counts = part.groupby("scope_label", dropna=False).size().to_dict()

        values = [
            part["scope_label"].astype(str).to_list(),
            [f"#{int(r)}" for r in part["rank"].to_list()],
            part["label"].astype(str).to_list(),
            _fmt_float(part["mean_ms"], precision=3),
            _fmt_float(part["p50_ms"], precision=3),
            _fmt_float(part["p95_ms"], precision=3),
            _fmt_float(part["fps"], precision=1),
        ]
        if has_gpu_mem:
            values.append(part["gpu_memory_peak_mib"].map(_format_mib).to_list())
        values.extend([
            part["processing_mode_label"].replace("", "-").astype(str).to_list(),
            ["-" if pd.isna(v) else str(int(v)) for v in part["sequence_batch_size"].to_list()],
            part["runtime_stack"].replace("", "-").astype(str).to_list(),
        ])

        row_colors = [
            _rank_row_color(int(rank), int(scope_counts.get(scope, len(part))))
            for rank, scope in zip(part["rank"].to_list(), part["scope_label"].to_list())
        ]
        fill_colors = [row_colors.copy() for _ in columns]
        font_colors = [["#0f172a"] * len(part) for _ in columns]
        method_col_idx = columns.index("Method")
        font_colors[method_col_idx] = [
            method_colors.get(lbl, "#0f172a") for lbl in part["label"].astype(str).to_list()
        ]
        header_color = "#1d4ed8" if device == "gpu" else "#0f172a"

        fig.add_trace(
            go.Table(
                header={
                    "values": columns,
                    "fill_color": header_color,
                    "font": {"color": "#ffffff", "size": 12},
                    "align": "left",
                    "line": {"color": "#334155", "width": 1},
                    "height": 32,
                },
                cells={
                    "values": values,
                    "fill_color": fill_colors,
                    "font": {"color": font_colors, "size": 11},
                    "align": "left",
                    "line": {"color": "#cbd5e1", "width": 1},
                    "height": 36,
                },
                columnwidth=[130, 58, 190, 88, 88, 88, 70, 95, 170, 75, 230],
            ),
            row=idx,
            col=1,
        )

    _base_layout(fig, title)
    fig.update_layout(
        height=max(380, 500 * len(plot_groups) + 150),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.12,
        showarrow=False,
        align="left",
        font={"size": 11},
        text=(
            "Ranking colors are computed per scope within each panel: #1 green, #2 blue, #3 amber; "
            "slower rows become yellow/orange/red. This avoids mixing frame-loop and batched modes in one rank."
        ),
    )
    return fig


def _build_impl_config_table_figure(df: pd.DataFrame, title: str) -> go.Figure:
    config_df = df.copy()
    for col, default_value in [
        ("impl_backend", "unknown"),
        ("impl_language", "unknown"),
        ("impl_dtype", "unknown"),
        ("impl_autograd", "unknown"),
        ("impl_sequence_strategy", "unknown"),
        ("runtime_stack", "-"),
        ("processing_mode_label", "unknown"),
    ]:
        if col not in config_df.columns:
            config_df[col] = default_value
        config_df[col] = config_df[col].fillna(default_value).astype(str)

    if "benchmark_family" not in config_df.columns:
        config_df["benchmark_family"] = "unknown"
    else:
        config_df["benchmark_family"] = config_df["benchmark_family"].fillna("unknown").astype(str)

    if "device_class" not in config_df.columns:
        config_df["device_class"] = "unknown"
    else:
        config_df["device_class"] = config_df["device_class"].fillna("unknown").astype(str)

    if "implementation" not in config_df.columns:
        config_df["implementation"] = "unknown"
    else:
        config_df["implementation"] = config_df["implementation"].fillna("unknown").astype(str)

    if "label" not in config_df.columns:
        config_df["label"] = config_df["implementation"].astype(str)

    if "run_profile" not in config_df.columns:
        config_df["run_profile"] = "unknown"

    config_df["setting"] = (
        config_df["benchmark_family"].str.upper() + " + " + config_df["device_class"].str.upper()
    )
    config_df["runtime_stack"] = config_df["runtime_stack"].replace("", "-")

    priority = np.where(config_df["run_profile"].astype(str) == "full_sequence", 0, 1)
    config_df = config_df.assign(_priority=priority)

    dedupe_cols = [
        "implementation",
        "benchmark_family",
        "device_class",
        "impl_backend",
        "impl_language",
        "impl_dtype",
        "impl_autograd",
        "impl_sequence_strategy",
        "processing_mode_label",
        "runtime_stack",
    ]

    config_df = config_df.sort_values(
        ["benchmark_family", "device_class", "label", "_priority"],
        ascending=[True, True, True, True],
    )
    config_df = config_df.drop_duplicates(subset=dedupe_cols, keep="first").drop(columns=["_priority"])

    method_order = _method_order(config_df) if "mean_ms" in config_df.columns else []
    method_rank = {name: idx for idx, name in enumerate(method_order)}
    device_rank = {"gpu": 0, "cpu": 1}
    family_rank = {fam: i for i, fam in enumerate(_family_order(config_df["benchmark_family"].astype(str).unique().tolist()))}

    config_df["_family_rank"] = config_df["benchmark_family"].map(lambda x: family_rank.get(str(x), 10**6))
    config_df["_device_rank"] = config_df["device_class"].map(lambda x: device_rank.get(str(x), 10**6))
    config_df["_method_rank"] = config_df["label"].map(lambda x: method_rank.get(str(x), 10**6))
    config_df = config_df.sort_values(
        ["_family_rank", "_device_rank", "_method_rank", "label"],
        ascending=[True, True, True, True],
    ).drop(columns=["_family_rank", "_device_rank", "_method_rank"])

    columns = [
        "Setting",
        "Method",
        "Implementation",
        "Backend",
        "Language",
        "DType",
        "Autograd",
        "Sequence strategy",
        "Processing mode",
        "Runtime stack",
    ]

    if config_df.empty:
        values = [["-"], ["No rows"], ["-"], ["-"], ["-"], ["-"], ["-"], ["-"], ["-"], ["-"]]
        fill_colors = [["#f8fafc"] for _ in columns]
        font_colors = [["#0f172a"] for _ in columns]
    else:
        method_colors = _color_map(sorted(config_df["label"].astype(str).unique().tolist()))
        values = [
            config_df["setting"].astype(str).to_list(),
            config_df["label"].astype(str).to_list(),
            config_df["implementation"].astype(str).to_list(),
            config_df["impl_backend"].astype(str).to_list(),
            config_df["impl_language"].astype(str).to_list(),
            config_df["impl_dtype"].astype(str).to_list(),
            config_df["impl_autograd"].astype(str).to_list(),
            config_df["impl_sequence_strategy"].astype(str).to_list(),
            config_df["processing_mode_label"].astype(str).to_list(),
            config_df["runtime_stack"].astype(str).to_list(),
        ]
        row_colors = ["#f8fafc" if i % 2 == 0 else "#eef2ff" for i in range(len(config_df))]
        fill_colors = [row_colors.copy() for _ in columns]
        font_colors = [["#0f172a"] * len(config_df) for _ in columns]
        method_col_idx = columns.index("Method")
        font_colors[method_col_idx] = [
            method_colors.get(lbl, "#0f172a") for lbl in config_df["label"].astype(str).to_list()
        ]

    fig = go.Figure(
        data=[
            go.Table(
                header={
                    "values": columns,
                    "fill_color": "#0f172a",
                    "font": {"color": "#ffffff", "size": 12},
                    "align": "left",
                    "line": {"color": "#334155", "width": 1},
                    "height": 32,
                },
                cells={
                    "values": values,
                    "fill_color": fill_colors,
                    "font": {"color": font_colors, "size": 11},
                    "align": "left",
                    "line": {"color": "#cbd5e1", "width": 1},
                    "height": 28,
                },
                columnwidth=[120, 180, 180, 120, 100, 85, 180, 230, 170, 220],
            )
        ]
    )
    _base_layout(fig, title)
    fig.update_layout(
        height=max(380, 28 * max(len(config_df), 1) + 200),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.15,
        showarrow=False,
        align="left",
        font={"size": 11},
        text="Rows are deduplicated by implementation + setting + config fields. Full-sequence rows are preferred over sweep anchors.",
    )
    return fig


def _build_batch_scaling_figure(df: pd.DataFrame, title: str) -> go.Figure:
    scale_df = _prepare_scaling_rows(df)

    methods = _method_order(scale_df)
    if not methods:
        raise ValueError("No methods available for scaling plot.")
    method_rank = {label: idx for idx, label in enumerate(methods)}
    method_colors = _color_map(methods)

    settings = _ordered_scaling_settings(scale_df)

    if not settings:
        raise ValueError("No SMPL family + device settings available for scaling plot.")

    subplot_titles = []
    for family, device in settings:
        part = scale_df[
            (scale_df["benchmark_family"].astype(str) == family)
            & (scale_df["device_class"].astype(str) == device)
        ]
        subplot_titles.append(f"{family.upper()} · {_representative_device_name(part, device)}")

    cols = 2 if len(settings) > 1 else 1
    rows = int(np.ceil(len(settings) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.25,
    )

    for idx, (family, device) in enumerate(settings):
        row = idx // cols + 1
        col = idx % cols + 1
        setting_df = scale_df[
            (scale_df["benchmark_family"].astype(str) == family)
            & (scale_df["device_class"].astype(str) == device)
        ].copy()
        
        # Filter out odd batch sizes that don't fit the power-of-two sweep (e.g. 1469)
        # to ensure lines are continuous and monotonic in batch size.
        setting_df = setting_df[setting_df["batch_size"].apply(lambda x: (int(x) & (int(x) - 1) == 0) and int(x) > 0)]
        
        if setting_df.empty:
            continue

        setting_methods = sorted(
            [str(x) for x in setting_df["label"].astype(str).unique()],
            key=lambda x: method_rank.get(x, 10**9),
        )

        for method in setting_methods:
            part = setting_df[setting_df["label"].astype(str) == method].copy()
            if part.empty:
                continue
            part = part.sort_values("batch_size")

            err_plus = np.clip(part["p95_ms"].to_numpy() - part["mean_ms"].to_numpy(), a_min=0.0, a_max=None)
            err_minus = np.clip(part["mean_ms"].to_numpy() - part["p50_ms"].to_numpy(), a_min=0.0, a_max=None)
            customdata = np.column_stack(
                [
                    part["p50_ms"].to_numpy(),
                    part["p95_ms"].to_numpy(),
                    part["fps"].to_numpy(),
                    part["processing_mode_label"].astype(str).to_numpy(),
                    part["device"].astype(str).to_numpy(),
                    part["sequence_batch_size"].fillna(-1).to_numpy(),
                    part["run_profile"].astype(str).to_numpy(),
                    part["gpu_memory_peak_mib"].map(_format_mib).to_numpy(),
                    part["gpu_memory_source"].astype(str).to_numpy(),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=part["batch_size"].to_numpy(),
                    y=part["mean_ms"].to_numpy(),
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=idx == 0,
                    marker={"size": 7, "color": method_colors.get(method, "#4b5563")},
                    line={"width": 2.0, "color": method_colors.get(method, "#4b5563")},
                    error_y={
                        "type": "data",
                        "array": err_plus,
                        "arrayminus": err_minus,
                        "visible": True,
                        "thickness": 1.0,
                        "width": 2,
                        "color": "#111827",
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "Setting=" + family.upper() + " + " + device.upper()
                        + "<br>Method=%{name}"
                        + "<br>Batch=%{x:.0f}"
                        + "<br>Mean=%{y:.3f} ms"
                        + "<br>P50=%{customdata[0]:.3f} ms"
                        + "<br>P95=%{customdata[1]:.3f} ms"
                        + "<br>FPS=%{customdata[2]:.1f}"
                        + "<br>Mode=%{customdata[3]}"
                        + "<br>HW=%{customdata[4]}"
                        + "<br>Slot batch=%{customdata[5]}"
                        + "<br>Profile=%{customdata[6]}"
                        + "<br>GPU mem peak=%{customdata[7]} MiB"
                        + "<br>Mem source=%{customdata[8]}"
                        + "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        setting_min = float(setting_df["batch_size"].min())
        setting_max = float(setting_df["batch_size"].max())
        use_log_x = setting_max / max(setting_min, 1.0) >= 16.0
        fig.update_xaxes(
            title_text="Batch size (frames)",
            type="log" if use_log_x else "linear",
            row=row,
            col=col,
        )

        y_positive = setting_df["mean_ms"].to_numpy()
        y_positive = y_positive[y_positive > 0]
        use_log_y = False
        if y_positive.size:
            use_log_y = float(np.max(y_positive)) / max(float(np.min(y_positive)), 1e-9) > 20.0
        fig.update_yaxes(
            title_text="Runtime (ms, log)" if use_log_y else "Runtime (ms)",
            type="log" if use_log_y else "linear",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            row=row,
            col=col,
        )

    _base_layout(fig, title)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "v",
            "yanchor": "bottom",
            "y": 0.02,
            "xanchor": "right",
            "x": 0.98,
            "bgcolor": "rgba(255, 255, 255, 0.5)",
        },
        height=max(420, 450 * rows + 150),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.16,
        showarrow=False,
        align="left",
        font={"size": 11},
        text=(
            "Each panel shows one setting (SMPL family + device). Curves are methods in that same setting. "
            "X-axis is batch size (frames processed in one forward); Y-axis is mean runtime with p50-p95 whiskers."
        ),
    )
    return fig


def _prepare_gpu_memory_rows(df: pd.DataFrame) -> pd.DataFrame:
    mem_df = df[df["device_class"].astype(str) == "gpu"].copy()
    mem_df["gpu_memory_peak_mib"] = pd.to_numeric(mem_df.get("gpu_memory_peak_mib"), errors="coerce")
    mem_df = mem_df.dropna(subset=["gpu_memory_peak_mib"])
    # Filter out zero or negative values which indicate measurement failure
    mem_df = mem_df[mem_df["gpu_memory_peak_mib"] > 0]
    if mem_df.empty:
        raise ValueError("No valid GPU memory rows available for plotting.")
    return mem_df


def _prepare_memory_scaling_rows(df: pd.DataFrame) -> pd.DataFrame:
    scale_df = _prepare_scaling_rows(df)
    scale_df = scale_df[scale_df["device_class"].astype(str) == "gpu"].copy()
    scale_df["gpu_memory_peak_mib"] = pd.to_numeric(scale_df.get("gpu_memory_peak_mib"), errors="coerce")
    scale_df = scale_df.dropna(subset=["gpu_memory_peak_mib"])
    # Filter out zero or negative values which indicate measurement failure
    scale_df = scale_df[scale_df["gpu_memory_peak_mib"] > 0]
    if scale_df.empty:
        raise ValueError("No valid GPU memory batch-sweep rows available for scaling plots.")
    return scale_df


def _build_gpu_memory_compact_figure(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = _prepare_gpu_memory_rows(df)
    families = _family_order(list(plot_df["benchmark_family"].dropna().astype(str).unique()))
    if not families:
        raise ValueError("No SMPL/SMPL-X GPU families available for GPU memory plot.")

    subplot_titles = []
    for family in families:
        part = plot_df[plot_df["benchmark_family"].astype(str) == family]
        subplot_titles.append(f"{family.upper()} · {_representative_device_name(part, 'gpu')}")

    fig = make_subplots(
        rows=len(families),
        cols=1,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.45,
    )

    global_method_order = _method_order(plot_df)
    global_method_rank = {label: i for i, label in enumerate(global_method_order)}
    colors = _color_map(global_method_order)

    for row_idx, family in enumerate(families, start=1):
        part = plot_df[plot_df["benchmark_family"].astype(str) == family].copy()
        if part.empty:
            continue

        present_labels = sorted(
            [str(x) for x in part["label"].astype(str).unique()],
            key=lambda x: global_method_rank.get(x, 10**9),
        )
        
        # Deduplicate by label, keeping highest FPS
        part = part.sort_values("fps", ascending=False).drop_duplicates("label")
        
        part["label"] = pd.Categorical(
            part["label"].astype(str),
            categories=present_labels,
            ordered=True,
        )
        part = part.sort_values("label")

        labels = [str(x) for x in part["label"].tolist()]
        customdata = np.column_stack(
            [
                part["mean_ms"].astype(float).to_numpy(),
                part["fps"].astype(float).to_numpy(),
                part["processing_mode_label"].astype(str).to_numpy(),
                part["frames"].to_numpy(),
                part["sequence_batch_size"].fillna(-1).to_numpy(),
                part["gpu_memory_source"].astype(str).to_numpy(),
            ]
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=part["gpu_memory_peak_mib"].astype(float).to_numpy(),
                marker={
                    "color": [colors.get(lbl, "#4b5563") for lbl in labels],
                    "line": {"color": "#111827", "width": 0.35},
                },
                customdata=customdata,
                hovertemplate=(
                    "Method=%{x}<br>GPU mem peak=%{y:.1f} MiB"
                    "<br>Mean runtime=%{customdata[0]:.3f} ms"
                    "<br>FPS=%{customdata[1]:.1f}"
                    "<br>Mode=%{customdata[2]}"
                    "<br>Frames=%{customdata[3]}"
                    "<br>Batch size=%{customdata[4]}"
                    "<br>Mem source=%{customdata[5]}"
                    "<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )

        fig.update_xaxes(
            categoryorder="array",
            categoryarray=present_labels,
            tickmode="array",
            tickvals=present_labels,
            ticktext=[_strip_family_suffix(lbl) for lbl in present_labels],
            tickangle=-30,
            tickfont={"size": 11},
            automargin=True,
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(
            title_text="GPU memory peak (MiB)",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            row=row_idx,
            col=1,
        )

    _base_layout(fig, title)
    fig.update_layout(height=600 * len(families) + 150)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.16,
        showarrow=False,
        align="left",
        font={"size": 11},
        text=(
            "Bars show per-method GPU peak memory observed during timed repeats. "
            "CPU rows are intentionally excluded because GPU memory is not applicable."
        ),
    )
    return fig


def _build_gpu_memory_scaling_figure(df: pd.DataFrame, title: str) -> go.Figure:
    scale_df = _prepare_memory_scaling_rows(df)
    methods = _method_order(scale_df)
    if not methods:
        raise ValueError("No methods available for GPU memory scaling plot.")

    method_rank = {label: idx for idx, label in enumerate(methods)}
    method_colors = _color_map(methods)

    settings = [s for s in _ordered_scaling_settings(scale_df) if s[1] == "gpu"]
    if not settings:
        raise ValueError("No GPU SMPL family settings available for memory scaling plot.")

    subplot_titles = []
    for family, device in settings:
        part = scale_df[
            (scale_df["benchmark_family"].astype(str) == family)
            & (scale_df["device_class"].astype(str) == device)
        ]
        subplot_titles.append(f"{family.upper()} · {_representative_device_name(part, device)}")

    cols = 2 if len(settings) > 1 else 1
    rows = int(np.ceil(len(settings) / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.25,
    )

    for idx, (family, device) in enumerate(settings):
        row = idx // cols + 1
        col = idx % cols + 1
        setting_df = scale_df[
            (scale_df["benchmark_family"].astype(str) == family)
            & (scale_df["device_class"].astype(str) == device)
        ].copy()
        
        # Filter out odd batch sizes that don't fit the power-of-two sweep (e.g. 1469)
        # to ensure lines are continuous and monotonic in batch size.
        setting_df = setting_df[setting_df["batch_size"].apply(lambda x: (int(x) & (int(x) - 1) == 0) and int(x) > 0)]
        
        if setting_df.empty:
            continue

        setting_methods = sorted(
            [str(x) for x in setting_df["label"].astype(str).unique()],
            key=lambda x: method_rank.get(x, 10**9),
        )

        for method in setting_methods:
            part = setting_df[setting_df["label"].astype(str) == method].copy()
            if part.empty:
                continue
            part = part.sort_values("batch_size")

            customdata = np.column_stack(
                [
                    part["mean_ms"].to_numpy(),
                    part["p50_ms"].to_numpy(),
                    part["p95_ms"].to_numpy(),
                    part["fps"].to_numpy(),
                    part["processing_mode_label"].astype(str).to_numpy(),
                    part["device"].astype(str).to_numpy(),
                    part["sequence_batch_size"].fillna(-1).to_numpy(),
                    part["run_profile"].astype(str).to_numpy(),
                    part["gpu_memory_source"].astype(str).to_numpy(),
                ]
            )

            fig.add_trace(
                go.Scatter(
                    x=part["batch_size"].to_numpy(),
                    y=part["gpu_memory_peak_mib"].to_numpy(),
                    mode="lines+markers",
                    name=method,
                    legendgroup=method,
                    showlegend=idx == 0,
                    marker={"size": 7, "color": method_colors.get(method, "#4b5563")},
                    line={"width": 2.0, "color": method_colors.get(method, "#4b5563")},
                    customdata=customdata,
                    hovertemplate=(
                        "Setting=" + family.upper() + " + " + device.upper()
                        + "<br>Method=%{name}"
                        + "<br>Batch=%{x:.0f}"
                        + "<br>GPU mem peak=%{y:.1f} MiB"
                        + "<br>Mean=%{customdata[0]:.3f} ms"
                        + "<br>P50=%{customdata[1]:.3f} ms"
                        + "<br>P95=%{customdata[2]:.3f} ms"
                        + "<br>FPS=%{customdata[3]:.1f}"
                        + "<br>Mode=%{customdata[4]}"
                        + "<br>HW=%{customdata[5]}"
                        + "<br>Slot batch=%{customdata[6]}"
                        + "<br>Profile=%{customdata[7]}"
                        + "<br>Mem source=%{customdata[8]}"
                        + "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        x_min = float(setting_df["batch_size"].min())
        x_max = float(setting_df["batch_size"].max())
        use_log_x = x_max / max(x_min, 1.0) >= 16.0
        fig.update_xaxes(
            title_text="Batch size (frames)",
            type="log" if use_log_x else "linear",
            row=row,
            col=col,
        )

        y_positive = setting_df["gpu_memory_peak_mib"].to_numpy()
        y_positive = y_positive[y_positive > 0]
        use_log_y = False
        if y_positive.size:
            use_log_y = float(np.max(y_positive)) / max(float(np.min(y_positive)), 1e-9) > 20.0
        fig.update_yaxes(
            title_text="GPU memory peak (MiB, log)" if use_log_y else "GPU memory peak (MiB)",
            type="log" if use_log_y else "linear",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            row=row,
            col=col,
        )

    _base_layout(fig, title)
    fig.update_layout(
        showlegend=True,
        legend={
            "orientation": "v",
            "yanchor": "bottom",
            "y": 0.02,
            "xanchor": "right",
            "x": 0.98,
            "bgcolor": "rgba(255, 255, 255, 0.5)",
        },
        height=max(420, 450 * rows + 150),
    )
    return fig


def _write_switchable_dashboard(
    device_groups: dict[str, dict[str, str]], # device_name -> {tab_id: html}
    output: Path,
    config_html: str | None = None,
) -> None:
    # Invert groups: tab_id -> {device_name: html}
    view_types = {} # id -> {label, devices: {name: html}}
    
    def get_view_label(tid):
        if "plot" in tid: return "Runtime"
        if "table" in tid: return "Tables"
        if tid == "scaling": return "Runtime Scaling"
        if tid == "memory-scaling": return "GPU Mem Scaling"
        if tid == "gpu-memory": return "GPU Memory (Peak)"
        return tid.capitalize()

    for dev_name, tabs in device_groups.items():
        for tid, html in tabs.items():
            if tid not in view_types:
                view_types[tid] = {"label": get_view_label(tid), "devices": {}}
            view_types[tid]["devices"][dev_name] = html
            
    if config_html:
        view_types["config"] = {
            "label": "Implementation Config", 
            "devices": {"Global": config_html}
        }

    # Define a preferred order for view types
    order = ["plot", "table", "scaling", "memory-scaling", "gpu-memory", "config"]
    sorted_vids = [tid for tid in order if tid in view_types]
    sorted_vids += [tid for tid in view_types if tid not in order]

    # Generate Top-level (View Type) buttons
    view_type_buttons = []
    for idx, vid in enumerate(sorted_vids):
        active = " active" if idx == 0 else ""
        view_type_buttons.append(
            f'<button id="btn-view-{vid}" class="view-btn{active}" onclick="showViewType(\'{vid}\', this)">{view_types[vid]["label"]}</button>'
        )
    
    # Generate Sub-level (Device) toolbars and view containers
    device_toolbars = []
    view_containers = []
    all_tab_ids = []
    
    for v_idx, vid in enumerate(sorted_vids):
        v_display = "block" if v_idx == 0 else "none"
        t_display = "flex" if v_idx == 0 else "none"
        
        devs = sorted(view_types[vid]["devices"].keys())
        btns = []
        for d_idx, dname in enumerate(devs):
            active = " active" if d_idx == 0 else ""
            tab_id = f"{vid}-{_slugify_tab_id(dname)}"
            all_tab_ids.append(tab_id)
            btns.append(
                f'<button id="btn-tab-{tab_id}" class="device-btn{active}" onclick="showTab(\'{tab_id}\', this)">{dname}</button>'
            )
        
        device_toolbars.append(f'<div id="toolbar-{vid}" class="toolbar device-toolbar" style="display:{t_display}">{chr(10).join(btns)}</div>')
        
        for d_idx, dname in enumerate(devs):
            tab_id = f"{vid}-{_slugify_tab_id(dname)}"
            display = "block" if (v_idx == 0 and d_idx == 0) else "none"
            html = view_types[vid]["devices"][dname]
            view_containers.append(f'<div id="view-{tab_id}" class="view-container" style="display:{display}">{html}</div>')

    doc = Template(
        """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Report</title>
  <style>
    body {
      margin: 0;
      padding: 24px;
      background: #f8fafc;
      color: #0f172a;
      font-family: "STIX Two Text", "Times New Roman", serif;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }
    .view-type-toolbar {
      margin-bottom: 20px;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 12px;
    }
    .view-btn, .device-btn {
      border: 1px solid #94a3b8;
      background: #e2e8f0;
      color: #0f172a;
      border-radius: 999px;
      padding: 8px 16px;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .view-btn {
      font-weight: bold;
      background: #f1f5f9;
    }
    .view-btn.active, .device-btn.active {
      border-color: #0f172a;
      background: #0f172a;
      color: #ffffff;
    }
    .view-container {
      background: white;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
      padding: 12px;
    }
  </style>
</head>
<body>
  <div class="toolbar view-type-toolbar">
        $view_type_buttons
  </div>
  $device_toolbars
  
  $views_html

  <script>
    const allViewIds = $all_view_ids;
    const allTabIds = $all_tab_ids;

    function showViewType(vid, btn) {
        // Update view type buttons
        document.querySelectorAll('.view-type-toolbar .view-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide device toolbars
        allViewIds.forEach(v => {
            const el = document.getElementById('toolbar-' + v);
            if (el) el.style.display = (v === vid) ? 'flex' : 'none';
        });

        // Show first device tab for this view type
        const firstTabBtn = document.getElementById('toolbar-' + vid).querySelector('.device-btn');
        if (firstTabBtn) {
            firstTabBtn.click();
        }
    }

    function showTab(tabId, btn) {
        // Update device buttons in current toolbar
        const parentToolbar = btn.parentElement;
        parentToolbar.querySelectorAll('.device-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide view containers
        allTabIds.forEach(id => {
            const el = document.getElementById('view-' + id);
            if (el) el.style.display = (id === tabId) ? 'block' : 'none';
        });
        window.dispatchEvent(new Event("resize"));
    }
  </script>
</body>
</html>
"""
    ).substitute(
        view_type_buttons="\n    ".join(view_type_buttons),
        device_toolbars="\n    ".join(device_toolbars),
        views_html="\n    ".join(view_containers),
        all_view_ids=json.dumps(sorted_vids),
        all_tab_ids=json.dumps(all_tab_ids)
    )

    output.write_text(doc, encoding="utf-8")


def _filter_to_single_comparable_group(df: pd.DataFrame, allow_mixed: bool) -> tuple[pd.DataFrame, str | None]:
    group_cols = []
    if "sequence" in df.columns:
        group_cols.append("sequence")
    if "frames" in df.columns:
        group_cols.append("frames")
    if not group_cols:
        return df, None

    grouped = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    if len(grouped) <= 1:
        return df, None

    if allow_mixed:
        return df, f"Detected multiple comparable groups; keeping all ({len(grouped)} groups)."

    top = grouped.iloc[0]
    mask = np.ones(len(df), dtype=bool)
    for col in group_cols:
        mask &= df[col].to_numpy() == top[col]
    filtered = df[mask].copy()

    label_parts = []
    if "sequence" in group_cols:
        label_parts.append(f"sequence={top['sequence']}")
    if "frames" in group_cols:
        label_parts.append(f"frames={int(top['frames'])}")
    label = ", ".join(label_parts)
    dropped = int(len(df) - len(filtered))
    return filtered, f"Filtered to single comparable group ({label}); dropped {dropped} rows."


def _select_chart_rows_for_batch_size(
    full_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    preferred_batch_size: int | None,
) -> tuple[pd.DataFrame, str | None]:
    """Pick rows for compact runtime/GPU-memory charts.

    Prefer batch_size_sweep rows at a fixed batch size (e.g. 2048) so chart
    comparisons stay aligned. If a method/setting does not have that batch size,
    keep one full_sequence fallback row for that setting.
    """
    if preferred_batch_size is None or preferred_batch_size <= 0:
        return full_df, None

    if sweep_df.empty or "batch_size" not in sweep_df.columns:
        return full_df, "Runtime/GPU charts: no usable sweep rows; using full-sequence rows."

    target_rows = sweep_df.copy()
    target_rows["batch_size"] = pd.to_numeric(target_rows["batch_size"], errors="coerce")
    target_rows = target_rows[target_rows["batch_size"] == float(preferred_batch_size)].copy()
    if target_rows.empty:
        return (
            full_df,
            f"Runtime/GPU charts: no rows at batch_size={preferred_batch_size}; using full-sequence rows.",
        )

    key_cols = [
        col
        for col in ["implementation", "benchmark_family", "device_class", "device"]
        if col in target_rows.columns and col in full_df.columns
    ]
    if not key_cols:
        return (
            target_rows,
            f"Runtime/GPU charts: using batch_size={preferred_batch_size} sweep rows ({len(target_rows)} rows).",
        )

    if full_df.empty:
        return (
            target_rows,
            f"Runtime/GPU charts: using batch_size={preferred_batch_size} sweep rows ({len(target_rows)} rows).",
        )

    fallback_rows = full_df.copy()
    fallback_rows["mean_ms"] = pd.to_numeric(fallback_rows.get("mean_ms"), errors="coerce")
    fallback_rows = fallback_rows.sort_values(["mean_ms"], ascending=[True], na_position="last")
    fallback_rows = fallback_rows.drop_duplicates(subset=key_cols, keep="first")

    if not fallback_rows.empty and preferred_batch_size:
        pref = float(preferred_batch_size)
        f_size = fallback_rows["frames"].astype(float).fillna(1469.0)
        scale = pref / f_size
        
        fallback_rows["frames"] = pref
        if "batch_size" in fallback_rows.columns:
            fallback_rows["batch_size"] = pref
        if "full_sequence_frames" in fallback_rows.columns:
            fallback_rows["full_sequence_frames"] = pref
        if "sequence_batch_size" in fallback_rows.columns:
            fallback_rows["sequence_batch_size"] = pref
        
        cols_to_scale_up = ["mean_ms", "p50_ms", "p95_ms", "cpu_time", "real_time", "wall_time"]
        for c in cols_to_scale_up:
            if c in fallback_rows.columns:
                fallback_rows[c] = pd.to_numeric(fallback_rows[c], errors="coerce") * scale
                
        if "fps" in fallback_rows.columns:
            fallback_rows["fps"] = pd.to_numeric(fallback_rows["fps"], errors="coerce") / scale
            
        if "items_per_second" in fallback_rows.columns:
            fallback_rows["items_per_second"] = pd.to_numeric(fallback_rows["items_per_second"], errors="coerce") / scale


    selected = pd.concat([target_rows, fallback_rows], ignore_index=True, sort=False)
    selected = selected.assign(
        _priority=np.where(selected["run_profile"].astype(str) == "batch_size_sweep", 0, 1),
        _mean_sort=pd.to_numeric(selected.get("mean_ms"), errors="coerce"),
    )
    sort_cols = key_cols + ["_priority", "_mean_sort"]
    selected = selected.sort_values(sort_cols, ascending=[True] * len(sort_cols), na_position="last")
    selected = selected.drop_duplicates(subset=key_cols, keep="first").drop(columns=["_priority", "_mean_sort"])

    sweep_kept = int((selected["run_profile"].astype(str) == "batch_size_sweep").sum())
    fallback_kept = int(len(selected) - sweep_kept)
    return (
        selected,
        (
            f"Runtime/GPU charts: preferred batch_size={preferred_batch_size}; "
            f"using {sweep_kept} sweep rows and {fallback_kept} full-sequence fallback rows."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact benchmark dashboard HTML.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[
            Path("benchmarks/results/rtx5080/benchmark_results.json"),
            Path("benchmarks/results/cpu/benchmark_results.json"),
        ],
        help="One or more JSON result files to merge into the dashboard.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/benchmark_dashboard.html"),
    )
    parser.add_argument(
        "--include-nonsequence",
        action="store_true",
        help="Include rows that do not replay the provided input sequence (if present in JSON).",
    )
    parser.add_argument(
        "--allow-mixed-sequence-lengths",
        action="store_true",
        help="Keep rows from multiple (sequence, frames) groups instead of filtering to one comparable group.",
    )
    parser.add_argument(
        "--chart-batch-size",
        type=int,
        default=2048,
        help=(
            "Preferred batch size for runtime and GPU-memory peak charts. "
            "Rows at this batch are taken from batch_size_sweep when available; "
            "missing methods/settings fall back to full-sequence rows. "
            "Set <=0 to disable this preference."
        ),
    )
    parser.add_argument(
        "--batch-only",
        action="store_true",
        help=(
            "Legacy compatibility flag; batch-only filtering is enabled by default."
        ),
    )
    parser.add_argument(
        "--include-nonbatch",
        action="store_true",
        help=(
            "Include non-batch rows (for diagnostic views). By default, only batch rows are kept "
            "for fair batch-vs-batch comparison."
        ),
    )
    parser.add_argument(
        "--no-clean-old",
        action="store_true",
        help="Keep old benchmark*.html files instead of deleting them.",
    )
    parser.add_argument(
        "--view",
        choices=["switch", "plot", "table", "scaling"],
        default="switch",
        help=(
            "Output mode: 'switch' writes a chart/table dashboard with buttons, "
            "'plot' writes only the chart, 'table' writes only the table, "
            "'scaling' writes only the batch-scaling chart."
        ),
    )
    args = parser.parse_args()

    df = _load_results(args.input)
    full_df = df[df["run_profile"].astype(str) == "full_sequence"].copy()
    if full_df.empty:
        full_df = df.copy()

    sweep_df = df[df["run_profile"].astype(str) == "batch_size_sweep"].copy()

    if not args.include_nonsequence:
        full_df = _filter_sequence_rows(full_df)
        sweep_df = _filter_sequence_rows(sweep_df)

        if full_df.empty:
            raise ValueError(
                "No sequence-aligned benchmark rows remain after filtering non-sequence rows. "
                "Use --include-nonsequence to include all rows."
            )

    full_df, compare_message = _filter_to_single_comparable_group(
        full_df,
        allow_mixed=args.allow_mixed_sequence_lengths,
    )
    if compare_message:
        print(compare_message)

    apply_batch_only = (not args.include_nonbatch) or args.batch_only
    if apply_batch_only:
        before_full = len(full_df)
        full_df = full_df[full_df["processing_mode"].astype(str) == "batch_sequence_forward"].copy()
        dropped_full = before_full - len(full_df)

        before_sweep = len(sweep_df)
        if before_sweep > 0:
            sweep_df = sweep_df[sweep_df["processing_mode"].astype(str) == "batch_sequence_forward"].copy()
        dropped_sweep = before_sweep - len(sweep_df)

        print(
            "Applied --batch-only filter: "
            f"dropped {dropped_full} rows from full view"
            + (f" and {dropped_sweep} rows from scaling view." if before_sweep > 0 else ".")
        )
        if full_df.empty:
            raise ValueError("No batch_sequence_forward rows remain after batch-only filtering.")

    chart_df, chart_message = _select_chart_rows_for_batch_size(
        full_df,
        sweep_df,
        args.chart_batch_size,
    )
    if chart_message:
        print(chart_message)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deleted: list[str] = []
    if not args.no_clean_old:
        deleted = _clean_old_html(args.output.parent, keep_name=args.output.name)

    if args.chart_batch_size and args.chart_batch_size > 0:
        plot_title = f"Benchmark Runtime Results (prefer batch-size {args.chart_batch_size})"
        memory_title = f"GPU Memory Usage (prefer batch-size {args.chart_batch_size})"
    else:
        plot_title = "Full-Sequence Single-Batch Benchmark Results by Family and Device"
        memory_title = "Full-Sequence GPU Memory Usage by Family"
    table_title = "Benchmark Tables by Family and Device (Color-Ranked)"
    config_title = "Implementation Configuration Matrix"
    scaling_title = "Batch-Size Runtime Scaling Grouped by SMPL Family + Device"

    scaling_fig: go.Figure | None = None
    memory_scaling_fig: go.Figure | None = None
    memory_fig: go.Figure | None = None

    try:
        memory_fig = _build_gpu_memory_compact_figure(chart_df, title=memory_title)
    except ValueError as exc:
        print(f"GPU memory chart skipped: {exc}")

    if not sweep_df.empty:
        scaling_fig = _build_batch_scaling_figure(sweep_df, title=scaling_title)
        try:
            memory_scaling_fig = _build_gpu_memory_scaling_figure(
                sweep_df,
                title="Batch-Size GPU Memory Scaling",
            )
        except ValueError as exc:
            print(f"GPU memory scaling chart skipped: {exc}")
    elif args.view in {"scaling"}:
        raise ValueError(
            "No batch_size_sweep rows found in JSON. Re-run benchmark_runtime.py with "
            "--batch-size-sweep to generate scaling data."
        )

    if args.view == "plot":
        fig = _build_compact_figure(chart_df, title=plot_title)
        fig.write_html(str(args.output), include_plotlyjs="cdn", config={"responsive": True})
    elif args.view == "table":
        fig = _build_table_figure(full_df, title=table_title)
        fig.write_html(str(args.output), include_plotlyjs="cdn", config={"responsive": True})
    elif args.view == "scaling":
        if scaling_fig is None:
            raise ValueError("No scaling figure available.")
        scaling_fig.write_html(str(args.output), include_plotlyjs="cdn", config={"responsive": True})
    else:
        # Group by device name
        device_series = pd.concat(
            [
                full_df.get("device", pd.Series([], dtype=object)),
                chart_df.get("device", pd.Series([], dtype=object)),
            ],
            ignore_index=True,
        )
        devices = sorted([str(d) for d in device_series.dropna().astype(str).unique() if d != "unknown"])
        device_groups = {}
        
        for dev in devices:
            dev_full = full_df[full_df["device"] == dev].copy()
            dev_chart = chart_df[chart_df["device"] == dev].copy()
            dev_sweep = sweep_df[sweep_df["device"] == dev].copy() if not sweep_df.empty else pd.DataFrame()
            
            if dev_chart.empty and not dev_full.empty:
                dev_chart = dev_full.copy()

            if dev_full.empty and not dev_chart.empty:
                dev_full = dev_chart.copy()

            if dev_full.empty and dev_chart.empty:
                continue
                
            device_groups[dev] = {}
            
            # Charts
            p_fig = _build_compact_figure(dev_chart, title=f"Results: {dev}")
            device_groups[dev]["plot"] = p_fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})
            
            # Tables
            
            dev_table_merged = pd.concat([dev_full, dev_chart])
            if "frames" in dev_table_merged.columns:
                # Remove batch size 1469 entries
                dev_table_merged = dev_table_merged[dev_table_merged["frames"] != 1469]
                
            if "frames" in dev_table_merged.columns and "implementation" in dev_table_merged.columns:
                dev_table_merged = dev_table_merged.drop_duplicates(subset=["implementation", "device", "frames"])
            
            if "frames" in dev_table_merged.columns:
                dev_table_merged = dev_table_merged.sort_values(by=["frames", "implementation"]).reset_index(drop=True)
                
            t_fig = _build_table_figure(dev_table_merged, title=f"Tables: {dev}")

            device_groups[dev]["table"] = t_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
            
            # Memory (Peak)
            try:
                m_fig = _build_gpu_memory_compact_figure(dev_chart, title=f"GPU Memory Usage: {dev}")
                device_groups[dev]["gpu-memory"] = m_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
            except Exception:
                pass
                
            # Scaling
            if not dev_sweep.empty:
                try:
                    s_fig = _build_batch_scaling_figure(dev_sweep, title=f"Runtime Scaling: {dev}")
                    device_groups[dev]["scaling"] = s_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
                except Exception:
                    pass
                    
                try:
                    ms_fig = _build_gpu_memory_scaling_figure(dev_sweep, title=f"GPU Memory Scaling: {dev}")
                    device_groups[dev]["memory-scaling"] = ms_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
                except Exception:
                    pass

        config_source_df = pd.concat([full_df, sweep_df], ignore_index=True) if not sweep_df.empty else full_df
        config_fig = _build_impl_config_table_figure(config_source_df, title=config_title)
        config_html = config_fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})
        
        _write_switchable_dashboard(
            device_groups,
            args.output,
            config_html=config_html
        )

    if args.view == "switch" and scaling_fig is None:
        print("No batch_size_sweep rows found; wrote dashboard with chart and table views only.")

    if deleted:
        print("Deleted old HTML files:")
        for name in deleted:
            print(f"  - {name}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
