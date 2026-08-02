"""Render the benchmark figures used by the README.

    python benchmarks/plot_figures.py                       # all figures, both themes
    python benchmarks/plot_figures.py --results <file.json> # a different run
    python benchmarks/plot_figures.py --figure throughput-bar

Each figure is written in a light and a dark variant (``*_dark.png``) so the
README can serve the right one per viewer theme, plus a vector PDF of the light
variant for papers.

Model family always comes from the result JSON's ``benchmark_family`` field.
Inferring it from the implementation name by substring is wrong: ``smplxpp_python``
and ``torchure_smplx_cpp`` both contain "smplx" but are SMPL benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_style import (
    FAMILY_LABELS,
    FAMILY_ORDER,
    FAMILY_VERTICES,
    IMPLEMENTATION_LABELS,
    IMPLEMENTATION_MARKERS,
    IMPLEMENTATION_ORDER,
    THEMES,
    THIS_WORK,
    Theme,
    apply_theme,
    compact,
    compact_label,
    figure_note,
    group_positions,
    log_ticks,
    rounded_bar,
    style_axes,
    thin_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "benchmarks" / "figures"

# The run the README quotes. Override with --results.
DEFAULT_RESULTS = REPO_ROOT / "benchmarks/results/rtx5080/all_methods_float32_fresh.json"

HEADLINE_BATCH = 2048

# Non-power-of-two sweep point left over from an older sequence-length run; it
# does not sit on the batch-size grid, so it is dropped from the sweep plots.
EXCLUDED_BATCH_SIZES = {1469}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_json(path)

    missing = {"implementation", "benchmark_family", "fps"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {sorted(missing)}")

    unknown = set(df["implementation"]) - set(IMPLEMENTATION_LABELS)
    if unknown:
        raise SystemExit(
            f"{path} contains implementations with no display label: {sorted(unknown)}.\n"
            "Add them to IMPLEMENTATION_LABELS / IMPLEMENTATION_ORDER in plot_style.py."
        )

    df = df.assign(
        library=df["implementation"].map(IMPLEMENTATION_LABELS),
        family=df["benchmark_family"].map(FAMILY_LABELS),
        fps=pd.to_numeric(df["fps"], errors="coerce"),
        batch_size=pd.to_numeric(df["batch_size"], errors="coerce"),
        mean_ms=pd.to_numeric(df["mean_ms"], errors="coerce"),
    )
    return df[df["fps"] > 0].dropna(subset=["family", "library"])


def drop_unsynchronized(df: pd.DataFrame, *, tolerance: float = 0.75) -> tuple[pd.DataFrame, list[str]]:
    """Drop timings that record kernel launch instead of completion.

    Total work grows with batch size, so for one implementation ``mean_ms`` must
    not fall as ``batch_size`` rises. When it drops sharply, the timer closed
    before the device finished — the known no-op-``sync_once`` path in the
    smplxpp binding, which reports launch time and yields an impossible speedup.

    Such a point is not a slow measurement to be averaged away; it measures the
    wrong thing, so it is removed rather than plotted with a caveat.
    """
    bad_index: list[int] = []
    reasons: list[str] = []

    for (impl, family), rows in df.dropna(subset=["batch_size", "mean_ms"]).groupby(
        ["library", "family"], sort=False
    ):
        rows = rows.sort_values("batch_size")
        best_ms = 0.0
        for idx, row in rows.iterrows():
            if row["mean_ms"] < best_ms * tolerance:
                bad_index.append(idx)
                reasons.append(
                    f"{impl} [{family}] batch {int(row['batch_size']):,}: "
                    f"{row['mean_ms']:.1f} ms after {best_ms:.1f} ms at a smaller batch"
                )
            else:
                best_ms = max(best_ms, float(row["mean_ms"]))

    return df.drop(index=bad_index), reasons


def device_label(df: pd.DataFrame) -> str:
    if "device" in df.columns and df["device"].notna().any():
        return str(df["device"].mode().iloc[0]).replace("NVIDIA GeForce ", "")
    return "GPU"


def present_libraries(df: pd.DataFrame) -> list[str]:
    """Implementations present in the data, in the fixed global order."""
    have = set(df["library"])
    return [lib for lib in IMPLEMENTATION_ORDER if lib in have]


def legend_handles(libs: list[str], theme: Theme, *, marker: bool = False):
    if marker:
        return [
            Line2D(
                [], [], color=theme.color_for(lib), marker=IMPLEMENTATION_MARKERS[lib],
                markersize=5, linewidth=2, markeredgecolor=theme.surface,
                markeredgewidth=1.0, label=lib,
            )
            for lib in libs
        ]
    return [Patch(facecolor=theme.color_for(lib), edgecolor="none", label=lib) for lib in libs]


# --------------------------------------------------------------------------
# Figure 1 — throughput by model family, at the headline batch size
# --------------------------------------------------------------------------


def throughput_bar(df: pd.DataFrame, theme: Theme, batch: int) -> plt.Figure:
    """Grouped columns: one group per model family, one column per library.

    Grouping by family puts every SMPL-X bar physically adjacent, so "who is
    fastest at SMPL-X" is a single left-to-right read. Library identity is
    carried by colour (fixed per library across all figures) plus the legend.
    """
    subset = df[df["batch_size"] == batch]
    subset = subset.sort_values("fps", ascending=False).drop_duplicates(["library", "family"])
    if subset.empty:
        raise SystemExit(f"no rows at batch_size={batch}")

    libs = present_libraries(subset)
    per_family = {
        fam: [lib for lib in libs if lib in set(subset.loc[subset["family"] == fam, "library"])]
        for fam in FAMILY_ORDER
    }
    offsets, ticks = group_positions([len(per_family[f]) for f in FAMILY_ORDER])

    fig = plt.figure(figsize=(6.6, 3.2))
    # Explicit bands: title, legend, plot, note. Letting the title and a legend
    # anchored above the axes negotiate the same strip is what makes them collide.
    ax = fig.add_axes((0.10, 0.19, 0.885, 0.585))

    ymax = float(subset["fps"].max())
    ax.set_xlim(-0.9, offsets[-1][-1] + 0.9)
    ax.set_ylim(0, ymax * 1.14)  # headroom for the cap labels
    bar_w = 0.66  # of a 1.0 slot — the remainder is the surface gap

    for family, centres in zip(FAMILY_ORDER, offsets):
        rows = subset[subset["family"] == family]
        for lib, x in zip(per_family[family], centres):
            fps = float(rows.loc[rows["library"] == lib, "fps"].iloc[0])
            rounded_bar(ax, x - bar_w / 2, 0, bar_w, fps, theme.color_for(lib))
            ax.text(
                x, fps + ymax * 0.018, compact_label(fps),
                ha="center", va="bottom", fontsize=7.2,
                color=theme.ink_primary if lib == THIS_WORK else theme.ink_secondary,
                fontweight="bold" if lib == THIS_WORK else "normal",
            )

    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{f}  ·  {FAMILY_VERTICES[f]:,} vertices" for f in FAMILY_ORDER],
        fontsize=9, color=theme.ink_primary,
    )
    ax.set_ylabel("Throughput (frames/s)", fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: compact(v))
    style_axes(ax, theme, axis="y")

    # Title says what is plotted; run conditions live in the footnote.
    fig.text(
        0.10, 0.975, "Forward-pass throughput",
        ha="left", va="top", fontsize=10.5, fontweight="semibold",
        color=theme.ink_primary,
    )
    fig.legend(
        handles=legend_handles(libs, theme),
        loc="upper left", bbox_to_anchor=(0.093, 0.915),
        ncol=3, handlelength=0.85, handleheight=0.85, fontsize=7.8,
    )
    return fig


# --------------------------------------------------------------------------
# Figure 2 — throughput across the batch-size sweep
# --------------------------------------------------------------------------


def throughput_sweep(df: pd.DataFrame, theme: Theme) -> plt.Figure:
    """One panel per model family, sharing a single y-axis.

    A shared axis is the point: with independent y-scales the two panels look
    alike while differing by an order of magnitude, which invents a similarity
    the data does not contain.
    """
    sweep = df[~df["batch_size"].isin(EXCLUDED_BATCH_SIZES)].dropna(subset=["batch_size"])
    if sweep.empty:
        raise SystemExit("no batch-size sweep rows")

    libs = present_libraries(sweep)
    batches = np.sort(sweep["batch_size"].unique())
    yticks = log_ticks(sweep["fps"].to_numpy())

    fps = sweep["fps"].to_numpy()
    fig = plt.figure(figsize=(7.0, 3.5))
    # Same explicit bands as the bar figure: title, legend, plots, note. The two
    # panels are placed by hand so the shared y-axis labels appear once, on the left.
    rects = [(0.093, 0.205, 0.435, 0.525), (0.556, 0.205, 0.435, 0.525)]
    axes = [fig.add_axes(rect) for rect in rects]

    for ax, family, rect in zip(axes, FAMILY_ORDER, rects):
        panel = sweep[sweep["family"] == family]
        for lib in libs:
            rows = panel[panel["library"] == lib].sort_values("batch_size")
            if len(rows) < 2:
                continue
            ax.plot(
                rows["batch_size"], rows["fps"],
                color=theme.color_for(lib),
                marker=IMPLEMENTATION_MARKERS[lib],
                markersize=4.2,
                # 2px surface ring keeps markers legible where lines cross.
                markeredgecolor=theme.surface, markeredgewidth=1.0,
                linewidth=1.8, zorder=4 if lib == THIS_WORK else 3,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.set_ylim(fps.min() * 0.6, fps.max() * 1.9)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(lambda v, _: compact(v))
        ax.minorticks_off()
        ax.set_xlabel("Batch size", fontsize=8.5)
        style_axes(ax, theme, axis="both")
        fig.text(
            rect[0], rect[1] + rect[3] + 0.025, family,
            ha="left", va="bottom", fontsize=9.5, fontweight="semibold",
            color=theme.ink_primary,
        )

    axes[0].set_ylabel("Throughput (frames/s)", fontsize=8.5)
    axes[1].set_yticklabels([])  # shared scale — label it once

    # Thinning measures rendered text, so it runs once the axes are final.
    tick_labels = [f"{int(b):,}" for b in batches]
    for ax in axes:
        ax.set_xticklabels(thin_labels(ax, batches, tick_labels, fontsize=7.8), fontsize=7.8)

    fig.text(
        0.093, 0.975, "Throughput scaling",
        ha="left", va="top", fontsize=10.5, fontweight="semibold",
        color=theme.ink_primary,
    )
    fig.legend(
        handles=legend_handles(libs, theme, marker=True),
        loc="upper left", bbox_to_anchor=(0.086, 0.925),
        ncol=3, fontsize=7.8, handlelength=1.6,
    )
    return fig


# --------------------------------------------------------------------------
# Figure 3 — mean runtime (README header)
# --------------------------------------------------------------------------


def runtime_bar(df: pd.DataFrame, theme: Theme, batch: int) -> plt.Figure:
    """Horizontal bars, lower is better — the README's header figure."""
    subset = df[(df["batch_size"] == batch) & df["mean_ms"].notna()]
    subset = subset.sort_values("mean_ms").drop_duplicates(["library", "family"])
    if subset.empty:
        raise SystemExit(f"no runtime rows at batch_size={batch}")

    libs = present_libraries(subset)
    # Families run top-to-bottom, so the group order is reversed against the
    # y-axis, which grows upward.
    families = list(reversed(FAMILY_ORDER))
    per_family = {
        fam: [lib for lib in libs if lib in set(subset.loc[subset["family"] == fam, "library"])]
        for fam in families
    }
    offsets, ticks = group_positions([len(per_family[f]) for f in families], gap=1.5)

    fig = plt.figure(figsize=(6.6, 3.2))
    ax = fig.add_axes((0.115, 0.235, 0.855, 0.53))

    xmax = float(subset["mean_ms"].max())
    ax.set_ylim(-0.9, offsets[-1][-1] + 0.9)
    ax.set_xlim(0, xmax * 1.22)  # room for the longest end-label
    bar_h = 0.62

    for family, centres in zip(families, offsets):
        rows = subset[subset["family"] == family]
        # Negated so the fixed library order still reads downward on screen.
        for lib, y in zip(per_family[family], -centres + offsets[-1][-1]):
            ms = float(rows.loc[rows["library"] == lib, "mean_ms"].iloc[0])
            rounded_bar(ax, 0, y - bar_h / 2, ms, bar_h, theme.color_for(lib), horizontal=True)
            ax.text(
                ms + xmax * 0.012, y, f"{ms:,.1f} ms",
                ha="left", va="center", fontsize=7.2,
                color=theme.ink_primary if lib == THIS_WORK else theme.ink_secondary,
                fontweight="bold" if lib == THIS_WORK else "normal",
            )

    ax.set_yticks([-t + offsets[-1][-1] for t in ticks])
    ax.set_yticklabels(families, fontsize=9, color=theme.ink_primary)
    # The title already names the measure; the axis carries units and direction.
    ax.set_xlabel("Milliseconds  —  lower is better", fontsize=8.5)
    style_axes(ax, theme, axis="x")

    fig.text(
        0.115, 0.975, "Mean runtime per forward pass",
        ha="left", va="top", fontsize=10.5, fontweight="semibold",
        color=theme.ink_primary,
    )
    fig.legend(
        handles=legend_handles(libs, theme),
        loc="upper left", bbox_to_anchor=(0.108, 0.915),
        ncol=3, handlelength=0.85, handleheight=0.85, fontsize=7.8,
    )
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

FIGURES = {
    "throughput-bar": ("throughput_by_model_batch2048", throughput_bar),
    "throughput-sweep": ("throughput_vs_batch_size", throughput_sweep),
    "runtime-bar": ("runtime_by_model_batch2048", runtime_bar),
}


def build_note(
    name: str, df: pd.DataFrame, results: Path, batch: int, dropped: list[str]
) -> str:
    """Run conditions for the footnote.

    Everything that parameterises the measurement — batch size, device, dtype,
    timing protocol, source file — belongs here rather than in the title. The
    title says what is plotted; the footnote says under what conditions.
    """
    device = device_label(df)
    dtype = str(df["impl_dtype"].mode().iloc[0]) if "impl_dtype" in df.columns else "float32"

    if name == "throughput-sweep":
        sizes = df["batch_size"].dropna()
        scope = f"batch sizes {int(sizes.min()):,}–{int(sizes.max()):,}"
    else:
        scope = f"batch {batch:,}"

    note = (
        f"{scope} · {device} · {dtype} · full forward pass · "
        "median of timed runs after untimed warmup\n"
        f"source: {results.name}"
    )
    # Only relevant where the excluded points fall inside the figure's scope;
    # the discarded timings are all at the top of the sweep.
    if dropped and name == "throughput-sweep":
        note += f" · {len(dropped)} unsynchronized timing(s) excluded"
    return note


def build(name: str, df: pd.DataFrame, theme: Theme, batch: int, note: str) -> plt.Figure:
    apply_theme(theme)
    builder = FIGURES[name][1]
    fig = builder(df, theme, batch) if builder is not throughput_sweep else builder(df, theme)
    figure_note(fig, note, theme)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS,
                        help=f"result JSON to plot (default: {DEFAULT_RESULTS.name})")
    parser.add_argument("--out-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--batch", type=int, default=HEADLINE_BATCH,
                        help="batch size for the single-batch figures")
    parser.add_argument("--figure", choices=sorted(FIGURES), action="append",
                        help="render only these figures (repeatable; default: all)")
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"results file not found: {args.results}")

    df = load_results(args.results)
    df, dropped = drop_unsynchronized(df)
    for reason in dropped:
        print(f"dropped unsynchronized timing — {reason}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.figure or sorted(FIGURES):
        stem = FIGURES[name][0]
        note = build_note(name, df, args.results, args.batch, dropped)
        for theme_name, theme in THEMES.items():
            fig = build(name, df, theme, args.batch, note)
            suffix = "" if theme_name == "light" else "_dark"
            png = args.out_dir / f"{stem}{suffix}.png"
            fig.savefig(png)
            if theme_name == "light":
                fig.savefig(args.out_dir / f"{stem}.pdf")
            plt.close(fig)
            # --out-dir may point outside the repo, where relative_to() raises.
            try:
                shown = png.relative_to(REPO_ROOT)
            except ValueError:
                shown = png
            print(f"wrote {shown}")


if __name__ == "__main__":
    main()
