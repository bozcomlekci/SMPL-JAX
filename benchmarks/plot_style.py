"""Shared plotting style for the benchmark figures.

Pure matplotlib — no seaborn, no tueplots — so the figures regenerate from a
bare ``pip install -e ".[viz]"`` and render identically everywhere.

The palette is a validated categorical set: hues are assigned to
implementations in a fixed order and never cycled or re-assigned by rank, so an
implementation keeps its colour across every figure and across filtered views.
Both themes were checked for lightness band, chroma floor, colour-vision-
deficiency separation of adjacent pairs, and contrast against their own surface.
Three light-mode hues sit below 3:1 on the light surface, so every bar carries a
visible value label rather than relying on fill colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# --------------------------------------------------------------------------
# Naming — one authoritative map, used by every figure and axis label.
# --------------------------------------------------------------------------

# Raw `implementation` keys in the result JSON -> display label.
# The same library benchmarked on both model families collapses to one label;
# the family it was run on comes from the authoritative `benchmark_family`
# field, never from substring-matching the implementation key.
IMPLEMENTATION_LABELS = {
    "smpl_jax_smpl": "bozcomlekci/SMPL-JAX",
    "smpl_jax_smplx": "bozcomlekci/SMPL-JAX",
    "smplx_torch": "vchoutas/smplx",
    "smplx_torch_smpl": "vchoutas/smplx",
    "smplxpp_python": "sxyu/smplxpp",
    "smplxpp_python_smplx": "sxyu/smplxpp",
    "smplpytorch_torch": "gulvarol/smplpytorch",
    "torchure_smplx_cpp": "Hydran00/torchure_smplx",
}

# Fixed draw / legend / colour order. Position in this list determines both the
# colour slot and the slot a bar occupies inside its group, so an implementation
# sits at the same offset in every group — a missing bar is then visibly a gap
# rather than a silent re-shuffle of its neighbours.
IMPLEMENTATION_ORDER = [
    "bozcomlekci/SMPL-JAX",
    "vchoutas/smplx",
    "sxyu/smplxpp",
    "gulvarol/smplpytorch",
    "Hydran00/torchure_smplx",
]

THIS_WORK = "bozcomlekci/SMPL-JAX"

# Secondary (non-colour) identity channel for line charts: survives greyscale
# printing and full-severity colour-vision deficiency.
IMPLEMENTATION_MARKERS = {
    "bozcomlekci/SMPL-JAX": "o",
    "vchoutas/smplx": "s",
    "sxyu/smplxpp": "^",
    "gulvarol/smplpytorch": "D",
    "Hydran00/torchure_smplx": "v",
}

# `benchmark_family` values -> display label.
FAMILY_LABELS = {"smpl": "SMPL", "smplx": "SMPL-X"}
FAMILY_ORDER = ["SMPL", "SMPL-X"]

# Vertex counts, for the axis subtitle.
FAMILY_VERTICES = {"SMPL": 6890, "SMPL-X": 10475}


# --------------------------------------------------------------------------
# Themes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink_primary: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    series: tuple[str, ...]

    def color_for(self, implementation: str) -> str:
        """Colour follows the entity, never its rank in the current view."""
        return self.series[IMPLEMENTATION_ORDER.index(implementation) % len(self.series)]


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink_primary="#0b0b0b",
    ink_secondary="#52514e",
    ink_muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink_primary="#ffffff",
    ink_secondary="#c3c2b7",
    ink_muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
)

THEMES = {"light": LIGHT, "dark": DARK}


def apply_theme(theme: Theme) -> None:
    """Install the theme as matplotlib rcParams.

    Grid and axis rules are solid hairlines one step off the surface — never
    dashed, which reads as a threshold rather than a grid.
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Nimbus Sans"],
            "font.size": 9,
            "figure.facecolor": theme.surface,
            "figure.dpi": 200,
            "savefig.facecolor": theme.surface,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.facecolor": theme.surface,
            "axes.edgecolor": theme.axis,
            "axes.linewidth": 0.8,
            "axes.labelcolor": theme.ink_secondary,
            "axes.labelsize": 9,
            "axes.titlesize": 10.5,
            "axes.titlecolor": theme.ink_primary,
            "axes.titleweight": "semibold",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": theme.ink_muted,
            "ytick.color": theme.ink_muted,
            "xtick.labelcolor": theme.ink_secondary,
            "ytick.labelcolor": theme.ink_secondary,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.handlelength": 1.0,
            "legend.handleheight": 1.0,
            "legend.borderpad": 0.0,
            "legend.columnspacing": 1.4,
            "legend.labelspacing": 0.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------
# Marks
# --------------------------------------------------------------------------


def data_radius(ax, pixels: float) -> tuple[float, float]:
    """Convert a radius in points/pixels into x- and y-data units.

    A corner radius has to be a *physical* size. Deriving it from a fraction of
    the bar's data width gives a radius in x-units, which is invisible once
    applied along a y-axis that spans 150,000. Axis limits must already be set
    when this is called.
    """
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0.0, 0.0))
    x1, y1 = inv.transform((pixels, pixels))
    return abs(x1 - x0), abs(y1 - y0)


def rounded_bar(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    *,
    horizontal: bool = False,
    radius_px: float = 4.0,
    zorder: float = 3,
) -> PathPatch:
    """A bar rounded at the data end and square at the baseline.

    matplotlib's ``bar`` draws plain rectangles; rounding only the growing end
    keeps the baseline reading as a hard zero while softening the mark. The
    radius is capped at half the bar's own extent so short bars degrade to a
    rectangle rather than a lozenge that would overstate their length.
    """
    rx, ry = data_radius(ax, radius_px)
    x0, x1, y0, y1 = x, x + width, y, y + height

    if horizontal:
        rx = min(rx, abs(width) * 0.5)
        ry = min(ry, abs(height) * 0.5)
        verts = [
            (x0, y0), (x1 - rx, y0),
            (x1, y0), (x1, y0 + ry),
            (x1, y1 - ry), (x1, y1), (x1 - rx, y1),
            (x0, y1), (x0, y0),
        ]
    else:
        rx = min(rx, abs(width) * 0.5)
        ry = min(ry, abs(height) * 0.5)
        verts = [
            (x0, y0), (x0, y1 - ry),
            (x0, y1), (x0 + rx, y1),
            (x1 - rx, y1), (x1, y1), (x1, y1 - ry),
            (x1, y0), (x0, y0),
        ]
    codes = [
        Path.MOVETO, Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CURVE3, Path.CURVE3,
        Path.LINETO, Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def compact(value: float) -> str:
    """Axis-tick formatting: 150000 -> '150k'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def compact_label(value: float) -> str:
    """Direct-label formatting: 147862 -> '147.9k'.

    Short enough to sit horizontally on a narrow column; the exact figures live
    in the README table, which is the chart's table view.
    """
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


def style_axes(ax, theme: Theme, *, axis: str = "y") -> None:
    """Recessive hairline grid behind the marks, on one axis only."""
    ax.set_axisbelow(True)
    ax.grid(True, axis=axis, color=theme.grid, linewidth=0.8, linestyle="-")
    ax.tick_params(length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(theme.axis)
    ax.spines["bottom"].set_color(theme.axis)


def figure_note(fig, text: str, theme: Theme, *, y: float = 0.015) -> None:
    """Provenance / caveat line, set in muted ink inside the figure's bottom band.

    Kept inside the figure rectangle rather than below it: with
    ``bbox_inches="tight"`` a negative y grows the canvas instead of using the
    margin that is already reserved, which is what opens a band of dead space.
    """
    fig.text(
        0.5, y, text, ha="center", va="bottom",
        fontsize=7.0, color=theme.ink_muted, linespacing=1.45,
    )


def group_positions(group_sizes: list[int], gap: float = 1.6) -> tuple[list[np.ndarray], list[float]]:
    """Lay bars out on a continuous index with a fixed gap between groups.

    Fixed-width bands sized to the largest group leave the smaller groups
    padded with dead space and push the groups apart; consuming exactly the
    slots each group needs keeps the figure compact and the gaps uniform.

    Returns the bar centres per group, and each group's tick position.
    """
    offsets: list[np.ndarray] = []
    ticks: list[float] = []
    cursor = 0.0
    for n in group_sizes:
        centres = cursor + np.arange(n)
        offsets.append(centres)
        ticks.append(float(centres.mean()) if n else cursor)
        cursor += n + gap
    return offsets, ticks


def log_ticks(values: np.ndarray) -> list[float]:
    """Decade ticks covering the data, for a log axis.

    The top decade is included only if the data actually enters it — rounding
    the maximum up unconditionally leaves an empty decade of dead space.
    """
    lo = int(np.floor(np.log10(values.min())))
    hi = int(np.floor(np.log10(values.max())))
    return [10.0**k for k in range(lo, hi + 1)]


def thin_labels(
    ax, ticks: np.ndarray, labels: list[str], *, fontsize: float, pad: float = 1.2
) -> list[str]:
    """Blank labels that would overprint, keeping the ticks themselves.

    On a log2 axis a sweep like 1, 8, …, 2048, 4096, 8192 is unevenly spaced:
    the early gaps span three doublings, the tail gaps only one, so the tail
    labels collide. Dropping the one or two that don't fit beats rotating the
    whole axis to fix them.

    Collisions are *measured* against the rendered text, not guessed from a
    fixed spacing threshold — the tail gap here is a full log2 unit, which any
    plausible threshold would wave through even though the labels overlap.
    Requires the axis scale and limits to be set first.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    widths = []
    for label in labels:
        probe = ax.text(0, 0, label, fontsize=fontsize)
        widths.append(probe.get_window_extent(renderer=renderer).width)
        probe.remove()

    xs = ax.transData.transform(
        np.column_stack([np.asarray(ticks, dtype=float), np.zeros(len(ticks))])
    )[:, 0]

    kept: list[str] = []
    last_right = -np.inf
    for x, width, label in zip(xs, widths, labels):
        half = width * pad / 2
        if x - half >= last_right:
            kept.append(label)
            last_right = x + half
        else:
            kept.append("")

    # The final tick anchors the axis; if it lost, drop the neighbour instead.
    if kept[-1] == "":
        for i in range(len(kept) - 2, -1, -1):
            if kept[i]:
                kept[i] = ""
                break
        kept[-1] = labels[-1]
    return kept
