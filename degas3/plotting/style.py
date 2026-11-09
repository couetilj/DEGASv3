"""Figure style: square 6.4 in panels, Okabe-Ito categories, blue-gray-red score colormap.

The score colormap shows an *ensemble percentile* in [0, 1] (rank among scored cells), not a
probability; 0.5 is the middle of the ranking.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402

PANEL_INCHES = 6.4
DPI = 300
SCORE_LABEL = "Ensemble DEGAS percentile"
CASE = "#D55E00"
CONTROL = "#0072B2"
SCORE_LOW, SCORE_MID, SCORE_HIGH = "#2166AC", "#E5E5E5", "#B2182B"
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def apply_style() -> None:
    """Set matplotlib rcParams for all degas3 figures."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "figure.titlesize": 12,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white", "savefig.bbox": None,
    })


def square_panel(title: str, xlabel: str = "", ylabel: str = ""):
    """New 6.4 x 6.4 in figure with a square axes; returns (fig, ax)."""
    apply_style()
    fig = plt.figure(figsize=(PANEL_INCHES, PANEL_INCHES))
    ax = fig.add_axes([0.19, 0.245, 0.63, 0.63])
    ax.set_box_aspect(1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.text(0.505, 0.955, title, ha="center", va="top", fontsize=12)
    return fig, ax


def colorbar(fig, ax, mappable, label: str = "", ticks=None):
    """Colorbar in the fixed slot to the right of the square axes."""
    cax = fig.add_axes([0.85, 0.32, 0.022, 0.47])
    cb = fig.colorbar(mappable, cax=cax, ticks=ticks)
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return cb


def save_panel(fig, ax, stem: str | Path, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save at 300 dpi without tight cropping; asserts the plot area and canvas are square."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    box = ax.get_window_extent()
    if abs(box.width / box.height - 1) > 1e-6 or abs(fig.get_figwidth() / fig.get_figheight() - 1) > 1e-6:
        raise ValueError("panel is not square; build it with square_panel() and do not resize the figure.")
    paths = []
    for ext in formats:
        p = stem.with_name(stem.name + "." + ext)
        fig.savefig(p, dpi=DPI, bbox_inches=None)
        paths.append(p)
    plt.close(fig)
    return paths


def score_cmap() -> LinearSegmentedColormap:
    """Blue -> gray90 -> red; gray sits at 0.5 of the 0-1 range."""
    return LinearSegmentedColormap.from_list("degas3_score", [SCORE_LOW, SCORE_MID, SCORE_HIGH], N=256)


def score_norm() -> Normalize:
    """Fixed 0-1 limits so the midpoint colour is exactly 0.5."""
    return Normalize(vmin=0.0, vmax=1.0)


def categorical_palette(n: int) -> list[str]:
    """``n`` hex colours: Okabe-Ito first, then progressively lighter cycles of the same hues."""
    colors: list[str] = []
    cycle = 0
    while len(colors) < n:
        for c in OKABE_ITO:
            mix = 0.35 * cycle
            r, g, b = to_rgb(c)
            colors.append("#%02X%02X%02X" % tuple(int(round((v + (1 - v) * mix) * 255)) for v in (r, g, b)))
        cycle = min(cycle + 1, 2)  # cap lightening at 70% so colours stay visible
    return colors[:n]
