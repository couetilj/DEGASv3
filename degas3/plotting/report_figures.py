"""Figure helpers for the explanation and report stages.

The attribution figures show *model attribution relative to low-score same-cluster
cells*. They are neither causal effects nor independent discovery evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from degas3.plotting.style import BLUE, DONOR, DPI, NEUTRAL, PANEL_INCHES, apply_style  # noqa: E402

DISCLAIMER = "Model attribution relative to low-score same-cluster cells; not causal, not independent discovery."


def _bars(ax, genes, values, lo=None, hi=None, signed=False, title="", xlabel="", unsigned_color=NEUTRAL):
    y = np.arange(len(genes))[::-1]
    colors = [(DONOR if v >= 0 else BLUE) if signed else unsigned_color for v in values]
    ax.barh(y, values, color=colors, height=0.7)
    if lo is not None and hi is not None and np.isfinite(lo).all() and np.isfinite(hi).all():
        ax.errorbar(values, y, xerr=[np.clip(values - lo, 0, None), np.clip(hi - values, 0, None)],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.set_box_aspect(1)
    if signed:
        ax.axvline(0, color="black", linewidth=0.8)


def top_genes_figure(attr: pd.DataFrame, stem: str | Path, *, signed: bool, cluster: str, n: int = 20) -> list[Path]:
    """Two square panels (pooled; equal-donor with 95% donor-bootstrap interval) of the top ``n`` genes.

    Genes are ranked by the equal-donor mean (absolute, or signed by absolute value). When the
    run has no donor column the ranking and both panels fall back to pooled values without
    intervals. Values are mean attributions of the mean raw score, not of percentiles.
    """
    apply_style()
    has_donor = "donor_abs" in attr and attr["donor_abs"].notna().any()
    if signed:
        key_d, key_p = "donor_signed", "pooled_signed"
    else:
        key_d, key_p = "donor_abs", "pooled_abs"
    rank_key = key_d if has_donor else key_p
    top = attr.reindex(attr[rank_key].abs().sort_values(ascending=False).index).head(n)
    genes = top["gene"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_INCHES, PANEL_INCHES))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.86, wspace=0.55)
    kind = "signed mean" if signed else "mean absolute"
    _bars(axes[0], genes, top[key_p].to_numpy(), signed=signed, title="Pooled (targets weighted equally)",
          xlabel=f"{kind} attribution")
    if has_donor:
        lo = top[key_d + "_lo"].to_numpy() if key_d + "_lo" in top else None
        hi = top[key_d + "_hi"].to_numpy() if key_d + "_hi" in top else None
        _bars(axes[1], genes, top[key_d].to_numpy(), lo, hi, signed=signed,
              title="Equal-donor (95% donor bootstrap)", xlabel=f"{kind} attribution", unsigned_color=DONOR)
    else:
        axes[1].text(0.5, 0.5, "no donor column:\nequal-donor summary unavailable", ha="center", va="center")
        axes[1].set_axis_off()
        axes[1].set_box_aspect(1)
    fig.suptitle(f"Cluster {cluster}: top {len(genes)} genes, {kind} attribution", fontsize=12)
    fig.text(0.5, 0.03, DISCLAIMER, ha="center", fontsize=9)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = stem.with_name(stem.name + "." + ext)
        fig.savefig(p, dpi=DPI, bbox_inches=None)
        paths.append(p)
    plt.close(fig)
    return paths
