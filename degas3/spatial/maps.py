"""Spatial maps of scores, clusters and high/low groups (``st_cell`` cells and ``st_spot`` spots).

Each map is a square panel (equal aspect, origin lower) with small rasterized points. The score
map shows the cell-level (or spot-level) ``ensemble_score``: a percentile rank among the scored
cells, not a probability. Maps are descriptive; neighbouring points are not independent and no
test is performed here. When more than ``MAX_POINTS`` points are present a fixed-seed random subset
is drawn and the figure footnote says so.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degas3.config import RunConfig
from degas3.plotting import style
from degas3.runs import RunDir
from degas3.spatial import read_cell_scores, read_spatial_cells, safe_name, sample_order, spatial_modality

log = logging.getLogger(__name__)

MAX_POINTS = 200_000
_GROUP_COLORS = {"high": style.SCORE_HIGH, "low": style.SCORE_LOW}
_OTHER_COLOR = "#BDBDBD"


def thin_indices(n: int, max_points: int, seed: int) -> tuple[np.ndarray, bool]:
    """Sorted indices of at most ``max_points`` of ``n`` points (fixed-seed subset); flag whether thinned."""
    if n <= max_points:
        return np.arange(n), False
    return np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False)), True


def point_size(n: int) -> float:
    """Marker area (pt^2) shrinking with the number of points shown."""
    return float(np.clip(4000.0 / max(n, 1) ** 0.75 * 4, 1.0, 20.0))


def _panel(title: str, xy: np.ndarray, thinned: bool, n_total: int, seed: int):
    fig, ax = style.square_panel(title, "x", "y")
    ax.set_aspect("equal", adjustable="datalim")
    note = f"{len(xy)} of {n_total} points shown (random subset, seed {seed})" if thinned else f"{n_total} points"
    fig.text(0.19, 0.06, note + "; neighbouring points are not independent", fontsize=8)
    return fig, ax


def plot_score_map(xy, score, title, stem, thinned=False, n_total=None, seed=0) -> list[Path]:
    fig, ax = _panel(title, xy, thinned, n_total or len(xy), seed)
    order = np.argsort(score, kind="stable")  # high scores drawn last
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=score[order], cmap=style.score_cmap(), norm=style.score_norm(),
                    s=point_size(len(xy)), linewidths=0, rasterized=True)
    style.colorbar(fig, ax, sc, style.SCORE_LABEL)
    return style.save_panel(fig, ax, stem)


def plot_category_map(xy, labels, title, stem, colors: Optional[dict] = None, thinned=False, n_total=None, seed=0) -> list[Path]:
    fig, ax = _panel(title, xy, thinned, n_total or len(xy), seed)
    cats = sorted(set(labels.tolist()))
    pal = style.categorical_palette(len(cats))
    cmap = {c: (colors or {}).get(c, pal[i]) for i, c in enumerate(cats)}
    s = point_size(len(xy))
    for c in cats:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=s, color=cmap[c], linewidths=0, rasterized=True,
                   label=(c if len(c) <= 18 else c[:17] + "~") + f" ({int(m.sum())})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, fontsize=7, markerscale=2.5,
              handletextpad=0.2, borderaxespad=0.0, ncol=1)
    return style.save_panel(fig, ax, stem)


def _read_groups(rd: RunDir, obs_names: pd.Index) -> Optional[pd.Series]:
    path = rd.path("strata", "cell_groups.csv.gz")
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if not {"cell_id", "group"} <= set(df.columns):
        log.warning("%s lacks cell_id/group columns; skipping the group map.", path)
        return None
    df["cell_id"] = df["cell_id"].astype(str)
    g = df.drop_duplicates("cell_id").set_index("cell_id")["group"].astype(str)
    return g.reindex(obs_names.astype(str)).fillna("not stratified")


def run_maps(run: str | Path, force: bool = False) -> dict:
    """Write score, cluster and (when strata exist) group maps per sample; skip when complete."""
    rd = RunDir(run)
    cfg = RunConfig.load(rd.root)
    spatial_modality(cfg, ("st_cell", "st_spot"), "maps")
    if rd.stage_complete("maps") and not force:
        log.info("maps already complete; skipping (use --force to redo).")
        return rd.read_complete("maps") or {}
    adata, samples, xy_all = read_spatial_cells(rd, cfg)
    scores = read_cell_scores(rd, adata.obs_names)
    ens = scores["ensemble_score"].to_numpy(dtype=float)
    if cfg.cluster_col and cfg.cluster_col in adata.obs:
        clusters = adata.obs[cfg.cluster_col].astype(str).to_numpy()
    elif "cluster" in scores.columns:
        clusters = scores["cluster"].astype(str).to_numpy()
    else:
        clusters = None
    groups = _read_groups(rd, adata.obs_names)
    written: dict[str, list[str]] = {}
    for si, name in enumerate(sample_order(samples)):
        idx = np.flatnonzero(samples == name)
        keep, thinned = thin_indices(len(idx), MAX_POINTS, cfg.selection_seed + si)
        sel = idx[keep]
        xy = xy_all[sel]
        mdir = rd.ensure("spatial", safe_name(name), "maps")
        kind = "spots" if cfg.modality == "st_spot" else "cells"
        common = dict(thinned=thinned, n_total=len(idx), seed=cfg.selection_seed + si)
        files = plot_score_map(xy, ens[sel], f"Ensemble score ({kind}): {name}", mdir / "score", **common)
        if clusters is not None:
            files += plot_category_map(xy, clusters[sel], f"Cluster ({kind}): {name}", mdir / "cluster", **common)
        if groups is not None:
            lab = groups.to_numpy()[sel]
            files += plot_category_map(xy, lab, f"High / low group ({kind}): {name}", mdir / "group",
                                       colors={**_GROUP_COLORS, "not stratified": _OTHER_COLOR}, **common)
        written[name] = [str(f.relative_to(rd.root)) for f in files]
        log.info("sample %s: %d maps written%s", name, len(files), " (thinned)" if thinned else "")
    payload = {"status": "complete", "modality": cfg.modality, "samples": written, "max_points": MAX_POINTS,
               "has_clusters": clusters is not None, "has_groups": groups is not None}
    rd.write_complete("maps", payload)
    return payload


def cli_maps(args: argparse.Namespace) -> int:
    run_maps(args.run, force=bool(getattr(args, "force", False)))
    return 0
