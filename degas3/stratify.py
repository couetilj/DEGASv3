"""Within-cluster high/low stratification of the ensemble score.

The ensemble score is a *rank percentile* (position among scored cells), not a probability.
Cutoffs are computed separately inside every cluster, so "high" and "low" are relative to the
cluster's own score distribution. Groups are descriptive labels for downstream contrasts;
cells are not independent replicates, so nothing here is donor-level inference.

Ties rule (SPEC): cells whose score equals a cutoff are ``tie`` and are excluded from both
groups. ``median``: low < median < high. ``quantiles [q_lo, q_hi]``: low < cutoff_lo,
high > cutoff_hi, cells strictly between are ``middle``, cells equal to either cutoff are ``tie``.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from degas3.config import RunConfig
from degas3.contracts import LAYOUT
from degas3.plotting import style
from degas3.runs import RunDir

log = logging.getLogger(__name__)

GROUPS = ["low", "high", "tie", "middle"]
NO_CLUSTER = "all"
NO_DONOR = "all"


def cluster_dirname(cluster: str) -> str:
    """Filesystem-safe directory name for a cluster label."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cluster)).strip("_")
    return name or "cluster"


def read_scores(run: RunDir) -> pd.DataFrame:
    """Read ``scores/cell_scores.csv.gz`` with string id columns.

    Empty ``cluster``/``donor`` (config column None) become ``"all"``; ``label`` and ``sample``
    stay empty strings when absent.
    """
    path = run.path(LAYOUT.SCORES, "cell_scores.csv.gz")
    if not path.is_file():
        raise ValueError(f"{path} not found; run `degas3 score {run.root}` first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("cell_id", "ensemble_score"):
        if col not in df.columns:
            raise ValueError(f"{path} lacks column {col!r}; re-run the score stage.")
    for col in ("donor", "cluster", "sample", "label"):
        if col not in df.columns:
            df[col] = ""
    df["ensemble_score"] = df["ensemble_score"].astype(float)
    if df["ensemble_score"].isna().any():
        raise ValueError("ensemble_score contains NaN; re-run the score stage.")
    df["cluster"] = df["cluster"].replace("", NO_CLUSTER)
    df["donor"] = df["donor"].replace("", NO_DONOR)
    return df


def assign_groups(scores: np.ndarray, split: str, quantiles=(0.25, 0.75)) -> tuple[np.ndarray, tuple[float, float]]:
    """Group labels for one cluster's scores and the (low, high) cutoffs used.

    Equality is exact floating-point equality with the cutoff (ties are excluded).
    """
    scores = np.asarray(scores, float)
    if split == "median":
        lo = hi = float(np.median(scores))
    elif split == "quantiles":
        lo, hi = (float(np.quantile(scores, q)) for q in quantiles)
    else:
        raise ValueError(f"split must be 'median' or 'quantiles', got {split!r}.")
    group = np.full(scores.shape, "middle", dtype=object)
    group[scores < lo] = "low"
    group[scores > hi] = "high"
    group[(scores == lo) | (scores == hi)] = "tie"
    return group, (lo, hi)


def stratify_scores(df: pd.DataFrame, split: str, quantiles=(0.25, 0.75)):
    """Return (cell_groups, cutoffs, coverage) tables for a scores frame."""
    groups = pd.Series("", index=df.index, dtype=object)
    cut_rows = []
    for cluster, idx in df.groupby("cluster", sort=True).groups.items():
        g, (lo, hi) = assign_groups(df.loc[idx, "ensemble_score"].to_numpy(), split, quantiles)
        groups.loc[idx] = g
        counts = pd.Series(g).value_counts()
        cut_rows.append({
            "cluster": cluster, "split": split,
            "cutoffs": f"{lo:.10g}" if split == "median" else f"{lo:.10g};{hi:.10g}",
            "cutoff_low": lo, "cutoff_high": hi, "n_cells": len(idx),
            "n_low": int(counts.get("low", 0)), "n_high": int(counts.get("high", 0)),
            "n_tie": int(counts.get("tie", 0)), "n_middle": int(counts.get("middle", 0)),
        })
    cell_groups = pd.DataFrame({"cell_id": df["cell_id"], "cluster": df["cluster"],
                                "donor": df["donor"], "group": groups})
    tab = cell_groups.groupby(["cluster", "donor"])["group"].value_counts().unstack(fill_value=0)
    for g in GROUPS:
        if g not in tab.columns:
            tab[g] = 0
    tab = tab.reset_index()
    cov = pd.DataFrame({
        "cluster": tab["cluster"], "donor": tab["donor"], "n_low": tab["low"], "n_high": tab["high"],
        "n_tie": tab["tie"], "fraction_high": tab["high"] / (tab["low"] + tab["high"] + tab["tie"] + tab["middle"]),
        "n_cells": tab[GROUPS].sum(axis=1),
    })
    return cell_groups, pd.DataFrame(cut_rows), cov


def _plot_groups(cutoffs: pd.DataFrame, stem: Path) -> None:
    fig, ax = style.square_panel("Cells per group within cluster", "Cluster", "Cells")
    pal = {"low": style.CONTROL, "high": style.CASE, "middle": "#BBBBBB", "tie": "#666666"}
    x = np.arange(len(cutoffs))
    bottom = np.zeros(len(cutoffs))
    for g in ["low", "middle", "tie", "high"]:
        vals = cutoffs[f"n_{g}"].to_numpy(float)
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, color=pal[g], label=g, width=0.8)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(cutoffs["cluster"].astype(str), rotation=45, ha="right")
    ax.legend(frameon=False, loc="upper right")
    style.save_panel(fig, ax, stem)


def run_stratify(run: RunDir, cfg: RunConfig) -> dict:
    scores = read_scores(run)
    cell_groups, cutoffs, cov = stratify_scores(scores, cfg.split, cfg.quantiles)
    out = run.ensure(LAYOUT.STRATA)
    cutoffs.to_csv(out / "cutoffs.csv", index=False)
    cell_groups.to_csv(out / "cell_groups.csv.gz", index=False)
    cov.to_csv(out / "coverage.csv", index=False)
    _plot_groups(cutoffs, out / "groups_by_cluster")
    return {
        "split": cfg.split, "quantiles": list(cfg.quantiles) if cfg.split == "quantiles" else None,
        "n_cells": int(len(scores)), "n_clusters": int(len(cutoffs)),
        "n_donors": int(scores["donor"].nunique()),
        "ties_rule": "cells equal to a cutoff are 'tie' and excluded from high and low",
        "inference_unit": "none (descriptive grouping of cells; not donor-level inference)",
        "score_semantics": "ensemble score is a rank percentile, not a probability",
    }


def cli_stratify(args: argparse.Namespace) -> int:
    """``degas3 stratify RUN [--split median|quantiles --quantiles LO HI]``."""
    run = RunDir(args.run)
    split = getattr(args, "split", None) or ("quantiles" if getattr(args, "quantiles", None) else None)
    cfg = RunConfig.load(run.root, split=split,
                         quantiles=list(args.quantiles) if getattr(args, "quantiles", None) else None)
    if run.stage_complete("stratify"):
        log.info("stratify already complete; skipping (delete %s to rerun)", run.complete_path("stratify"))
        return 0
    payload = run_stratify(run, cfg)
    run.write_complete("stratify", payload)
    log.info("stratify complete: %d clusters", payload["n_clusters"])
    return 0
