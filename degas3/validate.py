"""``select`` stage: cross-validation metrics and one-standard-error gene-list size selection.

Held-out predictions are averaged over CV seeds, scored per fold, and averaged over folds with
equal weight. The one-SE rule uses patient bootstraps within each fold (stratified by outcome or
event, identical resamples for every size, predictions held fixed): the standard error of the
best size's mean metric is the SD of that mean over resamples, and every size whose mean is at
least ``best - SE`` is retained.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

from degas3.config import RunConfig
from degas3.plotting.style import CASE, CONTROL, save_panel, square_panel
from degas3.prepare import load_prepared
from degas3.runs import RunDir

log = logging.getLogger(__name__)

OPTIMISM_NOTE = (
    "Sizes were chosen on the same held-out predictions that estimate performance, so the "
    "metric at the selected sizes is optimistic; it is not an independent test."
)


# -- metrics ---------------------------------------------------------------------
def _pair_weights(s: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Concordance weight of each ordered pair in ``valid``: 1 if s_i > s_j, 0.5 if tied, else 0."""
    s = np.asarray(s, float)
    return valid * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))


def auroc_pairs(y: np.ndarray) -> np.ndarray:
    """Indicator matrix of (case i, control j) pairs."""
    y = np.asarray(y).astype(int)
    return ((y[:, None] == 1) & (y[None, :] == 0)).astype(float)


def cindex_pairs(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Indicator matrix of comparable pairs: i has an event and an earlier time than j."""
    t, e = np.asarray(time, float), np.asarray(event).astype(int)
    return ((e[:, None] == 1) & (t[:, None] < t[None, :])).astype(float)


def auroc(y, score) -> float:
    """Area under the ROC curve: P(case score > control score) with ties counted 0.5. NaN if one class is missing."""
    d = auroc_pairs(y)
    return float(_pair_weights(score, d).sum() / d.sum()) if d.sum() else float("nan")


def concordance_index(time, event, score) -> float:
    """Harrell's C: among pairs where the earlier time is an event, the share with the higher score
    (hazard) on the earlier-failing patient; tied scores count 0.5; tied times are not comparable.
    NaN when no pair is comparable."""
    d = cindex_pairs(time, event)
    return float(_pair_weights(score, d).sum() / d.sum()) if d.sum() else float("nan")


def average_precision(y, score) -> float:
    """Step-wise area under the precision-recall curve (sum of recall increments x precision at each distinct threshold)."""
    y, s = np.asarray(y).astype(int), np.asarray(score, float)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last]
    prec = tp / (last + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def log_loss(y, score, eps: float = 1e-15) -> float:
    """Mean negative log-likelihood of the 0/1 outcome under the score read as P(case); scores are clipped to [eps, 1-eps]."""
    p = np.clip(np.asarray(score, float), eps, 1 - eps)
    y = np.asarray(y).astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, score) -> float:
    """Mean squared difference between the score (read as P(case)) and the 0/1 outcome."""
    return float(np.mean((np.asarray(score, float) - np.asarray(y, float)) ** 2))


def fold_metrics(kind: str, frame: pd.DataFrame) -> dict:
    """Metrics of one fold x size prediction frame (columns outcome or time,event; score)."""
    if kind == "binary":
        y, s = frame["outcome"].to_numpy(), frame["score"].to_numpy()
        return {"auroc": auroc(y, s), "average_precision": average_precision(y, s),
                "log_loss": log_loss(y, s), "brier": brier(y, s)}
    return {"c_index": concordance_index(frame["time"], frame["event"], frame["score"])}


# -- one-SE rule -------------------------------------------------------------------
def one_se_selection(metric_by_fold: list[np.ndarray], boot: np.ndarray) -> dict:
    """One-SE selection from fold metrics and bootstrap means.

    ``metric_by_fold``: list over folds of arrays (n_sizes,). ``boot``: (B, n_sizes) bootstrap
    means across folds. Returns mean, se, best index, cutoff and the boolean ``selected`` array.
    A tie for best goes to the smaller size (sizes are assumed ascending).
    """
    mean = np.nanmean(np.vstack(metric_by_fold), axis=0)
    se = np.nanstd(boot, axis=0, ddof=1) if len(boot) > 1 else np.full_like(mean, np.nan)
    best = int(np.nanargmax(mean))
    cutoff = mean[best] - (0.0 if np.isnan(se[best]) else se[best])
    return {"mean": mean, "se": se, "best": best, "cutoff": float(cutoff), "selected": mean >= cutoff}


def bootstrap_fold_means(kind: str, folds: list[pd.DataFrame], sizes: list[int], n_boot: int, seed: int) -> np.ndarray:
    """(B, n_sizes) bootstrap distribution of the equal-weight mean-over-folds primary metric.

    Within each fold patients are resampled with replacement inside each stratum (outcome class or
    event status); the same resample is applied to every size. Each statistic is the ratio of
    weighted concordant pairs to weighted comparable pairs, i.e. the AUROC / C-index of the
    resampled fold. Folds with no comparable pair in a resample are ignored for that resample.
    """
    rng = np.random.default_rng(seed)
    per_fold = []
    for fr in folds:
        first = fr[fr["size"] == sizes[0]].sort_values("sample_id")
        n = len(first)
        strat = first["outcome" if kind == "binary" else "event"].to_numpy().astype(int)
        d = auroc_pairs(strat) if kind == "binary" else cindex_pairs(first["time"], first["event"])
        counts = np.zeros((n_boot, n))
        for s in np.unique(strat):
            idx = np.flatnonzero(strat == s)
            counts[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
        den = ((counts @ d) * counts).sum(1)
        cols = []
        for sz in sizes:
            sc = fr[fr["size"] == sz].sort_values("sample_id")["score"].to_numpy()
            w = _pair_weights(sc, d)
            with np.errstate(invalid="ignore", divide="ignore"):
                cols.append(np.where(den > 0, ((counts @ w) * counts).sum(1) / den, np.nan))
        per_fold.append(np.column_stack(cols))
    return np.nanmean(np.stack(per_fold), axis=0)


# -- stage -----------------------------------------------------------------------
def collect_predictions(rd: RunDir, cfg: RunConfig, prep: dict) -> pd.DataFrame:
    """Long table (sample_id, fold_id, size, targets, score) with scores averaged over CV seeds."""
    frames = []
    for t in prep["tasks"]:
        seeds = []
        for s in range(cfg.cv_seeds):
            p = rd.cv_task_dir(t["task"], s) / "heldout_predictions.csv"
            if not p.is_file() or not rd.stage_complete("train", f"task_{t['task']:03d}", f"seed_{s}"):
                raise ValueError(f"task {t['task']} seed {s} is incomplete; run `degas3 train RUN --all` first.")
            seeds.append(pd.read_csv(p, dtype={"sample_id": str}))
        base = seeds[0].copy()
        for other in seeds[1:]:
            if not base["sample_id"].equals(other["sample_id"]):
                raise ValueError(f"task {t['task']}: seeds disagree on held-out samples; rerun training.")
        base["score"] = np.mean([f["score"].to_numpy() for f in seeds], axis=0)
        base["size"] = t["size"]
        base["task"] = t["task"]
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


def parse_selected(sel: dict | None) -> list[int]:
    """Sizes from a parsed selection.json: ``selected`` (ints or ``{"size": n}`` objects), else ``selected_sizes``."""
    if not sel:
        return []
    items = sel.get("selected")
    if isinstance(items, dict):
        items = list(items.get("sizes", items.keys()))
    if not items:
        items = sel.get("selected_sizes")
    if not isinstance(items, (list, tuple)):
        return []
    return sorted({int(i["size"]) if isinstance(i, dict) else int(i) for i in items})


def read_selected_sizes(run_dir, required: bool = True) -> list[int]:
    """Retained gene-list sizes from ``cv/selection.json`` (run directory path or ``RunDir``).

    Raises ``ValueError`` when the file is missing or lists no sizes and ``required`` is true;
    otherwise returns ``[]``.
    """
    root = run_dir.root if isinstance(run_dir, RunDir) else run_dir
    p = Path(root) / "cv" / "selection.json"
    sizes: list[int] = []
    if p.is_file():
        sizes = parse_selected(json.loads(p.read_text()))
    if not sizes and required:
        why = "not found" if not p.is_file() else "has no 'selected' sizes"
        raise ValueError(f"{p} {why}; run `degas3 select RUN` first.")
    return sizes


def _selection_figure(rd: RunDir, table: pd.DataFrame, sel: dict, metric: str) -> None:
    fig, ax = square_panel("Gene-list size selection", "Genes per list (log scale)", f"Mean {metric} across folds")
    x = table["size"].to_numpy()
    ax.errorbar(x, table["mean"], yerr=table["se"].fillna(0), color="#555555", lw=1.0, capsize=2, zorder=2)
    ax.plot(x, table["mean"], "-", color="#555555", zorder=2)
    chosen = table["selected"].to_numpy()
    ax.scatter(x[~chosen], table["mean"][~chosen], s=36, color=CONTROL, label="not retained", zorder=3)
    ax.scatter(x[chosen], table["mean"][chosen], s=48, color=CASE, label="retained", zorder=4)
    ax.axhline(sel["cutoff"], color="#999999", ls="--", lw=1.0, label="best - SE(best)")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(frameon=False, loc="lower right")
    save_panel(fig, ax, rd.path("cv", "selection"))


def cli_select(args: argparse.Namespace) -> int:
    rd = RunDir(args.run)
    if rd.stage_complete("select"):
        log.info("select already complete for %s; skipping.", rd.root)
        return 0
    cfg = RunConfig.load(rd.root)
    prep = load_prepared(rd)
    kind = prep["folds"]["outcome_type"]
    metric = "auroc" if kind == "binary" else "c_index"
    metric_label = "AUROC" if kind == "binary" else "C-index"
    preds = collect_predictions(rd, cfg, prep)
    sizes = sorted(preds["size"].unique().tolist())
    fold_ids = sorted(preds["fold_id"].unique().tolist())
    heldout = {f["fold_id"]: f["heldout_dataset"] for f in prep["folds"]["folds"]}
    rows, by_fold, fold_frames = [], [], []
    for f in fold_ids:
        fr = preds[preds["fold_id"] == f]
        fold_frames.append(fr)
        vals = []
        for sz in sizes:
            part = fr[fr["size"] == sz]
            m = fold_metrics(kind, part)
            rows.append({"fold_id": f, "heldout_dataset": heldout[f], "size": sz, "n_test": len(part), **m})
            vals.append(m[metric])
        by_fold.append(np.array(vals))
    metrics = pd.DataFrame(rows)
    cv = rd.ensure("cv")
    metrics.to_csv(cv / "metrics.csv", index=False)
    preds.drop(columns=["task"]).to_csv(cv / "oof_predictions.csv", index=False)
    boot = bootstrap_fold_means(kind, fold_frames, sizes, cfg.bootstraps, cfg.selection_seed)
    sel = one_se_selection(by_fold, boot)
    table = pd.DataFrame({"size": sizes, "mean": sel["mean"], "se": sel["se"], "selected": sel["selected"]})
    selected = [int(s) for s in table.loc[table["selected"], "size"]]
    out = {
        "metric": metric_label, "metric_column": metric, "outcome_type": kind, "sizes": sizes,
        "selected": selected, "selected_sizes": selected, "best_size": int(sizes[sel["best"]]), "best_mean": float(sel["mean"][sel["best"]]),
        "se_best": None if np.isnan(sel["se"][sel["best"]]) else float(sel["se"][sel["best"]]),
        "cutoff": sel["cutoff"], "bootstraps": cfg.bootstraps, "selection_seed": cfg.selection_seed,
        "n_folds": len(fold_ids), "cv": cfg.cv,
        "table": [{"size": int(r.size), "mean": float(r.mean), "se": None if np.isnan(r.se) else float(r.se),
                   "selected": bool(r.selected)} for r in table.itertuples()],
        "note": OPTIMISM_NOTE,
    }
    (cv / "selection.json").write_text(json.dumps(out, indent=2) + "\n")
    _selection_figure(rd, table, sel, metric_label)
    (rd.ensure("models") / "fit_tasks.json").write_text(json.dumps(
        {"tasks": [{"task": i, "size": z, "seed": sd} for i, (z, sd) in
                   enumerate((z, sd) for z in selected for sd in range(cfg.final_seeds))],
         "n_tasks": len(selected) * cfg.final_seeds}, indent=2) + "\n")
    rd.write_complete("select", {"selected_sizes": selected, "metric": metric_label, "n_folds": len(fold_ids)})
    log.info("selected sizes %s (best %d, %s %.3f). %s", selected, out["best_size"], metric_label, out["best_mean"], OPTIMISM_NOTE)
    return 0
