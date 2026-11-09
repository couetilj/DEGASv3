"""``fit`` and ``score`` stages: final models and the rank-based ensemble score.

Final models use all bulk samples, all cells, the full-data gene ranking, every retained size
and ``final_seeds`` seeds. For each size the raw cell score is the mean over seeds; its
*percentile* is the average-tie rank among scored cells divided by their number (a position in
the score distribution, not a probability); the *ensemble score* is the equal mean of the
per-size percentiles and is never re-ranked. The raw scores of the model are kept next to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from degas3.config import RunConfig
from degas3.core.models import load_checkpoint, predict_scores, resolve_device
from degas3.core.preprocess import preprocess_counts
from degas3.io.bulk import load_bulk_npz
from degas3.plotting.style import (CASE, CONTROL, SCORE_LABEL, colorbar, save_panel, score_cmap,
                                   score_norm, square_panel)
from degas3.prepare import load_prepared
from degas3.runs import RunDir
from degas3.validate import read_selected_sizes
from degas3.train import CELL_CHUNK, bulk_targets, cell_classes, fit_one, preprocess_cells

log = logging.getLogger(__name__)


# -- ensemble helpers --------------------------------------------------------------
def percentile_rank(values: np.ndarray, groups: Optional[Sequence] = None) -> np.ndarray:
    """Average-tie rank / N, over all values or within each group. Values lie in (0, 1]; not a probability."""
    values = np.asarray(values, float)
    if groups is None:
        return stats.rankdata(values, method="average") / len(values)
    groups = np.asarray(groups).astype(str)
    out = np.empty(len(values))
    for g in np.unique(groups):
        m = groups == g
        out[m] = stats.rankdata(values[m], method="average") / m.sum()
    return out


def rank_ensemble(raw: np.ndarray, groups: Optional[Sequence] = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-size percentiles (n, n_sizes) from raw scores (n, n_sizes) and their equal mean.

    The mean of percentiles is returned as is and must not be re-ranked.
    """
    raw = np.asarray(raw, float)
    pct = np.column_stack([percentile_rank(raw[:, j], groups) for j in range(raw.shape[1])])
    return pct, pct.mean(axis=1)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN and do not count toward m."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    adj = p[order] * len(ok) / np.arange(1, len(ok) + 1)
    out[order] = np.minimum(1.0, np.minimum.accumulate(adj[::-1])[::-1])
    return out


# -- fit -----------------------------------------------------------------------------
def selected_sizes(rd: RunDir) -> list[int]:
    p = rd.path("cv", "selection.json")
    if not p.is_file() or not rd.stage_complete("select"):
        raise ValueError("size selection is missing; run `degas3 select RUN` first.")
    return read_selected_sizes(rd)


def fit_tasks(rd: RunDir, cfg: RunConfig) -> list[tuple[int, int]]:
    """(size, seed) pairs of the final fits: sizes ascending x seeds ascending; ``fit --task N`` is the N-th (0-based)."""
    return [(int(s), seed) for s in sorted(selected_sizes(rd)) for seed in range(cfg.final_seeds)]


def _final_genes(prep: dict, size: int) -> list[str]:
    genes = prep["final_ranking"]["gene"].astype(str).tolist()
    if len(genes) < size:
        raise ValueError(f"final ranking has {len(genes)} genes but size {size} was selected; rerun `prepare`.")
    return genes[:size]


def cli_fit(args: argparse.Namespace) -> int:
    rd = RunDir(args.run)
    if rd.stage_complete("fit"):
        log.info("fit already complete for %s; skipping.", rd.root)
        return 0
    cfg = RunConfig.load(rd.root)
    prep = load_prepared(rd)
    sizes = selected_sizes(rd)
    tasks = fit_tasks(rd, cfg)
    task = getattr(args, "task", None)
    if task is not None:
        if not 0 <= task < len(tasks):
            raise ValueError(f"--task {task} is outside 0..{len(tasks) - 1} ({len(tasks)} fit tasks: sizes x seeds).")
        wanted = {tasks[task]}
    else:
        wanted = set(tasks)
    device = resolve_device("auto", cfg.allow_cpu or bool(getattr(args, "allow_cpu", False)))
    bulk = load_bulk_npz(rd.bulk_npz)
    y = bulk_targets(bulk.pheno, bulk.outcome_type)
    shared_idx = {g: i for i, g in enumerate(prep["genes"])}
    bidx = {g: i for i, g in enumerate(bulk.genes)}
    rows = np.arange(prep["counts"].shape[0])
    classes = cell_classes(prep["meta"], rows, cfg)
    for size in sizes:
        todo = [s for s in range(cfg.final_seeds)
                if (size, s) in wanted and not rd.stage_complete("fit", f"size_{size:04d}", f"seed_{s}")]
        if not todo:
            continue
        genes = _final_genes(prep, size)
        px = preprocess_counts(bulk.counts[:, [bidx[g] for g in genes]], already_log=bulk.already_log).astype(np.float32)
        sx = preprocess_cells(prep["counts"], rows, np.array([shared_idx[g] for g in genes]))
        for seed in todo:
            d = rd.model_dir(size, seed)
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            losses, model = fit_one(cfg, bulk.outcome_type, px, y, sx, classes, genes, seed, -1, device,
                                    d / "checkpoint.pt", size)
            losses.to_csv(d / "losses.csv", index=False)
            rd.write_complete("fit", {
                "size": size, "seed": seed, "iterations": cfg.iterations, "n_bulk": len(y), "n_cells": len(rows),
                "model_type": model.opt["model_type"], "device": str(device), "seconds": round(time.time() - t0, 2),
            }, f"size_{size:04d}", f"seed_{seed}")
            log.info("final model size %d seed %d done.", size, seed)
            del model
        del sx
    if all(rd.stage_complete("fit", f"size_{z:04d}", f"seed_{s}") for z, s in tasks):
        rd.write_complete("fit", {"sizes": sizes, "final_seeds": cfg.final_seeds, "iterations": cfg.iterations})
    else:
        log.info("fit task %s done; stage marker is written when every (size, seed) pair is complete.", task)
    return 0


# -- score ------------------------------------------------------------------------------
def _load_models(rd: RunDir, size: int, seeds: int):
    out = []
    for s in range(seeds):
        p = rd.model_dir(size, s) / "checkpoint.pt"
        if not p.is_file() or not rd.stage_complete("fit", f"size_{size:04d}", f"seed_{s}"):
            raise ValueError(f"final model size {size} seed {s} is missing; run `degas3 fit RUN --all` first.")
        out.append(load_checkpoint(p, "cpu"))
    return out


def _mean_raw(models, X: np.ndarray) -> np.ndarray:
    return np.mean([predict_scores(m, X) for m, _ in models], axis=0)


def _score_cells(counts, shared_idx: dict, models) -> np.ndarray:
    genes = models[0][1]["genes"]
    sub = counts[:, [shared_idx[g] for g in genes]].tocsr()
    out = np.empty(sub.shape[0])
    for s in range(0, sub.shape[0], CELL_CHUNK):
        block = preprocess_counts(sub[s:s + CELL_CHUNK].toarray()).astype(np.float32)
        out[s:s + len(block)] = _mean_raw(models, block)
    return out


def _patient_scores(rd: RunDir, bulk, models_by_size: dict) -> pd.DataFrame:
    """In-sample (final models) and out-of-fold (CV, selected sizes) ensemble per bulk sample."""
    sizes = sorted(models_by_size)
    bidx = {g: i for i, g in enumerate(bulk.genes)}
    base = bulk.pheno.copy()
    base.insert(0, "sample_id", bulk.pheno.index.astype(str))
    if "patient_id" not in base.columns:
        base["patient_id"] = base["sample_id"]
    base = base.drop(columns=[c for c in ("dataset",) if c in base.columns])
    raw = np.column_stack([
        _mean_raw(models_by_size[s], preprocess_counts(
            bulk.counts[:, [bidx[g] for g in models_by_size[s][0][1]["genes"]]], already_log=bulk.already_log).astype(np.float32))
        for s in sizes])
    _, ens = rank_ensemble(raw)
    ins = base.copy()
    ins["raw_mean"], ins["ensemble_score"], ins["in_sample"] = raw.mean(axis=1), ens, True
    oof_path = rd.path("cv", "oof_predictions.csv")
    if not oof_path.is_file():
        log.warning("cv/oof_predictions.csv is missing (select was not run on this run directory); "
                    "patient_scores.csv has in-sample rows only.")
        return ins
    oof = pd.read_csv(oof_path, dtype={"sample_id": str})
    wide = oof[oof["size"].isin(sizes)].pivot(index="sample_id", columns="size", values="score").loc[base["sample_id"]]
    _, ens_o = rank_ensemble(wide[sizes].to_numpy())
    out = base.copy()
    out["raw_mean"], out["ensemble_score"], out["in_sample"] = wide[sizes].to_numpy().mean(axis=1), ens_o, False
    return pd.concat([ins, out], ignore_index=True)


def label_association(df: pd.DataFrame) -> pd.DataFrame:
    """Per cluster: donor-mean ensemble scores of label 1 vs label 0 donors (two-sided Mann-Whitney U, BH FDR).

    The unit is the donor, never the cell. p and FDR are NaN when either group has < 2 donors.
    """
    d = df.assign(label_num=pd.to_numeric(df["label"], errors="coerce"))
    d = d[d["label_num"].isin([0, 1])]
    rows = []
    for cl, part in d.groupby("cluster", sort=True):
        dm = part.groupby("donor").agg(score=("ensemble_score", "mean"), label=("label_num", "first"))
        a, b = dm.loc[dm["label"] == 1, "score"], dm.loc[dm["label"] == 0, "score"]
        p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) >= 2 and len(b) >= 2 else np.nan
        rows.append({"cluster": cl, "n_donors_label1": len(a), "n_donors_label0": len(b),
                     "mean_donor_score_label1": a.mean() if len(a) else np.nan,
                     "mean_donor_score_label0": b.mean() if len(b) else np.nan, "p_value": p})
    out = pd.DataFrame(rows, columns=["cluster", "n_donors_label1", "n_donors_label0",
                                      "mean_donor_score_label1", "mean_donor_score_label0", "p_value"])
    out["fdr"] = bh_adjust(out["p_value"].to_numpy())
    return out


def _violins(rd: RunDir, df: pd.DataFrame, has_label: bool) -> None:
    clusters = sorted(df["cluster"].unique())
    fig, ax = square_panel("Ensemble score by cluster", "", SCORE_LABEL)
    data = [df.loc[df["cluster"] == c, "ensemble_score"].to_numpy() for c in clusters]
    pos = np.arange(1, len(clusters) + 1)
    ok = [i for i, v in enumerate(data) if len(v) > 1]
    if ok:
        parts = ax.violinplot([data[i] for i in ok], positions=pos[ok], showextrema=False, widths=0.8)
        for b in parts["bodies"]:
            b.set_facecolor("#CCCCCC"); b.set_edgecolor("#888888"); b.set_alpha(0.8)
    rng = np.random.default_rng(0)
    dm = df.groupby(["cluster", "donor"]).agg(score=("ensemble_score", "mean"), label=("label", "first")).reset_index()
    lab = pd.to_numeric(dm["label"], errors="coerce") if has_label else pd.Series(np.nan, index=dm.index)
    for i, c in enumerate(clusters):
        m = (dm["cluster"] == c).to_numpy()
        col = np.where(lab[m] == 1, CASE, np.where(lab[m] == 0, CONTROL, "#222222"))
        ax.scatter(pos[i] + rng.uniform(-0.15, 0.15, m.sum()), dm.loc[m, "score"], c=list(col), s=22, zorder=3,
                   edgecolor="white", linewidth=0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    if has_label:
        ax.scatter([], [], c=CASE, s=22, label="donor mean, label 1")
        ax.scatter([], [], c=CONTROL, s=22, label="donor mean, label 0")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    save_panel(fig, ax, rd.path("scores", "violins_by_cluster"))


def _umap(rd: RunDir, xy: np.ndarray, score: np.ndarray) -> None:
    fig, ax = square_panel("Ensemble score on UMAP", "UMAP 1", "UMAP 2")
    order = np.argsort(score, kind="mergesort")
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=score[order], cmap=score_cmap(), norm=score_norm(),
                    s=3, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    colorbar(fig, ax, sc, SCORE_LABEL, ticks=[0, 0.25, 0.5, 0.75, 1])
    save_panel(fig, ax, rd.path("scores", "umap_score"))


def cli_score(args: argparse.Namespace) -> int:
    rd = RunDir(args.run)
    if rd.stage_complete("score"):
        log.info("score already complete for %s; skipping.", rd.root)
        return 0
    cfg = RunConfig.load(rd.root)
    prep = load_prepared(rd)
    sizes = selected_sizes(rd)
    meta = prep["meta"]
    shared_idx = {g: i for i, g in enumerate(prep["genes"])}
    models_by_size = {s: _load_models(rd, s, cfg.final_seeds) for s in sizes}
    raw = np.column_stack([_score_cells(prep["counts"], shared_idx, models_by_size[s]) for s in sizes])
    groups = None
    if cfg.rank_scope == "sample":
        groups = meta["sample" if cfg.sample_col else "donor"].to_numpy()
    pct, ens = rank_ensemble(raw, groups)
    df = meta[["cell_id", "donor", "cluster", "sample", "label"]].copy()
    for j, s in enumerate(sizes):
        df[f"raw_size_{s:04d}"] = raw[:, j]
    for j, s in enumerate(sizes):
        df[f"percentile_size_{s:04d}"] = pct[:, j]
    df["raw_mean"], df["ensemble_score"] = raw.mean(axis=1), ens
    if not np.isfinite(df.select_dtypes("number").to_numpy()).all():
        raise ValueError("non-finite cell scores; check training losses in models/*/seed_*/losses.csv.")
    out = rd.ensure("scores")
    df.to_csv(out / "cell_scores.csv.gz", index=False, compression="gzip")
    bulk = load_bulk_npz(rd.bulk_npz)
    _patient_scores(rd, bulk, models_by_size).to_csv(out / "patient_scores.csv", index=False)

    plot = df.copy()
    if not cfg.cluster_col:
        plot["cluster"] = "all"
    has_label = bool(cfg.label_col) and bool((plot["label"] != "").any())
    _violins(rd, plot, has_label)
    assoc = None
    if has_label:
        assoc = label_association(plot)
        assoc.to_csv(out / "label_association.csv", index=False)
    umap = False
    import anndata as ad

    adata = ad.read_h5ad(rd.cells_h5ad, backed="r")
    try:
        if "X_umap" in adata.obsm:
            _umap(rd, np.asarray(adata.obsm["X_umap"]), ens)
            umap = True
    finally:
        adata.file.close()
    corr = np.nan
    if len(sizes) > 1:
        c = stats.spearmanr(raw).statistic
        corr = float(np.nanmean(np.atleast_2d(c)[np.triu_indices(len(sizes), 1)])) if np.ndim(c) else float(c)
    summary = {
        "sizes": sizes, "final_seeds": cfg.final_seeds, "n_cells": int(len(df)), "rank_scope": cfg.rank_scope,
        "ensemble_mean": float(ens.mean()), "ensemble_min": float(ens.min()), "ensemble_max": float(ens.max()),
        "mean_pairwise_spearman_between_sizes": None if np.isnan(corr) else corr,
        "umap_written": umap, "label_association_written": assoc is not None,
        "note": "Ensemble score is a mean of within-run percentile ranks in (0, 1], not a probability.",
    }
    (out / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rd.write_complete("score", summary)
    return 0
