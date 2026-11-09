"""Gradient SHAP attributions of the DEGAS score, per cluster (optional stage).

What is computed: for each cluster, ``shap.GradientExplainer`` (expected gradients) is run on
every retained final model with a reference of low-score same-cluster cells (below the
cluster's ``explain_reference_quantile`` of the ensemble score, donor-balanced). Attributions
are averaged over seeds and sizes with equal weight, so the explained quantity is the MEAN RAW
SCORE across the retained models, not the rank-based ensemble percentile. A gene that is absent
from a model's input contributes zero for that model.

What it is not: a causal effect of changing a gene, an independent discovery (the model
inputs are the attributed genes), or a statement about expression combinations that never
occur (independent substitution from the reference). Outputs are labelled "model attribution
relative to low-score same-cluster cells; not causal, not independent discovery".
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degas3.config import RunConfig
from degas3.contracts import LAYOUT
from degas3.runs import RunDir
from degas3.validate import read_selected_sizes

log = logging.getLogger(__name__)

LABEL = "model attribution relative to low-score same-cluster cells; not causal, not independent discovery"
NSAMPLES = 200
TARGET_BATCH = 64
NO_DONOR_MAX_TARGETS = 300


def safe_name(name: object) -> str:
    """Directory-safe form of a cluster label."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def _require_shap():
    try:
        import shap  # noqa: WPS433
    except ImportError as exc:
        raise ValueError(
            'the explain stage needs the optional dependency "shap"; install it with '
            '`pip install "degas3[explain]"`.'
        ) from exc
    return shap


# -- sampling ------------------------------------------------------------------------------
def balanced_sample(groups: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of up to ``n`` items, drawn round-robin so groups are balanced.

    Every group with at least one item contributes before any group contributes twice (when
    ``n`` is at least the number of groups). Returns sorted positions into ``groups``.
    """
    groups = np.asarray(groups)
    pools = {}
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        pools[g] = list(rng.permutation(idx))
    order = list(pools)
    chosen: list[int] = []
    while len(chosen) < n and any(pools.values()):
        for g in rng.permutation(len(order)):
            pool = pools[order[g]]
            if pool and len(chosen) < n:
                chosen.append(pool.pop())
    return np.sort(np.array(chosen, dtype=int))


def select_cells(
    scores: np.ndarray, donors: Optional[np.ndarray], *, ref_quantile: float, ref_cells: int, seed: int,
) -> dict:
    """Pick reference and target positions (into ``scores``) for one cluster.

    Reference: cells with score below the ``ref_quantile`` quantile, donor-balanced up to
    ``ref_cells``. Targets: one random cell per donor and score quartile Q2, Q3, Q4 among cells
    at or above the 25th percentile (up to 300 random cells when there is no donor column).
    Quartiles are the cluster's own score quartiles. Returns arrays plus the quartile labels.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    cut = np.quantile(scores, ref_quantile)
    q25, q50, q75 = np.quantile(scores, [0.25, 0.5, 0.75])
    ref_pool = np.flatnonzero(scores < cut)
    keys = donors if donors is not None else np.zeros(len(scores), dtype=int)
    ref = ref_pool[balanced_sample(keys[ref_pool], ref_cells, rng)] if len(ref_pool) else ref_pool
    quartile = np.where(scores >= q75, "Q4", np.where(scores >= q50, "Q3", np.where(scores >= q25, "Q2", "Q1")))
    if donors is None:
        pool = np.flatnonzero(quartile != "Q1")
        take = rng.permutation(pool)[:NO_DONOR_MAX_TARGETS]
        tgt = np.sort(take)
    else:
        picks = []
        for d in pd.unique(donors):
            for q in ("Q2", "Q3", "Q4"):
                cand = np.flatnonzero((donors == d) & (quartile == q))
                if len(cand):
                    picks.append(int(rng.choice(cand)))
        tgt = np.sort(np.array(picks, dtype=int))
    return {"reference": np.sort(ref), "targets": tgt, "quartile": quartile, "cutoff": float(cut)}


# -- model wrapper -------------------------------------------------------------------------
def _score_module(model):
    """torch module mapping preprocessed inputs to the outcome-head output (n, 1).

    Binary: softmax probability of class 1. Survival: sigmoid hazard. Dropout is off (eval).
    """
    import torch
    from torch import nn

    class Score(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.extractor = m.feature_extractor_layer
            self.head = m.low_reso_pred_layer
            self.cox = m.cox
            self.eval()

        def forward(self, x):
            _, emb = self.extractor(x)
            out = self.head(emb)
            return out if self.cox else torch.softmax(out, dim=1)[:, 1:2]

    return Score(model).eval()


def attribute_model(model, background: np.ndarray, targets: np.ndarray, seed: int) -> dict:
    """Gradient-SHAP values (n_targets x n_genes), predictions and base value for one model.

    ``base`` is the mean model output over the reference. Residual is
    ``prediction - base - sum(attributions)``; expected-gradients is a Monte-Carlo estimate,
    so a nonzero residual measures sampling error, not a model property.
    """
    import torch

    shap = _require_shap()
    module = _score_module(model)
    dev = model.device
    bg = torch.as_tensor(background, dtype=torch.float32, device=dev)
    with torch.no_grad():
        base = float(module(bg).mean().item())
    explainer = shap.GradientExplainer(module, bg)
    vals, preds = [], []
    for start in range(0, len(targets), TARGET_BATCH):
        xb = torch.as_tensor(targets[start:start + TARGET_BATCH], dtype=torch.float32, device=dev)
        with torch.no_grad():
            preds.append(module(xb).flatten().cpu().numpy())
        with torch.enable_grad():
            v = explainer.shap_values(xb, nsamples=NSAMPLES, rseed=seed)
        v = v[0] if isinstance(v, list) else v
        v = np.asarray(v)
        if v.ndim == 3 and v.shape[-1] == 1:
            v = v[..., 0]
        if v.shape != tuple(xb.shape) or not np.isfinite(v).all():
            raise ValueError("invalid SHAP output (wrong shape or non-finite); check the checkpoint and inputs.")
        vals.append(v)
    return {"values": np.concatenate(vals), "prediction": np.concatenate(preds), "base": base}


# -- summaries ------------------------------------------------------------------------------
def summarize_attributions(
    values: np.ndarray, genes: list[str], donors: Optional[np.ndarray], *, bootstraps: int, seed: int,
) -> pd.DataFrame:
    """Pooled and equal-donor attribution summaries per gene.

    ``pooled_*``: mean over targets, each target weighted equally. ``donor_*``: mean within each
    donor, then across donors (each donor weighted equally), with a 95% percentile interval
    from ``bootstraps`` resamples of donors. ``donor_*`` are NaN without a donor column.
    Signed means keep the direction; absolute means do not.
    """
    out = pd.DataFrame({
        "gene": genes,
        "pooled_signed": values.mean(axis=0),
        "pooled_abs": np.abs(values).mean(axis=0),
    })
    cols = ["donor_signed", "donor_signed_lo", "donor_signed_hi", "donor_abs", "donor_abs_lo", "donor_abs_hi"]
    if donors is None:
        for c in cols:
            out[c] = np.nan
        out["n_donors"] = 0
        return out[["gene", "pooled_signed", "pooled_abs", *cols, "n_donors"]]
    ud = pd.unique(donors)
    signed = np.stack([values[donors == d].mean(axis=0) for d in ud])
    absol = np.stack([np.abs(values[donors == d]).mean(axis=0) for d in ud])
    out["donor_signed"], out["donor_abs"] = signed.mean(axis=0), absol.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ud), size=(bootstraps, len(ud)))
    for name, mat in (("donor_signed", signed), ("donor_abs", absol)):
        boot = np.empty((bootstraps, mat.shape[1]), dtype=np.float32)
        for s in range(0, bootstraps, 100):
            boot[s:s + 100] = mat[idx[s:s + 100]].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        out[name + "_lo"], out[name + "_hi"] = lo, hi
    out["n_donors"] = len(ud)
    return out[["gene", "pooled_signed", "pooled_abs", *cols, "n_donors"]]


# -- orchestration --------------------------------------------------------------------------
def _find_checkpoints(rd: RunDir) -> list[Path]:
    """Checkpoints of the retained sizes (cv/selection.json) or, without it, of every fitted size."""
    sel_path = rd.path(LAYOUT.CV, "selection.json")
    sizes = (read_selected_sizes(rd, required=False) or None) if sel_path.is_file() else None
    found = []
    for p in sorted(rd.path(LAYOUT.MODELS).glob("size_*/seed_*/checkpoint.pt")):
        size = int(p.parent.parent.name.split("_")[1])
        if sizes is None or size in [int(s) for s in sizes]:
            found.append(p)
    if not found:
        raise ValueError(f"no final-model checkpoints under {rd.path(LAYOUT.MODELS)}; run `degas3 fit RUN --all` first.")
    return found


def explain_cluster(
    cluster: str, scores: pd.DataFrame, counts, var_index: pd.Index, models: list, cfg: RunConfig,
    out_dir: Path, cell_pos: np.ndarray, has_donor: bool,
) -> dict:
    """Run SHAP for one cluster and write its outputs; returns a summary row."""
    from degas3.core.preprocess import preprocess_counts

    row = {"cluster": cluster, "n_cells": int(len(scores))}
    donors = scores["donor"].astype(str).to_numpy() if has_donor else None
    sc = scores["ensemble_score"].to_numpy(dtype=float)
    sel = select_cells(sc, donors, ref_quantile=cfg.explain_reference_quantile,
                       ref_cells=cfg.explain_reference_cells, seed=cfg.explain_seed)
    ref, tgt = sel["reference"], sel["targets"]
    row.update(n_reference=int(len(ref)), n_targets=int(len(tgt)))
    if len(ref) == 0 or len(tgt) == 0:
        return {**row, "status": "skipped", "reason": "no reference or no target cells (score ties or too few cells)"}
    rows_idx = np.concatenate([ref, tgt])
    sub = counts[cell_pos[rows_idx]]
    sub = sub.toarray() if hasattr(sub, "toarray") else np.asarray(sub)
    union = list(dict.fromkeys(g for _, ck in models for g in ck["genes"]))
    pos_union = {g: i for i, g in enumerate(union)}
    total = np.zeros((len(tgt), len(union)))
    preds, bases = np.zeros(len(tgt)), 0.0
    for m_i, (model, ck) in enumerate(models):
        gi = var_index.get_indexer(ck["genes"])
        X = preprocess_counts(sub[:, gi])
        res = attribute_model(model, X[:len(ref)], X[len(ref):], seed=cfg.explain_seed + m_i)
        total[:, [pos_union[g] for g in ck["genes"]]] += res["values"]
        preds += res["prediction"]
        bases += res["base"]
    n_models = len(models)
    values, preds, base = total / n_models, preds / n_models, bases / n_models
    resid = preds - base - values.sum(axis=1)
    donor_t = donors[tgt] if donors is not None else None
    attr = summarize_attributions(values, union, donor_t, bootstraps=cfg.explain_bootstraps, seed=cfg.explain_seed)
    attr = attr.sort_values("pooled_abs", ascending=False, ignore_index=True)
    cell_ids = scores["cell_id"].to_numpy().astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "shap_values.npz", values=values.astype(np.float32), genes=np.array(union),
        target_cell_ids=cell_ids[tgt], predictions=preds, base=np.float64(base), residuals=resid,
    )
    attr.to_csv(out_dir / "gene_attributions.csv", index=False)
    pd.DataFrame({
        "cell_id": cell_ids[tgt], "donor": donor_t if donor_t is not None else "", "quartile": sel["quartile"][tgt],
        "ensemble_score": sc[tgt], "mean_raw_prediction": preds, "base": base, "residual": resid,
    }).to_csv(out_dir / "targets.csv", index=False)
    pd.DataFrame({
        "cell_id": cell_ids[ref], "donor": donors[ref] if donors is not None else "", "ensemble_score": sc[ref],
    }).to_csv(out_dir / "reference.csv", index=False)
    from degas3.plotting.report_figures import top_genes_figure

    top_genes_figure(attr, out_dir / "top_genes_abs", signed=False, cluster=cluster)
    top_genes_figure(attr, out_dir / "top_genes_signed", signed=True, cluster=cluster)
    pr = float(np.ptp(preds)) if len(preds) > 1 else float("nan")
    row.update(
        status="done", reason="", n_donors=int(attr["n_donors"].iloc[0]), n_models=n_models, n_genes=len(union),
        max_abs_residual=float(np.abs(resid).max()), median_abs_residual=float(np.median(np.abs(resid))),
        prediction_range=pr, base=float(base), reference_cutoff=sel["cutoff"], label=LABEL,
    )
    (out_dir / "complete.json").write_text(json.dumps(row, indent=2, default=str) + "\n")
    return row


def run_explain(run: str | Path) -> dict:
    """Explain every cluster of ``<run>``; idempotent (completed clusters and the stage are skipped)."""
    rd = RunDir(run)
    _require_shap()
    if rd.stage_complete("explain"):
        log.info("explain stage already complete; delete %s to redo it.", rd.complete_path("explain"))
        return rd.read_complete("explain") or {}
    scores_path = rd.path(LAYOUT.SCORES, "cell_scores.csv.gz")
    if not scores_path.is_file() or not rd.stage_complete("score"):
        raise ValueError("scores not found; run `degas3 score RUN` before `degas3 explain RUN`.")
    import anndata as ad

    from degas3.core.models import load_checkpoint, resolve_device
    from degas3.io.sc import get_counts

    cfg = RunConfig.load(rd.root)
    scores = pd.read_csv(scores_path)
    scores["cell_id"] = scores["cell_id"].astype(str)
    has_donor = bool(cfg.donor_col) and "donor" in scores and scores["donor"].notna().any()
    if has_donor:
        scores = scores[scores["donor"].notna()]
    adata = ad.read_h5ad(rd.cells_h5ad)
    cell_pos_all = pd.Index(adata.obs_names.astype(str)).get_indexer(scores["cell_id"])
    if (cell_pos_all < 0).any():
        raise ValueError("cell ids in scores/cell_scores.csv.gz are missing from inputs/cells.h5ad; rerun `degas3 score`.")
    counts = get_counts(adata)
    if hasattr(counts, "tocsr"):
        counts = counts.tocsr()
    device = resolve_device("auto", allow_cpu=True)
    models = [load_checkpoint(p, device) for p in _find_checkpoints(rd)]
    log.info("explaining with %d retained models on %s", len(models), device)
    rows = []
    out_root = rd.ensure(LAYOUT.EXPLAIN)
    scores = scores.reset_index(drop=True)
    for cluster, grp in scores.groupby("cluster", sort=True):
        name = safe_name(cluster)
        cdir = out_root / name
        if (cdir / "complete.json").is_file():
            rows.append(json.loads((cdir / "complete.json").read_text()))
            continue
        log.info("cluster %s: %d cells", cluster, len(grp))
        row = explain_cluster(str(cluster), grp, counts, adata.var_names, models, cfg, cdir,
                              cell_pos_all[grp.index.to_numpy()], has_donor)
        row["dir"] = name
        if row["status"] == "skipped":
            log.warning("cluster %s skipped: %s", cluster, row["reason"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "summary.csv", index=False)
    payload = {"clusters": [r["cluster"] for r in rows if r.get("status") == "done"],
               "skipped": {r["cluster"]: r.get("reason") for r in rows if r.get("status") != "done"},
               "n_models": len(models), "explained_quantity": "mean raw score across retained models (not the percentile)",
               "label": LABEL, "nsamples": NSAMPLES, "seed": cfg.explain_seed, "bootstraps": cfg.explain_bootstraps}
    rd.write_complete("explain", payload)
    return payload


def cli_explain(args: argparse.Namespace) -> int:
    """CLI entry point for ``degas3 explain RUN``."""
    run_explain(args.run)
    return 0
