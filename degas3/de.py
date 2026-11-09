"""High-vs-low differential expression within each cluster.

What the statistics are:

* ``pseudobulk_deseq2`` (and the edgeR engine): raw counts are summed per donor x group and
  tested with the design ``~ donor + group`` (paired within donor). Donors are the replicates, so
  the p-values are donor-level inference about the *within-cluster, within-donor* high-vs-low
  difference. They do not say the groups were defined independently of expression: groups come
  from a score that was trained on a gene set, so genes the model used show model attribution,
  not independent discovery.
* ``wilcoxon`` ("Cell-level Wilcoxon"): a rank-sum test treating cells as independent. Cells from
  one donor are correlated, so p-values are pseudo-replicated and descriptive/exploratory only;
  they are not donor-level inference. The AUC is the probability that a random high cell has a
  larger log-normalized value than a random low cell (ties count one half); it is an effect
  size, not a classifier accuracy.
* The clinical contrast compares donors with different clinical labels (``~ dataset + label``
  or ``~ label``) on pseudobulk of all cells of the cluster. It uses the donor's clinical label,
  not the DEGAS score.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite  # noqa: F401  (mmread re-exported for tests)

from degas3.config import RunConfig
from degas3.contracts import LAYOUT
from degas3.io.sc import get_counts
from degas3.plotting import style
from degas3.runs import RunDir
from degas3.stratify import cluster_dirname, read_scores
from degas3.validate import read_selected_sizes

log = logging.getLogger(__name__)

WILCOXON_NAME = "Cell-level Wilcoxon"
MIN_GENE_COUNTS = 10
ALPHA = 0.05
EDGER_SCRIPT = Path(__file__).resolve().parent.parent / "r" / "edger_pseudobulk.R"
SUMMARY_COLUMNS = ["cluster", "method", "n_donors", "n_high_cells", "n_low_cells", "n_genes_tested",
                   "n_padj_lt_0.05_up", "n_padj_lt_0.05_down", "skipped_reason", "directory"]
FOOTNOTE = "* model-input gene: association is model attribution, not independent discovery."


# -- small utilities -------------------------------------------------------------------
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values; NaN entries stay NaN and are not counted."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    n = int(ok.sum())
    if n == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    adj = pv[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def model_input_genes(run: RunDir) -> tuple[set[str], str]:
    """Union of genes over the selected sizes' ``seed_0`` checkpoints, and a note.

    Returns an empty set (with the reason) when selection or checkpoints are unavailable.
    """
    sel = run.path(LAYOUT.CV, "selection.json")
    if not sel.is_file():
        return set(), "cv/selection.json not found; no genes flagged as model_input."
    sizes = read_selected_sizes(run, required=False)
    genes: set[str] = set()
    found = []
    for size in sizes:
        ck = run.model_dir(int(size), 0) / "checkpoint.pt"
        if not ck.is_file():
            continue
        try:
            import torch

            data = torch.load(ck, map_location="cpu", weights_only=False)
            genes.update(str(g) for g in data["genes"])
            found.append(int(size))
        except Exception as exc:  # noqa: BLE001 - provenance helper must not abort DE
            log.warning("could not read %s: %s", ck, exc)
    if not found:
        return set(), "no checkpoints found for the selected sizes; no genes flagged as model_input."
    return genes, f"model_input = union of genes over sizes {found} (seed_0 checkpoints)."


def pseudobulk(X: sparse.csr_matrix, keys: np.ndarray) -> tuple[list, sparse.csr_matrix, np.ndarray]:
    """Sum rows of ``X`` (cells x genes) by ``keys``. Returns (unique keys, samples x genes, n_cells)."""
    codes, uniq = pd.factorize(pd.Series(keys), sort=True)
    ind = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    mat = (ind @ X).tocsr()
    return list(uniq), mat, np.bincount(codes, minlength=len(uniq))


def _write_pseudobulk(dest: Path, genes: list[str], mat: sparse.csr_matrix, samples: pd.DataFrame) -> None:
    """Write genes x samples ``pseudobulk_counts.mtx``, ``genes.txt`` and ``samples.csv``."""
    dest.mkdir(parents=True, exist_ok=True)
    mmwrite(str(dest / "pseudobulk_counts.mtx"), sparse.csr_matrix(mat.T), field="integer")
    (dest / "genes.txt").write_text("\n".join(genes) + "\n")
    samples.to_csv(dest / "samples.csv", index=False)


def _n_cpus() -> int:
    return max(1, min(4, os.cpu_count() or 1))


# -- engines -----------------------------------------------------------------------------
def _pydeseq2(counts: np.ndarray, genes: list[str], meta: pd.DataFrame, design: str, contrast: list[str]) -> pd.DataFrame:
    """Wald test with PyDESeq2 on samples x genes counts (genes < 10 total counts removed)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    keep = counts.sum(axis=0) >= MIN_GENE_COUNTS
    if keep.sum() < 2:
        raise ValueError("fewer than 2 genes have >= 10 total pseudobulk counts; nothing to test.")
    cdf = pd.DataFrame(counts[:, keep].astype(int), index=meta.index, columns=np.asarray(genes)[keep])
    meta = meta.copy()
    for col in meta.columns:
        if not pd.api.types.is_numeric_dtype(meta[col]):
            meta[col] = meta[col].astype(str).astype("category")
    dds = DeseqDataSet(counts=cdf, metadata=meta, design=design, refit_cooks=True,
                       n_cpus=_n_cpus(), quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True, n_cpus=_n_cpus())
    st.summary()
    res = st.results_df.copy()
    res.index.name = "gene"
    res = res.reset_index()
    return res[["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]


_EDGER_OK: Optional[bool] = None


def edger_available() -> bool:
    """True when ``Rscript`` is on PATH and ``library(edgeR)`` loads (checked once)."""
    global _EDGER_OK
    if _EDGER_OK is None:
        if shutil.which("Rscript") is None or not EDGER_SCRIPT.is_file():
            _EDGER_OK = False
        else:
            r = subprocess.run(["Rscript", "-e", "suppressMessages(library(edgeR))"],
                               capture_output=True, text=True)
            _EDGER_OK = r.returncode == 0
    return bool(_EDGER_OK)


def _edger(dest: Path, design: str) -> pd.DataFrame:
    """Run ``r/edger_pseudobulk.R`` on the files in ``dest`` and normalise its table."""
    out = dest / "edger_results.csv"
    proc = subprocess.run(["Rscript", str(EDGER_SCRIPT), str(dest / "pseudobulk_counts.mtx"),
                           str(dest / "samples.csv"), design, str(out)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"edgeR script failed: {proc.stderr.strip()[-500:]}")
    t = pd.read_csv(out)
    res = pd.DataFrame({
        "gene": t["gene"], "baseMean": 2.0 ** t["logCPM"], "log2FoldChange": t["logFC"],
        "stat": np.sign(t["logFC"]) * np.sqrt(t["F"].clip(lower=0)),
        "pvalue": t["PValue"], "padj": t["FDR"],
        "logFC": t["logFC"], "logCPM": t["logCPM"], "F": t["F"], "PValue": t["PValue"], "FDR": t["FDR"],
    })
    return res


def _fit(counts: np.ndarray, genes: list[str], meta: pd.DataFrame, samples: pd.DataFrame, dest: Path,
         kind: str, engine: str, design: str, contrast: list[str]) -> tuple[pd.DataFrame, str, dict]:
    """Write pseudobulk files, then test with the requested engine (edgeR falls back to PyDESeq2)."""
    _write_pseudobulk(dest, genes, sparse.csr_matrix(counts), samples)
    info: dict = {"engine_requested": engine, "design": design}
    if engine == "edger":
        if edger_available():
            try:
                res = _edger(dest, kind)
                info.update(engine_used="edger", status="complete")
                return res, "edger", info
            except RuntimeError as exc:
                log.warning("edgeR failed (%s); falling back to PyDESeq2.", exc)
                info["edger_error"] = str(exc)
        else:
            log.warning("engine 'edger' requested but Rscript/edgeR is unavailable; using PyDESeq2.")
        info["status"] = "edger_unavailable"
    res = _pydeseq2(counts, genes, meta, design, contrast)
    info["engine_used"] = "pydeseq2"
    info.setdefault("status", "complete")
    return res, "pydeseq2", info


# -- Wilcoxon ------------------------------------------------------------------------------
def wilcoxon_high_low(X: sparse.csr_matrix, genes: list[str], is_high: np.ndarray) -> pd.DataFrame:
    """Cell-level Wilcoxon of high vs low on log-normalized counts (scanpy, tie-corrected).

    ``auc`` is computed exactly from rank sums on the same log-normalized values
    (``AUC = U / (n_high * n_low)``); BH is applied over tested genes.
    """
    import anndata as ad
    import scanpy as sc
    from scipy.stats import rankdata

    n_high, n_low = int(is_high.sum()), int((~is_high).sum())
    expressed = np.asarray((X > 0).sum(axis=0)).ravel() > 0
    keep = np.flatnonzero(expressed)
    sub = ad.AnnData(X=X[:, keep].astype(np.float32), var=pd.DataFrame(index=np.asarray(genes)[keep]))
    pct_high = np.asarray((X[is_high][:, keep] > 0).mean(axis=0)).ravel()
    pct_low = np.asarray((X[~is_high][:, keep] > 0).mean(axis=0)).ravel()
    sub.obs["group"] = pd.Categorical(np.where(is_high, "high", "low"), categories=["low", "high"])
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)
    sc.tl.rank_genes_groups(sub, "group", groups=["high"], reference="low", method="wilcoxon",
                            tie_correct=True, n_genes=sub.n_vars, use_raw=False)
    df = sc.get.rank_genes_groups_df(sub, group="high").set_index("names").loc[sub.var_names]
    M = sub.X
    auc = np.empty(sub.n_vars)
    for s in range(0, sub.n_vars, 1000):
        blk = M[:, s:s + 1000]
        blk = blk.toarray() if sparse.issparse(blk) else np.asarray(blk)
        ranks = rankdata(blk, axis=0)
        u = ranks[is_high].sum(axis=0) - n_high * (n_high + 1) / 2.0
        auc[s:s + 1000] = u / (n_high * n_low)
    p = df["pvals"].to_numpy(float)
    return pd.DataFrame({
        "gene": sub.var_names, "log2FoldChange": df["logfoldchanges"].to_numpy(float), "auc": auc,
        "stat": df["scores"].to_numpy(float), "pvalue": p, "padj": bh_adjust(p),
        "pct_high": pct_high, "pct_low": pct_low,
    })


# -- figures ---------------------------------------------------------------------------------
def _neglog10(p: np.ndarray, cap: Optional[float] = None) -> np.ndarray:
    v = -np.log10(np.clip(np.asarray(p, float), 1e-300, 1.0))
    return np.minimum(v, cap) if cap else v


def plot_volcano(res: pd.DataFrame, title: str, stem: Path) -> None:
    d = res.dropna(subset=["log2FoldChange", "padj"])
    fig, ax = style.square_panel(title, "log2 fold change (high / low)", "-log10 adjusted p")
    y = _neglog10(d["padj"])
    up = (d["padj"] < ALPHA) & (d["log2FoldChange"] > 0)
    dn = (d["padj"] < ALPHA) & (d["log2FoldChange"] < 0)
    ax.scatter(d["log2FoldChange"], y, s=6, color="#BBBBBB", rasterized=True)
    ax.scatter(d["log2FoldChange"][up], y[up.to_numpy()], s=8, color=style.CASE, rasterized=True)
    ax.scatter(d["log2FoldChange"][dn], y[dn.to_numpy()], s=8, color=style.CONTROL, rasterized=True)
    ax.axhline(-np.log10(ALPHA), color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6)
    top = d.nsmallest(10, "padj")
    for r in top.itertuples():
        ax.annotate(r.gene, (r.log2FoldChange, _neglog10(np.array([r.padj]))[0]), fontsize=6,
                    xytext=(2, 2), textcoords="offset points")
    style.save_panel(fig, ax, stem)


def plot_clinical_vs_score(joined: pd.DataFrame, title: str, stem: Path) -> None:
    d = joined.dropna(subset=["log2FC_clinical", "log2FC_score", "padj_score"])
    fig, ax = style.square_panel(title, "Clinical log2 fold change", "High/low log2 fold change")
    c = _neglog10(d["padj_score"], cap=10)
    im = ax.scatter(d["log2FC_clinical"], d["log2FC_score"], c=c, vmin=0, vmax=10, cmap="magma_r", s=8,
                    rasterized=True)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    sig = d[d["padj_score"] < ALPHA]
    lab = pd.concat([sig[sig["log2FC_score"] > 0].nsmallest(8, "padj_score"),
                     sig[sig["log2FC_score"] < 0].nsmallest(8, "padj_score")])
    for r in lab.itertuples():
        ax.annotate(r.gene + ("*" if r.model_input else ""), (r.log2FC_clinical, r.log2FC_score),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")
    style.colorbar(fig, ax, im, "-log10 adjusted p (high/low), capped at 10")
    fig.text(0.19, 0.04, FOOTNOTE, fontsize=6.5, ha="left")
    style.save_panel(fig, ax, stem)


# -- clinical contrast -----------------------------------------------------------------------
def _norm_label(v: str) -> str:
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(v)


def _estimable(meta: pd.DataFrame, cols: list[str]) -> bool:
    X = pd.get_dummies(meta[cols].astype(str), drop_first=True).astype(float)
    X.insert(0, "const", 1.0)
    return np.linalg.matrix_rank(X.to_numpy()) == X.shape[1] and len(X) > X.shape[1]


def clinical_contrast(X: sparse.csr_matrix, genes: list[str], cells: pd.DataFrame, dest: Path,
                      cfg: RunConfig, engine: str, label: str) -> tuple[Optional[pd.DataFrame], dict]:
    """Donor pseudobulk (all cells of the cluster) tested between clinical label levels."""
    dest.mkdir(parents=True, exist_ok=True)
    meta_info: dict = {
        "contrast": "clinical label (donor-level)", "min_cells": cfg.min_cells,
        "inference_unit": "donor (pseudobulk across all cells of the cluster)",
        "note": "uses the donor's clinical label, not the DEGAS score",
    }

    def skip(reason: str):
        meta_info.update(status="skipped", skipped_reason=reason)
        (dest / "method.json").write_text(json.dumps(meta_info, indent=2) + "\n")
        return None, meta_info

    keys, mat, n_cells = pseudobulk(X, cells["donor"].to_numpy())
    donor_tab = cells.groupby("donor").agg(label=("label", "first"), sample=("sample", lambda s: s.mode().iloc[0]))
    tab = pd.DataFrame({"donor": keys, "n_cells": n_cells}).merge(donor_tab, left_on="donor", right_index=True)
    tab["library_size"] = np.asarray(mat.sum(axis=1)).ravel().astype(int)
    keep = (tab["n_cells"] >= cfg.min_cells).to_numpy()
    tab, mat = tab[keep].reset_index(drop=True), mat[np.flatnonzero(keep)]
    levels = sorted(tab["label"].unique())
    counts = tab["label"].value_counts()
    meta_info["donors_per_label"] = {k: int(v) for k, v in counts.items()}
    if len(levels) != 2 or counts.min() < 2:
        return skip(f"need exactly 2 label levels with >= 2 donors each (donors with >= {cfg.min_cells} cells); "
                    f"got {meta_info['donors_per_label']}")
    cols = ["label"]
    design, mode = "~ label", "clinical"
    if tab["sample"].nunique() >= 2 and _estimable(tab.assign(dataset=tab["sample"]), ["dataset", "label"]):
        tab["dataset"] = tab["sample"]
        design = "~ dataset + label"
        cols = ["dataset", "label"]
    elif tab["sample"].nunique() >= 2:
        meta_info["dataset_note"] = "sample column confounded with label; fitted ~ label"
    tab["sample_id"] = tab["donor"]
    out_cols = ["sample_id", "donor", "label"] + (["dataset"] if "dataset" in tab else []) + ["n_cells", "library_size"]
    meta = tab.set_index("sample_id")[cols]
    try:
        res, used, info = _fit(mat.toarray(), genes, meta, tab[out_cols], dest, mode, engine, design,
                               ["label", levels[1], levels[0]])
    except Exception as exc:  # noqa: BLE001
        log.warning("clinical contrast failed for %s: %s", label, exc)
        return skip(f"{type(exc).__name__}: {exc}")
    res.to_csv(dest / "results.csv", index=False)
    tab[out_cols].to_csv(dest / "included_donors.csv", index=False)
    meta_info.update(info, reference_level=levels[0], tested_level=levels[1], n_donors=len(tab),
                     n_genes_tested=len(res))
    (dest / "method.json").write_text(json.dumps(meta_info, indent=2) + "\n")
    return res, meta_info


# -- per cluster -------------------------------------------------------------------------------
def _count_sig(res: pd.DataFrame) -> tuple[int, int]:
    s = res["padj"] < ALPHA
    return int((s & (res["log2FoldChange"] > 0)).sum()), int((s & (res["log2FoldChange"] < 0)).sum())


def de_cluster(cluster: str, X: sparse.csr_matrix, genes: list[str], cells: pd.DataFrame, dest: Path,
               cfg: RunConfig, engine: str, model_genes: set[str], has_label: bool) -> dict:
    """All DE outputs for one cluster; returns the summary row."""
    dest.mkdir(parents=True, exist_ok=True)
    row = {c: None for c in SUMMARY_COLUMNS}
    row.update(cluster=cluster, directory=dest.name, skipped_reason="")
    hl = cells["group"].isin(["high", "low"]).to_numpy()
    n_high = int((cells["group"] == "high").sum())
    n_low = int((cells["group"] == "low").sum())
    row.update(n_high_cells=n_high, n_low_cells=n_low, n_donors=int(cells.loc[hl, "donor"].nunique()))
    meta: dict = {"cluster": cluster, "contrast": "high vs low (reference: low)", "requested_method": cfg.de_method,
                  "min_cells": cfg.min_cells, "n_high_cells": n_high, "n_low_cells": n_low}
    score_res: Optional[pd.DataFrame] = None
    if n_high < 2 or n_low < 2:
        reason = f"fewer than 2 cells in a group (high={n_high}, low={n_low})"
        row["skipped_reason"] = reason
        meta.update(method=None, status="skipped", skipped_reason=reason)
    else:
        sub = cells[hl]
        Xs = X[np.flatnonzero(hl)]
        piv = sub.groupby(["donor", "group"]).size().unstack(fill_value=0).reindex(columns=["high", "low"], fill_value=0)
        eligible = piv.index[(piv["high"] >= cfg.min_cells) & (piv["low"] >= cfg.min_cells)].tolist()
        rule = {"rule": "pseudobulk_deseq2 when >= 2 donors have >= min_cells cells in both groups",
                "n_eligible_donors": len(eligible), "eligible_donors": eligible}
        method = cfg.de_method
        if method == "auto":
            method = "pseudobulk_deseq2" if len(eligible) >= 2 else "wilcoxon"
        elif method == "pseudobulk_deseq2" and len(eligible) < 2:
            reason = (f"pseudobulk_deseq2 requested but only {len(eligible)} donor(s) have >= {cfg.min_cells} "
                      "cells in both groups; use --method wilcoxon or lower min_cells")
            row["skipped_reason"] = reason
            meta.update(rule, method=None, status="skipped", skipped_reason=reason)
            method = None
        meta["rule_outcome"] = rule
        if method == "pseudobulk_deseq2":
            m = sub["donor"].isin(eligible).to_numpy()
            key = (sub["donor"].astype(str) + "__" + sub["group"]).to_numpy()[m]
            names, mat, n_c = pseudobulk(Xs[np.flatnonzero(m)], key)
            samples = pd.DataFrame({"sample_id": names})
            samples["donor"] = [n.rsplit("__", 1)[0] for n in names]
            samples["group"] = [n.rsplit("__", 1)[1] for n in names]
            samples["n_cells"] = n_c
            samples["library_size"] = np.asarray(mat.sum(axis=1)).ravel().astype(int)
            md = samples.set_index("sample_id")[["donor", "group"]]
            try:
                score_res, used, info = _fit(mat.toarray(), genes, md, samples, dest, "paired", engine,
                                             "~ donor + group", ["group", "high", "low"])
                piv.assign(included=piv.index.isin(eligible)).reset_index().rename(
                    columns={"high": "n_high", "low": "n_low"}).to_csv(dest / "included_donors.csv", index=False)
                meta.update(info, method="pseudobulk_deseq2" if used == "pydeseq2" else "pseudobulk_edger",
                            n_donors=len(eligible), n_samples=len(samples),
                            inference_unit="donor (pseudobulk, paired design ~ donor + group)",
                            donors_excluded=[d for d in piv.index if d not in eligible])
            except Exception as exc:  # noqa: BLE001
                log.warning("pseudobulk DE failed for cluster %s: %s", cluster, exc)
                row["skipped_reason"] = f"{type(exc).__name__}: {exc}"
                meta.update(method=method, status="failed", skipped_reason=row["skipped_reason"])
        elif method == "wilcoxon":
            score_res = wilcoxon_high_low(Xs, genes, (sub["group"] == "high").to_numpy())
            meta.update(method=WILCOXON_NAME, status="complete", n_donors=int(sub["donor"].nunique()),
                        inference_unit="cells (pseudo-replicated; descriptive/exploratory, not donor-level inference)",
                        reason=("no replication: fewer than 2 donors with enough cells in both groups"
                                if cfg.de_method == "auto" else "requested"))
        if score_res is not None:
            score_res.to_csv(dest / "results.csv", index=False)
            up, dn = _count_sig(score_res)
            row.update(method=meta["method"], n_genes_tested=len(score_res))
            row["n_padj_lt_0.05_up"], row["n_padj_lt_0.05_down"] = up, dn
            meta["n_genes_tested"] = len(score_res)
            plot_volcano(score_res, f"{cluster}: high vs low", dest / "volcano")
        elif not row["skipped_reason"]:
            row["skipped_reason"] = "no result"
    # clinical contrast uses every cell of the cluster
    if has_label:
        clin, cinfo = clinical_contrast(X, genes, cells, dest / "clinical", cfg, engine, cluster)
        meta["clinical"] = {k: cinfo.get(k) for k in ("status", "skipped_reason", "design", "n_donors")}
        if clin is not None and score_res is not None:
            joined = clin[["gene", "log2FoldChange", "padj"]].rename(
                columns={"log2FoldChange": "log2FC_clinical", "padj": "padj_clinical"}).merge(
                score_res[["gene", "log2FoldChange", "padj"]].rename(
                    columns={"log2FoldChange": "log2FC_score", "padj": "padj_score"}), on="gene")
            joined["model_input"] = joined["gene"].isin(model_genes)
            joined.to_csv(dest / "clinical_vs_score.csv", index=False)
            plot_clinical_vs_score(joined, f"{cluster}: clinical vs high/low", dest / "clinical_vs_score")
    (dest / "method.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    if row["method"] is None:
        row["method"] = meta.get("method")
    return row


def run_de(run: RunDir, cfg: RunConfig) -> dict:
    import anndata as ad

    if not run.stage_complete("stratify"):
        raise ValueError(f"stratify is not complete; run `degas3 stratify {run.root}` first.")
    groups = pd.read_csv(run.path(LAYOUT.STRATA, "cell_groups.csv.gz"), dtype=str, keep_default_na=False)
    scores = read_scores(run)
    adata = ad.read_h5ad(run.cells_h5ad)
    if set(groups["cell_id"]) != set(adata.obs_names):
        raise ValueError("cell ids in strata/cell_groups.csv.gz differ from inputs/cells.h5ad; "
                         "re-run score and stratify on the same inputs.")
    cells = scores.set_index("cell_id").loc[adata.obs_names, ["donor", "cluster", "sample", "label"]].copy()
    cells["group"] = groups.set_index("cell_id").loc[adata.obs_names, "group"].to_numpy()
    cells["label"] = [_norm_label(v) if v != "" else "" for v in cells["label"]]
    has_label = bool((cells["label"] != "").all())
    clinical_skip = "" if has_label else "no clinical label column"
    if has_label and cells["label"].nunique() < 2:
        has_label = False
        clinical_skip = f"every donor has the same clinical label ({cells['label'].iloc[0]!r}); no contrast possible"
        log.warning("clinical contrast skipped: %s", clinical_skip)
    if has_label:
        per_donor = cells.groupby("donor")["label"].nunique()
        if (per_donor > 1).any():
            raise ValueError(f"label column varies within donor(s) {per_donor[per_donor > 1].index.tolist()}; "
                             "the clinical label must be constant per donor.")
    X = sparse.csr_matrix(get_counts(adata))
    genes = list(map(str, adata.var_names))
    engine = cfg.de_engine
    if engine == "edger" and not edger_available():
        log.warning("edgeR unavailable: pseudobulk contrasts will use PyDESeq2 (recorded in method.json).")
    model_genes, note = model_input_genes(run)
    out = run.ensure(LAYOUT.DE)
    rows = []
    for cluster in sorted(cells["cluster"].unique()):
        idx = np.flatnonzero((cells["cluster"] == cluster).to_numpy())
        log.info("DE cluster %s (%d cells)", cluster, len(idx))
        rows.append(de_cluster(str(cluster), X[idx], genes, cells.iloc[idx], out / cluster_dirname(cluster),
                               cfg, engine, model_genes, has_label))
    summary = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    summary.to_csv(out / "summary.csv", index=False)
    return {
        "de_method": cfg.de_method, "de_engine": engine, "min_cells": cfg.min_cells,
        "n_clusters": len(rows), "n_skipped": int((summary["skipped_reason"] != "").sum()),
        "clinical_contrast": has_label, "clinical_skipped_reason": clinical_skip, "model_input_note": note, "n_model_input_genes": len(model_genes),
        "inference_unit": "donor for pseudobulk methods; cells (exploratory) for Cell-level Wilcoxon",
    }


def cli_de(args: argparse.Namespace) -> int:
    """``degas3 de RUN [--method auto|pseudobulk_deseq2|wilcoxon] [--engine pydeseq2|edger]``."""
    run = RunDir(args.run)
    cfg = RunConfig.load(run.root, de_method=getattr(args, "method", None), de_engine=getattr(args, "engine", None))
    if run.stage_complete("de"):
        log.info("de already complete; skipping (delete %s to rerun)", run.complete_path("de"))
        return 0
    payload = run_de(run, cfg)
    run.write_complete("de", payload)
    return 0
