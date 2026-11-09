"""Outcome-blind clustering of the run's cells (SPEC "Clustering").

Pipeline: normalize_total(1e4) -> log1p -> 2000 HVG (seurat flavor, batch = donor when >= 2
donors) -> scale (clip 10) -> PCA 50 -> neighbors 15 -> Leiden -> UMAP. Raw counts in the
input object are never modified. Outcome labels, clinical labels, and scores are not read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from degas3.config import RunConfig, update_config_file
from degas3.io.sc import get_counts
from degas3.runs import RunDir

log = logging.getLogger(__name__)

CLUSTER_KEY = "degas3_cluster"


def cluster_adata(
    adata: ad.AnnData,
    *,
    donor_col: Optional[str] = None,
    resolution: float = 1.0,
    n_hvg: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    seed: int = 0,
    markers_per_cluster: int = 20,
) -> pd.DataFrame:
    """Write ``obs[degas3_cluster]`` (and ``obsm['X_umap']`` if absent) in place; return the marker table.

    The marker table lists the top Wilcoxon genes per cluster (one-vs-rest, log-normalized
    counts). It describes clusters; it is not a test of disease association.
    """
    log_adata = ad.AnnData(X=get_counts(adata).copy(), obs=pd.DataFrame(index=adata.obs_names),
                           var=pd.DataFrame(index=adata.var_names))
    sc.pp.normalize_total(log_adata, target_sum=1e4)
    sc.pp.log1p(log_adata)
    batch_key = None
    if donor_col and donor_col in adata.obs and adata.obs[donor_col].nunique() >= 2:
        log_adata.obs["_batch"] = adata.obs[donor_col].astype(str).values
        batch_key = "_batch"
    n_top = min(n_hvg, log_adata.n_vars)
    try:
        sc.pp.highly_variable_genes(log_adata, n_top_genes=n_top, flavor="seurat", batch_key=batch_key)
    except Exception as exc:  # per-batch HVG can fail for tiny batches; fall back to pooled
        log.warning("batch-aware HVG failed (%s); using pooled HVG.", exc)
        sc.pp.highly_variable_genes(log_adata, n_top_genes=n_top, flavor="seurat")
    hvg = log_adata[:, log_adata.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(hvg, max_value=10)
    n_comps = max(2, min(n_pcs, hvg.n_obs - 1, hvg.n_vars - 1))
    sc.pp.pca(hvg, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(hvg, n_neighbors=min(n_neighbors, hvg.n_obs - 1), n_pcs=n_comps, random_state=seed)
    sc.tl.leiden(hvg, resolution=resolution, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    adata.obs[CLUSTER_KEY] = pd.Categorical(hvg.obs["leiden"].astype(str).values)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(hvg, random_state=seed)
        adata.obsm["X_umap"] = np.asarray(hvg.obsm["X_umap"])
    log_adata.obs[CLUSTER_KEY] = adata.obs[CLUSTER_KEY].values
    if log_adata.obs[CLUSTER_KEY].nunique() < 2:
        log.warning("only one cluster found; no marker table. Raise --resolution to split the cells.")
        return pd.DataFrame(columns=["cluster", "rank", "gene", "score", "logfoldchange", "pval", "padj"])
    sc.tl.rank_genes_groups(log_adata, CLUSTER_KEY, method="wilcoxon", n_genes=markers_per_cluster, tie_correct=True)
    res = log_adata.uns["rank_genes_groups"]
    frames = []
    for grp in res["names"].dtype.names:
        frames.append(pd.DataFrame({
            "cluster": grp, "rank": np.arange(1, len(res["names"][grp]) + 1),
            "gene": res["names"][grp], "score": res["scores"][grp],
            "logfoldchange": res["logfoldchanges"][grp], "pval": res["pvals"][grp],
            "padj": res["pvals_adj"][grp],
        }))
    return pd.concat(frames, ignore_index=True)


def run_cluster(run: str | Path, resolution: Optional[float] = None, force: bool = False) -> dict:
    """Cluster ``<run>/inputs/cells.h5ad`` in place; write markers and completion; set ``cluster_col``."""
    rd = RunDir(run)
    if not rd.cells_h5ad.is_file():
        raise ValueError(f"{rd.cells_h5ad} not found; run `degas3 init` first.")
    if rd.stage_complete("cluster") and not force:
        log.info("cluster stage already complete; use --force to redo it.")
        return rd.read_complete("cluster") or {}
    cfg = RunConfig.load(rd.root)
    res = cfg.cluster_resolution if resolution is None else float(resolution)
    adata = ad.read_h5ad(rd.cells_h5ad)
    markers = cluster_adata(adata, donor_col=cfg.donor_col, resolution=res)
    adata.write_h5ad(rd.cells_h5ad)
    out = rd.ensure("cluster")
    markers.to_csv(out / "markers.csv", index=False)
    update_config_file(rd.root, cluster_col=CLUSTER_KEY, cluster_resolution=res)
    sizes = adata.obs[CLUSTER_KEY].value_counts().sort_index().to_dict()
    payload = {"resolution": res, "n_clusters": len(sizes), "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
               "batch_col": cfg.donor_col, "outcome_blind": True}
    rd.write_complete("cluster", payload)
    log.info("clustered %d cells into %d clusters", adata.n_obs, len(sizes))
    return payload
