"""``prepare`` stage: gene universe, folds, fold-internal gene rankings, and the task list.

Outputs below ``<run>/prepared``: ``shared_genes.txt``, ``folds.json``, ``rankings/<fold>.csv``,
``final_rankings.csv`` (all bulk, all cells), ``tasks.json`` (one task per fold x size),
``cells_shared_counts.npz`` (cells x shared genes, raw counts, sparse), ``cell_metadata.csv``,
``excluded_cells/<fold>.npy`` and ``complete.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from degas3.config import RunConfig
from degas3.contracts import BulkData
from degas3.features import bulk_variance_ranking, marker_ranking, mixed_list, nested_list
from degas3.folds import cell_exclusion_mask, make_folds
from degas3.io.bulk import load_bulk_npz
from degas3.io.sc import get_counts, intersect_genes
from degas3.runs import RunDir, sha256_file

log = logging.getLogger(__name__)

META_COLUMNS = ["cell_id", "donor", "cluster", "sample", "label", "dataset"]


def resolve_sizes(sizes: list[int], n_shared: int) -> list[int]:
    """Clip candidate sizes to the shared gene count; append the full universe when the largest size exceeds it."""
    kept = sorted({int(s) for s in sizes if int(s) <= n_shared})
    if max(sizes) > n_shared and n_shared not in kept:
        kept.append(n_shared)
    if not kept:
        raise ValueError(f"all candidate sizes {sizes} exceed the {n_shared} shared genes; lower 'sizes'.")
    return sorted(kept)


def cell_metadata(adata: ad.AnnData, cfg: RunConfig) -> pd.DataFrame:
    """Per-cell table (cell_id, donor, cluster, sample, label, dataset); absent columns are empty strings."""
    obs = adata.obs
    n = len(obs)

    def col(name, default=""):
        return obs[name].astype(str).to_numpy() if name and name in obs.columns else np.full(n, default, dtype=object)

    dataset_col = cfg.sample_col if cfg.sample_col in obs.columns else ("dataset" if "dataset" in obs.columns else None)
    return pd.DataFrame({
        "cell_id": adata.obs_names.astype(str), "donor": col(cfg.donor_col, "donor0"),
        "cluster": col(cfg.cluster_col), "sample": col(cfg.sample_col),
        "label": col(cfg.label_col), "dataset": col(dataset_col),
    })


def ranked_genes(bulk: BulkData, rows: np.ndarray, universe: list[str]) -> pd.DataFrame:
    """Bulk variance ranking on ``rows`` only, restricted to ``universe`` (library size uses all genes)."""
    ds = bulk.pheno["dataset"].to_numpy()[rows] if "dataset" in bulk.pheno.columns else None
    full = bulk_variance_ranking(bulk.counts[rows], ds, bulk.genes, already_log=bulk.already_log)
    return full[full["gene"].isin(set(universe))].reset_index(drop=True)


def build_gene_list(cfg: RunConfig, bulk: BulkData, rows: np.ndarray, universe: list[str],
                    counts, meta: pd.DataFrame, eligible: np.ndarray, size: int) -> pd.DataFrame:
    """Ranked gene table of length ``size`` (columns rank, gene, source, cluster, ...), using training data only."""
    rank = ranked_genes(bulk, rows, universe)
    if cfg.feature_selection == "variance":
        genes = nested_list(rank, universe, size)
        out = rank.set_index("gene").loc[genes].reset_index()
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        out["source"], out["cluster"] = "bulk_variance", ""
        return out
    if not cfg.cluster_col or (meta["cluster"] == "").all():
        raise ValueError("feature_selection='mixed' needs cluster labels; run `degas3 cluster RUN` or use 'variance'.")
    idx = np.flatnonzero(eligible)
    markers = marker_ranking(counts[idx], meta["cluster"].to_numpy()[idx], meta["donor"].to_numpy()[idx],
                             universe, cfg.marker_cells_per_cluster, cfg.selection_seed)
    picks = pd.DataFrame(mixed_list(rank, markers, universe, size))
    picks = picks.merge(rank, on="gene", how="left")
    picks.insert(0, "rank", np.arange(1, len(picks) + 1))
    return picks


def load_prepared(rd: RunDir) -> dict:
    """Read the prepared artefacts needed by train/fit/score."""
    prep = rd.stage_dir("prepare")
    if not (prep / "complete.json").is_file():
        raise ValueError("`prepare` has not completed; run `degas3 prepare RUN` first.")
    return {
        "genes": (prep / "shared_genes.txt").read_text().split(),
        "folds": json.loads((prep / "folds.json").read_text()),
        "tasks": json.loads((prep / "tasks.json").read_text()),
        "counts": sparse.load_npz(prep / "cells_shared_counts.npz").tocsr(),
        "meta": pd.read_csv(prep / "cell_metadata.csv", keep_default_na=False, dtype=str),
        "final_ranking": pd.read_csv(prep / "final_rankings.csv", keep_default_na=False),
    }


def cli_prepare(args: argparse.Namespace) -> int:
    rd = RunDir(args.run)
    if rd.stage_complete("prepare"):
        log.info("prepare already complete for %s; skipping.", rd.root)
        return 0
    cfg = RunConfig.load(rd.root)
    if not rd.bulk_npz.is_file() or not rd.cells_h5ad.is_file():
        raise ValueError(f"{rd.inputs} lacks bulk.npz/cells.h5ad; run `degas3 init` first.")
    bulk = load_bulk_npz(rd.bulk_npz)
    adata = ad.read_h5ad(rd.cells_h5ad)
    inter = intersect_genes(bulk, adata)  # ST sections share the cells' panel; no extra panels are required here
    universe = inter.genes
    if len(universe) < 2:
        raise ValueError(f"only {len(universe)} genes shared between bulk and cells; check gene identifiers (config 'gene_map').")
    sizes = resolve_sizes(cfg.sizes, len(universe))
    prep = rd.ensure("prepared")
    rd.ensure("prepared", "rankings")
    rd.ensure("prepared", "excluded_cells")
    (prep / "shared_genes.txt").write_text("\n".join(universe) + "\n")
    var_index = pd.Index(adata.var_names)
    cols = var_index.get_indexer(universe)
    counts = sparse.csr_matrix(get_counts(adata))[:, cols].astype(np.float32)
    sparse.save_npz(prep / "cells_shared_counts.npz", counts)
    meta = cell_metadata(adata, cfg)
    meta.to_csv(prep / "cell_metadata.csv", index=False)

    pheno = bulk.pheno
    folds = make_folds(pheno, cfg)
    max_size = max(sizes)
    tasks, fold_records = [], []
    for fd in folds:
        rows = np.asarray(fd["train_idx"])
        excl = cell_exclusion_mask(adata, cfg, fd, pheno)
        np.save(prep / "excluded_cells" / f"{fd['name']}.npy", np.flatnonzero(excl))
        table = build_gene_list(cfg, bulk, rows, universe, counts, meta, ~excl, max_size)
        table.to_csv(prep / "rankings" / f"{fd['name']}.csv", index=False)
        genes = table["gene"].tolist()
        test = pheno.iloc[fd["test_idx"]]
        held = sorted(set(test["patient_id"].astype(str))) if "patient_id" in test.columns else sorted(test.index.astype(str))
        fold_records.append({
            **fd, "heldout_ids": held, "n_cells_excluded": int(excl.sum()), "n_cells_eligible": int((~excl).sum()),
            "excluded_file": f"excluded_cells/{fd['name']}.npy",
        })
        for size in sizes:
            tasks.append({"task": len(tasks), "fold_id": fd["fold_id"], "fold": fd["name"], "size": int(size),
                          "genes": genes[:size], "heldout_dataset": fd["heldout_dataset"]})
        log.info("%s: train %d / test %d bulk samples; %d cells excluded.", fd["name"], fd["n_train"], fd["n_test"], int(excl.sum()))
    final = build_gene_list(cfg, bulk, np.arange(len(pheno)), universe, counts, meta, np.ones(len(meta), bool), max_size)
    final.to_csv(prep / "final_rankings.csv", index=False)
    (prep / "folds.json").write_text(json.dumps(
        {"cv": cfg.cv, "outcome_type": bulk.outcome_type, "folds": fold_records}, indent=2) + "\n")
    (prep / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    rd.write_complete("prepare", {
        "n_shared_genes": len(universe), "gene_counts": inter.counts, "sizes": sizes, "n_folds": len(folds),
        "n_tasks": len(tasks), "n_cells": int(counts.shape[0]), "n_bulk_samples": len(pheno),
        "feature_selection": cfg.feature_selection, "cv": cfg.cv, "selection_seed": cfg.selection_seed,
        "inputs": {"bulk_sha256": sha256_file(rd.bulk_npz), "cells_sha256": sha256_file(rd.cells_h5ad)},
        "config": cfg.to_dict(),
    })
    log.info("prepare complete: %d genes, sizes %s, %d folds, %d tasks.", len(universe), sizes, len(folds), len(tasks))
    return 0
