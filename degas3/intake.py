"""Intake: ``init`` creates a run from user files; ``inspect`` summarises it with warnings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import numpy as np

from degas3.config import RunConfig
from degas3.io.bulk import load_bulk, load_bulk_npz, save_bulk_npz
from degas3.io.sc import assert_integer_counts, get_counts, intersect_genes, load_cells
from degas3.runs import RunDir, sha256_path

log = logging.getLogger(__name__)

MIN_SHARED_GENES = 1000


def init_run(
    run: str | Path,
    *,
    bulk: str | Path,
    pheno: str | Path,
    cells: str | Path,
    outcome: Optional[str] = None,
    modality: Optional[str] = None,
    donor_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    label_col: Optional[str] = None,
    sample_col: Optional[str] = None,
    dataset_col: Optional[str] = None,
    patient_col: Optional[str] = None,
    pheno_id_col: str = "sample_id",
    spatial_key: Optional[str] = None,
    already_log: bool = False,
    force: bool = False,
    **config_overrides: Any,
) -> dict:
    """Create ``<run>`` with config, ``inputs/bulk.npz``, ``inputs/cells.h5ad`` and the input manifest."""
    rd = RunDir(run)
    if rd.config_path.exists() and not force:
        raise ValueError(f"{rd.config_path} exists; choose a new run directory or pass --force to overwrite inputs.")
    bd = load_bulk(bulk, pheno, outcome=outcome, already_log=already_log, pheno_id_col=pheno_id_col,
                   patient_col=patient_col or "patient_id", dataset_col=dataset_col or "dataset")
    cd = load_cells(cells, modality=modality, donor_col=donor_col, cluster_col=cluster_col,
                    label_col=label_col, sample_col=sample_col, spatial_key=spatial_key)
    cfg = RunConfig(
        bulk=str(Path(bulk).resolve()), pheno=str(Path(pheno).resolve()), cells=str(Path(cells).resolve()),
        outcome=bd.outcome_type, modality=cd.modality, already_log=already_log, donor_col=donor_col,
        cluster_col=cluster_col, label_col=label_col, sample_col=sample_col, dataset_col=dataset_col,
        patient_col=patient_col, spatial_key=cd.spatial_key,
    ).merged(**config_overrides)
    rd.ensure("inputs")
    cfg.save(rd.root)
    save_bulk_npz(bd, rd.bulk_npz)
    cd.adata.write_h5ad(rd.cells_h5ad)
    sources = {k: {"path": str(Path(v).resolve()), "sha256": sha256_path(v)}
               for k, v in (("bulk", bulk), ("pheno", pheno), ("cells", cells))}
    manifest = build_manifest(rd, sources=sources)
    log.info("run initialised at %s", rd.root)
    return manifest


def build_manifest(rd: RunDir, sources: Optional[dict] = None) -> dict:
    """(Re)build ``inputs/input_manifest.json`` from the saved inputs, keeping recorded source hashes."""
    cfg = RunConfig.load(rd.root)
    if sources is None and rd.input_manifest.is_file():
        sources = json.loads(rd.input_manifest.read_text()).get("sources", {})
    bulk = load_bulk_npz(rd.bulk_npz)
    adata = ad.read_h5ad(rd.cells_h5ad)
    shared = intersect_genes(bulk, adata)
    obs = adata.obs
    pheno = bulk.pheno
    donor_col, cluster_col = cfg.donor_col, cfg.cluster_col
    if cluster_col and cluster_col not in obs.columns:
        cluster_col = None
    outcome_counts = (
        {str(k): int(v) for k, v in pheno["outcome"].value_counts().sort_index().items()}
        if bulk.outcome_type == "binary"
        else {"events": int(pheno["event"].sum()), "censored": int((1 - pheno["event"]).sum())}
    )
    try:
        assert_integer_counts(get_counts(adata))
        cells_integer = True
    except ValueError:
        cells_integer = False
    manifest = {
        "sources": sources or {},
        "modality": cfg.modality,
        "outcome_type": bulk.outcome_type,
        "bulk": {
            "shape_samples_genes": list(bulk.counts.shape),
            "pheno_columns": list(pheno.columns),
            "already_log": bulk.already_log,
            "counts_integer": bool(bulk.already_log or np.allclose(bulk.counts, np.round(bulk.counts))),
            "n_patients": int(pheno["patient_id"].nunique()) if "patient_id" in pheno else int(len(pheno)),
            "datasets": sorted(pheno["dataset"].unique().tolist()) if "dataset" in pheno else [],
            "outcome_counts": outcome_counts,
        },
        "cells": {
            "shape_cells_genes": list(adata.shape),
            "obs_columns": list(obs.columns),
            "counts_integer": cells_integer,
            "donor_col": donor_col,
            "n_donors": int(obs[donor_col].nunique()) if donor_col and donor_col in obs else None,
            "cluster_col": cluster_col,
            "n_clusters": int(obs[cluster_col].nunique()) if cluster_col else None,
            "label_col": cfg.label_col,
            "sample_col": cfg.sample_col,
            "n_samples": int(obs[cfg.sample_col].nunique()) if cfg.sample_col and cfg.sample_col in obs else None,
            "spatial_key": cfg.spatial_key,
        },
        "shared_genes": shared.counts | {"n_shared": len(shared.genes)},
    }
    manifest["warnings"] = intake_warnings(manifest)
    rd.input_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def intake_warnings(m: dict) -> list[str]:
    """Plain-language problems visible in the manifest."""
    warns: list[str] = []
    cells, bulk = m["cells"], m["bulk"]
    if not cells.get("donor_col"):
        warns.append("No donor column: donor-aware inference (pseudobulk DE) is impossible; supply --donor-col.")
    if not cells.get("cluster_col"):
        warns.append("No cluster column: run `degas3 cluster RUN` or supply --cluster-col before stratification.")
    if len(bulk["datasets"]) < 2:
        warns.append("Fewer than 2 bulk datasets: leave-one-dataset-out validation is unavailable.")
    oc = bulk["outcome_counts"]
    if m["outcome_type"] == "binary":
        small, total = min(oc.values()), sum(oc.values())
        if small < 10 or small / total < 0.25:
            warns.append(f"Outcome imbalance: classes {oc}; AUROC will be noisy and folds may lack positives.")
    elif oc["events"] < 10:
        warns.append(f"Only {oc['events']} survival events; the Cox model and C-index will be unstable.")
    if m["shared_genes"]["n_shared"] < MIN_SHARED_GENES:
        warns.append(f"Only {m['shared_genes']['n_shared']} genes shared between bulk and cells (< {MIN_SHARED_GENES}).")
    if not bulk["counts_integer"]:
        warns.append("Bulk values are not integers; confirm they are counts or set --already-log for log2(count+1).")
    if not cells["counts_integer"]:
        warns.append("Cell values are not integer counts; DE and scoring need raw counts.")
    return warns


def format_summary(m: dict) -> str:
    """Human-readable intake summary."""
    b, c, g = m["bulk"], m["cells"], m["shared_genes"]
    lines = [
        "DEGAS3 intake summary",
        f"  modality        : {m['modality']}",
        f"  outcome         : {m['outcome_type']}  {b['outcome_counts']}",
        f"  bulk            : {b['shape_samples_genes'][0]} samples x {b['shape_samples_genes'][1]} genes; "
        f"{b['n_patients']} patients; datasets {b['datasets'] or 'none'}",
        f"  cells           : {c['shape_cells_genes'][0]} cells x {c['shape_cells_genes'][1]} genes",
        f"  donors          : {c['n_donors']} (column {c['donor_col']})",
        f"  clusters        : {c['n_clusters']} (column {c['cluster_col']})",
        f"  clinical label  : {c['label_col']}",
        f"  shared genes    : {g['n_shared']} (bulk {g['bulk']}, cells {g['cells']})",
    ]
    if m.get("sources"):
        lines.append("  sources (sha256):")
        lines += [f"    {k}: {v['path']}  {v['sha256'][:16]}" for k, v in m["sources"].items()]
    if m["warnings"]:
        lines.append("Warnings:")
        lines += [f"  - {w}" for w in m["warnings"]]
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


def inspect_run(run: str | Path) -> tuple[dict, str]:
    """Refresh the manifest from the run's inputs; return (manifest, printable summary)."""
    rd = RunDir(run)
    if not rd.bulk_npz.is_file() or not rd.cells_h5ad.is_file():
        raise ValueError(f"{rd.root} has no inputs; run `degas3 init` first.")
    m = build_manifest(rd)
    return m, format_summary(m)
