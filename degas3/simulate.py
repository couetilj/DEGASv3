"""Deterministic toy data generator with a planted signal.

A module of genes is raised in bulk samples with outcome 1 and, in cells, in a subset of
one cluster ("active" cells; more frequent in label-1 donors). A correct pipeline should
therefore score the active cells higher and recover the module in high-vs-low contrasts.
Gene names are generic (``GENE0001``...). Everything is a function of ``seed``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)

CLUSTERS = ["c0", "c1", "c2"]
AFFECTED_CLUSTER = "c1"
MODULE_FOLD_BULK = 1.8
MODULE_FOLD_CELL = 3.0
DISPERSION = 5.0


def _nb(rng: np.random.Generator, mu: np.ndarray, r: float = DISPERSION) -> np.ndarray:
    """Negative-binomial draw with mean ``mu`` and size ``r``."""
    return rng.negative_binomial(r, r / (r + mu))


def _obs_frame(n: int, prefix: str, **cols: Any) -> pd.DataFrame:
    df = pd.DataFrame(cols, index=[f"{prefix}{i:05d}" for i in range(n)])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
    return df


def simulate(
    out_dir: str | Path,
    seed: int = 0,
    n_patients: int = 120,
    n_cells: int = 3000,
    n_genes: int = 2000,
    n_donors: int = 8,
    n_module: int = 60,
) -> dict:
    """Write toy bulk, single-cell, single-cell spatial and Visium-like data to ``out_dir``.

    Files: ``bulk_counts.csv`` (genes x samples), ``bulk_pheno.csv``, ``cells.h5ad``,
    ``spatial_cells.h5ad``, ``visium_spots.h5ad``, ``simulate_manifest.json`` (the truth).
    Returns the manifest dict.
    """
    if n_genes < 300 or n_module < 5 or n_genes < 2 * n_module + 30 * len(CLUSTERS) + 50:
        raise ValueError("n_genes too small for the planted module and cluster markers; use n_genes >= 300.")
    if n_patients < 8 or n_donors < 4 or n_donors % 2:
        raise ValueError("need n_patients >= 8 and an even n_donors >= 4 (two donors per label at minimum).")
    if n_cells < 30 * n_donors:
        raise ValueError(f"n_cells must be at least {30 * n_donors} (30 per donor).")
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    genes = np.array([f"GENE{i + 1:04d}" for i in range(n_genes)])

    # -- gene programs -------------------------------------------------------
    base_bulk = np.exp(rng.normal(3.0, 1.2, n_genes))
    base_cell = np.exp(rng.normal(-1.5, 1.2, n_genes))
    upper = np.flatnonzero(base_cell >= np.median(base_cell))
    module = np.sort(rng.choice(upper, n_module, replace=False))
    rest = np.setdiff1d(np.arange(n_genes), module)
    markers = rng.choice(rest, 30 * len(CLUSTERS), replace=False).reshape(len(CLUSTERS), 30)
    cluster_fac = np.exp(rng.normal(0.0, 0.3, (len(CLUSTERS), n_genes)))
    for k in range(len(CLUSTERS)):
        cluster_fac[k, markers[k]] *= 4.0
    is_module = np.zeros(n_genes, bool)
    is_module[module] = True

    # -- bulk -----------------------------------------------------------------
    outcome = rng.permutation(np.arange(n_patients) % 2)
    dataset = rng.permutation(np.arange(n_patients) % 2)
    ds_fac = np.exp(rng.normal(0.0, 0.2, (2, n_genes)))
    depth = np.exp(rng.normal(0.0, 0.2, n_patients))
    mu = base_bulk[None, :] * depth[:, None] * ds_fac[dataset]
    mu[:, is_module] *= np.where(outcome == 1, MODULE_FOLD_BULK, 1.0)[:, None]
    counts = _nb(rng, mu, r=10.0)
    sample_ids = [f"S{i + 1:04d}" for i in range(n_patients)]
    pd.DataFrame(counts.T, index=genes, columns=sample_ids).to_csv(out / "bulk_counts.csv", index_label="gene")
    event_time = rng.exponential(1.0 / (0.1 * np.exp(0.9 * outcome)))
    censor = rng.uniform(5.0, 25.0, n_patients)
    pheno = pd.DataFrame({
        "sample_id": sample_ids,
        "patient_id": [f"P{i + 1:04d}" for i in range(n_patients)],
        "dataset": [f"dataset{d + 1}" for d in dataset],
        "outcome": outcome,
        "time": np.round(np.minimum(event_time, censor), 3),
        "event": (event_time <= censor).astype(int),
    })
    pheno.to_csv(out / "bulk_pheno.csv", index=False)

    # -- single-cell ------------------------------------------------------------
    donors = [f"donor{i + 1}" for i in range(n_donors)]
    donor_label = {d: int(i % 2) for i, d in enumerate(donors)}
    donor_idx = rng.integers(0, n_donors, n_cells)
    donor_idx[: n_donors] = np.arange(n_donors)  # every donor present
    cell_donor = np.array(donors)[donor_idx]
    label = np.array([donor_label[d] for d in cell_donor])
    cluster_idx = rng.choice(len(CLUSTERS), n_cells, p=[0.4, 0.35, 0.25])
    affected = CLUSTERS.index(AFFECTED_CLUSTER)
    p_active = np.where(label == 1, 0.5, 0.15)
    active = (cluster_idx == affected) & (rng.random(n_cells) < p_active)
    cdepth = np.exp(rng.normal(0.0, 0.25, n_cells))
    cmu = base_cell[None, :] * cdepth[:, None] * cluster_fac[cluster_idx]
    cmu[np.ix_(active, is_module)] *= MODULE_FOLD_CELL
    cells = ad.AnnData(
        X=sparse.csr_matrix(_nb(rng, cmu).astype(np.float32)),
        obs=_obs_frame(n_cells, "cell", donor=cell_donor, label=label,
                       cell_type=np.array(CLUSTERS)[cluster_idx], planted_active=active),
        var=pd.DataFrame(index=genes),
    )
    cells.write_h5ad(out / "cells.h5ad")

    # -- single-cell spatial -------------------------------------------------------
    per_sample = max(60, n_cells // 4)
    xy, ctype, sample, act = [], [], [], []
    for s in ("section1", "section2"):
        pos = rng.uniform(0, 1000, (per_sample, 2))
        ct = rng.choice(len(CLUSTERS), per_sample, p=[0.4, 0.35, 0.25])
        hot = (pos[:, 0] > 600) & (pos[:, 1] > 600)
        act.append((ct == affected) & hot)
        xy.append(pos); ctype.append(ct); sample += [s] * per_sample
    xy, ctype, act = np.vstack(xy), np.concatenate(ctype), np.concatenate(act)
    n_sp = len(ctype)
    smu = base_cell[None, :] * np.exp(rng.normal(0, 0.25, n_sp))[:, None] * cluster_fac[ctype]
    smu[np.ix_(act, is_module)] *= MODULE_FOLD_CELL
    spatial = ad.AnnData(
        X=sparse.csr_matrix(_nb(rng, smu).astype(np.float32)),
        obs=_obs_frame(n_sp, "stcell", sample=np.array(sample), donor=np.array(sample),
                       cell_type=np.array(CLUSTERS)[ctype], planted_active=act),
        var=pd.DataFrame(index=genes),
    )
    spatial.obsm["spatial"] = xy
    spatial.write_h5ad(out / "spatial_cells.h5ad")

    # -- Visium-like spots -----------------------------------------------------------
    n_spots = max(100, n_cells // 5)
    side = int(np.ceil(np.sqrt(n_spots)))
    grid = np.array([(i % side, i // side) for i in range(n_spots)], float) * 100.0
    grid[:, 0] += (grid[:, 1] / 100 % 2) * 50.0
    weights = rng.dirichlet([2.0, 2.0, 2.0], n_spots)
    hot = (grid[:, 0] > 0.6 * grid[:, 0].max()) & (grid[:, 1] > 0.6 * grid[:, 1].max())
    act_frac = np.where(hot, 0.8, 0.1)
    vmu = np.zeros((n_spots, n_genes))
    for k in range(len(CLUSTERS)):
        prof = np.tile(base_cell * cluster_fac[k], (n_spots, 1))
        if k == affected:
            prof[:, is_module] *= (1.0 + (MODULE_FOLD_CELL - 1.0) * act_frac)[:, None]
        vmu += weights[:, [k]] * prof
    vmu *= 5.0 * np.exp(rng.normal(0, 0.2, n_spots))[:, None]
    spots = ad.AnnData(
        X=sparse.csr_matrix(_nb(rng, vmu).astype(np.float32)),
        obs=_obs_frame(n_spots, "spot", sample=np.array(["section1"] * n_spots),
                       hotspot=hot, c1_fraction=weights[:, affected]),
        var=pd.DataFrame(index=genes),
    )
    spots.obsm["spatial"] = grid
    spots.write_h5ad(out / "visium_spots.h5ad")

    manifest = {
        "seed": seed,
        "sizes": {"n_patients": n_patients, "n_cells": n_cells, "n_genes": n_genes, "n_donors": n_donors,
                  "n_spatial_cells": int(n_sp), "n_spots": int(n_spots)},
        "files": ["bulk_counts.csv", "bulk_pheno.csv", "cells.h5ad", "spatial_cells.h5ad",
                  "visium_spots.h5ad", "simulate_manifest.json"],
        "planted_genes": genes[module].tolist(),
        "marker_genes": {c: genes[markers[i]].tolist() for i, c in enumerate(CLUSTERS)},
        "affected_cluster": AFFECTED_CLUSTER,
        "module_fold_bulk": MODULE_FOLD_BULK,
        "module_fold_cell": MODULE_FOLD_CELL,
        "active_probability": {"label_1_donors": 0.5, "label_0_donors": 0.15},
        "donor_labels": donor_label,
        "spatial_hotspot": "x > 600 and y > 600 within each section (cluster c1 cells are active there)",
        "visium_hotspot": "top-right 40% of the grid (80% of the c1 share is active there, else 10%)",
        "obs_columns": {"cells.h5ad": ["donor", "label", "cell_type", "planted_active"],
                        "spatial_cells.h5ad": ["sample", "donor", "cell_type", "planted_active"],
                        "visium_spots.h5ad": ["sample", "hotspot", "c1_fraction"]},
    }
    (out / "simulate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("simulated data written to %s", out)
    return manifest
