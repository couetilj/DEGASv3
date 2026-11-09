"""Spatial stages: neighborhood pooling for single-cell ST, score maps, and RCTD/C-SIDE export.

Shared helpers for reading the run's spatial cells live here so the three stages agree on how
samples, coordinates and scores are located.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from degas3.config import RunConfig
from degas3.runs import RunDir

DEFAULT_SAMPLE = "sample"


def safe_name(name: str) -> str:
    """File-system and R friendly version of a sample or cell-type name (``[A-Za-z0-9_.-]`` only)."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return out or "unnamed"


def spatial_modality(cfg: RunConfig, allowed: tuple[str, ...], stage: str) -> str:
    """Return ``cfg.modality`` if it is in ``allowed``; otherwise raise ValueError with a remedy."""
    if cfg.modality not in allowed:
        raise ValueError(
            f"stage '{stage}' needs modality in {list(allowed)}, but the run has modality={cfg.modality!r}; "
            "run 'degas3 init' with --modality, or skip this stage."
        )
    return cfg.modality


def read_spatial_cells(rd: RunDir, cfg: RunConfig) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """Load ``inputs/cells.h5ad``; return (adata, sample label per cell, float64 xy of shape (n, 2)).

    Coordinates come from ``obsm[cfg.spatial_key or "spatial"]`` in the sample's physical units.
    Without ``cfg.sample_col`` all cells form one sample named ``"sample"``.
    """
    if not rd.cells_h5ad.is_file():
        raise ValueError(f"{rd.cells_h5ad} not found; run 'degas3 init' first.")
    adata = ad.read_h5ad(rd.cells_h5ad)
    key = cfg.spatial_key or "spatial"
    if key not in adata.obsm:
        raise ValueError(f"coordinates adata.obsm[{key!r}] missing; found {list(adata.obsm)}. Set spatial_key in config.json.")
    xy = np.asarray(adata.obsm[key], dtype=np.float64)[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError(f"coordinates in obsm[{key!r}] contain NaN or inf; remove those cells before 'degas3 init'.")
    if cfg.sample_col:
        if cfg.sample_col not in adata.obs.columns:
            raise ValueError(f"sample column {cfg.sample_col!r} not in cell metadata; available: {list(adata.obs.columns)}.")
        samples = adata.obs[cfg.sample_col].astype(str).to_numpy()
    else:
        samples = np.full(adata.n_obs, DEFAULT_SAMPLE, dtype=object)
    return adata, samples, xy


def sample_order(samples: np.ndarray) -> list[str]:
    """Unique sample labels in sorted order (stable sample index for seeding)."""
    return sorted(set(samples.tolist()))


def read_cell_scores(rd: RunDir, obs_names: pd.Index) -> pd.DataFrame:
    """``scores/cell_scores.csv.gz`` aligned to ``obs_names`` (row order of ``cells.h5ad``)."""
    path = rd.path("scores", "cell_scores.csv.gz")
    if not path.is_file():
        raise ValueError(f"{path} not found; run 'degas3 score' first.")
    df = pd.read_csv(path)
    if "cell_id" not in df.columns or "ensemble_score" not in df.columns:
        raise ValueError(f"{path} needs columns cell_id and ensemble_score; found {list(df.columns)}.")
    df["cell_id"] = df["cell_id"].astype(str)
    df = df.drop_duplicates("cell_id").set_index("cell_id")
    missing = obs_names.astype(str).difference(df.index)
    if len(missing):
        raise ValueError(
            f"{len(missing)} cells in cells.h5ad have no score (e.g. {list(missing[:3])}); "
            "re-run 'degas3 score' on the current inputs."
        )
    return df.reindex(obs_names.astype(str))


def read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.is_file() else None
