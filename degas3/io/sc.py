"""Single-cell / spatial cell loaders returning :class:`CellData`, plus the gene intersection."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import numpy as np
from scipy import sparse

from degas3.contracts import BulkData, CellData
from degas3.io.bulk import strip_ensembl_version
from degas3.io.spatial import load_visium, load_xenium

log = logging.getLogger(__name__)


def get_counts(adata: ad.AnnData):
    """Raw counts of ``adata``: ``layers["counts"]`` when present, otherwise ``X``."""
    return adata.layers["counts"] if "counts" in adata.layers else adata.X


def assert_integer_counts(matrix, where: str = "counts") -> None:
    """Raise ValueError unless ``matrix`` is non-negative, finite, and integer-valued."""
    data = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if data.size == 0:
        raise ValueError(f"{where} is empty; supply a non-empty raw count matrix.")
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"{where} contains NaN or negative values; supply raw counts, not scaled data.")
    sample = data if data.size <= 2_000_000 else data[:: max(1, data.size // 2_000_000)]
    if not np.allclose(sample, np.round(sample)):
        raise ValueError(
            f"{where} is not integer-valued (looks normalised or log-transformed); "
            "supply raw counts in .X or in layers['counts']."
        )


def _clean_genes(adata: ad.AnnData) -> ad.AnnData:
    """Strip Ensembl versions; sum columns of duplicated gene names."""
    names = np.array([strip_ensembl_version(g) for g in adata.var_names])
    if len(set(names)) == len(names):
        adata.var_names = names
        return adata
    log.warning("duplicate gene identifiers; summing their counts.")
    uniq, inv = np.unique(names, return_inverse=True)
    agg = sparse.csr_matrix((np.ones(len(names)), (np.arange(len(names)), inv)), shape=(len(names), len(uniq)))
    X = sparse.csr_matrix(get_counts(adata)) @ agg
    out = ad.AnnData(X=X.astype(np.float32), obs=adata.obs.copy(), obsm=dict(adata.obsm))
    out.var_names = uniq
    return out


def _read_cells(path: Path, modality: Optional[str]) -> tuple[ad.AnnData, str]:
    """Dispatch on path type; return (adata, modality)."""
    if not path.exists():
        raise ValueError(f"cell data {path} does not exist.")
    if path.is_dir():
        if (path / "cell_feature_matrix.h5").is_file():
            return load_xenium(path), modality or "st_cell"
        if (path / "spatial").is_dir() or (path / "tissue_positions.csv").is_file():
            return load_visium(path), modality or "st_spot"
        import scanpy as sc

        files = {f.name for f in path.iterdir()}
        if files & {"matrix.mtx", "matrix.mtx.gz"}:
            return sc.read_10x_mtx(path, var_names="gene_symbols"), modality or "sc"
        raise ValueError(
            f"{path} is not a recognised directory (expected Xenium outs, Visium outs, or a 10x mtx folder)."
        )
    if path.suffix == ".h5ad":
        adata = ad.read_h5ad(path)
        return adata, modality or ("st_cell" if "spatial" in adata.obsm else "sc")
    if path.suffix == ".h5":
        import scanpy as sc

        return sc.read_10x_h5(path), modality or "sc"
    raise ValueError(f"unsupported cell file {path.name}; use .h5ad, 10x .h5, or a 10x/Xenium/Visium directory.")


def load_cells(
    path: str | Path,
    *,
    modality: Optional[str] = None,
    donor_col: Optional[str] = None,
    cluster_col: Optional[str] = None,
    label_col: Optional[str] = None,
    sample_col: Optional[str] = None,
    spatial_key: Optional[str] = None,
) -> CellData:
    """Load cells or spots, verify raw integer counts, strip gene versions, check obs columns.

    Accepts h5ad, 10x h5, a 10x mtx directory, a Xenium ``outs`` directory, or a Visium
    section (Space Ranger directory or h5ad). ``modality`` defaults from the path type
    (h5ad with ``obsm["spatial"]`` is taken as ``st_cell``; pass ``st_spot`` for spots).
    """
    adata, modality = _read_cells(Path(path), modality)
    if modality not in ("sc", "st_cell", "st_spot"):
        raise ValueError(f"modality must be sc, st_cell or st_spot, got {modality!r}.")
    where = "layers['counts']" if "counts" in adata.layers else ".X"
    assert_integer_counts(get_counts(adata), where)
    adata = _clean_genes(adata)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()
    for col_name, col in (("donor", donor_col), ("cluster", cluster_col), ("label", label_col), ("sample", sample_col)):
        if col is not None and col not in adata.obs.columns:
            raise ValueError(f"{col_name} column {col!r} not in cell metadata; available: {list(adata.obs.columns)}.")
    if modality != "sc":
        spatial_key = spatial_key or "spatial"
        if spatial_key not in adata.obsm:
            raise ValueError(f"spatial modality needs coordinates in adata.obsm[{spatial_key!r}]; found {list(adata.obsm)}.")
    else:
        spatial_key = None
    if label_col is not None:
        per_donor = adata.obs.groupby(donor_col)[label_col].nunique() if donor_col else None
        if per_donor is not None and (per_donor > 1).any():
            raise ValueError(f"label column {label_col!r} varies within a donor; it must be one clinical label per donor.")
    return CellData(adata, donor_col, cluster_col, label_col, modality, spatial_key, sample_col)


@dataclass
class GeneIntersection:
    """Gene counts of the intersection of bulk, cells, and any extra panels.

    ``genes`` keeps the bulk gene order. ``counts`` records each set size so the
    loss at every step is visible.
    """

    genes: list[str]
    counts: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"n_shared": len(self.genes), "counts": self.counts}, indent=2)


def intersect_genes(
    bulk: BulkData,
    cells: CellData | ad.AnnData | Iterable[str],
    extra_panels: Optional[list[Iterable[str]]] = None,
    out_path: Optional[str | Path] = None,
) -> GeneIntersection:
    """Genes present in the bulk matrix, the cells, and every extra panel (exact match after version stripping).

    Counts are logged and stored in ``.counts``; they are written as JSON to ``out_path`` when given.
    """
    if isinstance(cells, CellData):
        cell_genes = list(cells.adata.var_names)
    elif isinstance(cells, ad.AnnData):
        cell_genes = list(cells.var_names)
    else:
        cell_genes = list(cells)
    cell_set = set(cell_genes)
    counts = {"bulk": len(bulk.genes), "cells": len(cell_set)}
    shared = [g for g in bulk.genes if g in cell_set]
    counts["bulk_and_cells"] = len(shared)
    for i, panel in enumerate(extra_panels or []):
        panel_set = {strip_ensembl_version(g) for g in panel}
        counts[f"panel_{i}"] = len(panel_set)
        shared = [g for g in shared if g in panel_set]
        counts[f"after_panel_{i}"] = len(shared)
    result = GeneIntersection(shared, counts)
    log.info("gene intersection: %s", counts)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(result.to_json() + "\n")
    return result
