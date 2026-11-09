"""Spatial loaders: Xenium output directories, Visium sections, and spatial h5ad files."""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

log = logging.getLogger(__name__)


def load_xenium(outs_dir: str | Path) -> ad.AnnData:
    """Xenium ``outs`` directory -> AnnData of cells x genes with centroids in ``obsm["spatial"]``.

    Reads ``cell_feature_matrix.h5`` and keeps only features of type "Gene Expression"
    (control probes and codewords are dropped). Centroids come from ``cells.csv.gz``
    (``x_centroid``, ``y_centroid``) aligned on ``cell_id``.
    """
    import h5py

    outs = Path(outs_dir)
    h5 = outs / "cell_feature_matrix.h5"
    cells_csv = outs / "cells.csv.gz"
    for f in (h5, cells_csv):
        if not f.is_file():
            raise ValueError(f"{f} not found; point to the Xenium 'outs' directory.")
    dec = lambda arr: np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr])  # noqa: E731
    with h5py.File(h5, "r") as h:
        m = h["matrix"]
        shape = tuple(int(s) for s in m["shape"][:])  # features x cells
        mat = sparse.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        names = dec(m["features/name"][:])
        ftype = dec(m["features/feature_type"][:])
        barcodes = dec(m["barcodes"][:])
    keep = np.flatnonzero(ftype == "Gene Expression")
    if keep.size == 0:
        raise ValueError(f"no 'Gene Expression' features in {h5}; the file is not a Xenium cell-feature matrix.")
    X = mat[keep].T.tocsr().astype(np.float32)
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=pd.Index(barcodes, name="cell_id")),
                       var=pd.DataFrame(index=pd.Index(names[keep])))
    cells = pd.read_csv(cells_csv)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").reindex(adata.obs_names)
    if cells[["x_centroid", "y_centroid"]].isna().any().any():
        raise ValueError("some cells have no centroid in cells.csv.gz; use matching cell_feature_matrix.h5 and cells.csv.gz.")
    adata.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float64)
    return adata


def load_visium(path: str | Path) -> ad.AnnData:
    """Visium section: a Space Ranger ``outs`` directory (via scanpy) or an h5ad with ``obsm["spatial"]``."""
    p = Path(path)
    if p.is_dir():
        import scanpy as sc

        adata = sc.read_visium(p)
        adata.var_names_make_unique()
    else:
        adata = ad.read_h5ad(p)
    if "spatial" not in adata.obsm:
        raise ValueError("Visium data needs coordinates in adata.obsm['spatial']; supply a Space Ranger directory or add them.")
    return adata
