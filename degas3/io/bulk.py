"""Bulk loaders: CSV/TSV, h5ad and npz expression plus a phenotype table."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degas3.contracts import BulkData

log = logging.getLogger(__name__)

_ENSEMBL = re.compile(r"^(ENS[A-Z]*[GTEP]\d{6,})\.\d+$")


def strip_ensembl_version(name: str) -> str:
    """``ENSG00000123456.7`` -> ``ENSG00000123456``; other identifiers are returned unchanged."""
    m = _ENSEMBL.match(str(name))
    return m.group(1) if m else str(name)


def _collapse_duplicate_genes(df: pd.DataFrame) -> pd.DataFrame:
    """Sum columns that share a gene identifier (samples x genes frame)."""
    if df.columns.is_unique:
        return df
    log.warning("duplicate gene identifiers after version stripping; summing their counts.")
    return df.T.groupby(level=0, sort=False).sum().T


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") or path.name.lower().endswith((".tsv.gz", ".txt.gz")) else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def _read_pheno(pheno: str | Path | pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = pheno.copy() if isinstance(pheno, pd.DataFrame) else _read_table_noindex(Path(pheno))
    if id_col not in df.columns:
        raise ValueError(
            f"phenotype table has no {id_col!r} column (columns: {list(df.columns)}); "
            "name the sample identifier column 'sample_id' or pass --pheno-id-col."
        )
    df[id_col] = df[id_col].astype(str)
    if df[id_col].duplicated().any():
        raise ValueError(f"duplicate values in phenotype column {id_col!r}; each bulk sample needs one row.")
    return df.set_index(id_col, drop=True)


def _read_table_noindex(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def _finalize_pheno(
    pheno: pd.DataFrame, outcome: Optional[str], patient_col: Optional[str], dataset_col: Optional[str]
) -> tuple[pd.DataFrame, str]:
    cols = set(pheno.columns)
    if outcome is None:
        outcome = "survival" if {"time", "event"} <= cols and "outcome" not in cols else "binary"
    out = pd.DataFrame(index=pheno.index)
    if outcome == "binary":
        if "outcome" not in cols:
            raise ValueError("binary outcome needs an 'outcome' column coded 0 (reference) / 1 (case).")
        vals = pheno["outcome"]
        if vals.dtype == bool:
            vals = vals.astype(int)
        vals = pd.to_numeric(vals, errors="coerce")
        if vals.isna().any() or not set(vals.unique()) <= {0, 1}:
            raise ValueError("'outcome' must be coded 0/1 with no missing values; recode the phenotype table.")
        if vals.nunique() < 2:
            raise ValueError("'outcome' has a single class; both 0 and 1 are required.")
        out["outcome"] = vals.astype(int)
    elif outcome == "survival":
        if not {"time", "event"} <= cols:
            raise ValueError("survival outcome needs 'time' and 'event' columns.")
        t = pd.to_numeric(pheno["time"], errors="coerce")
        e = pd.to_numeric(pheno["event"], errors="coerce")
        if t.isna().any() or e.isna().any() or (t < 0).any() or not set(e.unique()) <= {0, 1}:
            raise ValueError("'time' must be non-negative and 'event' 0/1, with no missing values.")
        out["time"], out["event"] = t.astype(float), e.astype(int)
    else:
        raise ValueError(f"outcome must be 'binary' or 'survival', got {outcome!r}.")
    if patient_col and patient_col in cols:
        out["patient_id"] = pheno[patient_col].astype(str)
    if dataset_col and dataset_col in cols:
        out["dataset"] = pheno[dataset_col].astype(str)
    return out, outcome


def _orient(df: pd.DataFrame, sample_ids: set[str]) -> pd.DataFrame:
    """Return samples x genes, detecting orientation by overlap with phenotype ids."""
    idx_hits = len(set(map(str, df.index)) & sample_ids)
    col_hits = len(set(map(str, df.columns)) & sample_ids)
    if idx_hits == 0 and col_hits == 0:
        raise ValueError(
            "no expression row or column labels match the phenotype 'sample_id' values; "
            "check that sample identifiers agree between the two files."
        )
    return df if idx_hits > col_hits else df.T


def save_bulk_npz(bulk: BulkData, path: str | Path) -> Path:
    """Write the BulkData in the run-directory npz format (no pickles)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=bulk.counts,
        genes=np.asarray(bulk.genes, dtype=str),
        sample_ids=np.asarray(bulk.sample_ids, dtype=str),
        pheno_json=np.asarray(bulk.pheno.to_json(orient="split")),
        outcome_type=np.asarray(bulk.outcome_type),
        already_log=np.asarray(bulk.already_log),
    )
    return path


def load_bulk_npz(path: str | Path) -> BulkData:
    """Read the npz written by :func:`save_bulk_npz`."""
    z = np.load(path, allow_pickle=False)
    for key in ("counts", "genes", "sample_ids", "pheno_json"):
        if key not in z.files:
            raise ValueError(f"{path} lacks array {key!r}; expected a degas3 bulk npz (counts, genes, sample_ids, pheno_json).")
    from io import StringIO

    sample_ids = [str(s) for s in z["sample_ids"]]
    pheno = pd.read_json(StringIO(str(z["pheno_json"])), orient="split")
    pheno.index = [str(i) for i in pheno.index]
    return BulkData(
        counts=np.asarray(z["counts"], dtype=np.float64),
        genes=[str(g) for g in z["genes"]],
        sample_ids=sample_ids,
        pheno=pheno.loc[sample_ids],
        outcome_type=str(z["outcome_type"]) if "outcome_type" in z.files else _finalize_pheno(pheno, None, None, None)[1],
        already_log=bool(z["already_log"]) if "already_log" in z.files else False,
    )


def load_bulk(
    expr_path: str | Path,
    pheno_path: str | Path | pd.DataFrame | None = None,
    *,
    outcome: Optional[str] = None,
    already_log: bool = False,
    pheno_id_col: str = "sample_id",
    patient_col: Optional[str] = "patient_id",
    dataset_col: Optional[str] = "dataset",
) -> BulkData:
    """Load bulk expression (csv/tsv[.gz], h5ad, npz) and its phenotype table.

    CSV/TSV orientation (genes x samples or samples x genes) is detected from overlap with
    the phenotype ``sample_id`` values. h5ad uses ``obs_names`` as samples and, when no
    phenotype file is given, ``obs`` as the phenotype table. Gene identifiers lose any
    Ensembl version suffix; duplicate genes are summed. Samples absent from either file
    are dropped with a warning.
    """
    expr_path = Path(expr_path)
    if not expr_path.exists():
        raise ValueError(f"bulk expression file {expr_path} does not exist.")
    suffix = expr_path.suffix.lower()
    if suffix == ".npz":
        bulk = load_bulk_npz(expr_path)
        if pheno_path is None:
            return bulk
        expr = pd.DataFrame(bulk.counts, index=bulk.sample_ids, columns=bulk.genes)
    elif suffix == ".h5ad":
        import anndata as ad

        a = ad.read_h5ad(expr_path)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        expr = pd.DataFrame(x, index=a.obs_names.astype(str), columns=a.var_names.astype(str))
        if pheno_path is None:
            pheno_path = a.obs.reset_index().rename(columns={a.obs.index.name or "index": pheno_id_col})
    else:
        expr = _read_table(expr_path)
    if pheno_path is None:
        raise ValueError("a phenotype table is required (pheno CSV with sample_id and outcome or time,event).")
    pheno_df = _read_pheno(pheno_path, pheno_id_col)
    expr.index = expr.index.map(str)
    expr.columns = expr.columns.map(str)
    expr = _orient(expr, set(pheno_df.index))
    shared = [s for s in expr.index if s in pheno_df.index]
    dropped = len(expr.index) - len(shared), len(pheno_df.index) - len(shared)
    if dropped != (0, 0):
        log.warning("dropping %d expression-only and %d phenotype-only samples.", *dropped)
    if len(shared) < 4:
        raise ValueError(f"only {len(shared)} samples overlap between expression and phenotype; need at least 4.")
    expr = expr.loc[shared]
    expr.columns = [strip_ensembl_version(g) for g in expr.columns]
    expr = _collapse_duplicate_genes(expr)
    pheno, outcome = _finalize_pheno(pheno_df.loc[shared], outcome, patient_col, dataset_col)
    counts = expr.to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("bulk expression contains NaN or negative values; supply raw counts (or log2(count+1) with already_log).")
    if not already_log and not np.allclose(counts, np.round(counts)):
        log.warning("bulk values are not integers; if they are log2(count+1), set already_log (--already-log).")
    return BulkData(counts, list(expr.columns), shared, pheno, outcome, already_log)
