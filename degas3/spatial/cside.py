"""RCTD -> C-SIDE export, optional run, and result parsing for spot-resolution ST (``st_spot``).

Pipeline: export an SC reference (<= 1000 cells per cluster) and each section as MatrixMarket/CSV,
export the per-section score (``ensemble_score`` min-max scaled to [0, 1] within the section) as
the C-SIDE explanatory variable, and, when ``Rscript`` and ``spacexr`` are installed, run
``r/rctd_cside.R`` and parse its tables.

What the outputs are and are not: RCTD deconvolves each spot into cell-type weights; C-SIDE then
tests, per cell type, whether expression changes with the explanatory variable. The statistics are
spacexr's, estimated on spots that mix several cell types, and the explanatory variable is a model
percentile (scaled within section), so each result is an association with the score, not an effect
of the clinical outcome and not a cell-level measurement. Nothing is simulated: without R/spacexr
only the export is produced and the status is ``exported_only``.

Reference location: ``args.reference`` (``--reference``) or environment variable
``DEGAS3_CSIDE_REFERENCE``. The reference label column is ``args.reference_label`` /
``DEGAS3_CSIDE_REFERENCE_LABEL``, else ``cluster_col`` from the run config, else ``cell_type``.
Optional ``args.max_cores`` (default min(8, CPUs)), ``args.fdr`` (0.05) and
``args.cell_type_threshold`` (10, the spacexr default).
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

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as spio
from scipy import sparse

from degas3.config import RunConfig
from degas3.io.bulk import strip_ensembl_version
from degas3.io.sc import assert_integer_counts, get_counts
from degas3.plotting import style
from degas3.runs import RunDir
from degas3.spatial import read_cell_scores, read_spatial_cells, safe_name, sample_order, spatial_modality

log = logging.getLogger(__name__)

MAX_CELLS_PER_CLUSTER = 1000
MIN_CELLS_PER_CLUSTER = 25  # RCTD CELL_MIN_INSTANCE
R_SCRIPT = Path(__file__).resolve().parents[2] / "r" / "rctd_cside.R"
INSTALL_HINT = 'Rscript -e \'install.packages("devtools"); devtools::install_github("dmcable/spacexr")\''


# -- helpers -------------------------------------------------------------------------------
def minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] within one section; a constant vector maps to 0.5 (no information)."""
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo <= 0:
        log.warning("constant scores in a section; explanatory variable set to 0.5.")
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def write_mtx(path: Path, matrix: sparse.spmatrix) -> None:
    """Write a genes x cells integer MatrixMarket file."""
    coo = sparse.coo_matrix(matrix).astype(np.int64)
    spio.mmwrite(str(path), coo, field="integer")


def subsample_reference(labels: pd.Series, per_cluster: int, min_cells: int, seed: int) -> np.ndarray:
    """Positions of reference cells to keep: NA labels dropped, <= ``per_cluster`` per label, labels
    with fewer than ``min_cells`` cells dropped (RCTD cannot use them)."""
    rng = np.random.default_rng(seed)
    lab = labels.astype("object")
    keep: list[np.ndarray] = []
    for cl in sorted(lab.dropna().astype(str).unique()):
        pos = np.flatnonzero((lab == cl).to_numpy())
        if len(pos) < min_cells:
            log.warning("reference cluster %r has %d cells (< %d); dropped from the RCTD reference.", cl, len(pos), min_cells)
            continue
        keep.append(np.sort(rng.choice(pos, size=min(per_cluster, len(pos)), replace=False)))
    if not keep:
        raise ValueError(f"no reference cluster has >= {min_cells} cells; supply a reference with larger clusters.")
    return np.concatenate(keep)


def _reference_path(args: argparse.Namespace) -> Path:
    ref = getattr(args, "reference", None) or os.environ.get("DEGAS3_CSIDE_REFERENCE")
    if not ref:
        raise ValueError(
            "cside needs a single-cell reference with cluster labels: pass --reference PATH.h5ad "
            "or set the environment variable DEGAS3_CSIDE_REFERENCE. (The run's own cells are spots, "
            "which cannot serve as the RCTD reference.)"
        )
    p = Path(ref).expanduser()
    if not p.is_file():
        raise ValueError(f"reference {p} does not exist; give the path of an .h5ad with raw counts and cluster labels.")
    return p


def export_reference(ref_path: Path, label_col: Optional[str], cfg: RunConfig, out: Path, spot_genes: list[str],
                     seed: int) -> tuple[list[str], dict]:
    """Write ``reference/{matrix.mtx, genes.csv, barcodes.csv, cell_types.csv}``; returns (shared genes, info)."""
    ref = ad.read_h5ad(ref_path)
    label = label_col or os.environ.get("DEGAS3_CSIDE_REFERENCE_LABEL") or cfg.cluster_col or "cell_type"
    if label not in ref.obs.columns:
        raise ValueError(f"reference label column {label!r} not in the reference metadata {list(ref.obs.columns)}; "
                         "pass --reference-label or set DEGAS3_CSIDE_REFERENCE_LABEL.")
    counts = get_counts(ref)
    assert_integer_counts(counts, "reference counts")
    names = [strip_ensembl_version(g) for g in ref.var_names]
    if len(set(names)) != len(names):
        raise ValueError("reference has duplicate gene identifiers after version stripping; collapse them first.")
    index = {g: i for i, g in enumerate(names)}
    shared = [g for g in spot_genes if g in index]
    if len(shared) < 50:
        raise ValueError(f"only {len(shared)} genes shared between reference and spots; check gene identifiers (symbols vs Ensembl).")
    pos = subsample_reference(ref.obs[label], MAX_CELLS_PER_CLUSTER, MIN_CELLS_PER_CLUSTER, seed)
    X = sparse.csr_matrix(counts)[pos][:, [index[g] for g in shared]]
    barcodes = ref.obs_names.to_numpy()[pos].astype(str)
    out.mkdir(parents=True, exist_ok=True)
    write_mtx(out / "matrix.mtx", X.T)
    pd.DataFrame({"gene": shared}).to_csv(out / "genes.csv", index=False)
    pd.DataFrame({"barcode": barcodes}).to_csv(out / "barcodes.csv", index=False)
    ct = ref.obs[label].astype(str).to_numpy()[pos]
    pd.DataFrame({"barcode": barcodes, "cell_type": ct}).to_csv(out / "cell_types.csv", index=False)
    info = {"reference": str(ref_path), "label_column": label, "n_cells": int(len(pos)), "n_genes": len(shared),
            "cells_per_cell_type": pd.Series(ct).value_counts().sort_index().to_dict()}
    return shared, info


def export_sections(adata: ad.AnnData, samples: np.ndarray, xy: np.ndarray, ens: np.ndarray, genes: list[str],
                    out: Path) -> dict[str, str]:
    """Write per-section MTX/genes/coords and the explanatory variable; returns {sample: directory name}."""
    gi = [adata.var_names.get_loc(g) for g in genes]
    X = sparse.csr_matrix(get_counts(adata))[:, gi]
    mapping: dict[str, str] = {}
    for name in sample_order(samples):
        sec = safe_name(name)
        if sec in mapping.values():
            raise ValueError(f"sample names {name!r} collide after sanitising to {sec!r}; rename the samples.")
        mapping[name] = sec
        idx = np.flatnonzero(samples == name)
        bc = adata.obs_names.to_numpy()[idx].astype(str)
        d = out / "sections" / sec
        d.mkdir(parents=True, exist_ok=True)
        write_mtx(d / "matrix.mtx", X[idx].T)
        pd.DataFrame({"gene": genes}).to_csv(d / "genes.csv", index=False)
        pd.DataFrame({"barcode": bc, "x": xy[idx, 0], "y": xy[idx, 1]}).to_csv(d / "coords.csv", index=False)
        (out / "explanatory").mkdir(exist_ok=True)
        pd.DataFrame({"barcode": bc, "value": minmax_scale(ens[idx])}).to_csv(out / "explanatory" / f"{sec}.csv", index=False)
    return mapping


# -- R ---------------------------------------------------------------------------------------
def r_status() -> dict:
    """``{"rscript": bool, "spacexr": bool}`` from probing Rscript."""
    exe = shutil.which("Rscript")
    if not exe:
        return {"rscript": False, "spacexr": False}
    try:
        res = subprocess.run([exe, "-e", "suppressMessages(library(spacexr))"], capture_output=True, text=True, timeout=300)
        return {"rscript": True, "spacexr": res.returncode == 0}
    except (OSError, subprocess.SubprocessError):
        return {"rscript": True, "spacexr": False}


def rscript_command(script: Path, export_dir: Path, out_dir: Path, max_cores: int, fdr: float, threshold: int) -> list[str]:
    return ["Rscript", str(script), str(export_dir), str(out_dir), str(max_cores), str(fdr), str(threshold)]


def _how_to_run(cmd: list[str], reason: str) -> str:
    return (
        "# Running RCTD and C-SIDE\n\n"
        f"degas3 exported the inputs but did not run C-SIDE: {reason}.\n\n"
        "1. Install R packages (once):\n\n"
        f"   {INSTALL_HINT}\n\n"
        "2. Run (arguments: export_dir out_dir max_cores fdr cell_type_threshold):\n\n"
        f"   {' '.join(cmd)}\n\n"
        "3. Re-run `degas3 cside RUN --reference ...`; if `cside/rctd/population_de` exists the tables are parsed\n"
        "   into `cside/population_de/` and `cside/summary.csv` with a volcano per cell type.\n\n"
        "C-SIDE statistics are spacexr's, estimated on spots that mix cell types, and are associations with\n"
        "the min-max scaled ensemble score of each section.\n"
    )


_EFFECT_COLS = ("log_fc", "logfc", "log_fold_change", "mean", "estimate", "beta")
_P_COLS = ("p_val", "pval", "p_value", "p")
_Q_COLS = ("q_val", "fdr", "qval", "padj", "p_adj")


def _pick(df: pd.DataFrame, options: tuple[str, ...]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[o] for o in options if o in lower), None)


def parse_population_de(rctd_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Copy spacexr tables into ``population_de/`` and write ``summary.csv`` (n_sig_genes, n_up, n_down per cell type).

    Direction is the sign of the effect column (``log_fc`` or the first recognised alternative) of
    the significant-gene table. Cell types without significant genes appear with zeros only when a
    full table exists for them.
    """
    src = rctd_dir / "population_de"
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_files = sorted(src.glob("*_pop_DE.csv")) if src.is_dir() else []
    rows = []
    cell_types = set()
    for f in sig_files:
        df = pd.read_csv(f)
        ct = str(df["cell_type"].iloc[0]) if "cell_type" in df.columns and len(df) else f.name[: -len("_pop_DE.csv")]
        cell_types.add(f.name[: -len("_pop_DE.csv")])
        eff = _pick(df, _EFFECT_COLS)
        shutil.copy(f, out_dir / f.name)
        rows.append({"cell_type": ct, "n_sig_genes": len(df),
                     "n_up": int((df[eff] > 0).sum()) if eff else np.nan,
                     "n_down": int((df[eff] < 0).sum()) if eff else np.nan})
    for f in sorted(src.glob("*_pop_DE_all.csv")) if src.is_dir() else []:
        shutil.copy(f, out_dir / f.name)
        stem = f.name[: -len("_pop_DE_all.csv")]
        if stem not in cell_types:
            df = pd.read_csv(f, nrows=1)
            rows.append({"cell_type": str(df["cell_type"].iloc[0]) if "cell_type" in df.columns else stem,
                         "n_sig_genes": 0, "n_up": 0, "n_down": 0})
    summary = pd.DataFrame(rows, columns=["cell_type", "n_sig_genes", "n_up", "n_down"]).sort_values("cell_type")
    summary.to_csv(out_dir.parent / "summary.csv", index=False)
    return summary


def volcano(df: pd.DataFrame, cell_type: str, stem: Path, fdr: float) -> Optional[list[Path]]:
    """Volcano of effect vs -log10 p for one cell type; None when the table lacks recognisable columns."""
    eff, pcol = _pick(df, _EFFECT_COLS), _pick(df, _P_COLS)
    if eff is None or pcol is None:
        log.warning("cannot draw volcano for %s: no effect/p-value columns in %s.", cell_type, list(df.columns))
        return None
    qcol = _pick(df, _Q_COLS)
    x = df[eff].to_numpy(float)
    y = -np.log10(np.clip(df[pcol].to_numpy(float), 1e-300, 1))
    sig = (df[qcol].to_numpy(float) <= fdr) if qcol else np.zeros(len(df), bool)
    fig, ax = style.square_panel(f"C-SIDE: {cell_type}", f"{eff} (association with scaled score)", f"-log10 {pcol}")
    ax.scatter(x[~sig], y[~sig], s=6, color="#BDBDBD", linewidths=0, rasterized=True)
    ax.scatter(x[sig & (x > 0)], y[sig & (x > 0)], s=8, color=style.SCORE_HIGH, linewidths=0, rasterized=True, label="up")
    ax.scatter(x[sig & (x < 0)], y[sig & (x < 0)], s=8, color=style.SCORE_LOW, linewidths=0, rasterized=True, label="down")
    ax.axvline(0, color="black", lw=0.6)
    ax.legend(frameon=False)
    fig.text(0.19, 0.06, f"spacexr statistics on mixed-cell-type spots; {int(sig.sum())} genes at FDR <= {fdr}", fontsize=8)
    return style.save_panel(fig, ax, stem)


def run_cside(run: str | Path, args: Optional[argparse.Namespace] = None, force: bool = False) -> dict:
    """Export, optionally run, and parse; returns the completion record."""
    args = args or argparse.Namespace()
    rd = RunDir(run)
    cfg = RunConfig.load(rd.root)
    spatial_modality(cfg, ("st_spot",), "cside")
    status = r_status()
    prev = rd.read_complete("cside")
    if prev and not force and (prev.get("status") == "complete" or not status["spacexr"]):
        log.info("cside already %s; skipping (use --force to redo).", prev.get("status"))
        return prev
    max_cores = int(getattr(args, "max_cores", None) or min(8, os.cpu_count() or 1))
    fdr = float(getattr(args, "fdr", None) or 0.05)
    threshold = int(getattr(args, "cell_type_threshold", None) if getattr(args, "cell_type_threshold", None) is not None else 10)

    adata, samples, xy = read_spatial_cells(rd, cfg)
    scores = read_cell_scores(rd, adata.obs_names)
    ens = scores["ensemble_score"].to_numpy(float)
    spot_genes = [strip_ensembl_version(g) for g in adata.var_names]
    if len(set(spot_genes)) != len(spot_genes):
        raise ValueError("spot data has duplicate gene identifiers; collapse them before 'degas3 init'.")
    adata.var_names = spot_genes
    ref_path = _reference_path(args)

    cdir = rd.ensure("cside")
    export = cdir / "export"
    genes, ref_info = export_reference(ref_path, getattr(args, "reference_label", None), cfg, export / "reference",
                                       spot_genes, cfg.selection_seed)
    mapping = export_sections(adata, samples, xy, ens, genes, export)
    (export / "export_manifest.json").write_text(json.dumps(
        {"reference": ref_info, "sections": mapping, "explanatory": "ensemble_score min-max scaled to [0,1] within section"},
        indent=2) + "\n")
    script = cdir / "rctd_cside.R"
    shutil.copy(R_SCRIPT, script)
    rctd_dir = cdir / "rctd"
    cmd = rscript_command(script, export, rctd_dir, max_cores, fdr, threshold)
    payload = {"modality": cfg.modality, "n_sections": len(mapping), "reference": ref_info, "r": status,
               "command": " ".join(cmd), "fdr": fdr, "cell_type_threshold": threshold}
    if not (status["rscript"] and status["spacexr"]):
        reason = "Rscript was not found" if not status["rscript"] else "the R package spacexr is not installed"
        (cdir / "HOW_TO_RUN.md").write_text(_how_to_run(cmd, reason))
        log.warning("cside: %s; export written, see %s", reason, cdir / "HOW_TO_RUN.md")
        payload["status"] = "exported_only"
        rd.write_complete("cside", payload)
        return payload

    with open(cdir / "rctd.log", "w") as logfh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            logfh.write(line)
            logfh.flush()
            log.info("R: %s", line.rstrip())
        code = proc.wait()
    if code != 0:
        tail = "".join((cdir / "rctd.log").read_text().splitlines(keepends=True)[-15:])
        raise ValueError(f"Rscript failed with exit code {code}; see {cdir / 'rctd.log'}.\n{tail}")
    summary = parse_population_de(rctd_dir, cdir / "population_de")
    for f in sorted((cdir / "population_de").glob("*_pop_DE_all.csv")) or sorted((cdir / "population_de").glob("*_pop_DE.csv")):
        df = pd.read_csv(f)
        ct = str(df["cell_type"].iloc[0]) if "cell_type" in df.columns and len(df) else f.stem
        volcano(df, ct, cdir / "population_de" / f"volcano_{safe_name(ct)}", fdr)
    payload.update({"status": "complete", "summary": summary.to_dict(orient="records")})
    rd.write_complete("cside", payload)
    return payload


def cli_cside(args: argparse.Namespace) -> int:
    run_cside(args.run, args, force=bool(getattr(args, "force", False)))
    return 0
