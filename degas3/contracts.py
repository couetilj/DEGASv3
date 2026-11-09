"""Shared data containers and the run-directory layout used by every stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd

OutcomeType = Literal["binary", "survival"]
Modality = Literal["sc", "st_cell", "st_spot"]


@dataclass
class BulkData:
    """Bulk RNA-seq profiles with a patient-level outcome.

    ``counts`` is samples x genes. It holds raw counts unless ``already_log`` is
    True, in which case values are log2(count + 1). ``pheno`` is indexed by
    ``sample_ids`` and has ``outcome`` (0/1) or ``time`` and ``event``; the
    optional columns are ``patient_id`` and ``dataset``.
    """

    counts: np.ndarray
    genes: list[str]
    sample_ids: list[str]
    pheno: pd.DataFrame
    outcome_type: OutcomeType
    already_log: bool = False

    def __post_init__(self) -> None:
        if self.counts.shape != (len(self.sample_ids), len(self.genes)):
            raise ValueError(
                f"counts shape {self.counts.shape} does not match "
                f"{len(self.sample_ids)} samples x {len(self.genes)} genes; "
                "transpose the matrix or fix the id lists."
            )


@dataclass
class CellData:
    """Thin wrapper over an AnnData of cells or spots (cells x genes).

    Raw counts live in ``adata.layers["counts"]`` when present, otherwise in
    ``adata.X``. ``var_names`` are gene identifiers without version suffixes.
    The ``*_col`` fields name ``adata.obs`` columns and are None when absent.
    ``label_col`` is an optional donor-level clinical label (0/1), not a score.
    """

    adata: "anndata.AnnData"  # noqa: F821
    donor_col: Optional[str] = None
    cluster_col: Optional[str] = None
    label_col: Optional[str] = None
    modality: Modality = "sc"
    spatial_key: Optional[str] = None
    sample_col: Optional[str] = None


@dataclass(frozen=True)
class RunLayout:
    """Directory names below a run root. Stage modules must use these constants."""

    CONFIG: str = "config.json"
    DOCTOR: str = "doctor.json"
    INPUTS: str = "inputs"
    CLUSTER: str = "cluster"
    PREPARED: str = "prepared"
    CV: str = "cv"
    MODELS: str = "models"
    SCORES: str = "scores"
    STRATA: str = "strata"
    DE: str = "de"
    PATHWAYS: str = "pathways"
    EXPLAIN: str = "explain"
    SPATIAL: str = "spatial"
    CSIDE: str = "cside"
    REPORT: str = "report"
    COMPLETE: str = "complete.json"
    BULK_NPZ: str = "bulk.npz"
    CELLS_H5AD: str = "cells.h5ad"
    INPUT_MANIFEST: str = "input_manifest.json"


LAYOUT = RunLayout()

#: Stages in execution order. ``train`` runs per task; ``select`` closes ``cv/``.
STAGES: list[str] = [
    "init",
    "cluster",
    "prepare",
    "train",
    "select",
    "fit",
    "score",
    "stratify",
    "de",
    "pathways",
    "explain",
    "neighborhoods",
    "maps",
    "cside",
    "report",
]

#: Directory that holds each stage's outputs (relative to the run root).
STAGE_DIRS: dict[str, str] = {
    "init": LAYOUT.INPUTS,
    "cluster": LAYOUT.CLUSTER,
    "prepare": LAYOUT.PREPARED,
    "train": LAYOUT.CV,
    "select": LAYOUT.CV,
    "fit": LAYOUT.MODELS,
    "score": LAYOUT.SCORES,
    "stratify": LAYOUT.STRATA,
    "de": LAYOUT.DE,
    "pathways": LAYOUT.PATHWAYS,
    "explain": LAYOUT.EXPLAIN,
    "neighborhoods": LAYOUT.SPATIAL,
    "maps": LAYOUT.SPATIAL,
    "cside": LAYOUT.CSIDE,
    "report": LAYOUT.REPORT,
}

#: Completion-file name per stage. Stages that share a directory get distinct
#: names so one does not mark the other complete; the rest use ``complete.json``.
STAGE_COMPLETE_FILES: dict[str, str] = {
    "train": "train_complete.json",
    "select": "complete.json",
    "neighborhoods": "neighborhoods_complete.json",
    "maps": "maps_complete.json",
}
