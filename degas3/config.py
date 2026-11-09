"""Run configuration: every method default is a key here, stored in ``<run>/config.json``."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

DEFAULT_SIZES = [10, 20, 40, 60, 120, 240, 500, 1000, 2500, 5000, 10000]

_ENUMS: dict[str, tuple[str, ...]] = {
    "outcome": ("binary", "survival"),
    "modality": ("sc", "st_cell", "st_spot"),
    "cv": ("patient_kfold", "leave_one_dataset_out"),
    "feature_selection": ("variance", "mixed"),
    "rank_scope": ("global", "sample"),
    "split": ("median", "quantiles"),
    "de_method": ("auto", "pseudobulk_deseq2", "wilcoxon"),
    "de_engine": ("pydeseq2", "edger"),
    "species": ("human", "mouse"),
    "model": ("auto", "blank", "class"),
}
# Optional string fields (default None) accept str or None.
_OPTIONAL_STR = {
    "outcome", "modality", "bulk", "pheno", "cells", "donor_col", "cluster_col",
    "label_col", "sample_col", "dataset_col", "patient_col", "spatial_key",
    "gmt_path", "gene_map",
}


@dataclass
class RunConfig:
    """All run options. ``None`` means "not set / infer at intake"."""

    # inputs (set by ``degas3 init``)
    bulk: Optional[str] = None
    pheno: Optional[str] = None
    cells: Optional[str] = None
    outcome: Optional[str] = None
    modality: Optional[str] = None
    already_log: bool = False
    donor_col: Optional[str] = None
    cluster_col: Optional[str] = None
    label_col: Optional[str] = None
    sample_col: Optional[str] = None
    dataset_col: Optional[str] = None
    patient_col: Optional[str] = None
    spatial_key: Optional[str] = None
    gene_map: Optional[str] = None
    # gene universe and model
    sizes: list = field(default_factory=lambda: list(DEFAULT_SIZES))
    model: str = "auto"
    feature_dim: int = 25
    feature_selection: str = "variance"
    marker_cells_per_cluster: int = 500
    iterations: int = 300
    batch_size: int = 200
    allow_cpu: bool = False
    # validation
    cv: str = "patient_kfold"
    folds: int = 5
    cv_seeds: int = 3
    final_seeds: int = 5
    selection_seed: int = 42
    bootstraps: int = 2000
    paired_datasets: dict = field(default_factory=dict)
    # scoring and stratification
    rank_scope: str = "global"
    split: str = "median"
    quantiles: list = field(default_factory=lambda: [0.25, 0.75])
    # clustering
    cluster_resolution: float = 1.0
    # differential expression
    de_method: str = "auto"
    de_engine: str = "pydeseq2"
    min_cells: int = 10
    # pathways
    msigdb_collection: str = "h.all"
    msigdb_version: Optional[str] = None
    species: str = "human"
    gmt_path: Optional[str] = None
    pathway_permutations: int = 1000
    pathway_seed: int = 42
    pathway_min_size: int = 15
    pathway_max_size: int = 500
    # explanations
    explain_reference_cells: int = 100
    explain_reference_quantile: float = 0.25
    explain_bootstraps: int = 2000
    explain_seed: int = 42
    # spatial
    neighborhood_targets: list = field(default_factory=lambda: [5, 10, 20, 40])
    n_seeds: int = 2048
    calibration_seeds: int = 1024
    evaluation_seeds: int = 1024

    # -- validation ----------------------------------------------------------
    def validate(self) -> "RunConfig":
        """Check types and allowed values; raise ValueError with a remedy."""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _OPTIONAL_STR:
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"config key {f.name!r} must be a string or null, got {value!r}.")
                continue
            default = f.default if f.default is not MISSING else f.default_factory()
            if f.name == "msigdb_version":
                if value is not None and not isinstance(value, str):
                    raise ValueError("config key 'msigdb_version' must be a string or null.")
                continue
            expected = type(default)
            if expected is float:
                ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            elif expected is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            else:
                ok = isinstance(value, expected)
            if not ok:
                raise ValueError(
                    f"config key {f.name!r} must be {expected.__name__}, got {type(value).__name__} ({value!r})."
                )
        for key, allowed in _ENUMS.items():
            value = getattr(self, key)
            if value is not None and value not in allowed:
                raise ValueError(f"config key {key!r} must be one of {list(allowed)}, got {value!r}.")
        for key in ("feature_dim", "iterations", "batch_size", "folds", "cv_seeds", "final_seeds",
                    "bootstraps", "min_cells", "n_seeds"):
            if getattr(self, key) < 1:
                raise ValueError(f"config key {key!r} must be >= 1.")
        if self.folds < 2:
            raise ValueError("config key 'folds' must be >= 2 for patient_kfold cross-validation.")
        if not self.sizes or any((not isinstance(s, int)) or s < 1 for s in self.sizes):
            raise ValueError("config key 'sizes' must be a non-empty list of positive integers.")
        if len(self.quantiles) != 2 or not 0 < self.quantiles[0] < self.quantiles[1] < 1:
            raise ValueError("config key 'quantiles' must be [low, high] with 0 < low < high < 1.")
        return self

    # -- serialisation --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown config keys {unknown}; remove them or check spelling against RunConfig fields."
            )
        return cls(**data).validate()

    def merged(self, **overrides: Any) -> "RunConfig":
        """New config with non-None overrides applied (None means "keep")."""
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return RunConfig.from_dict(data)

    @classmethod
    def load(cls, run: str | Path, **overrides: Any) -> "RunConfig":
        """Defaults < ``<run>/config.json`` < non-None ``overrides``."""
        path = Path(run) / "config.json"
        data = json.loads(path.read_text()) if path.is_file() else {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(data)

    def save(self, run: str | Path) -> Path:
        path = Path(run) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    def resolved_model(self, has_clusters: Optional[bool] = None) -> str:
        """``blank`` or ``class``: ``auto`` picks class when a cluster column exists."""
        if self.model != "auto":
            return self.model
        has = bool(self.cluster_col) if has_clusters is None else has_clusters
        return "class" if has else "blank"


def update_config_file(run: str | Path, **updates: Any) -> RunConfig:
    """Set keys (including None) in ``<run>/config.json`` and return the new config."""
    cfg = RunConfig.load(run)
    data = cfg.to_dict()
    data.update(updates)
    new = RunConfig.from_dict(data)
    new.save(run)
    return new
