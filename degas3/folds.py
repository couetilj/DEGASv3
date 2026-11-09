"""Patient-grouped cross-validation folds and per-fold cell exclusion.

A fold never splits a patient: all samples with the same ``patient_id`` (``sample_id`` when
that column is absent) fall on one side. Folds are deterministic given ``selection_seed``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def outcome_kind(pheno: pd.DataFrame) -> str:
    """``binary`` when ``pheno`` has an ``outcome`` column, ``survival`` when it has ``time`` and ``event``."""
    if "outcome" in pheno.columns:
        return "binary"
    if {"time", "event"} <= set(pheno.columns):
        return "survival"
    raise ValueError("phenotype needs an 'outcome' column (binary) or 'time' and 'event' columns (survival).")


def _strata(pheno: pd.DataFrame) -> np.ndarray:
    return pheno["outcome" if outcome_kind(pheno) == "binary" else "event"].to_numpy().astype(int)


def _groups(pheno: pd.DataFrame) -> np.ndarray:
    col = pheno["patient_id"] if "patient_id" in pheno.columns else pd.Series(pheno.index, index=pheno.index)
    return col.astype(str).to_numpy()


def _check_test_fold(pheno: pd.DataFrame, test: np.ndarray, name: str) -> None:
    strata = _strata(pheno)
    if len(np.unique(strata[test])) < 2:
        what = "outcome classes" if outcome_kind(pheno) == "binary" else "event and censored samples"
        raise ValueError(
            f"held-out {name} lacks one of the two {what}, so its metric is undefined; "
            "reduce 'folds', merge datasets, or add patients."
        )
    if len(np.unique(strata[~test])) < 2:
        raise ValueError(f"training set for {name} lacks one outcome class; reduce 'folds' or add patients.")


def make_folds(pheno: pd.DataFrame, cfg) -> list[dict]:
    """Cross-validation folds as ``{fold_id, name, train_idx, test_idx, heldout_dataset}`` dicts.

    ``patient_kfold``: stratified group k-fold. Patients are the groups; each patient is
    labelled by its largest outcome/event value, patients of each stratum are shuffled with
    ``cfg.selection_seed`` and dealt to the fold with the fewest samples of that stratum.
    ``leave_one_dataset_out``: one fold per level of the ``dataset`` column (needs >= 2 levels).
    Indices are positional row numbers of ``pheno``.
    """
    n = len(pheno)
    groups = _groups(pheno)
    folds: list[dict] = []
    if cfg.cv == "leave_one_dataset_out":
        if "dataset" not in pheno.columns or pheno["dataset"].nunique() < 2:
            raise ValueError("leave_one_dataset_out needs a bulk 'dataset' column with at least 2 levels; "
                             "use cv='patient_kfold' or pass --dataset-col at init.")
        ds = pheno["dataset"].astype(str).to_numpy()
        for k, lev in enumerate(sorted(set(ds))):
            test = ds == lev
            _check_test_fold(pheno, test, f"dataset {lev!r}")
            folds.append(dict(fold_id=k, name=f"fold_{k}", train_idx=np.flatnonzero(~test).tolist(),
                              test_idx=np.flatnonzero(test).tolist(), heldout_dataset=lev))
    else:
        k = int(cfg.folds)
        strata = _strata(pheno)
        uniq = pd.unique(groups)
        if len(uniq) < k:
            raise ValueError(f"only {len(uniq)} patients for {k} folds; lower the 'folds' config key.")
        glabel = {g: int(strata[groups == g].max()) for g in uniq}
        rng = np.random.default_rng(cfg.selection_seed)
        size = {g: int((groups == g).sum()) for g in uniq}
        assign: dict[str, int] = {}
        total = np.zeros(k, int)
        by_stratum = {s: np.zeros(k, int) for s in sorted(set(glabel.values()))}
        for s in by_stratum:
            members = [g for g in uniq if glabel[g] == s]
            for g in [members[i] for i in rng.permutation(len(members))]:
                f = min(range(k), key=lambda j: (by_stratum[s][j], total[j], j))
                assign[g] = f
                by_stratum[s][f] += size[g]
                total[f] += size[g]
        fold_of = np.array([assign[g] for g in groups])
        for f in range(k):
            test = fold_of == f
            _check_test_fold(pheno, test, f"fold {f}")
            folds.append(dict(fold_id=f, name=f"fold_{f}", train_idx=np.flatnonzero(~test).tolist(),
                              test_idx=np.flatnonzero(test).tolist(), heldout_dataset=None))
    for fd in folds:
        tr, te = set(groups[fd["train_idx"]]), set(groups[fd["test_idx"]])
        if tr & te:
            raise ValueError(f"patients {sorted(tr & te)[:5]} appear in both train and test of {fd['name']}; "
                             "patient_id must identify one person per row group.")
        fd["n_train"], fd["n_test"] = len(fd["train_idx"]), len(fd["test_idx"])
    if sorted(i for fd in folds for i in fd["test_idx"]) != list(range(n)):
        raise ValueError("internal error: test folds do not partition the samples.")
    return folds


def _as_obs(adata) -> pd.DataFrame:
    return getattr(adata, "obs", adata)


def cell_exclusion_mask(adata, cfg, fold: Optional[dict], pheno: pd.DataFrame) -> np.ndarray:
    """Boolean mask over cells, True = excluded from training for this fold.

    Excludes (a) cells whose donor value equals the ``patient_id`` or ``sample_id`` of any
    held-out bulk row, and (b) cells whose dataset (``cfg.sample_col`` or an obs column named
    ``dataset``) is listed in ``cfg.paired_datasets[heldout_dataset]``. ``fold=None`` (the final
    fit) excludes nothing. Exact identifier matches only; aliases cannot be detected.
    """
    obs = _as_obs(adata)
    mask = np.zeros(len(obs), dtype=bool)
    if fold is None:
        return mask
    test = pheno.iloc[list(fold["test_idx"])]
    held = set(test.index.astype(str))
    if "patient_id" in test.columns:
        held |= set(test["patient_id"].astype(str))
    if cfg.donor_col and cfg.donor_col in obs.columns:
        mask |= obs[cfg.donor_col].astype(str).isin(held).to_numpy()
    ds = fold.get("heldout_dataset")
    paired = cfg.paired_datasets.get(ds) if ds is not None else None
    if paired is not None:
        paired = {paired} if isinstance(paired, str) else set(map(str, paired))
        col = cfg.sample_col if cfg.sample_col and cfg.sample_col in obs.columns else ("dataset" if "dataset" in obs.columns else None)
        if col is None:
            raise ValueError("paired_datasets is set but the cells have no sample_col or 'dataset' column to match; "
                             "set sample_col or add a 'dataset' column to the cell metadata.")
        mask |= obs[col].astype(str).isin(paired).to_numpy()
    return mask
