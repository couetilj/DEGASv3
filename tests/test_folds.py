import anndata as ad
import numpy as np
import pandas as pd
import pytest

from degas3.config import RunConfig
from degas3.folds import cell_exclusion_mask, make_folds


def _pheno(n=40, multi=True, survival=False):
    rng = np.random.default_rng(0)
    pid = [f"P{i // 2}" if multi else f"P{i}" for i in range(n)]
    df = pd.DataFrame({"patient_id": pid, "dataset": ["a" if (i // 2) % 2 == 0 else "b" for i in range(n)] if multi else ["a", "b"] * (n // 2)}, index=[f"S{i}" for i in range(n)])
    y = np.array([int(i // 2 % 3 == 0) for i in range(n)])  # constant within patient
    if survival:
        df["time"], df["event"] = rng.uniform(1, 10, n), y
    else:
        df["outcome"] = y
    return df


@pytest.mark.parametrize("survival", [False, True])
def test_kfold_no_patient_overlap_and_stratified(survival):
    ph = _pheno(survival=survival)
    cfg = RunConfig(folds=4)
    folds = make_folds(ph, cfg)
    assert len(folds) == 4
    assert sorted(i for f in folds for i in f["test_idx"]) == list(range(len(ph)))
    col = "event" if survival else "outcome"
    for f in folds:
        assert not set(ph.patient_id.iloc[f["train_idx"]]) & set(ph.patient_id.iloc[f["test_idx"]])
        assert ph[col].iloc[f["test_idx"]].nunique() == 2
        assert f["heldout_dataset"] is None
    pos = [ph[col].iloc[f["test_idx"]].sum() for f in folds]
    assert max(pos) - min(pos) <= 2  # two samples per patient
    assert folds == make_folds(ph, cfg)  # deterministic
    assert folds != make_folds(ph, RunConfig(folds=4, selection_seed=1))


def test_leave_one_dataset_out():
    ph = _pheno()
    folds = make_folds(ph, RunConfig(cv="leave_one_dataset_out"))
    assert [f["heldout_dataset"] for f in folds] == ["a", "b"]
    assert set(ph.dataset.iloc[folds[0]["test_idx"]]) == {"a"}
    with pytest.raises(ValueError, match="at least 2 levels"):
        make_folds(ph.drop(columns="dataset"), RunConfig(cv="leave_one_dataset_out"))


def test_cell_exclusion_by_donor_and_paired_dataset():
    ph = _pheno(multi=False)
    obs = pd.DataFrame({"donor": ["P0", "P1", "X", "S2", "Y"], "sample": ["s1", "s2", "s1", "s2", "s3"]},
                       index=[f"c{i}" for i in range(5)])
    adata = ad.AnnData(X=np.ones((5, 2)), obs=obs)
    folds = make_folds(ph, RunConfig(folds=2))
    fold = dict(folds[0], test_idx=[0, 1, 2, 3], heldout_dataset="a")
    cfg = RunConfig(donor_col="donor", sample_col="sample", paired_datasets={"a": ["s3"]})
    mask = cell_exclusion_mask(adata, cfg, fold, ph)
    # P0, P1 match patient ids; S2 matches a sample id; Y is in the paired dataset s3
    assert mask.tolist() == [True, True, False, True, True]
    assert not cell_exclusion_mask(adata, cfg, None, ph).any()
    no_pair = RunConfig(donor_col="donor")
    assert cell_exclusion_mask(adata, no_pair, fold, ph).tolist() == [True, True, False, True, False]
