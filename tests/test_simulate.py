import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from degas3.simulate import simulate


def test_shapes_and_integers(sim_dir):
    counts = pd.read_csv(sim_dir / "bulk_counts.csv", index_col=0)
    pheno = pd.read_csv(sim_dir / "bulk_pheno.csv")
    assert counts.shape == (400, 60)
    assert set(pheno.columns) >= {"sample_id", "patient_id", "dataset", "outcome", "time", "event"}
    assert pheno["dataset"].nunique() == 2 and pheno["outcome"].sum() == 30
    assert np.allclose(counts.values, np.round(counts.values))
    cells = ad.read_h5ad(sim_dir / "cells.h5ad")
    assert cells.shape == (800, 400)
    assert cells.obs["donor"].nunique() == 8 and cells.obs["cell_type"].nunique() == 3
    assert set(cells.obs["label"].unique()) == {0, 1}
    assert np.allclose(cells.X.data, np.round(cells.X.data))
    st = ad.read_h5ad(sim_dir / "spatial_cells.h5ad")
    assert st.obs["sample"].nunique() == 2 and st.obsm["spatial"].shape[1] == 2
    vis = ad.read_h5ad(sim_dir / "visium_spots.h5ad")
    assert vis.obsm["spatial"].shape == (vis.n_obs, 2)
    assert list(st.var_names) == list(cells.var_names) == list(counts.index)


def test_planted_signal_recoverable(sim_dir):
    truth = json.loads((sim_dir / "simulate_manifest.json").read_text())
    module = truth["planted_genes"]
    counts = pd.read_csv(sim_dir / "bulk_counts.csv", index_col=0)
    pheno = pd.read_csv(sim_dir / "bulk_pheno.csv").set_index("sample_id")
    logcpm = np.log2(counts / counts.sum() * 1e6 + 1)
    score = logcpm.loc[module].mean()
    case = score[pheno.loc[score.index, "outcome"] == 1]
    ctrl = score[pheno.loc[score.index, "outcome"] == 0]
    assert stats.ttest_ind(case, ctrl).pvalue < 1e-6

    cells = ad.read_h5ad(sim_dir / "cells.h5ad")
    aff = cells[cells.obs["cell_type"] == truth["affected_cluster"]]
    X = aff.X.toarray()
    X = np.log1p(X / X.sum(1, keepdims=True) * 1e4)
    idx = [list(aff.var_names).index(g) for g in module]
    s = X[:, idx].mean(1)
    active = aff.obs["planted_active"].to_numpy(bool)
    assert stats.ttest_ind(s[active], s[~active]).pvalue < 1e-6
    other = cells[cells.obs["cell_type"] != truth["affected_cluster"]]
    assert not other.obs["planted_active"].any()


def test_deterministic(tmp_path):
    a = simulate(tmp_path / "a", seed=3, n_patients=20, n_cells=300, n_genes=320, n_donors=4)
    b = simulate(tmp_path / "b", seed=3, n_patients=20, n_cells=300, n_genes=320, n_donors=4)
    assert a["planted_genes"] == b["planted_genes"]
    assert (tmp_path / "a" / "bulk_counts.csv").read_text() == (tmp_path / "b" / "bulk_counts.csv").read_text()
