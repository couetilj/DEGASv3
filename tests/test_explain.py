import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

from degas3 import cli  # noqa: E402
from degas3.config import RunConfig  # noqa: E402
from degas3.core.models import build_model, save_checkpoint  # noqa: E402
from degas3.core.options import make_options  # noqa: E402
from degas3.explain import balanced_sample, run_explain, select_cells, summarize_attributions  # noqa: E402
from degas3.runs import RunDir  # noqa: E402


def make_explain_run(root, sim_dir, sizes=(10, 6), donor=True):
    root.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(sim_dir / "cells.h5ad")
    (root / "inputs").mkdir()
    adata.write_h5ad(root / "inputs" / "cells.h5ad")
    RunConfig(donor_col="donor" if donor else None, cluster_col="cell_type", explain_bootstraps=50,
              explain_reference_cells=20).save(root)
    rng = np.random.default_rng(0)
    genes = list(adata.var_names)
    for k, size in enumerate(sizes):
        opt = make_options("binary", False, input_shape=size, feature_dim=8, seed=k)
        m = build_model(opt, "cpu")
        gsel = [genes[i] for i in rng.choice(len(genes), size, replace=False)]
        save_checkpoint(m, root / "models" / f"size_{size:04d}" / "seed_0" / "checkpoint.pt", genes=gsel, seed=k, size=size)
    scores = pd.DataFrame({
        "cell_id": adata.obs_names, "donor": adata.obs["donor"].astype(str).values,
        "cluster": adata.obs["cell_type"].astype(str).values, "ensemble_score": rng.uniform(size=adata.n_obs),
    })
    (root / "scores").mkdir()
    scores.to_csv(root / "scores" / "cell_scores.csv.gz", index=False)
    RunDir(root).write_complete("score", {})
    return root


def test_balanced_sample_and_selection():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 50 + ["b"] * 5 + ["c"] * 1)
    pick = balanced_sample(groups, 6, rng)
    assert len(pick) == 6 and {"a", "b", "c"} == set(groups[pick])
    scores = np.random.default_rng(1).uniform(size=400)
    donors = np.array(list("abcd") * 100)
    sel = select_cells(scores, donors, ref_quantile=0.25, ref_cells=20, seed=0)
    cut = np.quantile(scores, 0.25)
    assert len(sel["reference"]) == 20 and (scores[sel["reference"]] < cut).all()
    assert (scores[sel["targets"]] >= cut).all() and len(sel["targets"]) <= 12


def test_summaries_donor_vs_pooled():
    v = np.zeros((6, 3))
    v[:4, 0] = 1.0  # donor a has 4 targets, b has 2
    v[4:, 0] = -1.0
    donors = np.array(["a"] * 4 + ["b"] * 2)
    out = summarize_attributions(v, ["g1", "g2", "g3"], donors, bootstraps=100, seed=0)
    assert out.loc[0, "pooled_signed"] == pytest.approx(2 / 6)
    assert out.loc[0, "donor_signed"] == pytest.approx(0.0)  # equal-donor mean of (1, -1)
    assert out.loc[0, "donor_abs"] == pytest.approx(1.0) and out.loc[0, "n_donors"] == 2
    assert out.loc[0, "donor_signed_lo"] <= out.loc[0, "donor_signed"] <= out.loc[0, "donor_signed_hi"]


def test_explain_end_to_end(tmp_path, sim_dir):
    root = make_explain_run(tmp_path / "run", sim_dir)
    run_explain(root)
    rd = RunDir(root)
    assert rd.stage_complete("explain")
    summary = pd.read_csv(root / "explain" / "summary.csv")
    done = summary[summary["status"] == "done"]
    assert len(done) == 3
    cdir = root / "explain" / done["cluster"].iloc[0]
    attr = pd.read_csv(cdir / "gene_attributions.csv")
    assert list(attr.columns) == ["gene", "pooled_signed", "pooled_abs", "donor_signed", "donor_signed_lo",
                                  "donor_signed_hi", "donor_abs", "donor_abs_lo", "donor_abs_hi", "n_donors"]
    z = np.load(cdir / "shap_values.npz")
    assert z["values"].shape == (len(z["target_cell_ids"]), len(z["genes"])) and len(z["genes"]) <= 16
    assert z["values"].shape[1] >= 10
    # residual small relative to the spread of predictions/attributions
    assert np.abs(z["residuals"]).max() < 0.05
    for f in ("targets.csv", "reference.csv", "complete.json", "top_genes_abs.png", "top_genes_signed.png"):
        assert (cdir / f).is_file()
    assert "mean raw score" in rd.read_complete("explain")["explained_quantity"]
    mtime = (root / "explain" / "summary.csv").stat().st_mtime
    run_explain(root)  # idempotent
    assert (root / "explain" / "summary.csv").stat().st_mtime == mtime


def test_explain_without_donor_column(tmp_path, sim_dir):
    root = make_explain_run(tmp_path / "run", sim_dir, sizes=(8,), donor=False)
    run_explain(root)
    cdir = next(p for p in (root / "explain").iterdir() if p.is_dir())
    attr = pd.read_csv(cdir / "gene_attributions.csv")
    assert attr["donor_abs"].isna().all() and (attr["n_donors"] == 0).all()


def test_explain_requires_scores_and_cli(tmp_path, sim_dir, monkeypatch):
    with pytest.raises(ValueError, match="degas3 score"):
        run_explain(tmp_path / "empty")
    root = make_explain_run(tmp_path / "run", sim_dir, sizes=(8,))
    assert cli.main(["explain", str(root)]) == 0


def test_missing_shap_message(monkeypatch, tmp_path):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "shap":
            raise ImportError("no shap")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ValueError, match=r"degas3\[explain\]"):
        run_explain(tmp_path)
