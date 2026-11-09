import argparse
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from degas3 import de as de_mod
from degas3.config import RunConfig, update_config_file
from degas3.de import WILCOXON_NAME, bh_adjust, cli_de
from degas3.runs import RunDir
from degas3.stratify import cli_stratify


def make_run(tmp_path, sim_dir, single_donor=False, with_label=True, models=True):
    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    shutil.copy(sim_dir / "cells.h5ad", run / "inputs" / "cells.h5ad")
    adata = ad.read_h5ad(sim_dir / "cells.h5ad")
    rng = np.random.default_rng(3)
    score = adata.obs["planted_active"].to_numpy(float) * 0.5 + rng.random(adata.n_obs) * 0.4
    df = pd.DataFrame({
        "cell_id": adata.obs_names, "donor": "d1" if single_donor else adata.obs["donor"].astype(str).to_numpy(),
        "cluster": adata.obs["cell_type"].astype(str).to_numpy(), "sample": "",
        "label": adata.obs["label"].astype(str).to_numpy() if with_label else "",
        "raw_mean": score, "ensemble_score": score})
    (run / "scores").mkdir()
    df.to_csv(run / "scores" / "cell_scores.csv.gz", index=False)
    (run / "prepared").mkdir()
    (run / "prepared" / "shared_genes.txt").write_text("\n".join(adata.var_names) + "\n")
    manifest = json.loads((sim_dir / "simulate_manifest.json").read_text())
    if models:
        (run / "cv").mkdir()
        (run / "cv" / "selection.json").write_text(json.dumps({"selected": [10]}))
        ck = run / "models" / "size_0010" / "seed_0"
        ck.mkdir(parents=True)
        torch.save({"genes": manifest["planted_genes"][:10]}, ck / "checkpoint.pt")
    update_config_file(run, min_cells=5)
    cli_stratify(argparse.Namespace(run=str(run), split="median", quantiles=None))
    return run, manifest


def run_de(run, **kw):
    ns = argparse.Namespace(run=str(run), method=kw.get("method"), engine=kw.get("engine"))
    return cli_de(ns)


@pytest.fixture(scope="module")
def pseudobulk_run(tmp_path_factory, sim_dir):
    run, manifest = make_run(tmp_path_factory.mktemp("de"), sim_dir)
    assert run_de(run) == 0
    return run, manifest


def test_bh_adjust():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    adj = bh_adjust(p)
    assert np.isnan(adj[3])
    assert np.allclose(adj[[0, 1, 2, 4]], [0.04, 0.0533333333, 0.0533333333, 0.5])


def test_pseudobulk_recovers_planted_genes(pseudobulk_run):
    run, manifest = pseudobulk_run
    c1 = run / "de" / manifest["affected_cluster"]
    meta = json.loads((c1 / "method.json").read_text())
    assert meta["method"] == "pseudobulk_deseq2" and meta["rule_outcome"]["n_eligible_donors"] >= 2
    res = pd.read_csv(c1 / "results.csv")
    assert list(res.columns) == ["gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]
    top = set(res.nsmallest(30, "pvalue")["gene"])
    planted = set(manifest["planted_genes"])
    assert len(top & planted) >= 20
    assert (res[res.gene.isin(planted)]["log2FoldChange"] > 0).mean() > 0.9
    for f in ("pseudobulk_counts.mtx", "samples.csv", "included_donors.csv", "volcano.png", "genes.txt"):
        assert (c1 / f).is_file(), f
    samples = pd.read_csv(c1 / "samples.csv")
    assert set(samples.group) == {"high", "low"} and {"donor", "n_cells", "library_size"} <= set(samples.columns)
    summ = pd.read_csv(run / "de" / "summary.csv")
    row = summ.set_index("cluster").loc[manifest["affected_cluster"]]
    assert row["n_padj_lt_0.05_up"] > 0 and (run / "de" / "complete.json").is_file()


def test_clinical_contrast_and_scatter(pseudobulk_run):
    run, manifest = pseudobulk_run
    c1 = run / "de" / manifest["affected_cluster"]
    clin = pd.read_csv(c1 / "clinical" / "results.csv")
    assert {"gene", "log2FoldChange", "padj"} <= set(clin.columns)
    cvs = pd.read_csv(c1 / "clinical_vs_score.csv")
    assert list(cvs.columns) == ["gene", "log2FC_clinical", "padj_clinical", "log2FC_score", "padj_score", "model_input"]
    assert set(cvs[cvs.model_input].gene) == set(manifest["planted_genes"][:10]) & set(cvs.gene)
    assert (c1 / "clinical_vs_score.png").is_file()
    cm = json.loads((c1 / "clinical" / "method.json").read_text())
    assert cm["status"] == "complete" and cm["inference_unit"].startswith("donor")


def test_idempotent(pseudobulk_run):
    run, _ = pseudobulk_run
    stamp = (run / "de" / "complete.json").read_text()
    assert run_de(run) == 0
    assert (run / "de" / "complete.json").read_text() == stamp


def test_wilcoxon_fallback_single_donor(tmp_path, sim_dir):
    run, manifest = make_run(tmp_path, sim_dir, single_donor=True, with_label=False, models=False)
    assert run_de(run) == 0
    c1 = run / "de" / manifest["affected_cluster"]
    meta = json.loads((c1 / "method.json").read_text())
    assert meta["method"] == WILCOXON_NAME == "Cell-level Wilcoxon"
    assert "not donor-level" in meta["inference_unit"]
    res = pd.read_csv(c1 / "results.csv")
    assert {"log2FoldChange", "auc", "pvalue", "padj", "pct_high", "pct_low"} <= set(res.columns)
    assert res.auc.between(0, 1).all()
    top = set(res.nsmallest(20, "pvalue")["gene"])
    assert len(top & set(manifest["planted_genes"])) >= 12
    assert not (c1 / "clinical").exists()
    summ = pd.read_csv(run / "de" / "summary.csv")
    assert (summ.method == WILCOXON_NAME).all()


def test_forced_pseudobulk_without_replication_is_skipped(tmp_path, sim_dir):
    run, _ = make_run(tmp_path, sim_dir, single_donor=True, with_label=False, models=False)
    assert run_de(run, method="pseudobulk_deseq2") == 0
    summ = pd.read_csv(run / "de" / "summary.csv", keep_default_na=False)
    assert summ.skipped_reason.str.contains("pseudobulk_deseq2 requested").all()


def test_edger_unavailable_falls_back(tmp_path, sim_dir, monkeypatch):
    run, manifest = make_run(tmp_path, sim_dir, with_label=False, models=False)
    monkeypatch.setattr(de_mod, "_EDGER_OK", False)
    assert run_de(run, engine="edger") == 0
    meta = json.loads((run / "de" / manifest["affected_cluster"] / "method.json").read_text())
    assert meta["status"] == "edger_unavailable" and meta["engine_used"] == "pydeseq2"
    assert (run / "de" / manifest["affected_cluster"] / "pseudobulk_counts.mtx").is_file()


def test_requires_stratify(tmp_path):
    with pytest.raises(ValueError, match="stratify"):
        run_de(tmp_path)


def test_constant_label_skips_clinical_contrast(tmp_path, sim_dir):
    run, _ = make_run(tmp_path, sim_dir, models=False)
    p = run / "scores" / "cell_scores.csv.gz"
    df = pd.read_csv(p)
    df["label"] = 1
    df.to_csv(p, index=False)
    assert run_de(run) == 0
    done = json.loads((run / "de" / "complete.json").read_text())
    assert done["clinical_contrast"] is False and "same clinical label" in done["clinical_skipped_reason"]
    assert not list((run / "de").glob("*/clinical"))
