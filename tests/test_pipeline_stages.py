import json

import anndata as ad
import numpy as np
import pytest

from degas3 import cli
from degas3.config import RunConfig
from degas3.doctor import exit_code, run_checks
from degas3.intake import init_run, inspect_run
from degas3.runs import RunDir
from degas3.cluster import run_cluster


@pytest.fixture()
def run(sim_dir, tmp_path):
    root = tmp_path / "run"
    init_run(root, bulk=sim_dir / "bulk_counts.csv", pheno=sim_dir / "bulk_pheno.csv",
             cells=sim_dir / "cells.h5ad", donor_col="donor", cluster_col="cell_type",
             label_col="label", dataset_col="dataset")
    return root


def test_config_round_trip(tmp_path):
    cfg = RunConfig(donor_col="donor", sizes=[10, 20], quantiles=[0.2, 0.8])
    cfg.save(tmp_path)
    again = RunConfig.load(tmp_path)
    assert again == cfg and json.loads(cfg.to_json())["folds"] == 5
    assert RunConfig().cv_seeds == 3 and RunConfig().final_seeds == 5 and RunConfig().n_seeds == 2048
    assert RunConfig.load(tmp_path, split="quantiles").split == "quantiles"
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_dict({"nope": 1})
    with pytest.raises(ValueError, match="folds"):
        RunConfig.from_dict({"folds": "5"})
    with pytest.raises(ValueError, match="split"):
        RunConfig.from_dict({"split": "tertiles"})
    assert RunConfig(cluster_col="x").resolved_model() == "class" and RunConfig().resolved_model() == "blank"


def test_init_inspect(run):
    rd = RunDir(run)
    m = json.loads(rd.input_manifest.read_text())
    assert m["bulk"]["outcome_counts"] == {"0": 30, "1": 30}
    assert m["cells"]["n_donors"] == 8 and m["cells"]["n_clusters"] == 3
    assert m["shared_genes"]["n_shared"] == 400
    assert len(m["sources"]["bulk"]["sha256"]) == 64
    assert any("shared between bulk and cells" in w for w in m["warnings"])  # 400 < 1000
    _, text = inspect_run(run)
    assert "Warnings:" in text and "donors" in text
    with pytest.raises(ValueError, match="exists"):
        init_run(run, bulk="x", pheno="y", cells="z")


def test_cluster_is_outcome_blind_and_idempotent(run):
    info = run_cluster(run, resolution=0.5)
    rd = RunDir(run)
    adata = ad.read_h5ad(rd.cells_h5ad)
    assert "degas3_cluster" in adata.obs and "X_umap" in adata.obsm
    assert adata.obs["degas3_cluster"].nunique() == info["n_clusters"] >= 2
    assert RunConfig.load(run).cluster_col == "degas3_cluster"
    assert (rd.path("cluster", "markers.csv")).is_file() and rd.stage_complete("cluster")
    assert np.allclose(adata.X.data, np.round(adata.X.data))  # raw counts untouched
    assert run_cluster(run)["resolution"] == 0.5  # skipped, returns recorded payload
    assert json.loads(RunDir(run).input_manifest.read_text()) is not None


def test_doctor(tmp_path):
    result = run_checks(tmp_path / "r")
    assert exit_code(result) == 0
    assert (tmp_path / "r" / "doctor.json").is_file()
    assert {c["name"] for c in result["checks"]} >= {"python", "torch", "free disk"}


def test_cli_unimplemented_stage(monkeypatch, capsys):
    monkeypatch.setitem(cli.STAGE_MODULES, "report", ["degas3._does_not_exist"])
    assert cli.main(["report", "somewhere"]) == 2
    assert "stage 'report' is not implemented yet" in capsys.readouterr().err


def test_cli_all_subcommands_registered():
    parser = cli.build_parser()
    names = set(parser._subparsers._group_actions[0].choices)
    assert names >= {"doctor", "simulate", "init", "inspect", "cluster", "prepare", "train", "select", "fit",
                     "score", "stratify", "de", "pathways", "explain", "neighborhoods", "maps", "cside",
                     "report", "slurm", "run-all"}


def test_run_dir_complete(tmp_path):
    rd = RunDir(tmp_path)
    assert not rd.stage_complete("prepare")
    rd.write_complete("prepare", {"x": 1})
    rec = rd.read_complete("prepare")
    assert rec["x"] == 1 and rec["package_version"] and "completed_at" in rec
    assert rd.stage_complete("prepare") and not rd.stage_complete("select")
    assert rd.cv_task_dir(3, 1).as_posix().endswith("cv/task_003/seed_1")
