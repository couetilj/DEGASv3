import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from degas3 import cli
from degas3.config import RunConfig
from degas3.intake import init_run
from degas3.runs import RunDir
from degas3.score import bh_adjust, label_association, percentile_rank, rank_ensemble
from degas3.validate import (auroc, average_precision, bootstrap_fold_means, brier, concordance_index,
                             log_loss, one_se_selection)

TINY = dict(sizes=[10, 20], folds=2, cv_seeds=1, final_seeds=1, iterations=20, bootstraps=50, allow_cpu=True)


def make_run(sim_dir, root, outcome="binary", **extra):
    init_run(root, bulk=sim_dir / "bulk_counts.csv", pheno=sim_dir / "bulk_pheno.csv", cells=sim_dir / "cells.h5ad",
             outcome=outcome, donor_col="donor", cluster_col="cell_type", label_col="label",
             dataset_col="dataset", patient_col="patient_id", **{**TINY, **extra})
    return root


def run_cli(*argv):
    assert cli.main([str(a) for a in argv]) == 0


# -- metric and ensemble unit tests -------------------------------------------------
def test_c_index_hand_computed():
    t, e, s = [1, 2, 3, 4], [1, 1, 0, 1], [0.9, 0.3, 0.5, 0.1]
    # comparable pairs (0,1) (0,2) (0,3) (1,2) (1,3): concordant 4 of 5
    assert concordance_index(t, e, s) == pytest.approx(0.8)
    assert concordance_index([1, 2], [1, 0], [0.4, 0.4]) == 0.5
    assert concordance_index([1, 1], [1, 1], [0.4, 0.2]) != concordance_index([1, 1], [1, 1], [0.4, 0.2])  # NaN: tied times


def test_binary_metrics_hand_computed():
    y, s = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    assert auroc(y, s) == pytest.approx(0.75)
    assert average_precision(y, s) == pytest.approx(0.8333333, abs=1e-6)  # 0.5*1 + 0.5*(2/3)
    assert brier(y, s) == pytest.approx(np.mean([0.01, 0.16, 0.4225, 0.04]))
    assert log_loss(y, s) == pytest.approx(-np.mean(np.log([0.9, 0.6, 0.35, 0.8])))
    assert auroc(y, np.zeros(4)) == 0.5


def test_one_se_selection_synthetic():
    by_fold = [np.array([0.60, 0.80, 0.79, 0.70]), np.array([0.60, 0.82, 0.77, 0.70])]
    boot = 0.81 + np.array([[0.04], [-0.04], [0.04], [-0.04]]) * np.ones((1, 4))
    sel = one_se_selection(by_fold, boot)
    se = np.std(boot[:, 1], ddof=1)
    assert sel["best"] == 1 and sel["mean"][1] == pytest.approx(0.81)
    assert sel["cutoff"] == pytest.approx(0.81 - se)
    assert sel["selected"].tolist() == [False, True, True, False]  # 0.78 >= 0.81 - 0.046
    tie = one_se_selection([np.array([0.7, 0.7])], np.zeros((3, 2)) + np.array([[0.0], [0.0], [0.0]]))
    assert tie["best"] == 0 and tie["selected"].tolist() == [True, True]


def test_bootstrap_means_cover_point_estimate():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 30)
    frames = []
    for sz, shift in ((10, 0.0), (20, 1.0)):
        s = rng.normal(0, 1, 60) + shift * y
        frames.append(pd.DataFrame({"sample_id": [f"s{i}" for i in range(60)], "outcome": y, "score": s, "size": sz}))
    fold = pd.concat(frames)
    boot = bootstrap_fold_means("binary", [fold], [10, 20], 400, 3)
    assert boot.shape == (400, 2)
    for j, fr in enumerate(frames):
        assert boot[:, j].mean() == pytest.approx(auroc(fr["outcome"], fr["score"]), abs=0.03)


def test_rank_ensemble_hand_computed_and_not_reranked():
    raw = np.array([[0.1, 5.0], [0.3, 5.0], [0.2, 1.0], [0.9, 2.0]])
    pct, ens = rank_ensemble(raw)
    assert pct[:, 0].tolist() == [0.25, 0.75, 0.5, 1.0]
    assert pct[:, 1].tolist() == [0.875, 0.875, 0.25, 0.5]  # average ties
    assert ens.tolist() == [(0.25 + 0.875) / 2, (0.75 + 0.875) / 2, (0.5 + 0.25) / 2, (1.0 + 0.5) / 2]
    assert ((ens >= 0) & (ens <= 1)).all() and not np.allclose(np.sort(ens), np.arange(1, 5) / 4)
    grouped = percentile_rank(np.array([1, 2, 3, 4.0]), ["a", "a", "b", "b"])
    assert grouped.tolist() == [0.5, 1.0, 0.5, 1.0]


def test_bh_and_label_association_na_with_few_donors():
    adj = bh_adjust(np.array([0.01, 0.04, np.nan, 0.03]))
    assert adj[0] == pytest.approx(0.03) and adj[3] == pytest.approx(0.04) and np.isnan(adj[2])
    df = pd.DataFrame({"cluster": "c", "donor": ["a", "a", "b", "c"], "label": ["1", "1", "0", "0"],
                       "ensemble_score": [0.9, 0.8, 0.2, 0.3]})
    out = label_association(df)
    assert np.isnan(out.loc[0, "p_value"]) and np.isnan(out.loc[0, "fdr"])  # one label-1 donor


# -- stage chain --------------------------------------------------------------------------
@pytest.mark.parametrize("outcome", ["binary", "survival"])
def test_full_chain(sim_dir, tmp_path, outcome):
    run = make_run(sim_dir, tmp_path / "run", outcome)
    rd = RunDir(run)
    run_cli("prepare", run)
    prep = rd.path("prepared")
    for name in ("shared_genes.txt", "folds.json", "tasks.json", "complete.json", "final_rankings.csv",
                 "cells_shared_counts.npz", "cell_metadata.csv", "rankings/fold_0.csv", "rankings/fold_1.csv"):
        assert (prep / name).is_file(), name
    folds = json.loads((prep / "folds.json").read_text())["folds"]
    assert not set(folds[0]["train_idx"]) & set(folds[0]["test_idx"]) and folds[0]["n_cells_excluded"] == 0
    tasks = json.loads((prep / "tasks.json").read_text())
    assert [(t["fold_id"], t["size"]) for t in tasks] == [(0, 10), (0, 20), (1, 10), (1, 20)]
    assert tasks[1]["genes"][:10] == tasks[0]["genes"]  # nested prefixes
    run_cli("prepare", run)  # idempotent

    run_cli("train", run, "--all", "--allow-cpu")
    for t in range(4):
        d = rd.cv_task_dir(t, 0)
        for f in ("checkpoint.pt", "losses.csv", "heldout_predictions.csv", "complete.json"):
            assert (d / f).is_file()
    assert rd.stage_complete("train")
    pred = pd.read_csv(rd.cv_task_dir(0, 0) / "heldout_predictions.csv")
    assert np.isfinite(pred["score"]).all() and len(pred) == folds[0]["n_test"]

    run_cli("select", run)
    sel = json.loads(rd.path("cv", "selection.json").read_text())
    assert sel["metric"] == ("AUROC" if outcome == "binary" else "C-index")
    assert sel["selected_sizes"] and set(sel["selected_sizes"]) <= {10, 20} and "optimistic" in sel["note"]
    m = pd.read_csv(rd.path("cv", "metrics.csv"))
    assert len(m) == 4 and np.isfinite(m[sel["metric"].lower().replace("-", "_")]).all()
    for f in ("selection.png", "selection.pdf", "complete.json", "oof_predictions.csv"):
        assert rd.path("cv", f).is_file()

    run_cli("fit", run, "--all", "--allow-cpu")
    for size in sel["selected_sizes"]:
        assert (rd.model_dir(size, 0) / "checkpoint.pt").is_file() and rd.stage_complete("fit", f"size_{size:04d}", "seed_0")

    run_cli("score", run)
    sc = rd.path("scores")
    for f in ("cell_scores.csv.gz", "patient_scores.csv", "score_summary.json", "violins_by_cluster.png",
              "label_association.csv", "complete.json"):
        assert (sc / f).is_file(), f
    cells = pd.read_csv(sc / "cell_scores.csv.gz")
    assert len(cells) == 800 and True
    pcols = [c for c in cells if c.startswith("percentile_size_")]
    assert len(pcols) == len(sel["selected_sizes"])
    scols = [c for c in cells if c.startswith(("raw_", "percentile_")) or c == "ensemble_score"]
    assert np.isfinite(cells[scols].to_numpy()).all() and cells["ensemble_score"].between(0, 1).all()
    np.testing.assert_allclose(cells["ensemble_score"], cells[pcols].mean(axis=1))
    first = pcols[0]
    raw = cells[first.replace("percentile", "raw")]
    np.testing.assert_allclose(cells[first], stats.rankdata(raw) / len(cells))
    patients = pd.read_csv(sc / "patient_scores.csv")
    assert set(patients["in_sample"]) == {True, False} and len(patients) == 120
    assert np.isfinite(patients["ensemble_score"]).all()


def test_mixed_selection_prepares_alternating_lists(sim_dir, tmp_path):
    run = make_run(sim_dir, tmp_path / "mixed", feature_selection="mixed")
    run_cli("prepare", run)
    r = pd.read_csv(RunDir(run).path("prepared", "rankings", "fold_0.csv"))
    assert r["source"].tolist() == ["bulk_variance", "sc_marker"] * 10
    assert r["cluster"][r["source"] == "sc_marker"].nunique() == 3


def test_train_requires_gpu_or_flag(sim_dir, tmp_path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    run = make_run(sim_dir, tmp_path / "nogpu", allow_cpu=False)
    run_cli("prepare", run)
    assert cli.main(["train", str(run), "--all"]) == 1


def test_donor_leakage_excluded_cells_recorded(sim_dir, tmp_path):
    run = make_run(sim_dir, tmp_path / "leak")
    rd = RunDir(run)
    import anndata as ad
    from degas3.io.bulk import load_bulk_npz, save_bulk_npz

    bulk = load_bulk_npz(rd.bulk_npz)
    pheno = bulk.pheno.copy()
    pheno["patient_id"] = [f"donor{(i % 8) + 1}" if i < 16 else p for i, p in enumerate(pheno["patient_id"])]
    bulk.pheno = pheno
    save_bulk_npz(bulk, rd.bulk_npz)
    run_cli("prepare", run)
    folds = json.loads(rd.path("prepared", "folds.json").read_text())["folds"]
    assert sum(f["n_cells_excluded"] for f in folds) > 0
    obs = ad.read_h5ad(rd.cells_h5ad).obs
    for f in folds:
        held = set(f["heldout_ids"]) & set(obs["donor"])
        excl = np.load(rd.path("prepared", f["excluded_file"]))
        assert set(obs["donor"].iloc[excl]) == held
