import argparse
import json

import numpy as np
import pandas as pd

from degas3.stratify import assign_groups, cli_stratify, stratify_scores
from degas3.runs import RunDir
from degas3.config import RunConfig


def _write_scores(run, df):
    d = run / "scores"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / "cell_scores.csv.gz", index=False)


def test_median_ties():
    g, (lo, hi) = assign_groups(np.array([1, 2, 3, 3, 3, 4, 5.0]), "median")
    assert lo == hi == 3.0
    assert list(g) == ["low", "low", "tie", "tie", "tie", "high", "high"]


def test_quantile_ties_and_middle():
    g, (lo, hi) = assign_groups(np.arange(9, dtype=float), "quantiles", (0.25, 0.75))
    assert (lo, hi) == (2.0, 6.0)
    assert list(g) == ["low", "low", "tie", "middle", "middle", "middle", "tie", "high", "high"]


def test_tables_per_cluster_and_donor():
    df = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(8)], "donor": ["a", "a", "b", "b"] * 2,
        "cluster": ["x"] * 4 + ["y"] * 4, "ensemble_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.9]})
    groups, cut, cov = stratify_scores(df, "median")
    x = cut.set_index("cluster").loc["x"]
    assert (x.n_low, x.n_high, x.n_tie, x.n_middle) == (2, 2, 0, 0)
    y = cut.set_index("cluster").loc["y"]
    assert (y.n_tie, y.n_high, y.n_low) == (3, 1, 0)  # median 0.5, ties excluded
    assert cov.n_cells.sum() == 8
    assert set(groups.group) <= {"low", "high", "tie"}


def test_cli_stratify_outputs_and_idempotent(tmp_path):
    run = tmp_path / "run"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(60)], "donor": ["d1", "d2", "d3"] * 20,
        "cluster": ["k1"] * 30 + ["k2"] * 30, "sample": "", "label": "",
        "raw_mean": 0.0, "ensemble_score": rng.random(60)})
    _write_scores(run, df)
    ns = argparse.Namespace(run=str(run), split="quantiles", quantiles=[0.25, 0.75])
    assert cli_stratify(ns) == 0
    s = run / "strata"
    for f in ("cutoffs.csv", "cell_groups.csv.gz", "coverage.csv", "complete.json", "groups_by_cluster.png"):
        assert (s / f).is_file(), f
    cut = pd.read_csv(s / "cutoffs.csv")
    assert (cut.split == "quantiles").all() and cut.n_middle.sum() > 0
    assert list(pd.read_csv(s / "cell_groups.csv.gz").columns) == ["cell_id", "cluster", "donor", "group"]
    stamp = (s / "complete.json").read_text()
    assert cli_stratify(ns) == 0
    assert (s / "complete.json").read_text() == stamp


def test_empty_cluster_and_donor_columns(tmp_path):
    run = tmp_path / "run"
    df = pd.DataFrame({"cell_id": list("abcde"), "donor": "", "cluster": "", "sample": "", "label": "",
                       "raw_mean": 0.0, "ensemble_score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    _write_scores(run, df)
    cli_stratify(argparse.Namespace(run=str(run), split=None, quantiles=None))
    cut = pd.read_csv(run / "strata" / "cutoffs.csv")
    assert cut.cluster.tolist() == ["all"] and cut.n_tie.iloc[0] == 1
