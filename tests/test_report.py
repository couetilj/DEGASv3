import base64
import json
import re

import numpy as np
import pandas as pd
import pytest

from degas3 import cli
from degas3.intake import init_run
from degas3.report import build_report, methods_text
from degas3.runs import RunDir

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def run(sim_dir, tmp_path):
    root = tmp_path / "run"
    init_run(root, bulk=sim_dir / "bulk_counts.csv", pheno=sim_dir / "bulk_pheno.csv", cells=sim_dir / "cells.h5ad",
             donor_col="donor", cluster_col="cell_type", label_col="label", dataset_col="dataset")
    return root


def _full(root):
    rd = RunDir(root)
    (root / "cv").mkdir()
    pd.DataFrame({"fold_id": [0, 1, 0, 1], "heldout_dataset": ["a", "b", "a", "b"], "size": [10, 10, 20, 20],
                  "n_test": [40, 41, 40, 41], "auroc": [0.7, 0.8, 0.85, 0.9],
                  "average_precision": [0.6, 0.7, 0.8, 0.85]}).to_csv(root / "cv" / "metrics.csv", index=False)
    (root / "cv" / "selection.json").write_text(json.dumps(
        {"selected": [10, 20], "selected_sizes": [10, 20], "best_size": 20, "metric": "AUROC", "cutoff": 0.7, "se_best": 0.0512}))
    (root / "cv" / "selection.png").write_bytes(PNG)
    rd.write_complete("select", {})
    (root / "scores").mkdir()
    (root / "scores" / "violins_by_cluster.png").write_bytes(PNG)
    (root / "scores" / "score_summary.json").write_text(json.dumps({"n_cells": 800}))
    pd.DataFrame({"sample_id": ["a"], "score": [0.3]}).to_csv(root / "scores" / "patient_scores.csv", index=False)
    rd.write_complete("score", {})
    (root / "strata").mkdir()
    pd.DataFrame({"cluster": ["c0", "c1"], "cutoff": [0.5, 0.4]}).to_csv(root / "strata" / "cutoffs.csv", index=False)
    pd.DataFrame({"cluster": ["c0", "c1"], "donor": ["d1", "d1"], "n_high": [30, 30], "n_low": [30, 2]}).to_csv(
        root / "strata" / "coverage.csv", index=False)
    rd.write_complete("stratify", {})
    for c, m in (("c0", "pseudobulk_deseq2"), ("c1", "wilcoxon")):
        d = root / "de" / c
        d.mkdir(parents=True)
        (d / "method.json").write_text(json.dumps({"method": m}))
        pd.DataFrame({"gene": [f"G<{i}>" for i in range(40)], "log2FoldChange": np.linspace(-2, 2, 40),
                      "pvalue": 0.01, "padj": 0.02}).to_csv(d / "results.csv", index=False)
        (d / "volcano.png").write_bytes(PNG)
    pd.DataFrame({"cluster": ["c0", "c1"], "method": ["pseudobulk_deseq2", "wilcoxon"]}).to_csv(root / "de" / "summary.csv", index=False)
    rd.write_complete("de", {})


def test_report_minimal(run):
    html_path = build_report(run)
    text = html_path.read_text()
    assert "<h2>Overview</h2>" in text and "<h2>Inputs</h2>" in text and "<h2>Limitations</h2>" in text
    assert "<h2>Validation</h2>" not in text and "<h2>Differential expression</h2>" not in text
    assert "http://" not in text.replace("http://www.w3.org", "") and "<script" not in text
    man = json.loads((run / "report" / "manifest.json").read_text())
    assert len(man["input_hashes"]["bulk"]) == 64 and len(man["config_sha256"]) == 64
    assert man["package_version"] and "python" in man["environment"]
    assert (run / "report" / "METHODS.md").read_text().startswith("# Methods")
    assert RunDir(run).stage_complete("report")


def test_report_full(run):
    _full(run)
    text = build_report(run).read_text()
    for h in ("Validation", "Scores", "Stratification", "Differential expression", "Limitations", "Provenance"):
        assert f"<h2>{h}</h2>" in text
    assert text.count("data:image/png;base64,") >= 4
    assert "Selected sizes:</b> [10, 20]" in text and "optimistic" in text
    assert "SE 0.051" in text and "cutoff 0.700" in text.replace("0.7,", "0.700,") or "cutoff 0.7" in text
    val = text[text.index("<h2>Validation</h2>"):text.index("<h2>Scores</h2>")]
    head = val[val.index("<thead>"):val.index("</thead>")]
    assert "auroc" in head and "n folds" in head and "n_test" not in head and "fold_id" not in head
    assert "0.750 ± 0.071" in val and "<details><summary>Per-fold metrics</summary>" in val
    assert "Cell-level Wilcoxon was used in clusters c1" in text
    assert "Cluster c1: only 0 donor(s)" in text and "Cluster c0: only 1 donor(s)" in text
    assert "Percentile scores are ranks" in text
    assert "G&lt;" in text and "G<0>" not in text  # escaped
    assert sorted(p.name for p in (run / "report" / "figures").iterdir())
    methods = (run / "report" / "METHODS.md").read_text()
    assert "[10, 20]" in methods and "Muzellec" in methods
    assert "Johnson TS" in methods and "Lundberg" not in methods  # explain did not run
    man = json.loads((run / "report" / "manifest.json").read_text())
    assert {"select", "score", "stratify", "de"} <= set(man["stages"]) and man["selected_sizes"] == [10, 20]


def test_report_via_cli_and_rebuild(run, capsys):
    assert cli.main(["report", str(run)]) == 0
    assert cli.main(["report", str(run)]) == 0  # rebuilds unconditionally
    with pytest.raises(ValueError, match="config.json"):
        build_report(run / "missing")


def test_methods_only_for_ran_stages(run):
    rd = RunDir(run)
    from degas3.config import RunConfig
    t = methods_text(rd, RunConfig.load(run), None)
    assert "## Pathways" not in t and "## Data and preprocessing" in t
