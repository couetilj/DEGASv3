"""Regression tests for the integration pass: selection keys, MSigDB versions, CLI flags, robustness."""

import json

import numpy as np
import pandas as pd
import pytest

from degas3 import cli
from degas3.config import RunConfig
from degas3.pathways import latest_msigdb_version, resolve_gene_sets
from degas3.validate import read_selected_sizes


def _run(tmp_path, payload):
    root = tmp_path / "r"
    (root / "cv").mkdir(parents=True)
    if payload is not None:
        (root / "cv" / "selection.json").write_text(json.dumps(payload))
    return root


def test_read_selected_sizes_keys(tmp_path):
    assert read_selected_sizes(_run(tmp_path, {"selected": [40, 10], "selected_sizes": [1]})) == [10, 40]
    assert read_selected_sizes(_run(tmp_path / "b", {"selected_sizes": [20]})) == [20]
    assert read_selected_sizes(_run(tmp_path / "c", {"selected": []}), required=False) == []
    with pytest.raises(ValueError, match="no 'selected' sizes"):
        read_selected_sizes(_run(tmp_path / "d", {"selected": []}))
    with pytest.raises(ValueError, match="not found"):
        read_selected_sizes(_run(tmp_path / "e", None))


def test_latest_msigdb_version_from_dataframe():
    df = pd.DataFrame({"Name": ["2025.1.Hs", "2026.2.Hs", "2026.10.Hs", "2026.2.Mm", "msigdb_releases.json", "2026.3.Mm"]})
    assert latest_msigdb_version(df, "Hs") == "2026.10.Hs"
    assert latest_msigdb_version(df, "Mm") == "2026.3.Mm"
    with pytest.raises(RuntimeError):
        latest_msigdb_version(pd.DataFrame({"Name": ["msigdb_releases.json"]}), "Hs")


def test_msigdb_hallmark_download(tmp_path, monkeypatch):
    from degas3 import pathways

    monkeypatch.setattr(pathways, "CACHE_DIR", tmp_path / "cache")
    try:
        sets, src = resolve_gene_sets(RunConfig(msigdb_collection="h.all", species="human"), None)
    except Exception as exc:  # noqa: BLE001 - offline
        pytest.skip(f"MSigDB unavailable: {exc}")
    assert len(sets) == 50 and src["version"].endswith(".Hs") and list((tmp_path / "cache").glob("h.all.v*.gmt"))


def test_cli_new_flags_parse():
    p = cli.build_parser()
    a = p.parse_args(["init", "r", "--bulk", "b", "--pheno", "p", "--cells", "c", "--model", "class",
                      "--cv", "leave_one_dataset_out", "--feature-selection", "mixed", "--rank-scope", "sample"])
    assert (a.model, a.cv, a.feature_selection, a.rank_scope) == ("class", "leave_one_dataset_out", "mixed", "sample")
    assert p.parse_args(["fit", "r", "--task", "3"]).task == 3
    with pytest.raises(SystemExit):
        p.parse_args(["fit", "r", "--task", "3", "--all"])
    for st in ("neighborhoods", "maps", "cside", "explain", "report", "stratify", "de", "pathways"):
        assert p.parse_args([st, "r", "--force"]).force is True
    a = p.parse_args(["cside", "r", "--reference", "x.h5ad", "--reference-label", "ct", "--max-cores", "4",
                      "--fdr", "0.1", "--cell-type-threshold", "25"])
    assert (a.reference, a.reference_label, a.max_cores, a.fdr, a.cell_type_threshold) == ("x.h5ad", "ct", 4, 0.1, 25)
    a = p.parse_args(["slurm", "r", "--account", "A", "--partition", "P", "--cpu-partition", "C", "--time", "fit=01:00:00",
                      "--time", "train=02:00:00", "--venv", "v", "--modules", "m", "--explain", "--gpus", "2"])
    assert a.time == ["fit=01:00:00", "train=02:00:00"] and a.explain and a.gpus == 2 and a.cpu_partition == "C"
    a = p.parse_args(["run-all", "r", "--gmt", "g.gmt", "--skip-pathways"])
    assert a.gmt == "g.gmt" and a.skip_pathways


def test_init_model_flags_reach_config(sim_dir, tmp_path):
    root = tmp_path / "run"
    assert cli.main(["init", str(root), "--bulk", str(sim_dir / "bulk_counts.csv"), "--pheno", str(sim_dir / "bulk_pheno.csv"),
                     "--cells", str(sim_dir / "cells.h5ad"), "--donor-col", "donor", "--cluster-col", "cell_type",
                     "--model", "blank", "--feature-selection", "mixed", "--rank-scope", "sample"]) == 0
    cfg = RunConfig.load(root)
    assert (cfg.model, cfg.feature_selection, cfg.rank_scope) == ("blank", "mixed", "sample")


def test_slurm_fit_array_and_force_clears_marker(tmp_path):
    from degas3.runs import RunDir
    from degas3.slurm import build_script

    root = tmp_path / "run"
    RunConfig(final_seeds=2).save(root)
    s = build_script(root, account="a", partition="p")
    assert "fit" in s and "--task" not in s.split("J_FIT")[1].split("\n")[0]
    (root / "cv").mkdir()
    (root / "cv" / "selection.json").write_text(json.dumps({"selected": [10, 20, 40]}))
    line = [ln for ln in build_script(root, account="a", partition="p").splitlines() if ln.startswith("J_FIT=")][0]
    assert 'submit fit gpu "0-5"' in line and "fit" in line and "SLURM_ARRAY_TASK_ID" in line

    rd = RunDir(root)
    rd.write_complete("de", {})
    cli.main(["de", str(root), "--force"])  # no stratify yet: errors, but the marker must be gone
    assert not rd.stage_complete("de")


def test_selection_figure_ticks(tmp_path):
    from degas3.runs import RunDir
    from degas3.validate import _selection_figure

    rd = RunDir(tmp_path / "r")
    sizes = [10, 20, 40, 80, 120, 500, 10000]
    table = pd.DataFrame({"size": sizes, "mean": np.linspace(0.6, 0.8, 7), "se": 0.02,
                          "selected": [False, False, True, True, True, False, False]})
    import matplotlib.pyplot as plt

    captured = {}
    orig = plt.Figure.savefig
    plt.Figure.savefig = lambda self, *a, **k: captured.setdefault("ax", self.axes[0]) and None
    try:
        _selection_figure(rd, table, {"cutoff": 0.7}, "AUROC")
    finally:
        plt.Figure.savefig = orig
    ax = captured["ax"]
    assert [int(t) for t in ax.get_xticks()][:7] == sizes
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(s) for s in sizes] and not ax.get_xticks(minor=True).size
