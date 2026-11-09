import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from degas3.config import update_config_file
from degas3.pathways import cli_pathways, ora, read_gmt, resolve_gene_sets, ranking_from_results
from degas3.config import RunConfig
from tests.test_de import make_run, run_de

GMT = Path(__file__).parent / "fixtures" / "toy.gmt"


def test_toy_gmt():
    sets = read_gmt(GMT)
    assert len(sets) >= 20 and "TOY_PLANTED_MODULE" in sets


def test_ora_matches_scipy():
    sets = {"A": [f"g{i}" for i in range(10)], "B": [f"g{i}" for i in range(5, 25)], "C": ["g1", "g2"]}
    bg = [f"g{i}" for i in range(100)]
    lst = [f"g{i}" for i in range(8)] + ["g50", "g51", "not_in_background"]
    tab = ora(lst, bg, sets, min_size=5, max_size=500).set_index("term")
    assert set(tab.index) == {"A", "B"}  # C is below min_size
    exp = hypergeom.sf(8 - 1, 100, 10, 10)
    assert tab.loc["A", "pvalue"] == pytest.approx(exp) and tab.loc["A", "overlap"] == 8
    assert tab.loc["B", "pvalue"] == pytest.approx(hypergeom.sf(3 - 1, 100, 20, 10))
    assert tab.padj.max() <= 1 and tab.loc["A", "padj"] >= tab.loc["A", "pvalue"]


def test_ranking_rules():
    res = pd.DataFrame({"gene": ["a", "b", "a"], "stat": [2.0, -1.0, 5.0], "pvalue": [0.01, 0.5, 0.001],
                        "log2FoldChange": [1.0, -1.0, 1.0]})
    r, name = ranking_from_results(res, "pseudobulk_deseq2")
    assert name == "stat" and r.gene.is_unique and r.set_index("gene").loc["a", "score"] == 5.0
    r, name = ranking_from_results(res, "Cell-level Wilcoxon")
    assert r.set_index("gene").loc["b", "score"] == pytest.approx(np.log10(0.5))


def test_no_gmt_offline_raises(monkeypatch):
    import degas3.pathways as pw

    monkeypatch.setattr(pw, "CACHE_DIR", Path("/nonexistent/degas3_cache"))
    import gseapy

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(gseapy, "Msigdb", boom)
    with pytest.raises(ValueError, match="--gmt"):
        resolve_gene_sets(RunConfig(), None)


def test_prerank_recovers_planted_set(tmp_path, sim_dir):
    run, manifest = make_run(tmp_path, sim_dir, with_label=False, models=False)
    assert run_de(run) == 0
    update_config_file(run, pathway_permutations=200)
    ns = argparse.Namespace(run=str(run), gmt=str(GMT), collection=None, species=None)
    assert cli_pathways(ns) == 0
    c1 = manifest["affected_cluster"]
    g = pd.read_csv(run / "pathways" / c1 / "high_vs_low_gsea.csv")
    top = g.sort_values("NES", ascending=False)["Term"].head(3).tolist()
    assert "TOY_PLANTED_MODULE" in top
    assert (run / "pathways" / c1 / "ora_high.csv").is_file() and (run / "pathways" / c1 / "ora_low.csv").is_file()
    comp = json.loads((run / "pathways" / "complete.json").read_text())
    assert comp["gmt_sha256"] and comp["settings"]["permutations"] == 200
    summ = pd.read_csv(run / "pathways" / "summary.csv")
    assert (run / "pathways" / "gsea_dotplot.png").is_file() and len(summ) == 3
    ora_high = pd.read_csv(run / "pathways" / c1 / "ora_high.csv").set_index("term")
    assert ora_high.loc["TOY_PLANTED_MODULE", "padj"] < 0.05
