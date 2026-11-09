"""CPU tests for the spatial stages: graph geometry, calibration, pooling, scoring, maps, C-SIDE export."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import io as spio
from scipy import sparse

from degas3.config import RunConfig
from degas3.core.models import build_model, save_checkpoint
from degas3.core.options import make_options
from degas3.runs import RunDir
from degas3.spatial import cside, maps, neighborhoods as nb

REPO = Path(__file__).resolve().parents[1]


def grid_points(n: int = 20, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    if jitter:
        xy = xy + np.random.default_rng(seed).normal(0, jitter, xy.shape)
    return xy


# -- geometry ---------------------------------------------------------------------------------
def test_graph_trims_long_edges_and_stays_connected():
    xy = np.vstack([grid_points(40, jitter=0.05), [[90.0, 90.0], [91.0, 90.5]]])  # two far outliers
    graph, audit = nb.build_graph(xy)
    assert audit["n_edges_trimmed"] > 0
    assert audit["edge_length_threshold"] < 10.0  # hull edges to the outliers are gone
    assert audit["largest_component_fraction"] > 0.95
    assert audit["n_components"] >= 2  # outliers are cut off from the core
    assert graph.shape == (len(xy), len(xy))


def test_duplicate_locations_join_with_zero_edges():
    xy = grid_points(6, jitter=0.01)
    xy = np.vstack([xy, xy[:3]])  # exact duplicates
    graph, audit = nb.build_graph(xy)
    assert audit["n_duplicate_cells"] == 3
    d = nb._distance_rows(graph, np.array([len(xy) - 1]), np.inf)[0]
    assert d[2] == 0.0  # duplicate of cell 2 is at distance zero


def test_ball_counts_monotone_in_radius():
    graph, _ = nb.build_graph(grid_points(20, jitter=0.05))
    seeds = np.array([0, 55, 210, 399])
    radii = [0.5, 1.2, 2.5, 4.0]
    mem = nb.ball_membership(graph, seeds, radii)
    sizes = np.vstack([np.diff(m.indptr) for m in mem])
    assert (np.diff(sizes, axis=0) >= 0).all()
    assert (sizes[0] >= 1).all()  # the seed is always included
    assert sizes[-1].max() > sizes[0].max()


def test_calibration_recovers_k_on_regular_grid():
    graph, audit = nb.build_graph(grid_points(30, jitter=0.02))
    rng = np.random.default_rng(0)
    seeds = rng.choice(900, 200, replace=False)
    targets = [5, 10, 20]
    radii, valid = nb.calibrate_sample_radii(graph, seeds, targets, audit["median_edge_length"])
    assert (valid == len(seeds)).all() and (np.diff(radii) > 0).all()
    mem = nb.ball_membership(graph, rng.choice(900, 200, replace=False), [float(r) for r in radii])
    for k, m in zip(targets, mem):
        med = np.median(np.diff(m.indptr))
        assert 0.5 * k <= med <= 2.0 * k, (k, med)


# -- run fixture ----------------------------------------------------------------------------------
N_MODEL_GENES = 10


def make_run(tmp_path: Path, sim_dir: Path, modality: str, **cfg_over) -> RunDir:
    run = tmp_path / f"run_{modality}"
    (run / "inputs").mkdir(parents=True)
    src = "spatial_cells.h5ad" if modality == "st_cell" else "visium_spots.h5ad"
    adata = ad.read_h5ad(sim_dir / src)
    adata.write_h5ad(run / "inputs" / "cells.h5ad")
    cfg = RunConfig(modality=modality, sample_col="sample", spatial_key="spatial",
                    cluster_col="cell_type" if modality == "st_cell" else None,
                    neighborhood_targets=[5, 10], n_seeds=40, calibration_seeds=30, evaluation_seeds=30, **cfg_over)
    cfg.validate().save(run)
    genes = list(adata.var_names)
    (run / "prepared").mkdir()
    (run / "prepared" / "shared_genes.txt").write_text("\n".join(genes[:100]) + "\n")
    (run / "cv").mkdir()
    (run / "cv" / "selection.json").write_text(json.dumps({"selected": [10, 20]}))
    for size in (10, 20):
        for seed in (0, 1):
            opt = make_options("binary", False, input_shape=N_MODEL_GENES, feature_dim=4, seed=seed + size)
            save_checkpoint(build_model(opt, "cpu"), run / "models" / f"size_{size:04d}" / f"seed_{seed}" / "checkpoint.pt",
                            genes=genes[:N_MODEL_GENES], seed=seed, size=size)
    rng = np.random.default_rng(0)
    scores = pd.DataFrame({"cell_id": adata.obs_names, "donor": adata.obs["sample"].to_numpy(),
                           "cluster": adata.obs["cell_type"].to_numpy() if "cell_type" in adata.obs else "all",
                           "ensemble_score": rng.random(adata.n_obs)})
    (run / "scores").mkdir()
    scores.to_csv(run / "scores" / "cell_scores.csv.gz", index=False)
    return RunDir(run)


@pytest.fixture()
def cell_run(tmp_path, sim_dir):
    return make_run(tmp_path, sim_dir, "st_cell")


def test_pool_membership_sums_equal_raw_counts(cell_run):
    adata = ad.read_h5ad(cell_run.cells_h5ad)
    xy = np.asarray(adata.obsm["spatial"])
    graph, _ = nb.build_graph(xy)
    X = sparse.csr_matrix(adata.X)
    seeds = np.arange(0, adata.n_obs, 7)
    (M,) = nb.ball_membership(graph, seeds, [30.0])
    pooled = M @ X
    for i in (0, 5, len(seeds) - 1):
        members = M[i].indices
        np.testing.assert_allclose(np.asarray(pooled[i].todense()).ravel(), np.asarray(X[members].sum(axis=0)).ravel())
    assert pooled.sum() == pytest.approx(np.asarray(M.sum(axis=0)).ravel() @ np.asarray(X.sum(axis=1)).ravel())


def test_score_pools_finite_and_ensemble_in_unit_interval(cell_run):
    adata = ad.read_h5ad(cell_run.cells_h5ad)
    models = nb.load_models(cell_run, nb.selected_sizes(cell_run))
    pools = sparse.csr_matrix(adata.X[:50])
    raw = nb.score_pools(pools, list(adata.var_names), models)
    assert set(raw) == {10, 20}
    perc, ens = nb.ensemble_from_raw(raw)
    assert all(np.isfinite(v).all() for v in raw.values())
    assert np.isfinite(ens).all() and ens.min() > 0 and ens.max() <= 1.0


def test_score_pools_missing_gene_error(cell_run):
    models = nb.load_models(cell_run, [10])
    with pytest.raises(ValueError, match="absent"):
        nb.score_pools(sparse.csr_matrix(np.ones((3, 5))), ["a", "b", "c", "d", "e"], models)


def test_neighborhoods_stage_outputs_and_idempotence(cell_run):
    out = nb.run_neighborhoods(cell_run.root)
    assert out["status"] == "complete" and out["edge_trim_quantile"] == 0.99
    assert cell_run.stage_complete("neighborhoods")
    assert set(out["samples"]) == {"section1", "section2"}
    sdir = cell_run.path("spatial", "section1")
    for f in ("graph_audit.json", "selected_radii.csv", "evaluation_cell_counts.csv.gz", "neighborhoods.csv.gz",
              "pooled_counts.npz", "cell_counts_by_radius.png", "neighborhood_scores_r5.png", "neighborhood_scores_r10.png"):
        assert (sdir / f).is_file(), f
    df = pd.read_csv(sdir / "neighborhoods.csv.gz")
    assert set(df.target_cells) == {5, 10} and len(df) == 80
    assert df.ensemble_score.between(0, 1).all()
    assert {"raw_size_0010", "percentile_size_0020", "frac_c1"} <= set(df.columns)
    frac = df[[c for c in df.columns if c.startswith("frac_")]].sum(axis=1)
    np.testing.assert_allclose(frac, 1.0)
    radii = pd.read_csv(sdir / "selected_radii.csv")
    other = pd.read_csv(cell_run.path("spatial", "section2", "selected_radii.csv"))
    assert (radii.radius.to_numpy() == other.radius.to_numpy()).all()  # one shared radius per target
    pooled = nb.load_sparse_block(sdir / "pooled_counts.npz", "counts_k5")
    member = nb.load_sparse_block(sdir / "pooled_counts.npz", "membership_k5")
    assert pooled.shape[0] == member.shape[0] == 40
    assert member.sum(axis=1).min() >= 1
    first = cell_run.path("spatial", "section1", "neighborhoods.csv.gz").stat().st_mtime_ns
    nb.run_neighborhoods(cell_run.root)  # skipped
    assert cell_run.path("spatial", "section1", "neighborhoods.csv.gz").stat().st_mtime_ns == first


def test_neighborhoods_rejects_spots(tmp_path, sim_dir):
    run = make_run(tmp_path, sim_dir, "st_spot")
    with pytest.raises(ValueError, match="already"):
        nb.run_neighborhoods(run.root)


# -- maps ---------------------------------------------------------------------------------------------
def test_maps_cells_with_and_without_strata(cell_run):
    cells = ad.read_h5ad(cell_run.cells_h5ad)
    groups = pd.DataFrame({"cell_id": cells.obs_names, "cluster": cells.obs["cell_type"].to_numpy(),
                           "donor": cells.obs["sample"].to_numpy(),
                           "group": np.where(np.arange(cells.n_obs) % 2, "high", "low")})
    (cell_run.root / "strata").mkdir()
    groups.to_csv(cell_run.root / "strata" / "cell_groups.csv.gz", index=False)
    out = maps.run_maps(cell_run.root)
    assert out["has_groups"] and cell_run.stage_complete("maps")
    mdir = cell_run.path("spatial", "section1", "maps")
    for stem in ("score", "cluster", "group"):
        assert (mdir / f"{stem}.png").is_file() and (mdir / f"{stem}.pdf").is_file()


def test_maps_spots_without_clusters(tmp_path, sim_dir):
    run = make_run(tmp_path, sim_dir, "st_spot")
    out = maps.run_maps(run.root)
    assert not out["has_groups"]
    assert (run.path("spatial", "section1", "maps", "score.png")).is_file()


def test_thinning_is_fixed_seed():
    a, ta = maps.thin_indices(1000, 100, 3)
    b, tb = maps.thin_indices(1000, 100, 3)
    assert ta and tb and len(a) == 100 and (a == b).all()
    assert maps.thin_indices(50, 100, 3)[1] is False


# -- cside -------------------------------------------------------------------------------------------
def test_cside_export_valid_and_exported_only(tmp_path, sim_dir, monkeypatch):
    run = make_run(tmp_path, sim_dir, "st_spot")
    monkeypatch.setattr(cside, "r_status", lambda: {"rscript": False, "spacexr": False})
    monkeypatch.delenv("DEGAS3_CSIDE_REFERENCE", raising=False)
    with pytest.raises(ValueError, match="--reference"):
        cside.run_cside(run.root, argparse.Namespace())
    ref_path = sim_dir / "cells.h5ad"
    args = argparse.Namespace(reference=str(ref_path), reference_label="cell_type")
    out = cside.run_cside(run.root, args)
    assert out["status"] == "exported_only"
    assert run.read_complete("cside")["status"] == "exported_only"
    cdir = run.path("cside")
    howto = (cdir / "HOW_TO_RUN.md").read_text()
    assert "Rscript" in howto and "dmcable/spacexr" in howto and "rctd_cside.R" in howto

    ex = cdir / "export"
    genes = pd.read_csv(ex / "reference" / "genes.csv")["gene"].tolist()
    barcodes = pd.read_csv(ex / "reference" / "barcodes.csv")["barcode"].tolist()
    ct = pd.read_csv(ex / "reference" / "cell_types.csv")
    assert ct["barcode"].tolist() == barcodes and ct["cell_type"].notna().all()
    assert ct["cell_type"].value_counts().max() <= 1000
    mat = spio.mmread(ex / "reference" / "matrix.mtx").tocsr()
    assert mat.shape == (len(genes), len(barcodes))
    ref = ad.read_h5ad(ref_path)
    expect = sparse.csr_matrix(ref.X)[:, [ref.var_names.get_loc(g) for g in genes]]
    kept = [ref.obs_names.get_loc(b) for b in barcodes]
    np.testing.assert_array_equal(mat.toarray(), expect[kept].T.toarray())

    spots = ad.read_h5ad(run.cells_h5ad)
    sec = ex / "sections" / "section1"
    smat = spio.mmread(sec / "matrix.mtx").tocsr()
    coords = pd.read_csv(sec / "coords.csv")
    assert list(coords.columns) == ["barcode", "x", "y"] and smat.shape == (len(genes), len(coords))
    sub = spots[coords["barcode"].tolist()]
    np.testing.assert_array_equal(np.asarray(smat.sum(axis=0)).ravel(),
                                  np.asarray(sparse.csr_matrix(sub[:, genes].X).sum(axis=1)).ravel())
    expl = pd.read_csv(ex / "explanatory" / "section1.csv")
    assert expl["barcode"].tolist() == coords["barcode"].tolist()
    assert expl["value"].min() == 0.0 and expl["value"].max() == 1.0
    cside.run_cside(run.root, args)  # idempotent skip


def test_minmax_constant_and_subsample_rules():
    assert (cside.minmax_scale(np.ones(4)) == 0.5).all()
    labels = pd.Series(["a"] * 30 + ["b"] * 5 + [None] * 3)
    pos = cside.subsample_reference(labels, 10, 25, seed=1)
    assert len(pos) == 10 and set(labels.iloc[pos]) == {"a"}


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
def test_r_script_parses():
    res = subprocess.run(["Rscript", "-e", f'parse(file="{REPO / "r" / "rctd_cside.R"}")'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_parse_population_de_and_volcano(tmp_path):
    rctd = tmp_path / "rctd" / "population_de"
    rctd.mkdir(parents=True)
    rng = np.random.default_rng(0)
    full = pd.DataFrame({"gene": [f"g{i}" for i in range(50)], "log_fc": rng.normal(0, 1, 50),
                         "p_val": rng.random(50), "cell_type": "T"})
    full["q_val"] = full["p_val"]
    full.to_csv(rctd / "T_pop_DE_all.csv", index=False)
    full[full.p_val < 0.3].to_csv(rctd / "T_pop_DE.csv", index=False)
    summary = cside.parse_population_de(tmp_path / "rctd", tmp_path / "population_de")
    row = summary.iloc[0]
    assert row.cell_type == "T" and row.n_up + row.n_down == row.n_sig_genes
    assert (tmp_path / "summary.csv").is_file()
    paths = cside.volcano(full, "T", tmp_path / "volcano_T", 0.05)
    assert paths and paths[0].is_file()
