"""Geodesic-radius neighborhood pooling for single-cell-resolution spatial data (``st_cell``).

Per sample: Delaunay triangulation of cell centroids, long edges trimmed, shortest-path (geodesic)
radii calibrated so the median neighborhood holds a target number of cells, raw counts pooled over
the cells within each radius around sampled seed cells, and the pools scored with the final models.

What the outputs are and are not:

* A neighborhood is the set of cells within a geodesic distance of a seed *cell* (the seed is
  included). Neighborhoods around different seeds overlap, so pools are not independent samples.
* Pooled counts are sums of raw counts; they are re-normalised per pool by ``preprocess_counts``
  before scoring. A pool score is a model output for the summed profile, not an average of its
  cells' scores.
* ``ensemble_score`` of a pool is a percentile among the pools of the same sample and the same
  target size (average ties / number of pools, averaged over model sizes). The *pools* are the rank
  reference, not the cells; it is not comparable with the cell-level ``ensemble_score``.
* One radius per target is shared by all samples (median of the per-sample median radii, equal
  sample weight), so ball sizes differ between samples only through local cell density.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import rankdata

from degas3.config import RunConfig
from degas3.core.models import load_checkpoint, predict_scores
from degas3.core.preprocess import preprocess_counts
from degas3.io.sc import get_counts
from degas3.plotting import style
from degas3.runs import RunDir
from degas3.validate import read_selected_sizes
from degas3.spatial import read_json, read_spatial_cells, safe_name, sample_order, spatial_modality

log = logging.getLogger(__name__)

#: Edges longer than this quantile of positive edge lengths are removed (fixed; recorded in outputs).
EDGE_TRIM_QUANTILE = 0.99
_SCORE_CHUNK = 2048


# -- graph ---------------------------------------------------------------------------------
def build_graph(xy: np.ndarray, trim_quantile: float = EDGE_TRIM_QUANTILE) -> tuple[sparse.csr_matrix, dict]:
    """Delaunay graph on centroids with long edges removed.

    ``xy`` is (n, 2) in physical units. Cells at identical coordinates are joined to one
    representative by a zero-length edge. Edge weights are Euclidean lengths; edges longer than the
    ``trim_quantile`` quantile of the positive lengths are dropped. Returns a symmetric-use CSR
    matrix (one stored direction per edge; use ``directed=False`` in csgraph) and an audit dict.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    uniq, first_idx, inverse = np.unique(xy, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_uniq = len(uniq)
    if n_uniq < 3:
        raise ValueError(f"need at least 3 distinct cell locations for a Delaunay graph, found {n_uniq}.")
    try:
        tri = Delaunay(uniq)
    except Exception as exc:  # scipy raises QhullError (collinear or degenerate input)
        raise ValueError(f"Delaunay triangulation failed ({exc}); check that coordinates are 2-D and not collinear.") from exc
    s = tri.simplices
    pairs = np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    in_tri = np.zeros(n_uniq, dtype=bool)
    in_tri[np.unique(s)] = True
    n_omitted = int((~in_tri).sum())
    if n_omitted:  # Qhull may leave coplanar points out; attach each to its nearest triangulated location
        tree = cKDTree(uniq[in_tri])
        _, nn = tree.query(uniq[~in_tri])
        extra = np.column_stack([np.flatnonzero(~in_tri), np.flatnonzero(in_tri)[nn]])
        pairs = np.vstack([pairs, extra])
    lengths = np.linalg.norm(uniq[pairs[:, 0]] - uniq[pairs[:, 1]], axis=1)
    pos = lengths[lengths > 0]
    threshold = float(np.quantile(pos, trim_quantile)) if len(pos) else 0.0
    keep = lengths <= threshold
    rep = first_idx  # representative cell index per unique location
    rows = [rep[pairs[keep, 0]]]
    cols = [rep[pairs[keep, 1]]]
    vals = [lengths[keep]]
    dup = np.flatnonzero(rep[inverse] != np.arange(n))
    rows.append(dup)
    cols.append(rep[inverse[dup]])
    vals.append(np.zeros(len(dup)))
    graph = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    largest = int(np.bincount(labels).max())
    audit = {
        "n_cells": int(n),
        "n_unique_locations": int(n_uniq),
        "n_duplicate_cells": int(len(dup)),
        "n_edges_delaunay": int(len(pairs)),
        "n_edges_trimmed": int((~keep).sum()),
        "n_edges_kept": int(keep.sum()),
        "n_points_attached_outside_triangulation": n_omitted,
        "edge_trim_quantile": float(trim_quantile),
        "edge_length_threshold": threshold,
        "median_edge_length": float(np.median(pos)) if len(pos) else 0.0,
        "n_components": int(n_comp),
        "largest_component_cells": largest,
        "largest_component_fraction": float(largest / n),
    }
    return graph, audit


def _distance_rows(graph: sparse.csr_matrix, seeds: np.ndarray, limit: float) -> np.ndarray:
    return dijkstra(graph, directed=False, indices=np.asarray(seeds), limit=limit)


def _chunks(seeds: np.ndarray, n_nodes: int) -> Iterable[np.ndarray]:
    size = int(np.clip(20_000_000 // max(n_nodes, 1), 1, 256))
    for i in range(0, len(seeds), size):
        yield seeds[i:i + size]


def seed_kth_distances(graph: sparse.csr_matrix, seeds: np.ndarray, targets: list[int], start_limit: float) -> np.ndarray:
    """Geodesic distance from each seed to its k-th nearest cell (the seed counts as 1st, distance 0).

    Returns (len(seeds), len(targets)); NaN where the seed's component has fewer than k cells.
    The search radius starts at ``start_limit`` and doubles for seeds that have not yet reached
    ``max(targets)`` cells, finally unbounded.
    """
    kmax = max(targets)
    out = np.full((len(seeds), len(targets)), np.nan)
    pending = np.arange(len(seeds))
    limit = float(start_limit) if start_limit > 0 else np.inf
    for attempt in range(8):
        if len(pending) == 0:
            break
        if attempt == 7:
            limit = np.inf
        nxt: list[int] = []
        for sub in _chunks(pending, graph.shape[0]):
            D = _distance_rows(graph, seeds[sub], limit)
            for row, pos in zip(D, sub):
                d = np.sort(row[np.isfinite(row)])
                if len(d) >= kmax or not np.isfinite(limit):
                    for j, k in enumerate(targets):
                        if len(d) >= k:
                            out[pos, j] = d[k - 1]
                else:
                    nxt.append(int(pos))
        pending = np.array(nxt, dtype=int)
        limit = limit * 2 if np.isfinite(limit) else np.inf
    return out


def calibrate_sample_radii(graph: sparse.csr_matrix, seeds: np.ndarray, targets: list[int], median_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample radius per target: median over seeds of the distance to the k-th nearest cell.

    Returns (radii, n_valid_seeds), both of length ``len(targets)``. Radii are NaN when no seed
    reaches k cells.
    """
    start = 2.0 * np.sqrt(max(targets)) * median_edge
    d = seed_kth_distances(graph, seeds, targets, start)
    with np.errstate(all="ignore"):
        radii = np.array([np.nanmedian(d[:, j]) if np.isfinite(d[:, j]).any() else np.nan for j in range(d.shape[1])])
    return radii, np.isfinite(d).sum(axis=0)


def ball_membership(graph: sparse.csr_matrix, seeds: np.ndarray, radii: list[float]) -> list[sparse.csr_matrix]:
    """Cells within each geodesic radius of each seed (seed included): one (n_seeds, n_cells) 0/1 CSR per radius.

    Balls around different seeds overlap; membership rows are not disjoint.
    """
    rmax = float(max(radii))
    n = graph.shape[0]
    parts: list[list[sparse.csr_matrix]] = [[] for _ in radii]
    for chunk in _chunks(np.asarray(seeds), n):
        D = _distance_rows(graph, chunk, rmax)
        for j, r in enumerate(radii):
            mask = D <= r
            parts[j].append(sparse.csr_matrix(mask.astype(np.float64)))
    return [sparse.vstack(p, format="csr") if len(p) > 1 else p[0] for p in parts]


# -- scoring -------------------------------------------------------------------------------
def rank_percentile(values: np.ndarray) -> np.ndarray:
    """Average-tie rank divided by the number of values: a percentile in (0, 1] among these values."""
    return rankdata(values, method="average") / len(values)


def ensemble_from_raw(raw_by_size: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Per-size percentile ranks and their mean across sizes (the pool ``ensemble_score``)."""
    perc = {s: rank_percentile(v) for s, v in raw_by_size.items()}
    return perc, np.mean(np.vstack(list(perc.values())), axis=0)


def selected_sizes(rd: RunDir) -> list[int]:
    """Sizes in ``cv/selection.json`` (shared reader)."""
    return read_selected_sizes(rd)


def load_models(rd: RunDir, sizes: list[int]) -> dict[int, list[tuple]]:
    """Final models per selected size: ``{size: [(model, genes), ...]}`` over seeds."""
    out: dict[int, list[tuple]] = {}
    for s in sizes:
        ckpts = sorted(rd.path("models", f"size_{s:04d}").glob("seed_*/checkpoint.pt"))
        if not ckpts:
            raise ValueError(f"no checkpoints under models/size_{s:04d}/seed_*/; run 'degas3 fit --all'.")
        out[s] = [(m, list(c["genes"])) for m, c in (load_checkpoint(p, "cpu") for p in ckpts)]
    return out


def score_pools(counts: sparse.csr_matrix, var_names: list[str], models: dict[int, list[tuple]], already_log: bool = False) -> dict[int, np.ndarray]:
    """Mean raw model output per pool for each size (seeds averaged).

    ``counts`` is pools x genes (summed raw counts). Each model's own gene list selects and orders
    columns before ``preprocess_counts``; genes missing from ``var_names`` raise ValueError.
    """
    index = {g: i for i, g in enumerate(var_names)}
    raw: dict[int, np.ndarray] = {}
    counts = sparse.csr_matrix(counts)
    for size, plist in models.items():
        acc = np.zeros(counts.shape[0])
        for model, genes in plist:
            missing = [g for g in genes if g not in index]
            if missing:
                raise ValueError(f"{len(missing)} model genes are absent from the pooled counts (e.g. {missing[:3]}); "
                                 "the shared gene list must contain every model gene.")
            cols = np.array([index[g] for g in genes])
            sub = counts[:, cols]
            pred = np.empty(counts.shape[0])
            for i in range(0, counts.shape[0], _SCORE_CHUNK):
                X = preprocess_counts(sub[i:i + _SCORE_CHUNK].toarray(), already_log=already_log)
                pred[i:i + _SCORE_CHUNK] = predict_scores(model, X)
            acc += pred
        raw[size] = acc / len(plist)
    return raw


# -- io helpers ----------------------------------------------------------------------------
def save_sparse_blocks(path: Path, blocks: dict[str, sparse.csr_matrix], meta: dict[str, np.ndarray]) -> None:
    """Write named CSR matrices to one compressed npz (``<name>_data/_indices/_indptr/_shape``)."""
    arrays = dict(meta)
    for name, m in blocks.items():
        m = sparse.csr_matrix(m)
        arrays.update({f"{name}_data": m.data, f"{name}_indices": m.indices, f"{name}_indptr": m.indptr,
                       f"{name}_shape": np.array(m.shape)})
    np.savez_compressed(path, **arrays)


def load_sparse_block(path: Path, name: str) -> sparse.csr_matrix:
    z = np.load(path, allow_pickle=False)
    return sparse.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=tuple(z[f"{name}_shape"]))


def _shared_genes(rd: RunDir, var_names: pd.Index) -> list[str]:
    path = rd.path("prepared", "shared_genes.txt")
    if path.is_file():
        wanted = [g.strip() for g in path.read_text().splitlines() if g.strip()]
        present = set(var_names)
        genes = [g for g in wanted if g in present]
        if len(genes) < len(wanted):
            log.warning("%d of %d shared genes are absent from the spatial cells.", len(wanted) - len(genes), len(wanted))
        if not genes:
            raise ValueError("prepared/shared_genes.txt shares no genes with the spatial cells.")
        return genes
    log.warning("prepared/shared_genes.txt not found; pooling all genes.")
    return list(var_names)


# -- figures -------------------------------------------------------------------------------
def _plot_counts(counts_by_target: dict[int, np.ndarray], sample: str, stem: Path) -> None:
    fig, ax = style.square_panel(f"Cells per neighborhood: {sample}", "target cells per neighborhood", "cells within radius")
    keys = list(counts_by_target)
    ax.boxplot([counts_by_target[k] for k in keys], positions=range(len(keys)), widths=0.6, showfliers=False,
               medianprops={"color": style.CASE}, boxprops={"color": style.CONTROL}, whiskerprops={"color": style.CONTROL})
    ax.scatter(range(len(keys)), keys, marker="_", s=200, color="black", zorder=3, label="target")
    ax.set_xticks(range(len(keys)), [str(k) for k in keys])
    ax.legend(frameon=False, loc="upper left")
    style.save_panel(fig, ax, stem)


def _plot_scores(xy: np.ndarray, score: np.ndarray, sample: str, target: int, n_cells: int, stem: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: F401  (style selects the Agg backend)

    fig, ax = style.square_panel(f"Neighborhood score, ~{target} cells: {sample}", "x", "y")
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=score, cmap=style.score_cmap(), norm=style.score_norm(), s=6,
                    linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    style.colorbar(fig, ax, sc, style.SCORE_LABEL)
    fig.text(0.19, 0.06, f"{len(score)} seed cells; pools overlap; percentile among pools of this sample", fontsize=8)
    style.save_panel(fig, ax, stem)


# -- stage ---------------------------------------------------------------------------------
def _prepare_sample(adata, idx: np.ndarray, genes: list[str], xy_all: np.ndarray):
    X = sparse.csr_matrix(get_counts(adata[idx][:, genes]).astype(np.float64))
    return X, xy_all[idx]


def run_neighborhoods(run: str | Path, force: bool = False) -> dict:
    """Run the stage; returns the completion record. Skips when already complete (unless ``force``)."""
    rd = RunDir(run)
    cfg = RunConfig.load(rd.root)
    if cfg.modality == "st_spot":
        raise ValueError(
            "neighborhoods is not defined for modality 'st_spot': each Visium spot already pools several cells, "
            "so spots are the neighborhoods. Use 'degas3 maps' (and optionally 'degas3 cside') instead."
        )
    spatial_modality(cfg, ("st_cell",), "neighborhoods")
    if rd.stage_complete("neighborhoods") and not force:
        log.info("neighborhoods already complete; skipping (use --force to redo).")
        return rd.read_complete("neighborhoods") or {}
    targets = sorted(int(t) for t in cfg.neighborhood_targets)
    if not targets or targets[0] < 1:
        raise ValueError("config key 'neighborhood_targets' must hold positive integers.")
    sizes = selected_sizes(rd)
    models = load_models(rd, sizes)
    adata, samples, xy_all = read_spatial_cells(rd, cfg)
    genes = _shared_genes(rd, adata.var_names)
    names = sample_order(samples)
    cluster = adata.obs[cfg.cluster_col].astype(str).to_numpy() if cfg.cluster_col and cfg.cluster_col in adata.obs else None
    clusters = sorted(set(cluster.tolist())) if cluster is not None else []

    # pass 1: per-sample calibration
    per_sample: dict[str, dict] = {}
    for si, name in enumerate(names):
        idx = np.flatnonzero(samples == name)
        graph, audit = build_graph(xy_all[idx])
        if audit["largest_component_fraction"] < 0.9:
            log.warning("sample %s: largest graph component holds %.1f%% of cells.", name, 100 * audit["largest_component_fraction"])
        rng = np.random.default_rng([cfg.selection_seed + si, 0])
        seeds = rng.choice(len(idx), size=min(cfg.calibration_seeds, len(idx)), replace=False)
        radii, n_valid = calibrate_sample_radii(graph, seeds, targets, audit["median_edge_length"])
        per_sample[name] = {"audit": audit, "radii": radii, "n_valid": n_valid, "n_cal_seeds": len(seeds)}
        log.info("sample %s: per-sample radii %s", name, dict(zip(targets, np.round(radii, 3))))
        del graph
    med_matrix = np.vstack([per_sample[n]["radii"] for n in names])
    with np.errstate(all="ignore"):
        shared = np.array([np.nanmedian(med_matrix[:, j]) if np.isfinite(med_matrix[:, j]).any() else np.nan for j in range(len(targets))])
    if not np.isfinite(shared).all():
        raise ValueError(f"no calibration seed reaches k cells for targets {[t for t, r in zip(targets, shared) if not np.isfinite(r)]}; "
                         "lower neighborhood_targets or check that the graph is connected.")
    radii = [float(r) for r in shared]

    # pass 2: evaluate, pool, score
    summaries = {}
    for si, name in enumerate(names):
        idx = np.flatnonzero(samples == name)
        sdir = rd.ensure("spatial", safe_name(name))
        graph, audit = build_graph(xy_all[idx])
        (sdir / "graph_audit.json").write_text(json.dumps({**audit, "sample": name}, indent=2) + "\n")
        n = len(idx)
        rng = np.random.default_rng([cfg.selection_seed + si, 1])
        eval_seeds = rng.choice(n, size=min(cfg.evaluation_seeds, n), replace=False)
        pool_seeds = rng.choice(n, size=min(cfg.n_seeds, n), replace=False)
        obs_ids = adata.obs_names.to_numpy()[idx]

        ev = ball_membership(graph, eval_seeds, radii)
        ev_counts = {t: np.asarray(np.diff(m.indptr)) for t, m in zip(targets, ev)}
        pd.DataFrame([
            {"sample": name, "seed_cell_id": obs_ids[s], "target_cells": t, "radius": r, "n_cells": int(c)}
            for t, r in zip(targets, radii) for s, c in zip(eval_seeds, ev_counts[t])
        ]).to_csv(sdir / "evaluation_cell_counts.csv.gz", index=False)
        qs = (0.05, 0.25, 0.5, 0.75, 0.95)
        pd.DataFrame([{
            "target_cells": t, "radius": r, "sample_median_radius": per_sample[name]["radii"][j],
            "n_calibration_seeds": int(per_sample[name]["n_cal_seeds"]), "n_calibration_seeds_reaching_k": int(per_sample[name]["n_valid"][j]),
            "n_evaluation_seeds": len(eval_seeds), "mean_cells": float(ev_counts[t].mean()),
            **{f"q{int(q * 100):02d}_cells": float(np.quantile(ev_counts[t], q)) for q in qs},
        } for j, (t, r) in enumerate(zip(targets, radii))]).to_csv(sdir / "selected_radii.csv", index=False)
        _plot_counts(ev_counts, name, sdir / "cell_counts_by_radius")
        del ev

        X, xy = _prepare_sample(adata, idx, genes, xy_all)
        mem = ball_membership(graph, pool_seeds, radii)
        onehot = None
        if cluster is not None:
            codes = pd.Categorical(cluster[idx], categories=clusters).codes
            onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(clusters)))
        frames, blocks, meta = [], {}, {"genes": np.array(genes), "seed_cell_ids": obs_ids[pool_seeds],
                                       "targets": np.array(targets), "radii": np.array(radii)}
        for t, r, M in zip(targets, radii, mem):
            pooled = sparse.csr_matrix(M @ X)
            blocks[f"counts_k{t}"] = pooled
            blocks[f"membership_k{t}"] = M
            raw = score_pools(pooled, genes, models, cfg.already_log)
            perc, ens = ensemble_from_raw(raw)
            df = pd.DataFrame({"seed_cell_id": obs_ids[pool_seeds], "target_cells": t, "radius": r,
                               "n_cells": np.diff(M.indptr), "x": xy[pool_seeds, 0], "y": xy[pool_seeds, 1]})
            for s in sizes:
                df[f"raw_size_{s:04d}"] = raw[s]
            df["raw_mean"] = np.mean(np.vstack([raw[s] for s in sizes]), axis=0)
            for s in sizes:
                df[f"percentile_size_{s:04d}"] = perc[s]
            df["ensemble_score"] = ens
            if onehot is not None:
                frac = (M @ onehot).toarray() / np.maximum(np.diff(M.indptr), 1)[:, None]
                for ci, c in enumerate(clusters):
                    df[f"frac_{c}"] = frac[:, ci]
            frames.append(df)
            _plot_scores(xy[pool_seeds], ens, name, t, n, sdir / f"neighborhood_scores_r{t}")
        pd.concat(frames, ignore_index=True).to_csv(sdir / "neighborhoods.csv.gz", index=False)
        save_sparse_blocks(sdir / "pooled_counts.npz", blocks, meta)
        summaries[name] = {"n_cells": n, "n_pools_per_target": len(pool_seeds), "n_evaluation_seeds": len(eval_seeds),
                           "largest_component_fraction": audit["largest_component_fraction"]}
        del graph, X, mem

    payload = {
        "status": "complete", "modality": cfg.modality, "samples": summaries, "targets": targets,
        "shared_radii": dict(zip(map(str, targets), radii)), "model_sizes": sizes,
        "edge_trim_quantile": EDGE_TRIM_QUANTILE, "n_seeds": cfg.n_seeds, "calibration_seeds": cfg.calibration_seeds,
        "evaluation_seeds": cfg.evaluation_seeds, "seed": cfg.selection_seed, "n_genes_pooled": len(genes),
        "ranking": "pool percentiles within sample and target size (average ties / n pools), mean across model sizes",
    }
    rd.write_complete("neighborhoods", payload)
    return payload


def cli_neighborhoods(args: argparse.Namespace) -> int:
    run_neighborhoods(args.run, force=bool(getattr(args, "force", False)))
    return 0
