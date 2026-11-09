"""Fold-internal gene ranking: bulk variance and SC cluster markers, plus nested gene lists.

Everything here is a *feature-ranking* device. The marker p-values order genes by how well
they separate one cluster from the rest in a subsample of cells; they are not donor-level
inference and not a test of any outcome. Callers must pass only the training fold's bulk
rows (and, for markers, only the cells that are eligible for that fold).
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats

log = logging.getLogger(__name__)


def bulk_variance_ranking(
    counts: np.ndarray,
    datasets: Optional[Sequence] = None,
    genes: Optional[Sequence[str]] = None,
    already_log: bool = False,
) -> pd.DataFrame:
    """Rank genes by variance of log2-CPM across the supplied (training) bulk samples.

    ``counts`` is samples x genes raw counts (or log2(count+1) when ``already_log``, in which
    case values are used as is). When ``datasets`` has at least two levels the per-dataset mean
    of each gene is subtracted before the variance is computed (for ranking only; no
    transformed values are returned) and the residual sum of squares is divided by
    ``n - n_datasets``. With one level, or when that divisor would be below 1, the plain
    sample variance (divisor ``n - 1``, or 1 for a single sample) is used instead.
    The statistic is a variability score used to order genes; it is not an outcome association.

    Returns columns ``gene, variance, mean_logcpm`` sorted by variance desc, mean desc, gene asc.
    """
    x = np.asarray(counts, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"counts must be a non-empty samples x genes matrix, got shape {x.shape}.")
    genes = [str(g) for g in (genes if genes is not None else range(x.shape[1]))]
    if len(genes) != x.shape[1]:
        raise ValueError(f"{len(genes)} gene names for {x.shape[1]} count columns; fix the gene list.")
    if already_log:
        logcpm = x
    else:
        totals = x.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("a training bulk sample has zero total counts; drop it before ranking genes.")
        logcpm = np.log2(1.0 + 1e6 * x / totals)
    n = x.shape[0]
    levels = np.unique(np.asarray(datasets).astype(str)) if datasets is not None else np.array([])
    residual, dof = logcpm, n - 1
    if len(levels) >= 2 and n - len(levels) >= 1:
        ds = np.asarray(datasets).astype(str)
        residual = logcpm.copy()
        for lev in levels:
            take = ds == lev
            residual[take] -= logcpm[take].mean(axis=0)
        dof = n - len(levels)
    else:
        residual = logcpm - logcpm.mean(axis=0)
    variance = (residual ** 2).sum(axis=0) / max(dof, 1)
    out = pd.DataFrame({"gene": genes, "variance": variance, "mean_logcpm": logcpm.mean(axis=0)})
    return out.sort_values(["variance", "mean_logcpm", "gene"], ascending=[False, False, True],
                           kind="mergesort").reset_index(drop=True)


def _balanced_subsample(clusters: np.ndarray, donors: np.ndarray, per_cluster: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Up to ``per_cluster`` cells per cluster with donors contributing as equally as availability allows."""
    chosen: list[np.ndarray] = []
    for cl in np.unique(clusters):
        idx = np.flatnonzero(clusters == cl)
        if len(idx) <= per_cluster:
            chosen.append(idx)
            continue
        groups = [idx[donors[idx] == d] for d in np.unique(donors[idx])]
        groups.sort(key=len)
        left, quota = per_cluster, []
        for k, g in enumerate(groups):  # water-filling: small donors give everything, the rest share equally
            share = left // (len(groups) - k)
            take = min(len(g), share)
            quota.append(take)
            left -= take
        extra = left  # remainder from integer division goes to donors with spare cells
        for k in range(len(groups) - 1, -1, -1):
            if extra <= 0:
                break
            add = min(extra, len(groups[k]) - quota[k])
            quota[k] += add
            extra -= add
        for g, q in zip(groups, quota):
            chosen.append(rng.choice(g, q, replace=False))
    return np.sort(np.concatenate(chosen))


def marker_ranking(
    adata_counts,
    clusters: Sequence,
    donors: Sequence,
    universe: Sequence[str],
    cells_per_cluster: int = 500,
    seed: int = 0,
    block: int = 64,
) -> dict[str, pd.DataFrame]:
    """One-vs-rest positive cluster markers from a donor-balanced cell subsample.

    ``adata_counts`` is a cells x genes raw-count matrix (dense or sparse) whose columns are the
    genes in ``universe``. Up to ``cells_per_cluster`` cells per cluster are drawn, with donors
    contributing equally where they have enough cells. Counts are scaled to 1e4 per cell and
    log1p-transformed. For each cluster the statistic is the tie-corrected, asymptotic
    Wilcoxon rank-sum test with continuity correction (two-sided), the effect is the log2
    ratio of mean normalised expression + 1 (cluster vs rest), and ``pct_in`` is the fraction
    of in-cluster cells with a non-zero count. Benjamini-Hochberg adjustment is done in log
    space over all tested genes. Only genes with positive log2FC are returned.
    These values rank genes for feature selection; cells are not independent replicates, so
    they are not donor-level evidence.

    Returns ``{cluster: DataFrame[gene, log_p, log2fc, pct_in, log_p_adj]}`` sorted by
    log_p asc, log2fc desc, gene asc.
    """
    universe = [str(g) for g in universe]
    clusters = np.asarray(clusters).astype(str)
    donors = np.asarray(donors).astype(str)
    n_all = adata_counts.shape[0]
    if adata_counts.shape[1] != len(universe) or len(clusters) != n_all or len(donors) != n_all:
        raise ValueError("counts columns must match `universe` and clusters/donors must have one entry per cell.")
    if len(np.unique(clusters)) < 2:
        raise ValueError("marker ranking needs at least two clusters; use feature_selection='variance' or add clusters.")
    rng = np.random.default_rng(seed)
    sel = _balanced_subsample(clusters, donors, int(cells_per_cluster), rng)
    labels = clusters[sel]
    names = sorted(set(labels))
    n = len(sel)
    mat = sparse.csc_matrix(adata_counts[sel]) if sparse.issparse(adata_counts) else sparse.csc_matrix(np.asarray(adata_counts)[sel])
    totals = np.asarray(mat.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError("a cell in the marker subsample has zero total counts; filter empty cells first.")
    inside = {nm: labels == nm for nm in names}
    for nm in names:
        if min(inside[nm].sum(), n - inside[nm].sum()) < 2:
            raise ValueError(f"cluster {nm!r} has fewer than 2 eligible cells (or covers all but one); "
                             "merge small clusters or lower the number of clusters.")
    acc: dict[str, list[pd.DataFrame]] = {nm: [] for nm in names}
    for start in range(0, len(universe), block):
        stop = min(start + block, len(universe))
        expr = mat[:, start:stop].toarray()
        norm = expr * (1e4 / totals[:, None])
        lognorm = np.log1p(norm)
        ranks = stats.rankdata(lognorm, axis=0, method="average")
        tie = np.array([np.sum(c.astype(float) ** 3 - c)
                        for c in (np.unique(lognorm[:, j], return_counts=True)[1] for j in range(stop - start))])
        for nm in names:
            m = inside[nm]
            n1, n2 = int(m.sum()), int(n - m.sum())
            u = ranks[m].sum(axis=0) - n1 * (n1 + 1) / 2
            sd = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
            z = np.divide(np.abs(u - n1 * n2 / 2) - 0.5, sd, out=np.zeros_like(sd), where=sd > 0)
            logp = np.minimum(0.0, np.log(2) + stats.norm.logsf(np.maximum(z, 0)))
            lfc = np.log2((norm[m].mean(axis=0) + 1) / (norm[~m].mean(axis=0) + 1))
            acc[nm].append(pd.DataFrame({"gene": universe[start:stop], "log_p": logp, "log2fc": lfc,
                                         "pct_in": (expr[m] > 0).mean(axis=0)}))
    result: dict[str, pd.DataFrame] = {}
    for nm, pieces in acc.items():
        fr = pd.concat(pieces, ignore_index=True).sort_values(
            ["log_p", "log2fc", "gene"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        q = fr["log_p"].to_numpy() + np.log(len(fr) / np.arange(1, len(fr) + 1))
        fr["log_p_adj"] = np.minimum(0.0, np.minimum.accumulate(q[::-1])[::-1])
        result[nm] = fr[fr["log2fc"] > 0].reset_index(drop=True)
    return result


def _gene_order(ranking) -> list[str]:
    if isinstance(ranking, pd.DataFrame):
        return [str(g) for g in ranking["gene"]]
    return [str(g) for g in ranking]


def nested_list(ranking, universe: Iterable[str], size: int) -> list[str]:
    """First ``size`` ranked genes that are in ``universe`` (a prefix of every larger list).

    ``ranking`` is a ranked DataFrame with a ``gene`` column or an ordered gene sequence.
    Raises ValueError rather than returning fewer than ``size`` genes.
    """
    allowed = set(map(str, universe))
    genes = list(dict.fromkeys(g for g in _gene_order(ranking) if g in allowed))
    if len(genes) < size:
        raise ValueError(
            f"only {len(genes)} ranked genes are in the gene universe but a list of {size} was requested; "
            "remove sizes above the number of shared genes (config key 'sizes')."
        )
    return genes[:size]


def mixed_list(bulk_rank, markers: dict[str, pd.DataFrame], universe: Iterable[str], size: int) -> list[dict]:
    """Alternate bulk-variance genes and cluster markers (bulk first), without duplicates.

    Marker slots rotate across clusters: each marker slot goes to the cluster that has so far
    contributed the fewest genes (ties: smaller adjusted log p, larger log2FC, then cluster
    name), taking its best not-yet-used marker. Because genes are appended one at a time the
    lists for smaller ``size`` are prefixes of larger ones. Never pads: raises ValueError when
    the bulk ranking or every cluster's positive markers are exhausted.

    Returns a list of ``{gene, source, cluster}`` dicts (source is bulk_variance or sc_marker).
    """
    allowed = set(map(str, universe))
    bulk = [g for g in dict.fromkeys(_gene_order(bulk_rank)) if g in allowed]
    mk = {str(k): [r for r in df.itertuples(index=False) if r.gene in allowed] for k, df in markers.items()}
    pos = {k: 0 for k in mk}
    given = {k: 0 for k in mk}
    used: set[str] = set()
    bpos = 0
    out: list[dict] = []
    while len(out) < size:
        if len(out) % 2 == 0:
            while bpos < len(bulk) and bulk[bpos] in used:
                bpos += 1
            if bpos >= len(bulk):
                raise ValueError(f"bulk candidate genes exhausted after {len(out)} genes; lower 'sizes' or use feature_selection='variance'.")
            used.add(bulk[bpos])
            out.append({"gene": bulk[bpos], "source": "bulk_variance", "cluster": ""})
            continue
        cands = []
        for k, rows in mk.items():
            while pos[k] < len(rows) and rows[pos[k]].gene in used:
                pos[k] += 1
            if pos[k] < len(rows):
                r = rows[pos[k]]
                cands.append((given[k], r.log_p_adj, -r.log2fc, k, r.gene))
        if not cands:
            raise ValueError(
                f"positive cluster markers exhausted after {len(out)} genes (no silent padding); "
                "lower 'sizes' or use feature_selection='variance'."
            )
        _, _, _, k, gene = min(cands)
        given[k] += 1
        used.add(gene)
        out.append({"gene": gene, "source": "sc_marker", "cluster": k})
    return out
