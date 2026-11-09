import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from degas3.features import bulk_variance_ranking, marker_ranking, mixed_list, nested_list


def _counts(n=20, g=30, seed=0):
    return np.random.default_rng(seed).poisson(20, (n, g)).astype(float)


def test_ranking_uses_only_supplied_rows():
    counts = _counts()
    genes = [f"g{i}" for i in range(counts.shape[1])]
    train = np.arange(15)
    a = bulk_variance_ranking(counts[train], None, genes)
    changed = counts.copy()
    changed[17] = 1e5 * np.random.default_rng(1).random(counts.shape[1])  # held-out row
    b = bulk_variance_ranking(changed[train], None, genes)
    pd.testing.assert_frame_equal(a, b)


def test_variance_sorted_and_dataset_residual():
    counts = _counts()
    genes = [f"g{i}" for i in range(30)]
    ds = np.array(["a"] * 10 + ["b"] * 10)
    counts[10:, 3] *= 50  # a pure dataset shift in gene 3
    plain = bulk_variance_ranking(counts, None, genes).set_index("gene")
    resid = bulk_variance_ranking(counts, ds, genes).set_index("gene")
    assert resid.loc["g3", "variance"] < plain.loc["g3", "variance"]
    r = bulk_variance_ranking(counts, ds, genes)
    assert (np.diff(r["variance"]) <= 0).all()
    # manual check of residual variance for one gene
    lc = np.log2(1 + 1e6 * counts / counts.sum(1, keepdims=True))[:, 5]
    res = np.r_[lc[:10] - lc[:10].mean(), lc[10:] - lc[10:].mean()]
    assert resid.loc["g5", "variance"] == pytest.approx((res ** 2).sum() / 18)


def test_nested_prefixes_and_no_padding():
    rank = pd.DataFrame({"gene": list("abcdefgh")})
    uni = list("abcdfgh")
    l5, l3 = nested_list(rank, uni, 5), nested_list(rank, uni, 3)
    assert l3 == l5[:3] == ["a", "b", "c"] and "e" not in l5
    with pytest.raises(ValueError, match="requested"):
        nested_list(rank, uni, 8)


def _markers():
    mk = {}
    for k, name in enumerate(["c0", "c1", "c2"]):
        genes = [f"m{k}_{i}" for i in range(6)]
        mk[name] = pd.DataFrame({"gene": genes, "log_p": -np.arange(6, 0, -1.0) * (k + 1),
                                 "log2fc": 1.0, "pct_in": 1.0, "log_p_adj": -np.arange(6, 0, -1.0) * (k + 1)})
    return mk


def test_mixed_quotas_nested_and_exhaustion():
    mk = _markers()
    bulk = [f"b{i}" for i in range(40)]
    uni = set(bulk) | {g for d in mk.values() for g in d.gene}
    big = mixed_list(bulk, mk, uni, 18)
    assert [r["source"] for r in big] == ["bulk_variance", "sc_marker"] * 9
    assert len({r["gene"] for r in big}) == 18
    quota = pd.Series([r["cluster"] for r in big if r["source"] == "sc_marker"]).value_counts()
    assert quota.max() - quota.min() <= 0 and quota.sum() == 9  # 9 slots over 3 clusters -> 3 each
    small = mixed_list(bulk, mk, uni, 7)
    assert [r["gene"] for r in small] == [r["gene"] for r in big[:7]]
    with pytest.raises(ValueError, match="exhausted"):
        mixed_list(bulk, mk, uni, 40)  # 20 marker slots but only 18 markers
    with pytest.raises(ValueError, match="exhausted"):
        mixed_list(bulk[:3], mk, uni, 10)


def test_marker_ranking_positive_and_matches_scipy():
    rng = np.random.default_rng(0)
    n, g = 120, 20
    clusters = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
    donors = np.tile(["d1", "d2"], 60)
    mu = np.full((n, g), 5.0)
    mu[:40, 2] = 40.0
    mu[40:80, 7] = 40.0
    counts = rng.poisson(mu)
    genes = [f"g{i}" for i in range(g)]
    res = marker_ranking(sparse.csr_matrix(counts), clusters, donors, genes, cells_per_cluster=500, seed=1)
    assert res["a"].iloc[0].gene == "g2" and res["b"].iloc[0].gene == "g7"
    assert (res["a"]["log2fc"] > 0).all()
    assert (res["a"]["log_p_adj"] >= res["a"]["log_p"] - 1e-9).all()
    # all cells used (<= 500 per cluster): compare with scipy Mann-Whitney (asymptotic, continuity, ties)
    norm = np.log1p(counts * (1e4 / counts.sum(1, keepdims=True)))
    p = stats.mannwhitneyu(norm[:40, 2], norm[40:, 2], method="asymptotic", use_continuity=True).pvalue
    row = res["a"].set_index("gene").loc["g2"]
    assert np.exp(row.log_p) == pytest.approx(p, rel=1e-6)
    sub = marker_ranking(counts, clusters, donors, genes, cells_per_cluster=10, seed=1)
    assert sub["a"].iloc[0].gene == "g2"
    with pytest.raises(ValueError, match="two clusters"):
        marker_ranking(counts, np.zeros(n), donors, genes)
