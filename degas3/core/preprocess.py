"""DEGAS preprocessing for Python.

Python equivalent of `preprocessCounts` in
`DEGAS_R/R/DEGAS_preprocessing.R`:

    normalizeScale(1.5^log2(X + 1))

Per-sample (per-row of an AnnData-convention input) z-score followed by
per-sample min-max scaling to [0, 1].

The `already_log` flag handles bulk RNA-seq sources that ship
`log2(count + 1)` despite a 'counts'-suggesting filename (notably UCSC
Xena's `.star_counts.tsv.gz`). With `already_log=True` the internal
log2 step is skipped, satisfying

    preprocess_counts(raw)
    == preprocess_counts(np.log2(raw + 1), already_log=True)

to floating-point precision (verified in tests/test_preprocess.py).
"""

from __future__ import annotations

import numpy as np


def normalize_scale(X: np.ndarray) -> np.ndarray:
    """Per-row z-score then per-row min-max — matches R `normalizeScale`.

    R applies `apply(t(X), 1, normFunc)` then `apply(..., 1, scaleFunc)`,
    i.e. normalization runs over each sample's gene-expression vector.
    Inputs follow AnnData/scikit-learn convention `(n_samples, n_genes)`,
    so per-row in our convention IS per-sample.

    Denominator stabilizers (+1e-3) and ddof=1 (sample SD) match R exactly.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {X.shape}")
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, ddof=1, keepdims=True) + 1e-3
    Z = (X - mu) / sd
    lo = np.nanmin(Z, axis=1, keepdims=True)
    hi = np.nanmax(Z, axis=1, keepdims=True)
    return (Z - lo) / (hi - lo + 1e-3)


def preprocess_counts(X: np.ndarray, *, already_log: bool = False) -> np.ndarray:
    """DEGAS preprocessing: `normalizeScale(1.5^log2(X + 1))`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_genes)
        Expression in AnnData convention. For data in R/DEGAS convention
        (genes x samples), transpose before calling.
    already_log : bool, keyword-only
        If True, X is already `log2(count + 1)` transformed. Use for UCSC
        Xena `.star_counts.tsv.gz` bulk.

    Returns
    -------
    out : ndarray, shape (n_samples, n_genes)
        Each row is one sample's gene profile, z-scored across genes then
        min-max scaled to [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    Y = X if already_log else np.log2(X + 1.0)
    return normalize_scale(np.power(1.5, Y))
