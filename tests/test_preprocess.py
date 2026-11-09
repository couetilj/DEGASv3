import numpy as np

from degas3.core.preprocess import normalize_scale, preprocess_counts


def _independent(x):
    y = 1.5 ** np.log2(x + 1.0)
    z = (y - y.mean(1, keepdims=True)) / (y.std(1, ddof=1, keepdims=True) + 1e-3)
    return (z - z.min(1, keepdims=True)) / (z.max(1, keepdims=True) - z.min(1, keepdims=True) + 1e-3)


def test_matches_formula():
    x = np.random.default_rng(0).poisson(30, (7, 50)).astype(float)
    np.testing.assert_allclose(preprocess_counts(x), _independent(x), rtol=1e-12)
    out = preprocess_counts(x)
    assert out.min() >= 0 and out.max() <= 1


def test_already_log_equivalent():
    x = np.random.default_rng(1).poisson(30, (5, 40)).astype(float)
    np.testing.assert_allclose(preprocess_counts(x), preprocess_counts(np.log2(x + 1), already_log=True), atol=1e-12)


def test_normalize_scale_rejects_1d():
    import pytest
    with pytest.raises(ValueError):
        normalize_scale(np.arange(5.0))
