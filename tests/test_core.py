import numpy as np
import pytest

from degas3.core.models import build_model, fit, load_checkpoint, predict_scores, save_checkpoint
from degas3.core.options import make_options
from degas3.core.preprocess import preprocess_counts
from degas3.core.sampling import PatientBatchSampler, balance_sampling

G = 12


def _data(cox):
    rng = np.random.default_rng(0)
    px = preprocess_counts(rng.poisson(20, (30, G)).astype(float)).astype(np.float32)
    sx = preprocess_counts(rng.poisson(5, (60, G)).astype(float)).astype(np.float32)
    py = np.column_stack([rng.integers(1, 20, 30), rng.integers(0, 2, 30)]) if cox else np.arange(30) % 2
    return px, py, sx, rng.integers(0, 3, 60)


@pytest.mark.parametrize("outcome,labels", [("binary", False), ("binary", True), ("survival", False), ("survival", True)])
def test_fit_predict_checkpoint(tmp_path, outcome, labels):
    px, py, sx, sy = _data(outcome == "survival")
    extra = {"high_reso_output_shape": 3} if labels else {}
    opt = make_options(outcome, labels, input_shape=G, feature_dim=8, seed=3, batch_size=20, pat_batch_size=20, **extra)
    model = build_model(opt, "cpu")
    assert not list(tmp_path.iterdir())  # constructor touches no disk
    losses = fit(model, px, py, sx, sy if labels else None, iterations=5, seed=3)
    assert len(losses) == 5 and np.isfinite(losses.to_numpy()).all()
    assert ("high_reso_loss" in losses.columns) == labels
    s = predict_scores(model, sx, batch_size=16)
    assert s.shape == (60,) and s.min() >= 0 and s.max() <= 1
    path = save_checkpoint(model, tmp_path / "m" / "checkpoint.pt", genes=[f"g{i}" for i in range(G)], seed=3, size=G, fold=-1)
    m2, ck = load_checkpoint(path, "cpu")
    assert ck["genes"][0] == "g0" and ck["outcome_type"] == outcome and ck["size"] == G
    np.testing.assert_allclose(predict_scores(m2, sx), s, atol=1e-6)


def test_fit_is_deterministic():
    px, py, sx, sy = _data(False)
    opt = make_options("binary", False, input_shape=G, feature_dim=8, seed=1, batch_size=20, pat_batch_size=20)
    out = []
    for _ in range(2):
        m = build_model(opt, "cpu")
        fit(m, px, py, sx, None, iterations=4, seed=1)
        out.append(predict_scores(m, sx))
    np.testing.assert_allclose(out[0], out[1], atol=1e-6)


def test_class_model_needs_labels():
    px, py, sx, _ = _data(False)
    opt = make_options("binary", True, input_shape=G, feature_dim=8, high_reso_output_shape=3)
    with pytest.raises(ValueError, match="cell_y"):
        fit(build_model(opt, "cpu"), px, py, sx, None, iterations=1)


def test_balance_sampling_matches_legacy():
    labels = np.array([0] * 10 + [1] * 3)
    np.random.seed(7)
    legacy = []
    for c in np.unique(labels):
        legacy += list(np.random.choice(np.flatnonzero(labels == c), 4, replace=True))
    assert list(balance_sampling(labels, 8, 7)) == legacy
    state = np.random.get_state()[1][:3].copy()
    balance_sampling(labels, 8, 9)
    assert (np.random.get_state()[1][:3] == state).all()  # global state untouched


def test_cox_sampler_clips_batch():
    px, py, _, _ = _data(True)
    b = PatientBatchSampler(px, py, seed=0, batch_size=200, model_type="BlankCox").batch(0)
    assert len(b["pid"]) == 30
