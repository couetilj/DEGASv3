"""Class-balanced batch sampling for cells and patients (DEGASv2 STSCDataset / PatDataset logic).

Each call to ``batch(index)`` is deterministic in ``(seed, index)`` and uses a private
``RandomState``; no global numpy state is touched.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def balance_sampling(labels: np.ndarray, batch_size: int, random_seed: int) -> np.ndarray:
    """Sample ``batch_size // n_classes`` indices per class, with replacement.

    Identical to the DEGASv2 routine seeded with ``np.random.seed(random_seed)``.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D class codes; flatten or argmax one-hot labels first.")
    rs = np.random.RandomState(random_seed)
    picks: list[int] = []
    categories = np.unique(labels)
    for category in categories:
        idx = np.flatnonzero(labels == category)
        picks += list(rs.choice(idx, batch_size // len(categories), replace=True))
    return np.asarray(picks, dtype=np.int64)


class CellBatchSampler:
    """Balanced batches of cells (rows of ``X``); labels optional (single class when None)."""

    def __init__(self, X: np.ndarray, labels: Optional[np.ndarray] = None, *, seed: int = 0,
                 batch_size: int = 200, fold: int = -1, tot_folds: int = 1,
                 sample_method: str = "balance"):
        if sample_method != "balance":
            raise ValueError("only sample_method='balance' is implemented.")
        self.X = X
        n = X.shape[0]
        if labels is None:
            self.labels = np.zeros(n, dtype=np.int64)
        else:
            lab = np.asarray(labels)
            if lab.ndim > 1:
                lab = np.argmax(lab, axis=1) if lab.shape[1] > 1 else lab.reshape(-1)
            if lab.shape[0] != n:
                raise ValueError(f"cell labels have {lab.shape[0]} rows but X has {n}; align them.")
            self.labels = lab
        self.seed, self.batch_size = seed, batch_size
        self.fold_indices = None
        if fold >= 0 and tot_folds > 1:
            order = np.random.RandomState(0).choice(n, n, replace=False)
            size = n // tot_folds
            self.fold_indices = order[fold * size:(fold + 1) * size]
            self.sub_labels = self.labels[self.fold_indices]

    def batch(self, index: int) -> dict:
        if self.fold_indices is not None:
            idx = self.fold_indices[balance_sampling(self.sub_labels, self.batch_size, self.seed + index)]
        else:
            idx = balance_sampling(self.labels, self.batch_size, self.seed + index)
        return {"index": idx, "data": self.X[idx, :], "label": self.labels[idx]}


class PatientBatchSampler:
    """Batches of patients: class-balanced for binary outcomes, uniform without replacement for Cox.

    For Cox models ``y`` is an (n, 2) array of (time, event); otherwise 1-D class codes.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, *, seed: int = 0, batch_size: int = 200,
                 model_type: str = "BlankClass", sample_method: str = "balance"):
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"patient outcome has {y.shape[0]} rows but X has {X.shape[0]}; align them.")
        self.X, self.seed, self.batch_size = X, seed, batch_size
        self.cox = "Cox" in model_type
        self.n = X.shape[0]
        if self.cox:
            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError("Cox outcome must be an (n, 2) array of time and event.")
            self.time = y[:, 0].astype(int).reshape(-1)
            self.status = y[:, 1].reshape(-1)
        else:
            if y.ndim > 1:
                y = np.argmax(y, axis=1) if y.shape[1] > 1 else y.reshape(-1)
            if sample_method != "balance":
                raise ValueError("only sample_method='balance' is implemented.")
            self.labels = y

    def batch(self, index: int) -> dict:
        if self.cox:
            rs = np.random.RandomState(self.seed + index)
            idx = rs.choice(self.n, min(self.batch_size, self.n), replace=False)
            return {"pid": idx, "data": self.X[idx, :], "time": self.time[idx], "status": self.status[idx]}
        idx = balance_sampling(self.labels, self.batch_size, self.seed + index)
        return {"pid": idx, "data": self.X[idx, :], "label": self.labels[idx]}
