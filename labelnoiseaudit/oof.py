"""K-fold out-of-fold probabilities with a preprocessor fit on each train fold.

The preprocessor factory is called inside the loop. A fold never transforms
its holdout rows with a preprocessor that has seen them, and neighbor search
looks only at that fold's training rows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

Progress = Callable[[str], None]


def make_estimator(seed: int, *, sparse_input: bool) -> LogisticRegression:
    """Linear classifier with probabilities. Saga handles sparse text features."""

    if sparse_input:
        return LogisticRegression(
            solver="saga",
            C=1.0,
            max_iter=600,
            tol=1e-3,
            random_state=seed,
        )
    return LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=400,
        random_state=seed,
    )


def _slice(features, indices: np.ndarray):
    if isinstance(features, pd.DataFrame):
        return features.iloc[np.asarray(indices)]
    return features[np.asarray(indices)]


def _for_neighbors(matrix):
    if sparse.issparse(matrix):
        rows, cols = matrix.shape
        if rows * cols > 2_000_000:
            return matrix
        return matrix.toarray()
    return np.asarray(matrix, dtype=float)


def choose_n_splits(y: np.ndarray, requested: int) -> int:
    """Use as many stratified folds as the smallest class allows."""

    if requested < 2:
        raise ValueError("n_splits must be at least 2")
    y = np.asarray(y, dtype=int)
    n_classes = int(y.max()) + 1 if len(y) else 0
    if n_classes < 2:
        raise ValueError("Need at least two classes to score label noise.")
    counts = np.bincount(y, minlength=n_classes)
    smallest = int(counts.min()) if len(counts) else 0
    if smallest < 2:
        raise ValueError(
            "Each class needs at least 2 rows so a fold can hold one out. "
            f"The smallest class has {smallest}."
        )
    return min(requested, smallest)


def run_oof(
    features,
    y: np.ndarray,
    preprocessor_factory: Callable[[], object],
    *,
    n_splits: int,
    seed: int,
    n_neighbors: int,
    progress: Progress | None = None,
) -> tuple[np.ndarray, list[list[int]], list[list[int]], int]:
    """Return probabilities, neighbor labels, neighbor indexes, and the fold count.

    ``y`` is integer class indexes. ``preprocessor_factory`` must return a new,
    unfitted transformer every time it is called.
    """

    y = np.asarray(y, dtype=int)
    n_rows = len(y)
    n_classes = int(y.max()) + 1
    splits = choose_n_splits(y, n_splits)
    folder = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    proba = np.zeros((n_rows, n_classes), dtype=float)
    neighbor_labels: list[list[int]] = [[] for _ in range(n_rows)]
    neighbor_indices: list[list[int]] = [[] for _ in range(n_rows)]
    fold_list = list(folder.split(np.zeros(n_rows), y))
    for fold_number, (train_idx, test_idx) in enumerate(fold_list, start=1):
        if progress is not None:
            progress(f"Out-of-fold fold {fold_number}/{len(fold_list)}")
        preprocessor = preprocessor_factory()
        train_raw = _slice(features, train_idx)
        test_raw = _slice(features, test_idx)
        preprocessor.fit(train_raw, y[train_idx])
        x_train = preprocessor.transform(train_raw)
        x_test = preprocessor.transform(test_raw)
        estimator = make_estimator(seed, sparse_input=sparse.issparse(x_train))
        estimator.fit(x_train, y[train_idx])
        fold_proba = estimator.predict_proba(x_test)
        for column, class_index in enumerate(estimator.classes_):
            proba[test_idx, int(class_index)] = fold_proba[:, column]

        k = min(int(n_neighbors), len(train_idx))
        if k < 1:
            continue
        neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute")
        neighbors.fit(_for_neighbors(x_train))
        local_indices = neighbors.kneighbors(_for_neighbors(x_test), return_distance=False)
        for row_index, local in zip(test_idx, local_indices, strict=True):
            chosen = train_idx[np.asarray(local, dtype=int)]
            neighbor_indices[int(row_index)] = [int(j) for j in chosen]
            neighbor_labels[int(row_index)] = [int(y[j]) for j in chosen]
    return proba, neighbor_labels, neighbor_indices, splits
