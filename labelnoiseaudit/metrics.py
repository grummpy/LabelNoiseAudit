"""Retrieval metrics for 'did we rank the flipped labels first?'."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def precision_recall_at_k(scores, truth, k: int) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if len(scores) != len(truth):
        raise ValueError("scores and truth must have the same length")
    k = min(max(int(k), 1), len(scores))
    order = np.argsort(-scores, kind="mergesort")
    hits = float(truth[order[:k]].sum())
    positives = float(truth.sum())
    precision = hits / k
    recall = hits / positives if positives else 0.0
    return precision, recall


def ranking_report(scores, truth, *, fraction: float = 0.10) -> dict[str, float | int]:
    """Precision and recall at the noise count, plus the same pair at 10% of rows.

    When ``k`` equals the number of flipped labels, precision@k equals recall@k
    for every ranker. The 10% cutoff is where the two numbers come apart.
    """

    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    n_rows = int(len(scores))
    n_noisy = int(truth.sum())
    k = max(1, n_noisy)
    k_fraction = max(1, int(round(fraction * n_rows)))
    precision_at_k, recall_at_k = precision_recall_at_k(scores, truth, k)
    precision_at_fraction, recall_at_fraction = precision_recall_at_k(scores, truth, k_fraction)
    if 0 < n_noisy < n_rows:
        auroc = float(roc_auc_score(truth.astype(int), scores))
    else:
        auroc = float("nan")
    return {
        "n": n_rows,
        "n_noisy": n_noisy,
        "prevalence": (n_noisy / n_rows) if n_rows else 0.0,
        "k": k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "k_10pct": k_fraction,
        "precision_at_10pct": precision_at_fraction,
        "recall_at_10pct": recall_at_fraction,
        "auroc": auroc,
    }
