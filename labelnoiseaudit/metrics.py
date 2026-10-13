"""Retrieval metrics for 'did we rank the flipped labels first?'."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def wilson_interval(hits: int, total: int, *, z: float = 1.96) -> list[float]:
    """A deterministic 95% binomial interval for a human review sample."""

    if total <= 0:
        return [0.0, 0.0]
    proportion = hits / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * np.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total) / denominator
    return [float(max(0.0, center - spread)), float(min(1.0, center + spread))]


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


def _review_budget_report(
    scores: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    *,
    fraction: float,
) -> dict[str, object]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("review budget fraction must be in (0, 1]")
    n_rows = len(scores)
    budget = max(1, int(round(fraction * n_rows)))
    order = np.argsort(-scores, kind="mergesort")
    reviewed = order[:budget]
    hits = int(truth[reviewed].sum())
    positives = int(truth.sum())
    per_class: dict[str, dict[str, int | float | list[float]]] = {}
    for raw_label in sorted(set(labels.tolist()), key=str):
        label = str(raw_label)
        all_rows = labels == raw_label
        reviewed_rows = all_rows[reviewed]
        class_truth = truth[all_rows]
        class_hits = int(np.logical_and(all_rows[reviewed], truth[reviewed]).sum())
        class_reviewed = int(reviewed_rows.sum())
        class_positives = int(class_truth.sum())
        per_class[label] = {
            "n_rows": int(all_rows.sum()),
            "n_reviewed": class_reviewed,
            "n_injected_noise": class_positives,
            "true_positives": class_hits,
            "false_positives": int(class_reviewed - class_hits),
            "precision": (class_hits / class_reviewed) if class_reviewed else 0.0,
            "recall": (class_hits / class_positives) if class_positives else 0.0,
        }
    return {
        "fraction": fraction,
        "n_rows": n_rows,
        "n_reviewed": budget,
        "true_positives": hits,
        "false_positives": int(budget - hits),
        "precision": hits / budget,
        "recall": hits / positives if positives else 0.0,
        "precision_wilson_95": wilson_interval(hits, budget),
        "by_given_class": per_class,
        "human_review_required": True,
    }


def ranking_report(
    scores,
    truth,
    *,
    fraction: float = 0.10,
    labels=None,
    review_budget_fraction: float = 0.05,
) -> dict[str, object]:
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
    report: dict[str, object] = {
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
    if labels is not None:
        labels_array = np.asarray(labels)
        if len(labels_array) != n_rows:
            raise ValueError("labels and scores must have the same length")
        report["review_budget"] = _review_budget_report(
            scores, truth, labels_array, fraction=review_budget_fraction
        )
    return report
