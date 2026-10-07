"""Confident-learning scores, written out so the ranking is inspectable.

This is the core of Northcutt, Jiang, and Chuang's confident learning,
implemented here rather than hidden behind a library call:

* Self-confidence is the out-of-fold probability of the given label.
* Normalized margin is that probability minus the best competing class.
  Negative means the model prefers a different label.
* The confident joint counts a row in column ``j`` when ``j`` clears the
  per-class threshold (the mean self-confidence of rows labeled ``j``)
  and beats the other classes that also clear their threshold.
* The per-class noise rate is the off-diagonal share of that row of the joint.

An optional cleanlab cross-check can be layered on later. It is not required
for these scores.
"""

from __future__ import annotations

import numpy as np

SELF_CONFIDENCE_WEIGHT = 0.45
MARGIN_WEIGHT = 0.40
OFF_DIAGONAL_WEIGHT = 0.15
CL_ENSEMBLE_WEIGHT = 0.70
KNN_ENSEMBLE_WEIGHT = 0.30
KNN_CHECK_FRACTION = 0.5


def score_predictions(proba: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Score every row from an out-of-fold probability matrix.

    ``y`` holds integer class indexes into the columns of ``proba``.
    """

    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    if proba.ndim != 2:
        raise ValueError("proba must be a 2-d array of class probabilities")
    if len(y) != len(proba):
        raise ValueError("y and proba must have the same number of rows")
    n_rows, n_classes = proba.shape
    if n_classes < 2:
        raise ValueError("Scoring needs at least two classes")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("y indexes fall outside the probability columns")

    self_confidence = proba[np.arange(n_rows), y]
    competing = proba.copy()
    competing[np.arange(n_rows), y] = -np.inf
    alternative = competing.max(axis=1)
    normalized_margin = self_confidence - alternative

    thresholds = np.zeros(n_classes, dtype=float)
    for class_index in range(n_classes):
        mask = y == class_index
        if np.any(mask):
            thresholds[class_index] = float(self_confidence[mask].mean())
        else:
            thresholds[class_index] = 0.5

    confident_pred = np.empty(n_rows, dtype=int)
    for row_index in range(n_rows):
        probabilities = proba[row_index]
        above = probabilities >= thresholds
        if not np.any(above):
            confident_pred[row_index] = int(np.argmax(probabilities))
        else:
            eligible = np.where(above, probabilities, -np.inf)
            confident_pred[row_index] = int(np.argmax(eligible))

    joint = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(joint, (y, confident_pred), 1)
    row_sums = joint.sum(axis=1).astype(float)
    noise_rates = np.zeros(n_classes, dtype=float)
    for class_index in range(n_classes):
        if row_sums[class_index] > 0:
            off = row_sums[class_index] - joint[class_index, class_index]
            noise_rates[class_index] = off / row_sums[class_index]

    off_diagonal = confident_pred != y
    margin_suspicion = (1.0 - normalized_margin) / 2.0
    self_suspicion = 1.0 - self_confidence
    cl_score = (
        SELF_CONFIDENCE_WEIGHT * self_suspicion
        + MARGIN_WEIGHT * margin_suspicion
        + OFF_DIAGONAL_WEIGHT * off_diagonal.astype(float)
    )
    return {
        "self_confidence": self_confidence,
        "normalized_margin": normalized_margin,
        "thresholds": thresholds,
        "confident_pred": confident_pred,
        "confident_joint": joint,
        "noise_rates": noise_rates,
        "off_diagonal": off_diagonal,
        "cl_score": cl_score,
        "argmax": np.argmax(proba, axis=1).astype(int),
        "confidence": proba.max(axis=1),
    }


def knn_disagreement(given: int, neighbor_labels: list[int]) -> float:
    """Share of neighbors whose label differs from the given label."""

    if not neighbor_labels:
        return 0.0
    mismatches = sum(1 for label in neighbor_labels if int(label) != int(given))
    return mismatches / len(neighbor_labels)


def ensemble_scores(cl_score: np.ndarray, knn_score: np.ndarray) -> np.ndarray:
    """Blend the confident-learning score with neighbor disagreement."""

    return CL_ENSEMBLE_WEIGHT * np.asarray(cl_score, dtype=float) + KNN_ENSEMBLE_WEIGHT * np.asarray(
        knn_score, dtype=float
    )


def review_status(
    predicted: str,
    given: str,
    self_confidence: float,
    threshold: float,
    knn_score: float,
) -> str:
    """Mark a row Check when the model or its neighbors dispute the label."""

    if predicted != given:
        return "Check"
    if self_confidence < threshold and knn_score >= KNN_CHECK_FRACTION:
        return "Check"
    return "Clean"
