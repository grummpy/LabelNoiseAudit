import numpy as np

from labelnoiseaudit.audit import audit_matrix
from labelnoiseaudit.scoring import (
    CL_ENSEMBLE_WEIGHT,
    KNN_ENSEMBLE_WEIGHT,
    MARGIN_WEIGHT,
    OFF_DIAGONAL_WEIGHT,
    SELF_CONFIDENCE_WEIGHT,
    ensemble_scores,
    score_predictions,
)


def test_weights_sum_to_one():
    assert SELF_CONFIDENCE_WEIGHT + MARGIN_WEIGHT + OFF_DIAGONAL_WEIGHT == 1
    assert CL_ENSEMBLE_WEIGHT + KNN_ENSEMBLE_WEIGHT == 1


def test_confident_joint_on_a_constructed_matrix():
    proba = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.10, 0.90],
            [0.85, 0.15],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    scored = score_predictions(proba, labels)
    assert np.allclose(scored["thresholds"], [0.55, 0.525])
    assert scored["confident_pred"].tolist() == [0, 1, 1, 0]
    assert scored["confident_joint"].tolist() == [[1, 1], [1, 1]]
    assert np.allclose(scored["noise_rates"], [0.5, 0.5])
    assert scored["normalized_margin"][1] == np.float64(-0.6) or np.isclose(
        scored["normalized_margin"][1], -0.6
    )
    assert scored["cl_score"][1] > scored["cl_score"][0]
    assert scored["cl_score"][3] > scored["cl_score"][2]


def test_ensemble_formula():
    cl_score = np.array([0.2, 0.8])
    knn_score = np.array([1.0, 0.0])
    blended = ensemble_scores(cl_score, knn_score)
    assert np.allclose(blended, 0.7 * cl_score + 0.3 * knn_score)


def test_known_flipped_label_ranks_first():
    rng = np.random.default_rng(0)
    count = 50
    class_zero = rng.normal(loc=0.0, scale=0.35, size=(count, 4))
    class_one = rng.normal(loc=8.0, scale=0.35, size=(count, 4))
    features = np.vstack([class_zero, class_one])
    labels = np.array([0] * count + [1] * count)
    labels[0] = 1
    result = audit_matrix(features, labels, seed=0, n_splits=5, use_cleanlab=False)
    top = result.rows[0]
    assert top.source_index == 0
    assert top.given_label == "1"
    assert top.predicted_label == "0"
    assert top.self_confidence < 0.5
    assert top.status == "Check"
    assert any("predicts" in reason for reason in top.reasons)
    assert any("noise rate" in reason for reason in top.reasons)
    clean = [row for row in result.rows_by_source if row.source_index != 0]
    assert top.ensemble_score > float(np.median([row.ensemble_score for row in clean]))


def test_same_seed_repeats():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(40, 3))
    labels = np.array([0] * 20 + [1] * 20)
    first = audit_matrix(features, labels, seed=7, use_cleanlab=False)
    second = audit_matrix(features, labels, seed=7, use_cleanlab=False)
    assert np.allclose(
        [row.ensemble_score for row in first.rows_by_source],
        [row.ensemble_score for row in second.rows_by_source],
    )
