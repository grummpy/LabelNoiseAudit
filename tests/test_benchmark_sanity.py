import json
from pathlib import Path

import numpy as np
import pytest

from labelnoiseaudit.audit import audit_matrix
from labelnoiseaudit.datasets import load_builtin_frame
from labelnoiseaudit.metrics import ranking_report
from labelnoiseaudit.noise import inject_noise

ROOT = Path(__file__).resolve().parents[1]


def test_detection_beats_random_on_separated_clusters():
    rng = np.random.default_rng(4)
    features = np.vstack(
        [
            rng.normal(0.0, 0.4, size=(80, 4)),
            rng.normal(5.0, 0.4, size=(80, 4)),
            rng.normal([0.0, 5.0, 0.0, 5.0], 0.4, size=(80, 4)),
        ]
    )
    labels = np.array([0] * 80 + [1] * 80 + [2] * 80)
    noisy, mask = inject_noise(labels, rate=0.10, kind="uniform", seed=4)
    result = audit_matrix(features, noisy, seed=4, use_cleanlab=False)
    scores = np.array([row.ensemble_score for row in result.rows_by_source])
    report = ranking_report(scores, mask)
    assert report["auroc"] > 0.8
    assert report["precision_at_k"] > report["prevalence"]
    assert report["recall_at_k"] == report["precision_at_k"]


def test_published_benchmark_beats_random_and_iris_matches():
    published = json.loads((ROOT / "docs" / "benchmark.json").read_text(encoding="utf-8"))
    assert published["seed"] == 42
    for name, payload in published["datasets"].items():
        assert payload["runs"], name
        for run in payload["runs"]:
            assert run["auroc"] > 0.5, (name, run)
            assert run["precision_at_k"] > run["prevalence"], (name, run)

    cell = next(
        run
        for run in published["datasets"]["iris"]["runs"]
        if run["noise"] == "uniform" and run["rate"] == 0.1
    )
    frame, label, _mode = load_builtin_frame("iris")
    features = frame.drop(columns=[label]).to_numpy(dtype=float)
    noisy, mask = inject_noise(frame[label].to_numpy(), rate=0.1, kind="uniform", seed=42)
    result = audit_matrix(features, noisy, seed=42, n_splits=5, use_cleanlab=False)
    scores = np.array([row.ensemble_score for row in result.rows_by_source])
    fresh = ranking_report(scores, mask)
    assert fresh["auroc"] == pytest.approx(cell["auroc"], abs=1e-9)
    assert fresh["precision_at_k"] == pytest.approx(cell["precision_at_k"], abs=1e-9)
