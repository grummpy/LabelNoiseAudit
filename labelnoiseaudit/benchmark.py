"""Benchmark noise detection on iris, wine, digits, and synthetic text."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from labelnoiseaudit.audit import audit_matrix, audit_table
from labelnoiseaudit.datasets import benchmark_tasks
from labelnoiseaudit.metrics import ranking_report
from labelnoiseaudit.noise import inject_noise

RATES = (0.05, 0.10, 0.20)
KINDS = ("uniform", "class_conditional")
Progress = Callable[[str], None]


def _scores_in_source_order(result) -> np.ndarray:
    scores = np.zeros(result.summary["n_rows"], dtype=float)
    for row in result.rows_by_source:
        scores[row.source_index] = row.ensemble_score
    return scores


def run_benchmark(seed: int = 42, progress: Progress | None = None) -> dict:
    """Inject 5/10/20% noise, both uniform and class-conditional, and score it."""

    report: dict = {
        "seed": seed,
        "k_definition": (
            "k is the number of injected flips. At that cutoff precision@k equals "
            "recall@k. precision_at_10pct and recall_at_10pct use k = 10% of rows."
        ),
        "rates": list(RATES),
        "kinds": list(KINDS),
        "datasets": {},
    }
    for name, kind, payload, label_column in benchmark_tasks(seed):
        runs = []
        if kind == "matrix":
            features, labels = payload
            n_rows = int(len(labels))
            n_classes = int(len(set(np.asarray(labels).tolist())))
        else:
            n_rows = int(len(payload))
            n_classes = int(payload[label_column].nunique())
        for noise_kind in KINDS:
            for rate in RATES:
                if progress is not None:
                    progress(f"{name} {noise_kind} {rate:.0%}")
                if kind == "matrix":
                    features, labels = payload
                    noisy, mask = inject_noise(labels, rate=rate, kind=noise_kind, seed=seed)
                    result = audit_matrix(
                        features,
                        noisy,
                        seed=seed,
                        n_splits=5,
                        use_cleanlab=False,
                    )
                else:
                    frame: pd.DataFrame = payload
                    noisy, mask = inject_noise(
                        frame[label_column].to_numpy(),
                        rate=rate,
                        kind=noise_kind,
                        seed=seed,
                    )
                    scored = frame.copy()
                    scored[label_column] = noisy
                    result = audit_table(
                        scored,
                        label_column,
                        feature_columns=["note"],
                        seed=seed,
                        n_splits=5,
                        use_cleanlab=False,
                    )
                metrics = ranking_report(_scores_in_source_order(result), mask)
                runs.append(
                    {
                        "noise": noise_kind,
                        "rate": rate,
                        **metrics,
                    }
                )
        report["datasets"][name] = {
            "n_rows": n_rows,
            "n_classes": n_classes,
            "task": kind,
            "runs": runs,
        }
    return report


def markdown_table(report: dict) -> str:
    lines = [
        "| Dataset | Noise | Rate | k | Precision@k | Recall@k | Precision@10% | Recall@10% | AUROC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in report["datasets"].items():
        for run in payload["runs"]:
            lines.append(
                "| {name} | {noise} | {rate:.0%} | {k} | {precision_at_k:.3f} | {recall_at_k:.3f} | "
                "{precision_at_10pct:.3f} | {recall_at_10pct:.3f} | {auroc:.3f} |".format(
                    name=name,
                    **run,
                )
            )
    return "\n".join(lines) + "\n"
