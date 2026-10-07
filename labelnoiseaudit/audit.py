"""Run an audit: out-of-fold probabilities, confident learning, and neighbors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from labelnoiseaudit.cleanlab_check import cleanlab_crosscheck
from labelnoiseaudit.features import (
    build_preprocessor,
    infer_column_kinds,
    normalize_kinds,
    prepare_frame,
)
from labelnoiseaudit.images import classic_features
from labelnoiseaudit.oof import run_oof
from labelnoiseaudit.scoring import (
    ensemble_scores,
    knn_disagreement,
    review_status,
    score_predictions,
)

Progress = Callable[[str], None]


@dataclass
class RowFinding:
    row_id: str
    source_index: int
    given_label: str
    predicted_label: str
    confidence: float
    self_confidence: float
    normalized_margin: float
    cl_score: float
    knn_score: float
    ensemble_score: float
    neighbor_labels: list[str]
    neighbor_indices: list[int]
    class_noise_rate: float
    off_diagonal: bool
    status: str
    reasons: list[str]
    image_path: str | None = None
    cleanlab_flag: bool = False


@dataclass
class AuditResult:
    rows: list[RowFinding]
    class_names: list[str]
    thresholds: list[float]
    confident_joint: list[list[int]]
    noise_rates: dict[str, float]
    cleanlab: dict
    n_splits: int
    seed: int
    summary: dict
    rows_by_source: list[RowFinding] = field(default_factory=list)


def _encode(labels) -> tuple[np.ndarray, list[str]]:
    raw = [str(value) for value in np.asarray(labels).tolist()]
    class_names = sorted(set(raw))
    if len(class_names) < 2:
        raise ValueError("Need at least two classes to score label noise.")
    index = {name: position for position, name in enumerate(class_names)}
    encoded = np.array([index[value] for value in raw], dtype=int)
    return encoded, class_names


def _reasons(
    *,
    given: str,
    predicted: str,
    confidence: float,
    self_confidence: float,
    normalized_margin: float,
    neighbor_labels: list[str],
    noise_rate: float,
    off_diagonal: bool,
    cleanlab_flag: bool,
) -> list[str]:
    reasons = []
    if predicted != given:
        reasons.append(
            f"Given label {given!r}, model predicts {predicted!r} with confidence {confidence:.2f}."
        )
    else:
        reasons.append(f"Model agrees with {given!r} at confidence {confidence:.2f}.")
    reasons.append(
        f"Self-confidence on the given label is {self_confidence:.2f} "
        f"(normalized margin {normalized_margin:+.2f})."
    )
    if neighbor_labels:
        counts = Counter(neighbor_labels)
        mix = ", ".join(f"{label} ×{count}" for label, count in counts.most_common())
        disagree = sum(1 for label in neighbor_labels if label != given)
        reasons.append(
            f"{disagree} of {len(neighbor_labels)} nearest neighbors disagree ({mix})."
        )
    else:
        reasons.append("No nearest neighbors were available for this row.")
    reasons.append(f"Estimated noise rate for class {given!r} is {noise_rate:.0%}.")
    if off_diagonal:
        reasons.append(
            "Confident joint places this row off the diagonal: the thresholded "
            "prediction differs from the given label."
        )
    if cleanlab_flag:
        reasons.append("Cleanlab also ranks this row as a label issue.")
    return reasons


def assemble_result(
    *,
    proba: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    neighbor_labels: list[list[int]],
    neighbor_indices: list[list[int]],
    n_splits: int,
    seed: int,
    row_ids: list[str] | None = None,
    image_paths: list[str | None] | None = None,
    use_cleanlab: bool = True,
) -> AuditResult:
    scored = score_predictions(proba, y)
    cleanlab = cleanlab_crosscheck(y, proba) if use_cleanlab else {
        "available": False,
        "error": None,
        "flagged_indices": [],
    }
    flagged = set(int(index) for index in cleanlab.get("flagged_indices", []))
    knn_scores = np.array(
        [
            knn_disagreement(int(label), neighbors)
            for label, neighbors in zip(y, neighbor_labels, strict=True)
        ],
        dtype=float,
    )
    blended = ensemble_scores(scored["cl_score"], knn_scores)
    findings: list[RowFinding] = []
    for index in range(len(y)):
        given_index = int(y[index])
        given = class_names[given_index]
        predicted = class_names[int(scored["argmax"][index])]
        neighbors = [class_names[int(label)] for label in neighbor_labels[index]]
        noise_rate = float(scored["noise_rates"][given_index])
        cleanlab_flag = index in flagged
        self_confidence = float(scored["self_confidence"][index])
        margin = float(scored["normalized_margin"][index])
        confidence = float(scored["confidence"][index])
        knn_score = float(knn_scores[index])
        status = review_status(
            predicted,
            given,
            self_confidence,
            float(scored["thresholds"][given_index]),
            knn_score,
        )
        row_id = row_ids[index] if row_ids is not None else str(index)
        image_path = image_paths[index] if image_paths is not None else None
        findings.append(
            RowFinding(
                row_id=row_id,
                source_index=index,
                given_label=given,
                predicted_label=predicted,
                confidence=confidence,
                self_confidence=self_confidence,
                normalized_margin=margin,
                cl_score=float(scored["cl_score"][index]),
                knn_score=knn_score,
                ensemble_score=float(blended[index]),
                neighbor_labels=neighbors,
                neighbor_indices=list(neighbor_indices[index]),
                class_noise_rate=noise_rate,
                off_diagonal=bool(scored["off_diagonal"][index]),
                status=status,
                reasons=_reasons(
                    given=given,
                    predicted=predicted,
                    confidence=confidence,
                    self_confidence=self_confidence,
                    normalized_margin=margin,
                    neighbor_labels=neighbors,
                    noise_rate=noise_rate,
                    off_diagonal=bool(scored["off_diagonal"][index]),
                    cleanlab_flag=cleanlab_flag,
                ),
                image_path=image_path,
                cleanlab_flag=cleanlab_flag,
            )
        )
    by_source = list(findings)
    ranked = sorted(
        findings,
        key=lambda row: (-row.ensemble_score, -row.cl_score, row.source_index),
    )
    noise_rates = {
        class_names[index]: float(scored["noise_rates"][index]) for index in range(len(class_names))
    }
    n_flagged = sum(1 for row in ranked if row.status == "Check")
    summary = {
        "n_rows": len(ranked),
        "n_classes": len(class_names),
        "n_flagged": n_flagged,
        "flag_rate": (n_flagged / len(ranked)) if ranked else 0.0,
        "n_splits": n_splits,
        "seed": seed,
    }
    return AuditResult(
        rows=ranked,
        class_names=list(class_names),
        thresholds=[float(value) for value in scored["thresholds"]],
        confident_joint=scored["confident_joint"].astype(int).tolist(),
        noise_rates=noise_rates,
        cleanlab=cleanlab,
        n_splits=n_splits,
        seed=seed,
        summary=summary,
        rows_by_source=by_source,
    )


def result_to_dict(result: AuditResult) -> dict:
    def row_dict(row: RowFinding, rank: int | None) -> dict:
        payload = {
            "rank": rank,
            "row_id": row.row_id,
            "source_index": row.source_index,
            "given_label": row.given_label,
            "predicted_label": row.predicted_label,
            "confidence": row.confidence,
            "self_confidence": row.self_confidence,
            "normalized_margin": row.normalized_margin,
            "cl_score": row.cl_score,
            "knn_score": row.knn_score,
            "ensemble_score": row.ensemble_score,
            "neighbor_labels": row.neighbor_labels,
            "neighbor_indices": row.neighbor_indices,
            "class_noise_rate": row.class_noise_rate,
            "off_diagonal": row.off_diagonal,
            "status": row.status,
            "reasons": row.reasons,
            "image_path": row.image_path,
            "cleanlab_flag": row.cleanlab_flag,
        }
        return payload

    return {
        "class_names": result.class_names,
        "thresholds": result.thresholds,
        "confident_joint": result.confident_joint,
        "noise_rates": result.noise_rates,
        "cleanlab": result.cleanlab,
        "n_splits": result.n_splits,
        "seed": result.seed,
        "summary": result.summary,
        "rows": [row_dict(row, rank) for rank, row in enumerate(result.rows, start=1)],
    }


def audit_matrix(
    features,
    labels,
    *,
    n_splits: int = 5,
    seed: int = 42,
    n_neighbors: int = 5,
    row_ids: list[str] | None = None,
    image_paths: list[str | None] | None = None,
    progress: Progress | None = None,
    preprocessor_factory: Callable[[], object] | None = None,
    use_cleanlab: bool = True,
) -> AuditResult:
    array = np.asarray(features, dtype=float)
    y, class_names = _encode(labels)
    if len(array) != len(y):
        raise ValueError("Feature rows and labels differ in length")

    def default_factory():
        return StandardScaler()

    factory = preprocessor_factory or default_factory
    proba, neighbor_labels, neighbor_indices, used_splits = run_oof(
        array,
        y,
        factory,
        n_splits=n_splits,
        seed=seed,
        n_neighbors=n_neighbors,
        progress=progress,
    )
    return assemble_result(
        proba=proba,
        y=y,
        class_names=class_names,
        neighbor_labels=neighbor_labels,
        neighbor_indices=neighbor_indices,
        n_splits=used_splits,
        seed=seed,
        row_ids=row_ids,
        image_paths=image_paths,
        use_cleanlab=use_cleanlab,
    )


def audit_table(
    frame: pd.DataFrame,
    label_column: str,
    feature_columns: list[str] | None = None,
    *,
    column_kinds: dict[str, list[str]] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    n_neighbors: int = 5,
    row_ids: list[str] | None = None,
    progress: Progress | None = None,
    preprocessor_factory: Callable[[], object] | None = None,
    use_cleanlab: bool = True,
) -> AuditResult:
    if label_column not in frame.columns:
        raise ValueError(f"Label column {label_column!r} is not in the table")
    features = list(feature_columns) if feature_columns is not None else [
        str(column) for column in frame.columns if str(column) != label_column
    ]
    if label_column in features:
        raise ValueError("The label column cannot also be a feature")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if not features:
        raise ValueError("Select at least one feature column")
    kinds = normalize_kinds(column_kinds or infer_column_kinds(frame, features), features)
    prepared = prepare_frame(frame, kinds)
    y, class_names = _encode(frame[label_column].tolist())

    def default_factory():
        return build_preprocessor(kinds)

    factory = preprocessor_factory or default_factory
    if progress is not None:
        progress("Fitting out-of-fold models")
    proba, neighbor_labels, neighbor_indices, used_splits = run_oof(
        prepared,
        y,
        factory,
        n_splits=n_splits,
        seed=seed,
        n_neighbors=n_neighbors,
        progress=progress,
    )
    ids = row_ids
    if ids is None:
        ids = [str(value) for value in frame.index.tolist()]
    return assemble_result(
        proba=proba,
        y=y,
        class_names=class_names,
        neighbor_labels=neighbor_labels,
        neighbor_indices=neighbor_indices,
        n_splits=used_splits,
        seed=seed,
        row_ids=ids,
        use_cleanlab=use_cleanlab,
    )


def audit_image_records(
    records: list[tuple[str, str]],
    *,
    n_splits: int = 5,
    seed: int = 42,
    n_neighbors: int = 5,
    embeddings: bool = False,
    allow_download: bool = False,
    progress: Progress | None = None,
    use_cleanlab: bool = True,
) -> AuditResult:
    if not records:
        raise ValueError("No images to audit")
    paths = [path for path, _label in records]
    labels = [label for _path, label in records]
    if embeddings:
        from labelnoiseaudit.embeddings import embed_paths

        matrix = embed_paths(paths, allow_download=allow_download, progress=progress)
    else:
        vectors = []
        for index, path in enumerate(paths, start=1):
            if progress is not None and (index == 1 or index % 10 == 0 or index == len(paths)):
                progress(f"Image features {index}/{len(paths)}")
            vectors.append(classic_features(path))
        matrix = np.vstack(vectors)
    row_ids = [f"{label}/{Path(path).name}" for path, label in records]
    return audit_matrix(
        matrix,
        labels,
        n_splits=n_splits,
        seed=seed,
        n_neighbors=n_neighbors,
        row_ids=row_ids,
        image_paths=paths,
        progress=progress,
        use_cleanlab=use_cleanlab,
    )

