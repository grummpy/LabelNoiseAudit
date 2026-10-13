"""Local, versioned provenance for out-of-fold audit reports.

The report is evidence for ranking and human review.  It is deliberately not
presented as a final performance evaluation: callers must provide a separate,
untouched evaluation set for that purpose.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "labelnoiseaudit.oof-provenance/v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(encoded.encode("utf-8"))


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash table values and order without exposing their contents in the report."""

    csv = frame.to_csv(index=True, lineterminator="\n")
    return sha256_bytes(csv.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_revision() -> str:
    """Return a checked-out revision when available, without requiring Git."""

    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dependency_versions() -> dict[str, str]:
    names = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pillow",
        "flask",
        "torch",
        "torchvision",
        "cleanlab",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def table_source(frame: pd.DataFrame, *, source_path: Path | None = None) -> dict[str, object]:
    source: dict[str, object] = {
        "kind": "table",
        "row_count": int(len(frame)),
        "content_sha256": dataframe_sha256(frame),
    }
    if source_path is not None and source_path.is_file():
        # A basename identifies the local source without unnecessarily exposing
        # a complete home-directory path in a report that users may share.
        source["file_name"] = source_path.name
        source["file_sha256"] = file_sha256(source_path)
    return source


def image_source(records: list[tuple[str, str]]) -> dict[str, object]:
    """Hash image bytes and labels; no image is changed or transmitted."""

    rows = []
    for path_text, label in records:
        path = Path(path_text)
        rows.append(
            {
                "row_id": f"{label}/{path.name}",
                "label": str(label),
                "file_sha256": file_sha256(path),
            }
        )
    return {
        "kind": "images",
        "row_count": len(rows),
        "content_sha256": json_sha256(rows),
        "row_ids_are": "label/basename",
    }


def make_provenance(
    *,
    source: dict[str, object],
    class_order: list[str],
    seed: int,
    n_splits: int,
    n_neighbors: int,
    feature_backend: str,
    configuration: dict[str, object],
) -> dict[str, object]:
    settings = {
        "seed": int(seed),
        "n_splits": int(n_splits),
        "n_neighbors": int(n_neighbors),
        "class_order": list(class_order),
        "feature_backend": feature_backend,
        **configuration,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "ranking_and_selection_only",
        "final_evaluation": {
            "included": False,
            "note": "These out-of-fold scores support human review and must not be used as a final performance claim.",
        },
        "source": source,
        "configuration": {**settings, "sha256": json_sha256(settings)},
        "runtime": {
            "project_revision": project_revision(),
            "python": platform.python_version(),
            "dependencies": dependency_versions(),
        },
        "local_only": True,
    }
