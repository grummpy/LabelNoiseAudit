"""Write a cleaned CSV and an audit log without touching the original file."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from labelnoiseaudit.serialize import to_builtin, write_json

SOURCE_INDEX = "__lna_source_index"


class OverwriteRefused(Exception):
    """Raised when an export path is the original dataset."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return to_builtin(value)


def export_review(
    frame: pd.DataFrame,
    label_column: str,
    decisions: dict,
    cleaned_path: Path,
    log_path: Path,
    *,
    original_path: Path | None = None,
    audit_meta: dict | None = None,
) -> dict:
    """Apply keep / relabel / drop decisions.

    Rows without a decision are kept. Dropped rows are omitted from the cleaned
    CSV and still recorded in the log. Image files are never modified.
    """

    if label_column not in frame.columns:
        raise ValueError(f"Label column {label_column!r} is not in the table")
    cleaned_path = Path(cleaned_path)
    log_path = Path(log_path)
    original = Path(original_path).resolve() if original_path is not None else None
    before = None
    if original is not None:
        cleaned_resolved = cleaned_path.resolve()
        log_resolved = log_path.resolve()
        if original.is_file() and (cleaned_resolved == original or log_resolved == original):
            raise OverwriteRefused(
                "Refusing to overwrite the original dataset. Choose a new file name."
            )
        if original.is_dir() and (
            original in cleaned_resolved.parents
            or original in log_resolved.parents
            or cleaned_resolved == original
            or log_resolved == original
        ):
            raise OverwriteRefused(
                "Refusing to write the export inside the original image folder."
            )
        if original.is_file():
            before = sha256_file(original)

    work = frame.copy()
    if SOURCE_INDEX not in work.columns:
        work.insert(0, SOURCE_INDEX, np.arange(len(work)))

    kept_rows = []
    log_entries = []
    for _index, row in work.iterrows():
        source_index = int(row[SOURCE_INDEX])
        decision = decisions.get(str(source_index), {"action": "keep"})
        if not isinstance(decision, dict):
            raise ValueError(f"Decision for row {source_index} must be an object")
        action = decision.get("action", "keep")
        if action not in {"keep", "relabel", "drop"}:
            raise ValueError(f"Unknown action {action!r} for row {source_index}")
        original_label = _cell(row[label_column])
        exported_label = original_label
        if action == "relabel":
            if "new_label" not in decision or str(decision["new_label"]).strip() == "":
                raise ValueError(f"Relabel for row {source_index} needs a new_label")
            exported_label = _cell(decision["new_label"])
        log_entries.append(
            {
                "source_index": source_index,
                "action": action,
                "original_label": original_label,
                "exported_label": None if action == "drop" else exported_label,
            }
        )
        if action == "drop":
            continue
        exported = row.drop(labels=[SOURCE_INDEX])
        exported[label_column] = decision["new_label"] if action == "relabel" else row[label_column]
        kept_rows.append(exported)

    export_columns = [column for column in work.columns if column != SOURCE_INDEX]
    if kept_rows:
        cleaned = pd.DataFrame(kept_rows)
        cleaned = cleaned.loc[:, [column for column in export_columns if column in cleaned.columns]]
    else:
        cleaned = work.loc[:, export_columns].iloc[0:0]
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(cleaned_path, index=False)
    payload = {
        "original_path": str(original) if original else None,
        "original_sha256": before,
        "cleaned_path": str(cleaned_path.resolve()),
        "log_path": str(log_path.resolve()),
        "label_column": label_column,
        "n_in": int(len(work)),
        "n_out": int(len(cleaned)),
        "n_dropped": sum(1 for entry in log_entries if entry["action"] == "drop"),
        "n_relabeled": sum(1 for entry in log_entries if entry["action"] == "relabel"),
        "n_kept": sum(1 for entry in log_entries if entry["action"] == "keep"),
        "images_modified": False,
        "decisions": log_entries,
        "audit": audit_meta or {},
    }
    if original is not None and original.is_file():
        after = sha256_file(original)
        payload["original_sha256_after"] = after
        if after != before:
            raise RuntimeError("The original dataset changed during export.")
    write_json(log_path, payload)
    return payload
