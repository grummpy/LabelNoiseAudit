from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labelnoiseaudit.audit import audit_image_records, audit_table
from labelnoiseaudit.datasets import materialize_shapes, synthetic_text_frame
from labelnoiseaudit.features import build_preprocessor, infer_column_kinds, infer_kind
from labelnoiseaudit.images import (
    classic_features,
    feature_length,
    iter_class_folder,
    safe_extract_zip,
)


def test_column_kinds_distinguish_numbers_categories_and_text():
    frame = pd.DataFrame(
        {
            "age": [1, 2, 3, 4],
            "city": ["lima", "oslo", "lima", "oslo"],
            "note": [
                "satellite thruster telemetry from the ground station",
                "soil sprout watering can under the trellis",
                "simmer the broth then fold in the herb",
                "the buoy line held the dock against the tide",
            ],
        }
    )
    assert infer_kind(frame["age"]) == "numeric"
    assert infer_kind(frame["city"]) == "categorical"
    assert infer_kind(frame["note"]) == "text"
    kinds = infer_column_kinds(frame, list(frame.columns))
    assert kinds == {
        "numeric": ["age"],
        "categorical": ["city"],
        "text": ["note"],
    }
    first = build_preprocessor(kinds)
    second = build_preprocessor(kinds)
    assert first is not second


def test_image_features_are_deterministic(tmp_path: Path):
    root = materialize_shapes(tmp_path / "shapes", n_per_class=2, seed=3)
    records = iter_class_folder(root)
    assert len(records) == 4
    first = classic_features(records[0][0])
    second = classic_features(records[0][0])
    other = classic_features(records[-1][0])
    assert first.shape == (feature_length(),)
    assert np.allclose(first, second)
    assert not np.allclose(first, other)
    assert np.isfinite(first).all()


def test_synthetic_text_is_generated_not_copied():
    frame = synthetic_text_frame(n_per_class=5, seed=1)
    assert set(frame["topic"]) == {"orbit", "garden", "kitchen", "harbor"}
    assert frame["note"].str.contains(" ").all()
    again = synthetic_text_frame(n_per_class=5, seed=1)
    assert frame["note"].tolist() == again["note"].tolist()


def test_text_and_image_audits_run(tmp_path: Path):
    notes = synthetic_text_frame(n_per_class=8, seed=0)
    text_result = audit_table(
        notes,
        "topic",
        feature_columns=["note"],
        seed=0,
        n_splits=4,
        use_cleanlab=False,
    )
    assert text_result.summary["n_rows"] == len(notes)
    assert text_result.n_splits == 4

    root = materialize_shapes(tmp_path / "shapes", n_per_class=6, seed=1)
    image_result = audit_image_records(
        iter_class_folder(root),
        n_splits=3,
        seed=1,
        use_cleanlab=False,
    )
    assert image_result.summary["n_rows"] == 12
    assert all(row.image_path for row in image_result.rows)
    assert all(row.source_index not in row.neighbor_indices for row in image_result.rows_by_source)


def test_zip_slip_is_rejected(tmp_path: Path):
    import zipfile

    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "nope")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_zip(archive_path, tmp_path / "out")
