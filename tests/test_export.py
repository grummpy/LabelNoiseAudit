import pandas as pd
import pytest

from labelnoiseaudit.export import OverwriteRefused, export_review, sha256_file
from labelnoiseaudit.serialize import read_json


def _frame():
    return pd.DataFrame(
        {
            "path": ["a.png", "b.png", "c.png", "d.png"],
            "label": ["sun", "sun", "tile", "tile"],
            "note": ["one", "two", "three", "four"],
        }
    )


def test_export_applies_decisions_and_leaves_the_original(tmp_path):
    source = tmp_path / "original.csv"
    frame = _frame()
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    digest = sha256_file(source)
    image = tmp_path / "a.png"
    image.write_bytes(b"not-a-real-png")
    image_before = image.read_bytes()

    cleaned = tmp_path / "cleaned.csv"
    log_path = tmp_path / "audit-log.json"
    decisions = {
        "1": {"action": "relabel", "new_label": "tile"},
        "2": {"action": "drop"},
        "3": {"action": "keep"},
    }
    payload = export_review(
        pd.read_csv(source),
        "label",
        decisions,
        cleaned,
        log_path,
        original_path=source,
    )
    assert source.read_bytes() == before
    assert sha256_file(source) == digest
    assert image.read_bytes() == image_before
    assert payload["images_modified"] is False
    assert payload["original_sha256_after"] == digest
    exported = pd.read_csv(cleaned)
    assert list(exported["label"]) == ["sun", "tile", "tile"]
    assert "b.png" in set(exported["path"])
    assert "c.png" not in set(exported["path"])
    assert list(exported["note"]) == ["one", "two", "four"]
    log = read_json(log_path)
    actions = {entry["source_index"]: entry["action"] for entry in log["decisions"]}
    assert actions == {0: "keep", 1: "relabel", 2: "drop", 3: "keep"}
    assert log["n_out"] == 3
    assert log["n_dropped"] == 1
    assert log["n_relabeled"] == 1


def test_export_refuses_to_overwrite_the_original(tmp_path):
    source = tmp_path / "original.csv"
    _frame().to_csv(source, index=False)
    with pytest.raises(OverwriteRefused):
        export_review(_frame(), "label", {}, source, tmp_path / "log.json", original_path=source)
    with pytest.raises(OverwriteRefused):
        export_review(_frame(), "label", {}, tmp_path / "cleaned.csv", source, original_path=source)


def test_export_refuses_to_write_inside_an_image_folder(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"png")
    with pytest.raises(OverwriteRefused, match="image folder"):
        export_review(
            _frame(),
            "label",
            {},
            folder / "cleaned.csv",
            tmp_path / "audit-log.json",
            original_path=folder,
        )
    assert (folder / "a.png").read_bytes() == b"png"
    exported = export_review(
        _frame(),
        "label",
        {"0": {"action": "drop"}},
        tmp_path / "cleaned.csv",
        tmp_path / "audit-log.json",
        original_path=folder,
    )
    assert exported["images_modified"] is False
    assert (folder / "a.png").read_bytes() == b"png"


def test_relabel_requires_a_new_label(tmp_path):
    with pytest.raises(ValueError, match="new_label"):
        export_review(
            _frame(),
            "label",
            {"0": {"action": "relabel"}},
            tmp_path / "cleaned.csv",
            tmp_path / "log.json",
        )
