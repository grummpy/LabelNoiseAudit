from pathlib import Path

import pandas as pd
import pytest

import labelnoiseaudit.export as export_module
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


def test_export_refuses_hard_linked_output_before_changing_the_original(tmp_path):
    source = tmp_path / "original.csv"
    _frame().to_csv(source, index=False)
    cleaned = tmp_path / "cleaned.csv"
    cleaned.hardlink_to(source)
    before = source.read_bytes()

    with pytest.raises(OverwriteRefused, match="original dataset"):
        export_review(
            _frame(),
            "label",
            {"0": {"action": "relabel", "new_label": "moon"}},
            cleaned,
            tmp_path / "log.json",
            original_path=source,
        )

    assert source.read_bytes() == before


def test_export_refuses_identical_cleaned_and_log_destinations(tmp_path):
    destination = tmp_path / "export.json"

    with pytest.raises(OverwriteRefused, match="different file paths"):
        export_review(_frame(), "label", {}, destination, destination)

    assert not destination.exists()


def test_export_refuses_cleaned_and_log_destinations_with_same_file_identity(tmp_path):
    cleaned = tmp_path / "cleaned.csv"
    log_path = tmp_path / "audit-log.json"
    cleaned.write_text("previous-cleaned-output", encoding="utf-8")
    log_path.hardlink_to(cleaned)

    with pytest.raises(OverwriteRefused, match="different file paths"):
        export_review(_frame(), "label", {}, cleaned, log_path)

    assert cleaned.read_text(encoding="utf-8") == "previous-cleaned-output"
    assert log_path.read_text(encoding="utf-8") == "previous-cleaned-output"


def test_export_does_not_publish_cleaned_csv_when_staged_log_write_fails(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned.csv"
    log_path = tmp_path / "audit-log.json"
    cleaned.write_text("previous-cleaned-output", encoding="utf-8")
    log_path.write_text('{"previous": "receipt"}', encoding="utf-8")

    def fail_log_write(_path, _payload):
        raise OSError("synthetic log write failure")

    monkeypatch.setattr(export_module, "write_json", fail_log_write)

    with pytest.raises(OSError, match="synthetic log write failure"):
        export_review(_frame(), "label", {}, cleaned, log_path)

    assert cleaned.read_text(encoding="utf-8") == "previous-cleaned-output"
    assert log_path.read_text(encoding="utf-8") == '{"previous": "receipt"}'
    assert list(tmp_path.glob(".*.stage*")) == []


def _fail_final_log_replace(monkeypatch, log_path):
    original_replace = Path.replace

    def fail_log_publish(path, target):
        if Path(target) == log_path:
            raise OSError("synthetic final log publish failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_log_publish)


def test_export_restores_prior_csv_when_final_log_publish_fails(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned.csv"
    log_path = tmp_path / "audit-log.json"
    cleaned.write_text("previous-cleaned-output", encoding="utf-8")
    log_path.write_text('{"previous": "receipt"}', encoding="utf-8")
    _fail_final_log_replace(monkeypatch, log_path)

    with pytest.raises(OSError, match="synthetic final log publish failure"):
        export_review(_frame(), "label", {}, cleaned, log_path)

    assert cleaned.read_text(encoding="utf-8") == "previous-cleaned-output"
    assert log_path.read_text(encoding="utf-8") == '{"previous": "receipt"}'
    assert list(tmp_path.glob(".*.stage*")) == []


def test_export_removes_new_csv_when_final_log_publish_fails(tmp_path, monkeypatch):
    cleaned = tmp_path / "cleaned.csv"
    log_path = tmp_path / "audit-log.json"
    _fail_final_log_replace(monkeypatch, log_path)

    with pytest.raises(OSError, match="synthetic final log publish failure"):
        export_review(_frame(), "label", {}, cleaned, log_path)

    assert not cleaned.exists()
    assert not log_path.exists()
    assert list(tmp_path.glob(".*.stage*")) == []


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
