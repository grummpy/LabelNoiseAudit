import json
from pathlib import Path

import pandas as pd
import pytest

from labelnoiseaudit.cli import main


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_cli_audit_writes_json_without_touching_the_csv(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2, 3.0, 3.1, 3.2, 0.05, 3.05],
            "y": [0.0, 0.2, -0.1, 3.0, 2.8, 3.2, 0.1, 2.9],
            "label": ["a", "a", "a", "b", "b", "b", "a", "b"],
        }
    )
    source = tmp_path / "tiny.csv"
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    out = tmp_path / "audit.json"
    code = main(
        [
            "audit",
            "--csv",
            str(source),
            "--label",
            "label",
            "--features",
            "x",
            "y",
            "--out",
            str(out),
            "--folds",
            "2",
            "--seed",
            "1",
        ]
    )
    assert code == 0
    assert out.is_file()
    assert "ensemble_score" in out.read_text(encoding="utf-8")
    assert source.read_bytes() == before
    payload = json.loads(out.read_text(encoding="utf-8"))
    provenance = payload["provenance"]
    assert provenance["schema_version"] == "labelnoiseaudit.oof-provenance/v1"
    assert provenance["purpose"] == "ranking_and_selection_only"
    assert provenance["final_evaluation"]["included"] is False
    assert provenance["source"]["file_name"] == "tiny.csv"
    assert len(provenance["source"]["file_sha256"]) == 64
    assert len(provenance["configuration"]["sha256"]) == 64
    for row in payload["rows"]:
        assert 1 <= row["oof_fold"] <= payload["n_splits"]
        assert list(row["oof_probabilities"]) == payload["class_names"]
        assert sum(row["oof_probabilities"].values()) == pytest.approx(1.0)


def test_cli_refuses_to_overwrite_the_csv(tmp_path: Path):
    source = tmp_path / "tiny.csv"
    source.write_text("x,label\n0,a\n1,b\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="overwrite"):
        main(["audit", "--csv", str(source), "--label", "label", "--out", str(source)])
