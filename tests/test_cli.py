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


def test_cli_refuses_to_overwrite_the_csv(tmp_path: Path):
    source = tmp_path / "tiny.csv"
    source.write_text("x,label\n0,a\n1,b\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="overwrite"):
        main(["audit", "--csv", str(source), "--label", "label", "--out", str(source)])
