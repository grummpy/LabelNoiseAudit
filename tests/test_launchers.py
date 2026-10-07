import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_files_exist_and_point_at_the_module():
    command = ROOT / "Launch LabelNoiseAudit.command"
    windows = ROOT / "Launch LabelNoiseAudit.bat"
    shell = ROOT / "launch.sh"
    desktop = ROOT / "labelnoiseaudit.desktop"
    for path in (command, windows, shell, desktop):
        assert path.is_file(), path
    assert os.access(command, os.X_OK)
    assert os.access(shell, os.X_OK)
    for path in (command, windows, shell):
        text = path.read_text(encoding="utf-8")
        assert "labelnoiseaudit" in text
        assert "https://www.python.org/downloads/" in text
        assert "-m labelnoiseaudit" in text
    desktop_text = desktop.read_text(encoding="utf-8")
    assert "launch.sh" in desktop_text
    assert "icon" in desktop_text.lower()
    assert b"\r\n" in windows.read_bytes()
