import os
import shutil
import subprocess
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


def test_linux_launcher_repairs_an_interrupted_environment(tmp_path: Path):
    """A package __init__ can import while the actual CLI dependencies are absent."""
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "launch.sh", project / "launch.sh")
    (project / "requirements.txt").write_text("# mocked installer input\n", encoding="utf-8")
    venv_python = project / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    fake_python = tmp_path / "python3.13"
    fake_python.write_text(
        """#!/bin/sh
set -eu
if [ \"$1\" = \"-c\" ]; then
  case \"$2\" in
    *version_info*) exit 0 ;;
    *labelnoiseaudit*) test -f .venv/.labelnoiseaudit-installed ;;
  esac
fi
if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then
  echo \"$*\" >> \"$LNA_TRACE\"
  case \"$*\" in
    *\" pip check\") test -f .venv/.dependencies-installed ;;
    *\" -r requirements.txt\") touch .venv/.dependencies-installed .venv/.runtime-dependencies-installed ;;
    *\" -e .\"*) touch .venv/.labelnoiseaudit-installed ;;
  esac
  exit 0
fi
if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"labelnoiseaudit\" ] && [ \"$3\" = \"--help\" ]; then
  test -f .venv/.runtime-dependencies-installed
  exit 0
fi
if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"labelnoiseaudit\" ]; then
  echo \"started\" >> \"$LNA_TRACE\"
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    shutil.copy(fake_python, venv_python)
    venv_python.chmod(0o755)
    # The old probe accepted this state: the stdlib-only package __init__ and
    # pip metadata are present, but the imports required by the CLI are not.
    (project / ".venv" / ".labelnoiseaudit-installed").touch()
    (project / ".venv" / ".dependencies-installed").touch()
    trace = tmp_path / "trace.txt"
    environment = os.environ | {
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "LNA_TRACE": str(trace),
    }
    result = subprocess.run(
        [str(project / "launch.sh"), "serve"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "-m pip check",
        "-m pip install --upgrade pip",
        "-m pip install -r requirements.txt",
        "-m pip install -e . --no-deps",
        "started",
    ]
