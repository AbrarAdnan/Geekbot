"""Smoke tests for step S0.1: the package installs, imports from src/ and exposes its command.

They test the wiring (pyproject metadata, the editable install, the console script), not
features. If one fails, the environment itself is broken and every other test result would
be meaningless.
"""

from importlib.metadata import entry_points, version
from pathlib import Path

import pytest

import geekbot
from geekbot.cli import main


def test_version_comes_from_installed_metadata() -> None:
    # pyproject.toml is the single source of truth; the package must not hard-code a copy.
    assert geekbot.__version__ == version("geekbot")


def test_package_is_imported_from_this_checkouts_src() -> None:
    # The editable install must point at this repo's src/ folder. If this fails, the tests
    # are running against some other copy of geekbot (e.g. a stale non-editable install).
    assert geekbot.__file__ is not None
    package_dir = Path(geekbot.__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]
    assert package_dir == repo_root / "src" / "geekbot"


def test_console_script_is_registered() -> None:
    # pip generates the `geekbot` launcher from this entry point at install time. Exactly
    # one must exist; two would mean two installs of geekbot are visible.
    (script,) = entry_points(group="console_scripts", name="geekbot")
    assert script.value == "geekbot.cli:main"


def test_version_flag_prints_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse's version action prints, then exits with code 0.
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"geekbot {geekbot.__version__}"
