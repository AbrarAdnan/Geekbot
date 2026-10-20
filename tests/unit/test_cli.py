"""Tests for step S0.2: the Typer CLI. Stub commands must say which step builds them."""

import re

import pytest
import typer
from typer.testing import CliRunner

from geekbot import __version__
from geekbot.cli import NOT_BUILT_EXIT_CODE, _not_built, app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"geekbot {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "config" in result.output
    assert "doctor" in result.output


def test_not_built_exits_2_and_names_its_step() -> None:
    """No command is a stub right now (`serve` was the last, S4.1); the helper stays for the next."""
    with pytest.raises(typer.Exit) as raised:
        _not_built("future", "S9.9")
    assert raised.value.exit_code == NOT_BUILT_EXIT_CODE == 2


def test_serve_has_no_host_or_workers_option() -> None:
    """PLAN §6.2: one process, this machine only; the command offers no way around that."""
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    # CI turns colours on, and the colour codes split an option name into styled pieces.
    text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--port" in text
    assert "--workers" not in text
    assert "--host" not in text
