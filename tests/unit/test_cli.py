"""Tests for step S0.2: the Typer CLI. Stub commands must say which step builds them."""

import pytest
from typer.testing import CliRunner

from geekbot import __version__
from geekbot.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"geekbot {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "config" in result.output
    assert "doctor" in result.output


@pytest.mark.parametrize(
    ("args", "step"),
    [
        (["ingest"], "S1.6"),
        (["search", "anything", "--top-k", "3"], "S2.1"),
        (["ask", "what is RAG?"], "S3.1"),
        (["serve"], "S4.1"),
        (["csv", "describe", "x.csv"], "C2"),
        (["csv", "ask", "x.csv", "how many rows?"], "C2"),
    ],
)
def test_stub_commands_exit_2_and_name_their_step(args: list[str], step: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert f"step {step}" in result.output
