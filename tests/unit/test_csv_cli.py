"""Tests for step C2: ``geekbot csv describe`` and ``geekbot csv ask``.

The CLI is kept offline: the Ollama check is replaced and the real provider is swapped for a fake
model, so no server or GPU is involved. ``result.output`` holds stdout and stderr together, which
is how the user sees them in a terminal.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from csv_analyst.llm.base import LLMProviderError
from geekbot.cli import app
from geekbot.csv_mode import adapter
from geekbot.csv_mode.adapter import CsvAnalyst

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sales_small.csv"

SQL = "SELECT region, SUM(revenue) AS total FROM data GROUP BY region ORDER BY total DESC"
PLAN = json.dumps(
    {
        "answerability": "answerable",
        "plan": ["sum the revenue per region"],
        "columns_used": ["revenue", "region"],
        "code": SQL,
    }
)
ANSWER = json.dumps({"answer": "North has the highest total revenue, 520."})
REFUSAL = json.dumps(
    {
        "answerability": "data_missing",
        "reason": "There is no salary column.",
        "plan": [],
        "columns_used": [],
        "code": "",
    }
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The CLI builds its own Settings(), which reads GEEKBOT_* variables and a `.env` in the
    # working directory. Neither may leak into the tests.
    for name in list(os.environ):
        if name.startswith("GEEKBOT_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


class FakeLLM:
    """A stand-in model: canned replies in order, or an error on every call."""

    def __init__(self, replies: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.replies = list(replies or [])
        self.error = error

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        if self.error is not None:
            raise self.error
        return self.replies.pop(0)


def _offline(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLLM, *, problem: str | None = None
) -> list[Path]:
    """Skip the Ollama check and use ``fake`` as the model. Returns the files that got opened."""
    opened: list[Path] = []
    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: problem)
    real_open: Callable[..., CsvAnalyst] = CsvAnalyst.open.__func__  # type: ignore[attr-defined]

    def open_with_fake(cls: type[CsvAnalyst], path: Path, settings: Any, **kwargs: Any) -> Any:
        opened.append(path)
        return real_open(cls, path, settings, provider=fake)

    monkeypatch.setattr(CsvAnalyst, "open", classmethod(open_with_fake))
    return opened


# --- csv describe ---------------------------------------------------------------------------


def test_describe_prints_what_was_loaded() -> None:
    result = runner.invoke(app, ["csv", "describe", str(FIXTURE)])
    assert result.exit_code == 0
    assert "sales_small.csv" in result.output
    assert "12 rows, 4 columns" in result.output


def test_describe_of_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["csv", "describe", str(tmp_path / "nope.csv")])
    assert result.exit_code == 2


def test_describe_of_an_unusable_file_exits_1_with_one_error_line(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    result = runner.invoke(app, ["csv", "describe", str(empty)])
    assert result.exit_code == 1
    assert "error:" in result.output


# --- csv ask --------------------------------------------------------------------------------


def test_ask_prints_the_reply_and_a_footer_naming_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offline(monkeypatch, FakeLLM([PLAN, PLAN, ANSWER]))
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "Which region sells the most?"])
    assert result.exit_code == 0
    assert "North" in result.output
    assert "[csv] model=qwen2.5-coder:3b profile=proto status=answered time=" in result.output
    assert "model_load=0ms" in result.output


def test_a_question_the_data_cannot_answer_still_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch, FakeLLM([REFUSAL, REFUSAL]))
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "What is the average salary?"])
    assert result.exit_code == 0
    assert "There is no salary column." in result.output
    assert "status=abstained" in result.output


def test_a_failed_question_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch, FakeLLM(error=LLMProviderError("the model is down")))
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "Which region sells the most?"])
    assert result.exit_code == 1
    assert "status=failed" in result.output


def test_ask_stops_early_when_ollama_or_the_model_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = "the CSV model qwen2.5-coder:3b is not pulled: run `ollama pull qwen2.5-coder:3b`"
    opened = _offline(monkeypatch, FakeLLM(), problem=problem)
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "Which region sells the most?"])
    assert result.exit_code == 1
    assert f"error: {problem}" in result.output
    assert opened == []  # the CSV was never even loaded


def test_ask_with_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["csv", "ask", str(tmp_path / "nope.csv"), "anything?"])
    assert result.exit_code == 2


def test_ask_needs_a_question() -> None:
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE)])
    assert result.exit_code == 2
