"""Tests for step C2: ``CsvAnalyst``, Geekbot's side of the connection to ``csv_analyst``.

Most tests use a fake model (a class with ``generate_json``) so no Ollama is needed. One test
runs the real ``OllamaCsvProvider`` against a fake HTTP server, so it fails until you write
``generate_json`` (your piece).
"""

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from csv_analyst import CsvAnalystSession, LLMProviderError
from geekbot.config import Settings
from geekbot.csv_mode.adapter import CsvAnalyst, CsvLoadError, LoadTimer, find_problem

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sales_small.csv"

# North 100+120+140+160 = 520, South 380, East 300: the SQL below puts North first.
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
# Two planner samples that agree, then the explanation (the same pattern as the package's own
# tests/csv_analyst/vendored/test_session.py).
HAPPY_PATH = [PLAN, PLAN, ANSWER]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A GEEKBOT_* variable set in the shell running pytest must not change the results.
    for name in list(os.environ):
        if name.startswith("GEEKBOT_"):
            monkeypatch.delenv(name)


class FakeLLM:
    """A stand-in model: returns canned replies in order and counts the calls."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.closed = False

    def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls += 1
        return self.replies.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeOllama:
    """Plays an Ollama server: each /api/generate gets the next canned reply, 2 ms of load time."""

    LOAD_NS = 2_000_000

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        reply = {"response": self.replies.pop(0), "load_duration": self.LOAD_NS}
        return httpx.Response(200, json=reply)


# --- describe and ask, with a fake model ---------------------------------------------------


def test_describe_reports_rows_and_columns_without_calling_the_model() -> None:
    fake = FakeLLM([])
    with CsvAnalyst.open(FIXTURE, Settings(_env_file=None), provider=fake) as analyst:
        text = analyst.describe()
    assert "sales_small.csv" in text
    assert "12 rows, 4 columns" in text
    assert fake.calls == 0


def test_ask_answers_from_the_data_and_names_the_model() -> None:
    settings = Settings(_env_file=None)
    with CsvAnalyst.open(FIXTURE, settings, provider=FakeLLM(HAPPY_PATH)) as analyst:
        answer = analyst.ask("Which region has the highest total revenue?")
    assert answer.status == "answered"
    assert not answer.failed
    assert "North" in answer.markdown
    assert "520" in answer.markdown
    assert answer.model == settings.resolved_csv_model
    assert answer.seconds >= 0
    assert answer.model_load_ms == 0  # a fake model reports no load time


def test_ask_passes_the_per_question_time_cap_to_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}
    real_ask = CsvAnalystSession.ask

    def spy(self: CsvAnalystSession, question: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real_ask(self, question, **kwargs)

    monkeypatch.setattr(CsvAnalystSession, "ask", spy)
    settings = Settings(_env_file=None, csv_question_time_limit_s=7)
    with CsvAnalyst.open(FIXTURE, settings, provider=FakeLLM(HAPPY_PATH)) as analyst:
        analyst.ask("Which region has the highest total revenue?")
    assert seen["time_limit_s"] == 7


def test_ask_passes_the_seed_offset_to_the_package_and_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    real_ask = CsvAnalystSession.ask

    def spy(self: CsvAnalystSession, question: str, **kwargs: Any) -> Any:
        seen.append(kwargs["seed_offset"])
        return real_ask(self, question, **kwargs)

    monkeypatch.setattr(CsvAnalystSession, "ask", spy)
    with CsvAnalyst.open(FIXTURE, Settings(_env_file=None), provider=FakeLLM(HAPPY_PATH * 2)) as a:
        a.ask("Which region has the highest total revenue?")
        a.ask("Which region has the highest total revenue?", seed_offset=20_000)
    assert seen == [0, 20_000]


def test_a_refusal_carries_its_reason_and_no_retries() -> None:
    refusal = json.dumps(
        {
            "answerability": "data_missing",
            "reason": "There is no salary column.",
            "plan": [],
            "columns_used": [],
            "code": "",
        }
    )
    with CsvAnalyst.open(
        FIXTURE, Settings(_env_file=None), provider=FakeLLM([refusal, refusal])
    ) as analyst:
        answer = analyst.ask("What is the average salary?")
    assert answer.status == "abstained"
    assert "There is no salary column." in (answer.reason or "")
    assert answer.retries == 0


def test_a_happy_path_question_reports_its_steps() -> None:
    with CsvAnalyst.open(FIXTURE, Settings(_env_file=None), provider=FakeLLM(HAPPY_PATH)) as a:
        answer = a.ask("Which region has the highest total revenue?")
    assert answer.failed_steps == 0
    assert answer.slowest_step_ms >= 0


def test_a_failing_model_call_counts_as_a_failed_step() -> None:
    class Broken(FakeLLM):
        def generate_json(self, prompt: str, **kwargs: Any) -> str:
            raise LLMProviderError("timed out")

    with CsvAnalyst.open(FIXTURE, Settings(_env_file=None), provider=Broken([])) as analyst:
        answer = analyst.ask("Which region has the highest total revenue?")
    assert answer.failed
    assert answer.failed_steps >= 1


def test_a_provider_you_pass_in_is_not_closed_by_the_analyst() -> None:
    fake = FakeLLM([])
    with CsvAnalyst.open(FIXTURE, Settings(_env_file=None), provider=fake):
        pass
    assert not fake.closed


def test_a_file_that_is_not_a_csv_raises_csv_load_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(CsvLoadError):
        CsvAnalyst.open(empty, Settings(_env_file=None), provider=FakeLLM([]))


# --- with our real provider, against a fake Ollama (needs your generate_json) --------------


def test_the_real_provider_uses_geekbots_settings_and_the_load_time_is_summed() -> None:
    settings = Settings(_env_file=None)
    server = FakeOllama(HAPPY_PATH)
    transport = httpx.MockTransport(server)
    with CsvAnalyst.open(FIXTURE, settings, transport=transport) as analyst:
        answer = analyst.ask("Which region has the highest total revenue?")
    assert answer.status == "answered"
    assert server.bodies, "the model was never called"
    for body in server.bodies:
        assert body["model"] == settings.resolved_csv_model
        assert body["keep_alive"] == settings.keep_alive
        assert body["options"]["num_ctx"] == settings.csv_num_ctx
    assert answer.model_load_ms == 2 * len(server.bodies)


# --- LoadTimer and find_problem -------------------------------------------------------------


def test_load_timer_adds_up_load_durations_and_ignores_anything_else() -> None:
    timer = LoadTimer()
    timer({"load_duration": 5})
    timer({})  # no load_duration at all
    timer({"load_duration": True})  # a bool is not a number of nanoseconds
    timer({"load_duration": "7"})  # nor is text
    timer({"load_duration": 10})
    assert timer.total_ns == 15


def _ollama_with_models(*names: str) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda _: httpx.Response(200, json={"models": [{"name": name} for name in names]})
    )


def test_find_problem_is_none_when_ollama_has_the_csv_model() -> None:
    settings = Settings(_env_file=None)
    transport = _ollama_with_models(settings.resolved_csv_model, "gemma3:4b")
    assert find_problem(settings, transport=transport) is None


def test_find_problem_names_the_model_when_it_is_not_pulled() -> None:
    settings = Settings(_env_file=None)
    problem = find_problem(settings, transport=_ollama_with_models("gemma3:4b"))
    assert problem is not None
    assert settings.resolved_csv_model in problem
    assert "ollama pull" in problem


def test_find_problem_says_when_ollama_cannot_be_reached() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    problem = find_problem(Settings(_env_file=None), transport=httpx.MockTransport(refuse))
    assert problem is not None
    assert "cannot reach Ollama" in problem
