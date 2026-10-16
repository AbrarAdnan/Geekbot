"""Tests for step C3: the command-line side of the swap and the reproduction run.

Ollama, the GPU commands and the run itself are replaced by fakes, so these tests check the
wiring only: options reach the right function, errors become one clear line, exit codes are
right. The behaviour behind each command is tested in test_swap.py, test_swapcheck.py,
test_doctor.py and test_csv_repro.py.
"""

from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from typer.testing import CliRunner

from geekbot import cli
from geekbot.cli import app
from geekbot.csv_mode import adapter
from geekbot.evals import csv_repro
from geekbot.evals.csv_repro import Preflight, ReproResult, RunContext, summarize
from geekbot.llm.vram import CommandError, GpuMemory, LoadedModel

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

runner = CliRunner()
CHAT, CSV = "gemma3:4b", "qwen2.5-coder:3b"


class FakeOllama:
    """Stands in for ``OllamaClient`` (a context manager). One instance is shared per test."""

    def __init__(self, pulled: list[str]) -> None:
        self.pulled = pulled
        self.loaded: list[str] = []
        self.hosts: list[str] = []

    def __call__(self, host: str, **kwargs: Any) -> "FakeOllama":
        self.hosts.append(host)
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def version(self) -> str:
        return "0.34.2"

    def list_models(self) -> list[str]:
        return self.pulled

    def loaded_models(self) -> list[str]:
        return list(self.loaded)

    def unload(self, model: str) -> None:
        self.loaded.remove(model)

    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:
        self.loaded.append(model)
        return 1.0


def _fake_machine(
    monkeypatch: pytest.MonkeyPatch, *, pulled: list[str] | None = None, gpu_percent: int = 100
) -> FakeOllama:
    fake = FakeOllama(pulled if pulled is not None else [CHAT, CSV])
    monkeypatch.setattr(cli, "OllamaClient", fake)
    monkeypatch.setattr(cli, "_read_gpu", lambda: GpuMemory(5000, 10240))
    monkeypatch.setattr(
        cli,
        "_read_ps",
        lambda: [
            LoadedModel(m, gpu_percent, 100 - gpu_percent, None, "Forever") for m in fake.loaded
        ],
    )
    return fake


# --- doctor --mode --------------------------------------------------------------------------


def test_doctor_checks_the_chat_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_machine(monkeypatch)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "checking chat" in result.output
    assert fake.loaded == [CHAT]


def test_doctor_mode_csv_swaps_to_the_csv_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _machine_with_chat_loaded(monkeypatch)
    result = runner.invoke(app, ["doctor", "--mode", "csv"])
    assert result.exit_code == 0, result.output
    assert "checking csv" in result.output
    assert f"unloaded {CHAT}" in result.output
    assert fake.loaded == [CSV]


def _machine_with_chat_loaded(monkeypatch: pytest.MonkeyPatch) -> FakeOllama:
    fake = _fake_machine(monkeypatch)
    fake.loaded.append(CHAT)
    return fake


def test_doctor_rejects_an_unknown_mode() -> None:
    result = runner.invoke(app, ["doctor", "--mode", "bogus"])
    assert result.exit_code == 2


def test_doctor_exits_1_when_the_new_model_is_split_across_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_machine(monkeypatch, gpu_percent=60)
    result = runner.invoke(app, ["doctor", "--mode", "csv"])
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


# --- csv swap -------------------------------------------------------------------------------


def test_csv_swap_prints_the_timing_table(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch)
    result = runner.invoke(app, ["csv", "swap", "--cycles", "2"])
    assert result.exit_code == 0, result.output
    assert "chat->csv: 2 swap(s)" in result.output
    assert "csv->chat: 2 swap(s)" in result.output
    assert f"profile=proto: chat {CHAT} <-> csv {CSV}" in result.output


def test_csv_swap_names_a_model_that_is_not_pulled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake_machine(monkeypatch, pulled=[CHAT])
    result = runner.invoke(app, ["csv", "swap"])
    assert result.exit_code == 1
    assert f"ollama pull {CSV}" in result.output
    assert fake.loaded == []  # nothing was loaded before the problem was found


def test_csv_swap_exits_1_when_a_residency_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch, gpu_percent=40)
    result = runner.invoke(app, ["csv", "swap", "--cycles", "1"])
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


def test_csv_swap_survives_unreadable_machine_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_machine(monkeypatch)

    def broken() -> Any:
        raise CommandError("not found")

    monkeypatch.setattr(cli, "_read_ps", broken)
    monkeypatch.setattr(cli, "_read_gpu", broken)
    result = runner.invoke(app, ["csv", "swap", "--cycles", "1"])
    assert result.exit_code == 0, result.output


def test_csv_swap_needs_at_least_one_cycle() -> None:
    assert runner.invoke(app, ["csv", "swap", "--cycles", "0"]).exit_code == 2


# --- csv repro ------------------------------------------------------------------------------


def _canned_result() -> ReproResult:
    context = RunContext(
        created="2026-10-16T12:00:00+00:00",
        model=CSV,
        profile="proto",
        host="http://127.0.0.1:11434",
        ollama_version="0.34.2",
        source="src",
        suites=["hospital"],
        runs=1,
        limit=None,
        time_limit_s=600,
        geekbot_cap_s=60,
        preflight=Preflight(654, 5900, 4.2, (), "PASS: ok", True),
        official=False,
        official_notes=["profile"],
    )
    return ReproResult(context, [], summarize([], runs=1, cap_s=60))


def _fake_repro(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    _fake_machine(monkeypatch)
    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: None)

    def fake_run(settings: Any, suites: Any, **kwargs: Any) -> ReproResult:
        calls.append({"suites": suites, **kwargs})
        return _canned_result()

    monkeypatch.setattr(csv_repro, "run_repro", fake_run)


def _fake_suite_loader(
    monkeypatch: pytest.MonkeyPatch, loaded: list[tuple[str, int | None]]
) -> None:
    def fake_load(source: Path, name: str, *, limit: int | None = None) -> str:
        loaded.append((name, limit))
        return name

    monkeypatch.setattr(csv_repro, "load_suite", fake_load)


def test_repro_runs_both_suites_with_the_documented_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    loaded: list[tuple[str, int | None]] = []
    _fake_repro(monkeypatch, calls)
    _fake_suite_loader(monkeypatch, loaded)
    result = runner.invoke(app, ["csv", "repro"])
    assert result.exit_code == 0, result.output
    assert loaded == [("hospital", None), ("sales", None)]
    (call,) = calls
    assert call["suites"] == ["hospital", "sales"]
    assert call["runs"] == 3
    assert call["time_limit_s"] == 600
    assert call["limit"] is None
    assert call["source"] == Path("scratch/gigalogy-latest")
    assert call["out"].parent == Path("evals/csv/results")
    assert "Reproduction run" in result.output
    assert "not the target profile" in result.output  # the default profile is proto


def test_repro_passes_its_options_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    loaded: list[tuple[str, int | None]] = []
    _fake_repro(monkeypatch, calls)
    _fake_suite_loader(monkeypatch, loaded)
    out = tmp_path / "result.json"
    args = [
        "csv", "repro", "--suite", "sales", "--runs", "1", "--limit", "2",
        "--time-limit-s", "90", "--source", str(tmp_path), "--out", str(out),
    ]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert loaded == [("sales", 2)]
    (call,) = calls
    assert (call["runs"], call["limit"], call["time_limit_s"]) == (1, 2, 90)
    assert call["source"] == tmp_path
    assert call["out"] == out


def test_repro_with_a_wrong_source_says_so_in_one_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    _fake_repro(monkeypatch, calls)
    result = runner.invoke(app, ["csv", "repro", "--source", str(tmp_path / "nowhere")])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "--source" in result.output
    assert calls == []  # nothing was loaded onto the GPU


def test_repro_stops_early_when_ollama_or_the_model_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    _fake_repro(monkeypatch, calls)
    problem = "the CSV model qwen2.5-coder:3b is not pulled: run `ollama pull qwen2.5-coder:3b`"
    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: problem)
    result = runner.invoke(app, ["csv", "repro"])
    assert result.exit_code == 1
    assert f"error: {problem}" in result.output
    assert calls == []


def test_repro_rejects_a_zero_run_count() -> None:
    assert runner.invoke(app, ["csv", "repro", "--runs", "0"]).exit_code == 2
