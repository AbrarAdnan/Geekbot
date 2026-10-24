"""Tests for step C6: the command-line side of ``geekbot eval csv-followups``.

The model, the datasets and the run are replaced by fakes, so these check the wiring only: options reach
the runner, a bad item file or a missing model becomes one clear line, the exit codes are right. The
rules behind the command are tested in test_csv_followups.py."""

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.csv_mode import adapter
from geekbot.evals import csv_followups_run
from geekbot.llm.models import OllamaError

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

runner = CliRunner()
REPOSITORY = Path(__file__).resolve().parents[2]
ITEMS = REPOSITORY / "evals" / "datasets" / "csv-followups.jsonl"


def fake_run(
    monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], *, error: Exception | None = None
) -> None:
    def run(settings: Any, items: Any, **kwargs: Any) -> str:
        calls.append({"items": list(items), **kwargs})
        if error is not None:
            raise error
        return "REPORT TEXT"

    monkeypatch.setattr(csv_followups_run, "run_csv_followups", run)
    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: None)


def test_the_defaults_run_every_item_three_times_into_a_folder_named_for_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    fake_run(monkeypatch, calls)
    result = runner.invoke(app, ["eval", "csv-followups", "--items", str(ITEMS)])
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert len(call["items"]) == 15
    assert call["runs"] == 3
    assert call["name"] == "csv-followups"
    assert call["out_dir"] == Path("evals/results") / "csv-followups"
    assert call["source"] == Path("scratch/gigalogy-latest")
    assert "REPORT TEXT" in result.output


def test_the_options_reach_the_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    fake_run(monkeypatch, calls)
    args = [
        "eval", "csv-followups", "--items", str(ITEMS), "--name", "c6-trial", "--runs", "1",
        "--limit", "4", "--source", str(tmp_path), "--results-dir", str(tmp_path / "out"),
    ]  # fmt: skip
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    (call,) = calls
    assert [item.id for item in call["items"]] == ["F-001", "F-002", "F-003", "F-004"]
    assert (call["runs"], call["name"], call["source"]) == (1, "c6-trial", tmp_path)
    assert call["out_dir"] == tmp_path / "out" / "c6-trial"


def test_a_bad_item_file_is_one_line_and_nothing_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    fake_run(monkeypatch, calls)
    broken = tmp_path / "items.jsonl"
    broken.write_text('{"id": "F-1"}\n', encoding="utf-8")
    result = runner.invoke(app, ["eval", "csv-followups", "--items", str(broken)])
    assert result.exit_code == 1
    assert "error:" in result.output and "items.jsonl line 1" in result.output
    assert calls == []
    missing = runner.invoke(app, ["eval", "csv-followups", "--items", str(tmp_path / "none.jsonl")])
    assert missing.exit_code == 1 and "error:" in missing.output


def test_a_missing_model_stops_before_anything_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    fake_run(monkeypatch, calls)
    problem = "the CSV model qwen2.5-coder:3b is not pulled: run `ollama pull qwen2.5-coder:3b`"
    monkeypatch.setattr(adapter, "find_problem", lambda settings, **kwargs: problem)
    result = runner.invoke(app, ["eval", "csv-followups", "--items", str(ITEMS)])
    assert result.exit_code == 1
    assert f"error: {problem}" in result.output
    assert calls == []


def test_a_failure_in_the_run_is_one_line_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run(monkeypatch, [], error=OllamaError("connection refused"))
    result = runner.invoke(app, ["eval", "csv-followups", "--items", str(ITEMS)])
    assert result.exit_code == 1
    assert "error: " in result.output and "connection refused" in result.output
    assert "Traceback" not in result.output


def test_a_missing_dataset_folder_is_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_run(
        monkeypatch,
        [],
        error=ValueError(
            "x/data/sales/sales.csv not found; point --source at the Gigalogy folder (sales)"
        ),
    )
    result = runner.invoke(app, ["eval", "csv-followups", "--items", str(ITEMS)])
    assert result.exit_code == 1 and "--source" in result.output


def test_a_zero_run_count_is_refused() -> None:
    assert runner.invoke(app, ["eval", "csv-followups", "--runs", "0"]).exit_code == 2
