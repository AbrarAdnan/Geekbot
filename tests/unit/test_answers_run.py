"""Tests for step S5.2: the phases of the answer run that need no model (``geekbot.evals.answers_run``).

Generating and judging load models and are live-tested; what is checked here is the order and the
guards: a judge of the generator's family is refused before anything is loaded, the report is built
from the saved files, the labels are only read, and a trial run covers only the first questions.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from geekbot.cli import app
from geekbot.config import Settings
from geekbot.evals.answers import AnswerRecord, SourceView, load_answers, save_answers
from geekbot.evals.answers_run import PHASES, AnswerRun, _in_item_order, run_answers
from geekbot.evals.datasets import GoldenItem, load_golden
from geekbot.evals.judge import JudgeConfigError
from geekbot.labelling import LabelStore

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

# The fixture moves the working folder into a temporary one, so the repository's files are found by
# the position of this file, not by a relative path.
REPOSITORY = Path(__file__).resolve().parents[2]
GOLDEN = REPOSITORY / "evals" / "datasets" / "golden.jsonl"
CORPUS = REPOSITORY / "evals" / "corpus"


def items() -> list[GoldenItem]:
    return load_golden(GOLDEN)


def make_record(item: GoldenItem, *, abstained: bool = False) -> AnswerRecord:
    gold = item.sources[0].path if item.sources else "x.md"
    return AnswerRecord(
        id=item.id,
        type=item.type,
        split="dev",
        question=item.question,
        earlier_turns=item.earlier_turns,
        standalone="",
        answer="I could not find it in the documents." if abstained else "An answer [1].",
        raw_answer="",
        abstained=abstained,
        cited=() if abstained else (1,),
        dropped=(),
        sources=() if abstained else (SourceView(1, gold, "t", "", "text"),),
        seconds=1.0,
        first_token_s=0.3,
        output_tokens=20,
    )


@pytest.fixture
def run(tmp_path: Path) -> AnswerRun:
    return AnswerRun("r1", tmp_path / "results", tmp_path / "labels" / "labels.jsonl")


def write_answers(
    run: AnswerRun, count: int = 4, generator: str = "gemma4:12b"
) -> list[AnswerRecord]:
    records = [make_record(item) for item in items()[:count]]
    save_answers(
        run.answers,
        {"name": run.name, "model": generator, "profile": "target", "top_k": 5},
        records,
    )
    return records


def settings() -> Settings:
    return Settings(_env_file=None)


def silent(_line: str) -> None:
    return None


def test_the_phases_run_in_the_plans_order() -> None:
    assert PHASES == ("generate", "judge", "report")


def test_the_run_names_its_files(run: AnswerRun) -> None:
    assert run.answers == run.folder / "answers.json"
    assert run.judge_cache.name == "judge-cache.json" and run.judge_file.name == "judge.json"
    assert run.report.name == "answers-report.md" and run.folder.name == "r1"


def test_the_report_is_built_from_the_saved_answers(run: AnswerRun) -> None:
    write_answers(run)
    text = run_answers(
        run,
        settings(),
        phases=("report",),
        golden=GOLDEN,
        corpus=CORPUS,
        judge_model="qwen3:8b",
        limit=4,
        log=silent,
    )
    assert text is not None and "# Answer evaluation: r1" in text
    assert run.report.read_text(encoding="utf-8") == text
    assert "The judge" not in text  # no judge file yet: the report says nothing about a judge


def test_a_judge_file_and_labels_add_their_sections_and_the_labels_are_only_read(
    run: AnswerRun,
) -> None:
    records = write_answers(run)
    verdicts = {
        r.id: {"supported": True, "answers_question": True, "reason": "ok"} for r in records
    }
    run.judge_file.write_text(
        json.dumps(
            {
                "name": "r1",
                "judge": {
                    "model": "qwen2.5-coder:7b",
                    "family": "qwen",
                    "generator_family": "gemma",
                    "judged": 4,
                    "skipped_refusals": 0,
                    "supported": 4,
                    "answers_question": 4,
                    "by_type": {},
                },
                "verdicts": verdicts,
            }
        ),
        encoding="utf-8",
    )
    store = run.labels()
    store.add(records[0].id, True)
    store.add(records[1].id, False)
    before = run.labels_file.read_text(encoding="utf-8")
    text = run_answers(
        run,
        settings(),
        phases=("report",),
        golden=GOLDEN,
        corpus=CORPUS,
        judge_model="qwen3:8b",
        limit=4,
        log=silent,
    )
    assert text is not None
    assert "## The judge" in text and "supporting evidence only" in text
    assert "no verdict yet: 2 labels" in text  # two labels are too few for a verdict
    assert run.labels_file.read_text(encoding="utf-8") == before  # the report wrote no label


def test_a_judge_of_the_generators_family_is_refused_before_any_model_is_touched(
    run: AnswerRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_answers(run, generator="gemma4:12b")

    def no_ollama(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("no model may be contacted before the family check")

    monkeypatch.setattr("geekbot.llm.models.OllamaClient", no_ollama)
    with pytest.raises(JudgeConfigError, match="same family"):
        run_answers(
            run,
            settings(),
            phases=("judge",),
            golden=GOLDEN,
            corpus=CORPUS,
            judge_model="gemma3:4b",
            limit=4,
            log=silent,
        )


def test_a_trial_run_covers_only_the_first_questions(run: AnswerRun) -> None:
    write_answers(run, count=8)
    text = run_answers(
        run,
        settings(),
        phases=("report",),
        golden=GOLDEN,
        corpus=CORPUS,
        judge_model="qwen3:8b",
        limit=3,
        log=silent,
    )
    assert text is not None
    _meta, records = load_answers(run.answers)
    assert len(records) == 8  # the file is as it was written; the report counted three
    assert re.search(r"\| all \| 3 \|", text)


def test_a_report_with_no_answers_file_says_so(run: AnswerRun) -> None:
    with pytest.raises(ValueError, match="not an answers file"):
        run_answers(
            run,
            settings(),
            phases=("report",),
            golden=GOLDEN,
            corpus=CORPUS,
            judge_model="qwen3:8b",
            limit=3,
            log=silent,
        )


def test_records_are_kept_in_the_order_of_the_golden_items_without_repeats() -> None:
    chosen = items()[:4]
    scrambled = [
        make_record(chosen[2]),
        make_record(chosen[0]),
        make_record(chosen[2]),
        make_record(chosen[1]),
    ]
    ordered = _in_item_order(scrambled, chosen)
    assert [r.id for r in ordered] == [chosen[0].id, chosen[1].id, chosen[2].id]


def test_the_labels_of_the_run_are_the_runs_own(run: AnswerRun) -> None:
    run.labels().add("S-001", True)
    other = AnswerRun("r2", run.results_dir, run.labels_file)
    assert other.labels().labels() == {}
    assert isinstance(run.labels(), LabelStore)


# --- the command -----------------------------------------------------------------------------

runner = CliRunner()


def test_the_command_offers_its_options() -> None:
    text = re.sub(r"\x1b\[[0-9;]*m", "", runner.invoke(app, ["eval", "answers", "--help"]).output)
    for wanted in (
        "--name",
        "--phase",
        "--judge-model",
        "--golden",
        "--results-dir",
        "--labels-file",
        "--data-dir",
        "--limit",
    ):
        assert wanted in text, wanted


def test_the_command_stops_with_one_line_on_an_unknown_phase_or_a_missing_golden_file(
    tmp_path: Path,
) -> None:
    unknown = runner.invoke(app, ["eval", "answers", "--phase", "everything"])
    assert unknown.exit_code == 1 and "unknown phase everything" in unknown.output
    missing = runner.invoke(app, ["eval", "answers", "--golden", str(tmp_path / "nope.jsonl")])
    assert missing.exit_code == 1 and "does not exist" in missing.output


def test_the_command_reports_a_judge_of_the_wrong_family_as_one_line(tmp_path: Path) -> None:
    run = AnswerRun("same", tmp_path / "results", tmp_path / "labels.jsonl")
    write_answers(run, generator="gemma4:12b")
    result = runner.invoke(
        app,
        [
            "eval",
            "answers",
            "--name",
            "same",
            "--phase",
            "judge",
            "--judge-model",
            "gemma3:4b",
            "--golden",
            str(GOLDEN),
            "--results-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path / "data"),
        ],
    )
    assert result.exit_code == 1
    assert "same family" in result.output and "Traceback" not in result.output
