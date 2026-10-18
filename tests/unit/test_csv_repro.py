"""Tests for step C3: the reproduction run (``geekbot.evals.csv_repro``).

Everything is offline: the benchmark files are tiny made-up ones in a temporary folder, the
analyst is a fake, and Ollama and the GPU are faked, so no model is loaded.
"""

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from geekbot.config import Settings
from geekbot.evals.csv_repro import (
    LOAD_EVENT_MS,
    MAX_NOTE_CHARS,
    SEED_OFFSET_STRIDE,
    Preflight,
    QuestionRun,
    ReproError,
    default_output_path,
    format_progress,
    format_report,
    load_suite,
    nearest_rank,
    official_notes,
    prepare_gpu,
    run_repro,
    run_suite_once,
    shorten,
    summarize,
)
from geekbot.llm.vram import CommandError, GpuMemory, LoadedModel

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

TARGET_CSV = "qwen2.5-coder:7b"
CHAT = "gemma3:4b"

QUESTIONS_YAML = """\
# a comment, like the real files have
- id: e01
  difficulty: easy
  category: aggregation
  dataset: hospital.csv
  question: "SECRET QUESTION TEXT one?"
- id: m01
  difficulty: medium
  category: filter
  question: "SECRET QUESTION TEXT two?"
- id: u01
  difficulty: unanswerable
  category: abstain
  question: "SECRET QUESTION TEXT three?"
"""
GOLD = {
    "e01": {"expected_abstain": False, "compare": "scalar", "tolerance": 0, "value": 3},
    "m01": {"expected_abstain": False, "compare": "scalar", "tolerance": 0, "value": 7},
    "u01": {"expected_abstain": True, "abstain_reason": "column_absent"},
}


def _source(root: Path, *, suites: tuple[str, ...] = ("hospital", "sales")) -> Path:
    """A fake Gigalogy folder with the files a suite needs, laid out as the real one is."""
    layout = {
        "hospital": ("questions/questions.yaml", "gold/gold.json", "hospital/hospital.csv"),
        "sales": ("questions/questions_sales.yaml", "gold/gold_sales.json", "sales/sales.csv"),
    }
    for name in suites:
        questions, gold, csv = layout[name]
        for path, text in (
            (root / "evaluation" / questions, QUESTIONS_YAML),
            (root / "evaluation" / gold, json.dumps(GOLD)),
            (root / "data" / csv, "a,b\n1,2\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return root


# --- loading --------------------------------------------------------------------------------


def test_a_suite_is_read_from_the_source_folder(tmp_path: Path) -> None:
    suite = load_suite(_source(tmp_path), "hospital")
    assert suite.name == "hospital"
    assert suite.csv == tmp_path / "data" / "hospital" / "hospital.csv"
    assert [(q.id, q.difficulty) for q in suite.questions] == [
        ("e01", "easy"),
        ("m01", "medium"),
        ("u01", "unanswerable"),
    ]
    assert suite.questions[0].text == "SECRET QUESTION TEXT one?"
    assert suite.gold["u01"]["expected_abstain"] is True


def test_limit_keeps_only_the_first_questions(tmp_path: Path) -> None:
    suite = load_suite(_source(tmp_path), "sales", limit=2)
    assert [q.id for q in suite.questions] == ["e01", "m01"]


def test_a_missing_file_says_which_one_and_how_to_fix_it(tmp_path: Path) -> None:
    with pytest.raises(ReproError, match=r"--source") as error:
        load_suite(tmp_path / "nowhere", "hospital")
    assert "questions.yaml" in str(error.value)


def test_a_question_without_gold_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "evaluation" / "gold" / "gold.json").write_text(
        json.dumps({"e01": GOLD["e01"]}), encoding="utf-8"
    )
    with pytest.raises(ReproError, match="no gold answer for m01, u01"):
        load_suite(source, "hospital")


def test_a_question_file_that_is_not_a_list_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "evaluation" / "questions" / "questions.yaml").write_text("a: 1\n", encoding="utf-8")
    with pytest.raises(ReproError, match="expected a list"):
        load_suite(source, "hospital")


def test_a_question_missing_a_field_is_refused(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "evaluation" / "questions" / "questions.yaml").write_text(
        "- id: e01\n  difficulty: easy\n", encoding="utf-8"
    )
    with pytest.raises(ReproError, match="missing"):
        load_suite(source, "hospital")


def test_an_unknown_suite_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ReproError, match="unknown suite"):
        load_suite(tmp_path, "chinook")


# --- summary --------------------------------------------------------------------------------


def _row(
    suite: str,
    qid: str,
    difficulty: str,
    seed_set: int,
    *,
    correct: bool,
    status: str = "answered",
    elapsed_s: float = 3.0,
    load_ms: int = 5,
) -> QuestionRun:
    return QuestionRun(
        suite, qid, difficulty, "cat", seed_set, status, correct, "note", 0, elapsed_s, load_ms
    )


def test_nearest_rank_percentiles() -> None:
    values = [float(n) for n in range(1, 11)]
    assert nearest_rank(values, 50) == 5.0
    assert nearest_rank(values, 95) == 10.0
    assert nearest_rank([], 95) == 0.0
    assert nearest_rank([7.0], 95) == 7.0


def test_solved_means_right_in_more_than_half_of_the_runs() -> None:
    rows = [
        # right in 2 of 3 -> solved, but unstable
        _row("h", "e01", "easy", 0, correct=True),
        _row("h", "e01", "easy", 1, correct=True),
        _row("h", "e01", "easy", 2, correct=False),
        # right in 1 of 3 -> not solved, unstable
        _row("h", "m01", "medium", 0, correct=False),
        _row("h", "m01", "medium", 1, correct=True),
        _row("h", "m01", "medium", 2, correct=False),
        # right in 0 of 3 -> not solved, stable
        _row("h", "d01", "hard", 0, correct=False),
        _row("h", "d01", "hard", 1, correct=False),
        _row("h", "d01", "hard", 2, correct=False),
    ]
    summary = summarize(rows, runs=3, cap_s=60)
    assert summary.questions == 3
    assert summary.question_runs == 9
    assert summary.solved == 1
    assert summary.solved_by_difficulty["easy"] == [1, 1]
    assert summary.solved_by_difficulty["medium"] == [0, 1]
    assert summary.solved_by_difficulty["hard"] == [0, 1]
    assert summary.unstable_questions == 2
    assert summary.correct_per_run == [1, 2, 0]
    assert summary.mean_correct_per_run == pytest.approx(1.0)


def test_the_same_question_in_two_suites_is_two_questions() -> None:
    rows = [
        _row("hospital", "e01", "easy", 0, correct=True),
        _row("sales", "e01", "easy", 0, correct=False),
    ]
    summary = summarize(rows, runs=1, cap_s=60)
    assert summary.questions == 2
    assert summary.solved == 1


def test_invented_answers_are_answered_unanswerable_questions_only() -> None:
    rows = [
        _row("h", "u01", "unanswerable", 0, correct=False, status="answered"),  # invented
        _row("h", "u01", "unanswerable", 1, correct=True, status="abstained"),
        _row("h", "u01", "unanswerable", 2, correct=False, status="failed"),  # no number made up
    ]
    summary = summarize(rows, runs=3, cap_s=60)
    assert (summary.invented, summary.invented_of) == (1, 3)
    assert summary.failures == 1
    assert summary.solved_by_difficulty["unanswerable"] == [0, 1]  # right in only 1 of 3


def test_false_abstentions_are_wrong_answerable_questions_that_were_refused() -> None:
    rows = [
        _row("h", "e01", "easy", 0, correct=False, status="abstained"),
        _row("h", "e01", "easy", 1, correct=False, status="needs_clarification"),
        _row("h", "e01", "easy", 2, correct=False, status="answered"),  # wrong, not refused
        _row("h", "u01", "unanswerable", 0, correct=True, status="abstained"),  # right to refuse
    ]
    assert summarize(rows, runs=3, cap_s=60).false_abstentions == 2


def test_latency_cap_and_load_event_counts() -> None:
    times = [1.0, 2.0, 3.0, 61.0, 100.0]
    loads = [0, 10, LOAD_EVENT_MS, LOAD_EVENT_MS + 1, 7000]
    rows = [
        _row("h", f"q{i}", "easy", 0, correct=True, elapsed_s=t, load_ms=ms)
        for i, (t, ms) in enumerate(zip(times, loads, strict=True))
    ]
    summary = summarize(rows, runs=1, cap_s=60)
    assert summary.latency_mean_s == pytest.approx(33.4)
    assert summary.latency_p50_s == 3.0
    assert summary.latency_p95_s == 100.0
    assert summary.latency_max_s == 100.0
    assert summary.over_cap == 2  # 61 s and 100 s
    assert summary.load_events == 2  # strictly more than the threshold
    assert summary.load_ms_median == 500
    assert summary.load_ms_max == 7000


def test_a_question_that_ran_long_because_one_step_ran_away_is_visible() -> None:
    # The first live run (proto) had two questions of exactly 61.2 s: one 60 s model-call
    # timeout plus the rest. The summary must tell that apart from many ordinary steps.
    runaway = QuestionRun(
        "h", "m03", "medium", "cat", 0, "answered", False, "n", 0, 61.2, 5, 1, 60.0
    )
    ordinary = QuestionRun("h", "e01", "easy", "cat", 0, "answered", True, "n", 0, 2.0, 5, 0, 0.9)
    summary = summarize([runaway, ordinary], runs=1, cap_s=60)
    assert summary.steps_failed_runs == 1
    assert summary.slowest_step_s == 60.0
    assert summary.over_cap == 1


def test_older_call_sites_without_step_data_still_work() -> None:
    row = _row("h", "e01", "easy", 0, correct=True)
    assert (row.failed_steps, row.slowest_step_s) == (0, 0.0)
    summary = summarize([row], runs=1, cap_s=60)
    assert (summary.steps_failed_runs, summary.slowest_step_s) == (0, 0.0)


def test_an_empty_run_summarises_to_zeros_instead_of_crashing() -> None:
    summary = summarize([], runs=3, cap_s=60)
    assert (summary.questions, summary.solved, summary.latency_max_s) == (0, 0, 0.0)
    assert (summary.steps_failed_runs, summary.slowest_step_s) == (0, 0.0)


# --- the run loop ---------------------------------------------------------------------------


class FakeAnalyst:
    """Answers with canned statuses and remembers what it was asked and with which seeds."""

    def __init__(self, log: list[tuple[str, str, int]], name: str) -> None:
        self.log, self.name = log, name

    def ask(self, question: str, *, seed_offset: int = 0) -> Any:
        self.log.append((self.name, question, seed_offset))
        return SimpleNamespace(
            status="answered",
            retries=1,
            seconds=2.0,
            model_load_ms=3,
            failed=False,
            failed_steps=2,
            slowest_step_ms=1500,
        )


def _always(correct: bool = True, note: str = "matches gold") -> Any:
    return lambda answer, gold: (correct, note)


def test_every_question_is_asked_with_the_seeds_of_its_seed_set(tmp_path: Path) -> None:
    suite = load_suite(_source(tmp_path), "hospital")
    log: list[tuple[str, str, int]] = []
    lines: list[str] = []
    rows = run_suite_once(
        suite, FakeAnalyst(log, "h"), seed_set=2, grade_fn=_always(), report=lines.append
    )
    assert [entry[2] for entry in log] == [2 * SEED_OFFSET_STRIDE] * 3
    assert [r.id for r in rows] == ["e01", "m01", "u01"]
    assert all(r.seed_set == 2 and r.retries == 1 and r.elapsed_s == 2.0 for r in rows)
    assert all(r.failed_steps == 2 and r.slowest_step_s == 1.5 for r in rows)
    assert len(lines) == 3
    assert lines[0].startswith("[s2] [PASS] hospital")


def test_a_huge_mismatch_note_is_cut_in_the_progress_line_and_in_the_result(
    tmp_path: Path,
) -> None:
    suite = load_suite(_source(tmp_path), "hospital", limit=1)
    lines: list[str] = []
    huge = "expected " + "[{'city': 'Hamilton', 'patient_count': 1938}] " * 200
    rows = run_suite_once(
        suite,
        FakeAnalyst([], "h"),
        seed_set=0,
        grade_fn=_always(correct=False, note=huge),
        report=lines.append,
    )
    assert len(rows[0].note) == MAX_NOTE_CHARS
    assert rows[0].note.endswith("...")
    assert len(lines[0]) < 2 * MAX_NOTE_CHARS


def test_a_short_note_is_left_alone() -> None:
    assert shorten("matches gold") == "matches gold"
    assert shorten("x" * MAX_NOTE_CHARS) == "x" * MAX_NOTE_CHARS


def test_progress_line_marks_a_failure() -> None:
    row = _row("hospital", "e01", "easy", 0, correct=False)
    assert "[FAIL]" in format_progress(row)


# --- the whole run, with a fake Ollama and GPU ----------------------------------------------


class FakeOllama:
    def __init__(self, loaded: list[str] | None = None) -> None:
        self.loaded = list(loaded or [])
        self.warmed: list[tuple[str, int]] = []

    def version(self) -> str:
        return "0.34.2"

    def loaded_models(self) -> list[str]:
        return list(self.loaded)

    def unload(self, model: str) -> None:
        self.loaded.remove(model)

    def warm_up(self, model: str, *, num_ctx: int, keep_alive: str) -> float:
        self.warmed.append((model, num_ctx))
        self.loaded.append(model)
        return 4.2


def _ps_for(client: FakeOllama, *, gpu: int = 100, context: int = 16384) -> Any:
    return lambda: [LoadedModel(m, gpu, 100 - gpu, context, "Forever") for m in client.loaded]


def _gpu(*used: int) -> Any:
    readings = iter(GpuMemory(u, 10240) for u in used)
    return lambda: next(readings)


class OpenSpy:
    """Replaces ``CsvAnalyst.open``: hands out fake analysts and records how they were opened."""

    def __init__(self) -> None:
        self.log: list[tuple[str, str, int]] = []
        self.opened: list[tuple[Path, int]] = []
        self.closed = 0

    @contextmanager
    def __call__(self, path: Path, settings: Settings) -> Any:
        self.opened.append((path, settings.csv_question_time_limit_s))
        try:
            yield FakeAnalyst(self.log, path.parent.name)
        finally:
            self.closed += 1


def _run(
    tmp_path: Path,
    settings: Settings,
    *,
    runs: int = 3,
    limit: int | None = None,
    client: FakeOllama | None = None,
    read_gpu: Any = None,
    read_ps: Any = None,
    out: Path | None = None,
    suites: tuple[str, ...] = ("hospital", "sales"),
    spy: OpenSpy | None = None,
    lines: list[str] | None = None,
) -> Any:
    client = client or FakeOllama()
    source = _source(tmp_path / "gigalogy")
    return run_repro(
        settings,
        [load_suite(source, name, limit=limit) for name in suites],
        runs=runs,
        limit=limit,
        time_limit_s=600,
        source=source,
        client=client,
        read_gpu=read_gpu or _gpu(654, 5900),
        read_ps=read_ps or _ps_for(client),
        open_analyst=spy or OpenSpy(),
        grade_fn=_always(),
        report=(lines if lines is not None else []).append,
        out=out,
    )


def test_a_full_run_on_the_target_model_with_an_idle_gpu_is_official(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV, chat_model=CHAT)
    result = _run(tmp_path, settings)
    assert result.context.official, result.context.official_notes
    assert result.context.model == TARGET_CSV
    assert result.context.preflight.gpu_idle_mib == 654
    assert result.context.preflight.gpu_loaded_mib == 5900
    assert result.context.preflight.cold_load_s == 4.2
    assert len(result.rows) == 3 * 2 * 3  # 3 seed sets x 2 suites x 3 questions
    assert result.summary.question_runs == 18


def test_the_run_uses_the_package_cap_but_the_report_compares_against_geekbots(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None)
    spy = OpenSpy()
    result = _run(tmp_path, settings, spy=spy)
    assert {cap for _, cap in spy.opened} == {600}
    assert result.context.time_limit_s == 600
    assert result.context.geekbot_cap_s == settings.csv_question_time_limit_s == 60
    assert result.summary.cap_s == 60


def test_the_loop_goes_seed_set_then_suite_then_question_and_closes_the_analysts(
    tmp_path: Path,
) -> None:
    spy = OpenSpy()
    _run(tmp_path, Settings(_env_file=None), runs=2, spy=spy)
    order = [(name, offset) for name, _, offset in spy.log]
    assert order == (
        [("hospital", 0)] * 3
        + [("sales", 0)] * 3
        + [("hospital", SEED_OFFSET_STRIDE)] * 3
        + [("sales", SEED_OFFSET_STRIDE)] * 3
    )
    assert len(spy.opened) == 2  # one session per suite, not one per run
    assert spy.closed == 2


def test_the_models_are_emptied_from_the_gpu_before_the_csv_model_is_warmed(
    tmp_path: Path,
) -> None:
    settings = Settings(_env_file=None)
    client = FakeOllama(loaded=[CHAT, "qwen2.5-coder:3b"])
    result = _run(tmp_path, settings, client=client)
    assert set(result.context.preflight.unloaded) == {CHAT, "qwen2.5-coder:3b"}
    assert client.loaded == ["qwen2.5-coder:3b"]  # warmed again, alone
    assert client.warmed == [("qwen2.5-coder:3b", settings.csv_num_ctx)]


def test_the_proto_model_is_never_official(tmp_path: Path) -> None:
    result = _run(tmp_path, Settings(_env_file=None))
    assert not result.context.official
    assert any("not the target model" in note for note in result.context.official_notes)


def test_a_busy_gpu_is_not_official(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    result = _run(tmp_path, settings, read_gpu=_gpu(6500, 9000))
    assert not result.context.official
    assert any("not idle" in note for note in result.context.official_notes)


def test_a_gpu_that_cannot_be_read_is_not_official(tmp_path: Path) -> None:
    def broken() -> GpuMemory:
        raise CommandError("nvidia-smi not found")

    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    result = _run(tmp_path, settings, read_gpu=broken)
    assert not result.context.official
    assert result.context.preflight.gpu_idle_mib is None


def test_a_model_split_across_the_cpu_is_not_official(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    client = FakeOllama()
    result = _run(tmp_path, settings, client=client, read_ps=_ps_for(client, gpu=52))
    assert not result.context.official
    assert any("residency (R-1)" in note for note in result.context.official_notes)


@pytest.mark.parametrize(
    "kwargs",
    [{"runs": 1}, {"limit": 2}, {"suites": ("hospital",)}],
    ids=["one run", "limit", "one suite"],
)
def test_a_partial_run_is_not_official(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    result = _run(tmp_path, settings, **kwargs)
    assert not result.context.official
    assert any("partial run" in note for note in result.context.official_notes)


def test_official_notes_are_empty_for_a_clean_full_run() -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    preflight = Preflight(654, 5900, 4.2, (), "PASS: ok", True)
    notes = official_notes(settings, preflight, suites=["sales", "hospital"], runs=3, limit=None)
    assert notes == []


def test_the_result_file_is_written_after_each_seed_set_and_holds_no_question_text(
    tmp_path: Path,
) -> None:
    out = tmp_path / "results" / "repro.json"
    _run(tmp_path, Settings(_env_file=None), runs=2, out=out)
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"context", "rows", "summary"}
    assert len(data["rows"]) == 2 * 2 * 3
    assert data["context"]["model"] == "qwen2.5-coder:3b"
    assert data["context"]["preflight"]["gpu_idle_mib"] == 654
    assert "SECRET QUESTION TEXT" not in text  # ids and numbers only


def test_progress_is_reported_per_question_and_per_seed_set(tmp_path: Path) -> None:
    lines: list[str] = []
    _run(tmp_path, Settings(_env_file=None), runs=1, lines=lines)
    assert any("loaded cold in 4.2 s" in line for line in lines)
    assert sum("[PASS]" in line for line in lines) == 6
    assert "seed set 0 done: 6 of 6 correct" in lines


# --- report and file name -------------------------------------------------------------------


def test_the_report_shows_our_numbers_next_to_the_source_projects_and_says_whose_they_are(
    tmp_path: Path,
) -> None:
    text = format_report(_run(tmp_path, Settings(_env_file=None)))
    assert "model qwen2.5-coder:3b" in text
    assert "profile proto" in text
    assert "official: NO" in text
    assert "source project*" in text
    assert "Gigalogy's README" in text
    assert "over Geekbot's 60 s cap" in text
    assert "with a failed step" in text
    assert "35/40" in text  # the source project's published figure


def test_an_official_report_says_so(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, csv_model=TARGET_CSV)
    assert "official: yes" in format_report(_run(tmp_path, settings))


def test_the_result_file_name_says_the_profile_and_the_model() -> None:
    path = default_output_path(Settings(_env_file=None), datetime(2026, 10, 16, 14, 5))
    assert path == Path("evals/csv/results/repro-proto-qwen2.5-coder-3b-20261016-1405.json")
    target = Settings(_env_file=None, profile="target")
    assert "repro-target-qwen2.5-coder-7b-" in default_output_path(target).name


def test_prepare_gpu_reports_an_unreadable_ps_as_not_confirmed() -> None:
    def broken() -> list[LoadedModel]:
        raise CommandError("ollama not found")

    preflight = prepare_gpu(
        Settings(_env_file=None), FakeOllama(), read_gpu=_gpu(654, 5900), read_ps=broken
    )
    assert not preflight.residency_ok
    assert preflight.residency.startswith("unknown")
