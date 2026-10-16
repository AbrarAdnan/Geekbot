"""The C3 reproduction run: Gigalogy's CSV benchmark, asked through Geekbot's adapter.

Question: does the copied ``csv_analyst`` package, wired to Geekbot's own Ollama client and
settings, behave like the original? The benchmark is Gigalogy's 40 questions (hospital and sales
datasets: 30 answerable, 10 that the data cannot answer). Because local models vary from run to
run, every question is asked 3 times with different seeds and counts as *solved* when it is right
in more than half of the runs, i.e. at least 2 of 3 (Gigalogy's rule). PLAN.md §9.

What this is and is not (D-041):

- It is a *check on the copy*. Gigalogy's published numbers are shown next to ours, labelled as
  the source project's numbers; ours are Geekbot's own measurements and name their model.
- It is read-only towards Gigalogy's folder. Questions, gold answers and CSVs are read from
  ``--source`` (default ``scratch/gigalogy-latest``) and nothing from there is copied into this
  repository: the datasets come from third parties, and result files hold ids and numbers only.
- A number is *official* only if it came from the target model on an idle GPU in a full run
  (D-028, D-043). Anything else is labelled as a plumbing check.

The module imports no ``csv_analyst`` code: everything goes through ``geekbot.csv_mode``.
"""

import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from geekbot.config import PROFILES, Settings
from geekbot.csv_mode.adapter import CsvAnalyst, CsvAnswer
from geekbot.csv_mode.grading import grade
from geekbot.doctor import BUSY_GPU_USED_MIB, check_residency
from geekbot.evals import DEFAULT_RUNS, DEFAULT_SOURCE, SUITE_NAMES
from geekbot.llm.models import Patience, SwapClient, swap_to_mode, unload_and_wait
from geekbot.llm.vram import GpuMemory, LoadedModel, try_read

# Gigalogy gives each repeat ("seed set") its own block of seeds, 10,000 apart.
SEED_OFFSET_STRIDE = 10_000
# A heuristic, not a measurement: Ollama reports a small load_duration even for a model that is
# already loaded, so only a reply that spent this long loading counts as a load event (R-2).
# The report also shows the median and maximum, so the threshold can be judged from the data.
LOAD_EVENT_MS = 500
# A mismatch note holds the expected and the actual table in full: hundreds of rows for a
# per-city question. That buries the progress lines and would copy rows of a third-party
# dataset into the result file, so a note is cut to its start (found in the first live run).
MAX_NOTE_CHARS = 160
DIFFICULTIES = ("easy", "medium", "hard", "unanswerable")

# Where each suite's files are, relative to the Gigalogy folder.
_SUITE_FILES = {
    "hospital": (
        "evaluation/questions/questions.yaml",
        "evaluation/gold/gold.json",
        "data/hospital/hospital.csv",
    ),
    "sales": (
        "evaluation/questions/questions_sales.yaml",
        "evaluation/gold/gold_sales.json",
        "data/sales/sales.csv",
    ),
}


class ReproError(Exception):
    """The benchmark files are missing or not in the expected shape."""


# --- loading the benchmark (read-only) -------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    difficulty: str
    category: str
    text: str


@dataclass(frozen=True)
class Suite:
    name: str
    csv: Path
    questions: list[Question]
    gold: dict[str, dict[str, Any]]


def load_suite(source: Path, name: str, *, limit: int | None = None) -> Suite:
    """Read one suite (its questions, gold answers and CSV path) from the Gigalogy folder.

    ``limit`` keeps only the first N questions (a quick smoke test). Raises ``ReproError`` with
    a message that says what to fix, because the usual cause is a wrong ``--source``.
    """
    if name not in _SUITE_FILES:
        raise ReproError(f"unknown suite {name!r}; expected one of {', '.join(SUITE_NAMES)}")
    questions_path, gold_path, csv_path = (source / part for part in _SUITE_FILES[name])
    for path in (questions_path, gold_path, csv_path):
        if not path.is_file():
            raise ReproError(
                f"{path} not found; point --source at the Gigalogy folder "
                f"(default {DEFAULT_SOURCE})"
            )

    # Imported here: PyYAML has no type stubs installed, and only this function needs it.
    import yaml  # type: ignore[import-untyped]

    raw: object = yaml.safe_load(questions_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ReproError(f"{questions_path}: expected a list of questions")
    questions: list[Question] = []
    for item in raw:
        try:
            questions.append(
                Question(
                    id=str(item["id"]),
                    difficulty=str(item["difficulty"]),
                    category=str(item["category"]),
                    text=str(item["question"]),
                )
            )
        except (KeyError, TypeError) as error:
            raise ReproError(f"{questions_path}: a question is missing {error}") from error

    gold: object = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold, dict):
        raise ReproError(f"{gold_path}: expected an object keyed by question id")
    missing = [q.id for q in questions if q.id not in gold]
    if missing:
        raise ReproError(f"{gold_path}: no gold answer for {', '.join(missing)}")
    if limit is not None:
        questions = questions[:limit]
    return Suite(name=name, csv=csv_path, questions=questions, gold=gold)


# --- running ---------------------------------------------------------------------------------


class AnalystLike(Protocol):
    """What the run needs from ``CsvAnalyst`` (so a test can pass a fake)."""

    def ask(self, question: str, *, seed_offset: int = 0) -> CsvAnswer: ...


class ReproClient(SwapClient, Protocol):
    """The Ollama client the run needs: the swap calls plus the server version."""

    def version(self) -> str: ...


GradeFn = Callable[[CsvAnswer, Mapping[str, Any]], tuple[bool, str]]
OpenAnalyst = Callable[[Path, Settings], AbstractContextManager[AnalystLike]]


@dataclass(frozen=True)
class QuestionRun:
    """One question asked once. No question text or answer text: the id is enough to look it up."""

    suite: str
    id: str
    difficulty: str
    category: str
    seed_set: int
    status: str  # answered, abstained, needs_clarification or failed
    correct: bool
    note: str
    retries: int
    elapsed_s: float
    model_load_ms: int
    # Added after the first live run; they have defaults so older call sites keep working.
    failed_steps: int = 0  # pipeline steps that failed on the way (e.g. a model-call timeout)
    slowest_step_s: float = 0.0  # the longest single step of the question


def shorten(note: str, limit: int = MAX_NOTE_CHARS) -> str:
    """Cut ``note`` to ``limit`` characters, marking the cut with ``...``."""
    return note if len(note) <= limit else note[: limit - 3] + "..."


def format_progress(row: QuestionRun) -> str:
    mark = "PASS" if row.correct else "FAIL"
    return (
        f"[s{row.seed_set}] [{mark}] {row.suite:<8} {row.id:<4} ({row.difficulty:<12}) "
        f"{row.elapsed_s:6.1f} s  {row.note}"
    )


def run_suite_once(
    suite: Suite,
    analyst: AnalystLike,
    *,
    seed_set: int,
    grade_fn: GradeFn,
    report: Callable[[str], None],
) -> list[QuestionRun]:
    """Ask every question of ``suite`` once, with the seeds of ``seed_set``, and grade it."""
    rows: list[QuestionRun] = []
    for question in suite.questions:
        answer = analyst.ask(question.text, seed_offset=seed_set * SEED_OFFSET_STRIDE)
        correct, note = grade_fn(answer, suite.gold[question.id])
        row = QuestionRun(
            suite=suite.name,
            id=question.id,
            difficulty=question.difficulty,
            category=question.category,
            seed_set=seed_set,
            status=answer.status,
            correct=correct,
            note=shorten(note),
            retries=answer.retries,
            elapsed_s=answer.seconds,
            model_load_ms=answer.model_load_ms,
            failed_steps=answer.failed_steps,
            slowest_step_s=answer.slowest_step_ms / 1000,
        )
        report(format_progress(row))
        rows.append(row)
    return rows


# --- summary ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceReference:
    """The source project's published numbers for the shipped configuration.

    From Gigalogy's README ("Does it actually work?"): 40 questions, 3 runs, ``qwen2.5-coder:7b``.
    They were measured on the commit that README names, not on the working-tree copy Geekbot
    vendored (VENDORED.md), and on Gigalogy's own harness. Quoted as the source project's
    measurements only (D-041).
    """

    solved: int = 35
    questions: int = 40
    easy: int = 10
    medium: int = 8
    hard: int = 7
    unanswerable_refused: int = 10
    correct_per_run: float = 34.0
    invented: int = 0
    invented_of: int = 30
    seconds_per_question: float = 3.0


SOURCE_REFERENCE = SourceReference()


@dataclass(frozen=True)
class Summary:
    runs: int
    questions: int  # distinct questions
    question_runs: int
    correct_per_run: list[int]
    mean_correct_per_run: float
    solved: int
    solved_by_difficulty: dict[str, list[int]]  # difficulty -> [solved, total]
    unstable_questions: int  # right in some runs and wrong in others
    invented: int  # unanswerable question-runs that were answered
    invented_of: int
    false_abstentions: int
    failures: int  # status "failed" (no reliable answer, time limit or backend down)
    latency_mean_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_max_s: float
    cap_s: int
    over_cap: int  # question-runs that took longer than the cap
    load_events: int  # question-runs whose model_load_ms exceeded LOAD_EVENT_MS
    load_ms_median: float
    load_ms_max: int
    steps_failed_runs: int  # question-runs in which at least one pipeline step failed
    slowest_step_s: float  # the longest single step in any question-run


def nearest_rank(values: Sequence[float], percent: int) -> float:
    """The nearest-rank percentile (the same one Gigalogy's harness uses)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
    return ordered[index]


def summarize(rows: Sequence[QuestionRun], *, runs: int, cap_s: int) -> Summary:
    by_question: dict[tuple[str, str], list[QuestionRun]] = {}
    for row in rows:
        by_question.setdefault((row.suite, row.id), []).append(row)

    solved_by_difficulty = {d: [0, 0] for d in DIFFICULTIES}
    solved = unstable = 0
    for question_rows in by_question.values():
        right = sum(r.correct for r in question_rows)
        is_solved = 2 * right > runs
        bucket = solved_by_difficulty.setdefault(question_rows[0].difficulty, [0, 0])
        bucket[0] += is_solved
        bucket[1] += 1
        solved += is_solved
        unstable += 0 < right < len(question_rows)

    seed_sets = sorted({r.seed_set for r in rows})
    correct_per_run = [sum(r.correct for r in rows if r.seed_set == s) for s in seed_sets]
    unanswerable = [r for r in rows if r.difficulty == "unanswerable"]
    answerable = [r for r in rows if r.difficulty != "unanswerable"]
    times = [r.elapsed_s for r in rows]
    loads = [r.model_load_ms for r in rows]
    return Summary(
        runs=runs,
        questions=len(by_question),
        question_runs=len(rows),
        correct_per_run=correct_per_run,
        mean_correct_per_run=statistics.fmean(correct_per_run) if correct_per_run else 0.0,
        solved=solved,
        solved_by_difficulty=solved_by_difficulty,
        unstable_questions=unstable,
        invented=sum(r.status == "answered" and not r.correct for r in unanswerable),
        invented_of=len(unanswerable),
        false_abstentions=sum(
            not r.correct and r.status in ("abstained", "needs_clarification") for r in answerable
        ),
        failures=sum(r.status == "failed" for r in rows),
        latency_mean_s=statistics.fmean(times) if times else 0.0,
        latency_p50_s=nearest_rank(times, 50),
        latency_p95_s=nearest_rank(times, 95),
        latency_max_s=max(times, default=0.0),
        cap_s=cap_s,
        over_cap=sum(t > cap_s for t in times),
        load_events=sum(ms > LOAD_EVENT_MS for ms in loads),
        load_ms_median=statistics.median(loads) if loads else 0.0,
        load_ms_max=max(loads, default=0),
        steps_failed_runs=sum(r.failed_steps > 0 for r in rows),
        slowest_step_s=max((r.slowest_step_s for r in rows), default=0.0),
    )


# --- the whole run ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preflight:
    """The state of the GPU before and after the CSV model was warmed up."""

    gpu_idle_mib: int | None  # GPU memory in use with none of our models loaded
    gpu_loaded_mib: int | None  # ... and with the CSV model warm
    cold_load_s: float  # time to load the CSV model onto an empty GPU
    unloaded: tuple[str, ...]  # our models that were loaded when the run began
    residency: str  # the R-1 check for the CSV model: "PASS: ..." / "FAIL: ..." / "unknown: ..."
    residency_ok: bool


@dataclass(frozen=True)
class RunContext:
    created: str
    model: str  # the CSV model that answered (D-043)
    profile: str
    host: str
    ollama_version: str
    source: str
    suites: list[str]
    runs: int
    limit: int | None
    time_limit_s: int  # the cap this run used
    geekbot_cap_s: int  # Geekbot's own cap, which the report compares the times against
    preflight: Preflight
    official: bool
    official_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReproResult:
    context: RunContext
    rows: list[QuestionRun]
    summary: Summary


def prepare_gpu(
    settings: Settings,
    client: SwapClient,
    *,
    read_gpu: Callable[[], GpuMemory],
    read_ps: Callable[[], list[LoadedModel]],
    patience: Patience | None = None,
) -> Preflight:
    """Empty the GPU of our models, note how busy it is, load the CSV model, check R-1.

    Unloading first gives a clean "is anything else using the GPU?" reading (D-028) and makes
    the CSV model's first load a cold one, which is also worth knowing.
    """
    unloaded = unload_and_wait(
        client, [settings.resolved_chat_model, settings.resolved_csv_model], patience=patience
    )
    idle = try_read(read_gpu)
    swap = swap_to_mode(client, settings, "csv", patience=patience)
    loaded = try_read(read_gpu)
    ps = try_read(read_ps)
    if ps is None:
        residency, ok = "unknown: could not read `ollama ps`", False
    else:
        check = check_residency(
            settings.resolved_csv_model, ps, expected_context=settings.csv_num_ctx
        )
        residency, ok = f"{check.status}: {check.detail}", check.status == "PASS"
    return Preflight(
        gpu_idle_mib=None if idle is None else idle.used_mib,
        gpu_loaded_mib=None if loaded is None else loaded.used_mib,
        cold_load_s=swap.warm_up_s,
        unloaded=unloaded,
        residency=residency,
        residency_ok=ok,
    )


def official_notes(
    settings: Settings, preflight: Preflight, *, suites: Sequence[str], runs: int, limit: int | None
) -> list[str]:
    """Why this run's numbers are *not* official (empty = official). D-028, D-043."""
    notes: list[str] = []
    target = PROFILES["target"].csv
    if settings.resolved_csv_model != target:
        notes.append(f"model is {settings.resolved_csv_model}, not the target model {target}")
    if preflight.gpu_idle_mib is None:
        notes.append("GPU use could not be read, so an idle GPU is not verified")
    elif preflight.gpu_idle_mib > BUSY_GPU_USED_MIB:
        notes.append(
            f"the GPU was not idle: {preflight.gpu_idle_mib} MiB in use with our models unloaded"
        )
    if not preflight.residency_ok:
        notes.append(f"residency (R-1) not confirmed: {preflight.residency}")
    if sorted(suites) != sorted(SUITE_NAMES) or runs < DEFAULT_RUNS or limit is not None:
        notes.append("partial run (not both suites, fewer than 3 runs, or --limit)")
    return notes


def run_repro(
    settings: Settings,
    suites: Sequence[Suite],
    *,
    runs: int,
    limit: int | None,
    time_limit_s: int,
    source: Path,
    client: ReproClient,
    read_gpu: Callable[[], GpuMemory],
    read_ps: Callable[[], list[LoadedModel]],
    open_analyst: OpenAnalyst = CsvAnalyst.open,
    grade_fn: GradeFn = grade,
    report: Callable[[str], None] = print,
    out: Path | None = None,
) -> ReproResult:
    """Warm the CSV model, then ask every suite's questions ``runs`` times and summarise.

    The loop order is seed set, then suite, then question (as Gigalogy's harness does), so slow
    drift in the machine hits every question alike. With ``out``, the result so far is written
    after each seed set, so a crash or Ctrl-C keeps the finished runs.
    """
    preflight = prepare_gpu(settings, client, read_gpu=read_gpu, read_ps=read_ps)
    report(f"CSV model {settings.resolved_csv_model} loaded cold in {preflight.cold_load_s:.1f} s")
    report(f"residency: {preflight.residency}")
    notes = official_notes(
        settings, preflight, suites=[s.name for s in suites], runs=runs, limit=limit
    )
    context = RunContext(
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        model=settings.resolved_csv_model,
        profile=settings.profile,
        host=settings.ollama_host,
        ollama_version=client.version(),
        source=str(source),
        suites=[s.name for s in suites],
        runs=runs,
        limit=limit,
        time_limit_s=time_limit_s,
        geekbot_cap_s=settings.csv_question_time_limit_s,
        preflight=preflight,
        official=not notes,
        official_notes=notes,
    )
    # The run asks questions under the package's cap (see DEFAULT_TIME_LIMIT_S); settings are
    # frozen, so make a copy with that one value changed.
    run_settings = settings.model_copy(update={"csv_question_time_limit_s": time_limit_s})

    rows: list[QuestionRun] = []
    with ExitStack() as stack:
        analysts = {
            suite.name: stack.enter_context(open_analyst(suite.csv, run_settings))
            for suite in suites
        }
        for seed_set in range(runs):
            for suite in suites:
                rows += run_suite_once(
                    suite, analysts[suite.name], seed_set=seed_set, grade_fn=grade_fn, report=report
                )
            done = sum(r.correct for r in rows if r.seed_set == seed_set)
            report(
                f"seed set {seed_set} done: {done} of "
                f"{sum(r.seed_set == seed_set for r in rows)} correct"
            )
            if out is not None:
                write_result(
                    out,
                    ReproResult(
                        context, list(rows), summarize(rows, runs=runs, cap_s=context.geekbot_cap_s)
                    ),
                )
    return ReproResult(context, rows, summarize(rows, runs=runs, cap_s=context.geekbot_cap_s))


# --- output ----------------------------------------------------------------------------------


def default_output_path(settings: Settings, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
    model = settings.resolved_csv_model.replace(":", "-").replace("/", "-")
    return Path("evals/csv/results") / f"repro-{settings.profile}-{model}-{stamp}.json"


def write_result(path: Path, result: ReproResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


def format_report(result: ReproResult) -> str:
    c, s = result.context, result.summary
    ref = SOURCE_REFERENCE
    sbd = s.solved_by_difficulty
    easy, medium, hard = (sbd[d][0] for d in ("easy", "medium", "hard"))
    refused, refused_of = sbd["unanswerable"]
    lines = [
        "Reproduction run: Gigalogy's CSV benchmark through Geekbot's adapter",
        f"model {c.model}   profile {c.profile}   Ollama {c.ollama_version}   {c.created}",
        (
            f"suites {', '.join(c.suites)}   runs (seed sets) {c.runs}   "
            f"questions {s.questions}   question-runs {s.question_runs}   "
            f"per-question cap used {c.time_limit_s} s"
        ),
        "official: yes" if c.official else "official: NO (a plumbing check, not a result)",
        *(f"  - {note}" for note in c.official_notes),
        (
            f"CSV model: cold load {c.preflight.cold_load_s:.1f} s; GPU memory "
            f"{c.preflight.gpu_idle_mib} MiB idle -> {c.preflight.gpu_loaded_mib} MiB loaded; "
            f"{c.preflight.residency}"
        ),
        "",
        f"{'measure':<46}{'this run':>10}{'source project*':>17}",
        (
            f"{'questions solved (right in >half the runs)':<46}{s.solved:>7}/{s.questions:<2}"
            f"{ref.solved:>14}/{ref.questions}"
        ),
        (
            f"{'easy / medium / hard solved (of 10 each)':<46}{f'{easy}/{medium}/{hard}':>10}"
            f"{f'{ref.easy}/{ref.medium}/{ref.hard}':>17}"
        ),
        (
            f"{'unanswerable refused (solved rule)':<46}{f'{refused}/{refused_of}':>10}"
            f"{f'{ref.unanswerable_refused}/10':>17}"
        ),
        (
            f"{'correct answers per run (mean)':<46}{s.mean_correct_per_run:>10.1f}"
            f"{ref.correct_per_run:>17.1f}"
        ),
        (
            f"{'invented answers (unanswerable question-runs)':<46}"
            f"{f'{s.invented} of {s.invented_of}':>10}"
            f"{f'{ref.invented} of {ref.invented_of}':>17}"
        ),
        (
            f"{'seconds per question (mean)':<46}{s.latency_mean_s:>10.1f}"
            f"{ref.seconds_per_question:>17.1f}"
        ),
        "* Gigalogy's README, measured on its own harness and commit; the vendored copy is a",
        "  later working tree (VENDORED.md). Gigalogy: 2-3 answers of 120 differ between identical",
        "  runs, so a small gap is noise.",
        "",
        (
            f"correct per run: {s.correct_per_run}   "
            f"unstable questions (right in some runs only): {s.unstable_questions}"
        ),
        f"false abstentions: {s.false_abstentions}   failed (no reliable answer): {s.failures}",
        (
            f"seconds per question: mean {s.latency_mean_s:.1f}, p50 {s.latency_p50_s:.1f}, "
            f"p95 {s.latency_p95_s:.1f}, max {s.latency_max_s:.1f}"
        ),
        (
            f"question-runs over Geekbot's {s.cap_s} s cap: {s.over_cap} of {s.question_runs} "
            "(an upper bound on what the cap would cut: it is checked only between model calls)"
        ),
        (
            "question-runs with a failed step (e.g. a model call that timed out): "
            f"{s.steps_failed_runs} of {s.question_runs}; "
            f"slowest single step {s.slowest_step_s:.1f} s"
        ),
        (
            f"model loads during questions (> {LOAD_EVENT_MS} ms of load_duration): "
            f"{s.load_events}; load_duration median {s.load_ms_median:.0f} ms, "
            f"max {s.load_ms_max} ms"
        ),
    ]
    return "\n".join(lines)
