"""Measuring how fast a question gets answered, properly (step S3.3, PLAN.md §6.5).

A single timed question says almost nothing. The same question can take three times as long on its
first call as on its tenth, and the plan's targets are written on percentiles, not on one run. So
this harness separates two situations and reports each with its spread:

- **The first question after a fresh load of the model** ("cold", but the model *was* warmed up on
  purpose, as the server does at start: rule R-2). S3.1 once saw 5.9 s for such a first call against
  0.4 s for the next; whether that repeats, and how big it is, is what these rounds measure.
- **Steady state** (the model is warm and has just answered): many runs, the questions taken in
  turns so the start of the prompt (the fixed system message) is shared but the sources differ, as in
  real use.

For every run it keeps Geekbot's own clock readings (time to first token, whole answer, retrieval)
and Ollama's (model load, prompt tokens, tokens per second). It counts the runs where Ollama spent
0.5 s or more loading the model, because the plan's rule R-2 says that number must be 0.

Why percentiles, and how few runs they can bear. The median (p50) is the typical run and p95 is the
slow end a user notices, but with 15 runs the 95th percentile sits between the two slowest, so it is
close to the maximum: the table prints the run count next to it, and nothing here should be quoted
without it. The harness takes its model, its retrieval and its clock as parameters (the command
supplies the real ones), so it is tested with fakes. What it prints is a measurement of one machine
at one moment, for one named model.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

from geekbot.domain import RunStats
from geekbot.retrieval.rerank_timing import percentile

__all__ = [
    "LOAD_NOTICE_SECONDS",
    "LatencyError",
    "LatencyReport",
    "LatencyRun",
    "Spread",
    "count_loads",
    "format_latency_report",
    "measure_latency",
    "spread",
]

# Ollama's own ``load_duration`` for a call is a few hundredths of a second when the model is
# already in memory and seconds when it had to be loaded again; half a second separates the two.
LOAD_NOTICE_SECONDS: Final = 0.5

_QUESTION_WIDTH: Final = 28  # how much of a question the report prints
_LABEL_WIDTH: Final = 24  # the width of a row's label in the table
_MILLISECONDS: Final = 1000.0


class LatencyError(RuntimeError):
    """A timed question could not be timed (for example, it found nothing to answer from)."""


@dataclass(frozen=True)
class LatencyRun:
    """One question through the whole graph: the model's readings and the retrieval time.

    ``retrieval_ms`` is the retriever's own total; ``stages_ms`` its stages (dense, lexical,
    fusion, rerank, ...), so a slow total can be traced to the stage that made it slow.
    """

    question: str
    stats: RunStats
    retrieval_ms: float | None = None
    cold: bool = False
    stages_ms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencyReport:
    """Every run in the order it happened, and the one-off retrieval warm-up."""

    model: str
    runs: tuple[LatencyRun, ...]
    first_retrieval_ms: float | None = None

    @property
    def cold(self) -> tuple[LatencyRun, ...]:
        return tuple(run for run in self.runs if run.cold)

    @property
    def warm(self) -> tuple[LatencyRun, ...]:
        return tuple(run for run in self.runs if not run.cold)


@dataclass(frozen=True)
class Spread:
    """The count, median, 95th percentile and maximum of some readings."""

    n: int
    p50: float
    p95: float
    maximum: float


def spread(values: Sequence[float]) -> Spread | None:
    """The spread of ``values``, or ``None`` for no values (a reading that never existed)."""
    if not values:
        return None
    return Spread(
        n=len(values),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        maximum=max(values),
    )


def count_loads(runs: Sequence[LatencyRun], *, threshold_s: float = LOAD_NOTICE_SECONDS) -> int:
    """How many runs had Ollama spend ``threshold_s`` or more loading the model."""
    return sum(1 for run in runs if (run.stats.load_s or 0.0) >= threshold_s)


def measure_latency(
    run_once: Callable[[str], LatencyRun],
    cold_start: Callable[[], None],
    questions: Sequence[str],
    *,
    model: str,
    cold_rounds: int = 3,
    warm_runs: int = 5,
    first_retrieval_ms: float | None = None,
) -> LatencyReport:
    """Run the cold rounds, then the steady-state runs, and collect every reading.

    A cold round is ``cold_start()`` (the command unloads the chat model and warms it up again, the
    way the server does at start) followed by the *first* question, once. Then each of the
    ``warm_runs`` passes asks every question in turn. ``run_once`` must not hide a failure:
    an exception ends the measurement. ``ValueError`` for no questions, a negative number of cold
    rounds or fewer than one warm pass.
    """
    if not questions:
        raise ValueError("give at least one question")
    if cold_rounds < 0:
        raise ValueError(f"cold_rounds must not be negative, got {cold_rounds}")
    if warm_runs < 1:
        raise ValueError(f"warm_runs must be at least 1, got {warm_runs}")
    runs: list[LatencyRun] = []
    for _ in range(cold_rounds):
        cold_start()
        runs.append(replace(run_once(questions[0]), cold=True))
    for _ in range(warm_runs):
        runs.extend(replace(run_once(question), cold=False) for question in questions)
    return LatencyReport(model=model, runs=tuple(runs), first_retrieval_ms=first_retrieval_ms)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _seconds(value: float) -> str:
    return f"{value:.2f} s"


def _millis(value: float) -> str:
    return f"{value:.0f} ms"


def _cold_line(number: int, run: LatencyRun) -> str:
    stats = run.stats
    facts: list[str] = []
    if stats.first_token_s is not None:
        facts.append(f"first token {_seconds(stats.first_token_s)}")
    facts.append(f"whole answer {_seconds(stats.seconds)}")
    if stats.load_s is not None:
        facts.append(f"model load {_seconds(stats.load_s)}")
    if stats.prompt_tokens is not None:
        facts.append(f"prompt {stats.prompt_tokens} tokens")
    return f"  round {number}: " + ", ".join(facts)


def _row(label: str, values: Sequence[float], show: Callable[[float], str]) -> str | None:
    found = spread(values)
    if found is None:
        return None
    return (
        f"  {label:<{_LABEL_WIDTH}} {found.n:>4}  p50 {show(found.p50):>9}  "
        f"p95 {show(found.p95):>9}  max {show(found.maximum):>9}"
    )


def _steady_rows(runs: Sequence[LatencyRun]) -> list[str]:
    """The table of the warm runs.

    The model's own times, then retrieval with its stages, then what the user waits for (retrieval
    first, then the model), which is what the plan's latency targets are written on.
    """
    first = [r.stats.first_token_s for r in runs if r.stats.first_token_s is not None]
    whole = [r.stats.seconds for r in runs]
    retrieval = [r.retrieval_ms for r in runs if r.retrieval_ms is not None]
    rows = [
        _row("time to first token", first, _seconds),
        _row("whole answer", whole, _seconds),
        _row("retrieval", retrieval, _millis),
    ]
    stages: dict[str, list[float]] = {}
    for run in runs:
        for stage, value in run.stages_ms.items():
            stages.setdefault(stage, []).append(value)
    rows.extend(_row(f"  of which {stage}", values, _millis) for stage, values in stages.items())
    # What the user waits for: the retrieval, then the model. Summed run by run, so these are the
    # spread of the sums, not the sum of the spreads.
    asked_to_first = [
        r.retrieval_ms / _MILLISECONDS + r.stats.first_token_s
        for r in runs
        if r.retrieval_ms is not None and r.stats.first_token_s is not None
    ]
    asked_to_done = [
        r.retrieval_ms / _MILLISECONDS + r.stats.seconds for r in runs if r.retrieval_ms is not None
    ]
    rows.append(_row("question to first token", asked_to_first, _seconds))
    rows.append(_row("question to whole answer", asked_to_done, _seconds))
    speeds = [r.stats.tokens_per_second for r in runs if r.stats.tokens_per_second is not None]
    if speeds:
        rows.append(
            f"  {'tokens per second':<{_LABEL_WIDTH}} {len(speeds):>4}  "
            f"p50 {percentile(speeds, 50):>9.1f}  slowest {min(speeds):.1f}"
        )
    return [row for row in rows if row is not None]


def _prompt_reading(stats: RunStats) -> str:
    """``812 (0.12 s)``: the prompt token count and the time Ollama says it spent reading it."""
    if stats.prompt_tokens is None:
        return "?"
    if stats.prompt_eval_s is None:
        return str(stats.prompt_tokens)
    return f"{stats.prompt_tokens} ({_seconds(stats.prompt_eval_s)})"


def _prompt_lines(runs: Sequence[LatencyRun]) -> list[str]:
    """Per question, the prompt readings of each run, in the order the runs happened."""
    order: list[str] = []
    counts: dict[str, list[str]] = {}
    for run in runs:
        if run.question not in counts:
            order.append(run.question)
            counts[run.question] = []
        counts[run.question].append(_prompt_reading(run.stats))
    lines: list[str] = []
    for question in order:
        shown = question if len(question) <= _QUESTION_WIDTH else question[:_QUESTION_WIDTH] + "…"
        lines.append(f"  {shown!r}: " + ", ".join(counts[question]))
    return lines


def format_latency_report(report: LatencyReport, *, setup: str) -> str:
    """A report for the terminal. ``setup`` says how the run was set up (model, mode, thinking)."""
    lines = [setup]
    if report.first_retrieval_ms is not None:
        lines.append(
            f"first retrieval in this process: {_millis(report.first_retrieval_ms)} "
            "(loads the embedder client and the reranker; every later retrieval is warm)"
        )
    cold = report.cold
    if cold:
        rounds = _plural(len(cold), "round")
        lines.extend(["", f"first question after a fresh load of the model ({rounds}):"])
        lines.extend(_cold_line(number, run) for number, run in enumerate(cold, start=1))
    warm = report.warm
    runs = _plural(len(warm), "run")
    questions = _plural(len({run.question for run in warm}), "question")
    lines.extend(["", f"steady state, model warm ({runs} over {questions}):"])
    lines.append(f"  {'':<{_LABEL_WIDTH}} {'runs':>4}")
    lines.extend(_steady_rows(warm))
    loads = count_loads(report.runs)
    lines.append("")
    lines.append(
        f"model loads during questions: {loads} of {len(report.runs)} runs spent "
        f"{LOAD_NOTICE_SECONDS} s or more loading the model (rule R-2 wants 0 once it is warm)"
    )
    lines.append(
        "prompt per run, as Ollama reports it: tokens (time spent reading them), in the order the "
        "runs happened. If Ollama reuses the start of the prompt, the read time should fall on a "
        "repeat; the token count may stay the same:"
    )
    lines.extend(_prompt_lines(report.runs))
    lines.append("")
    lines.append(
        f"wall-clock readings on this machine, now, for {report.model}. With few runs the "
        "95th percentile is close to the slowest one. A number from a light prototyping model "
        "is a plumbing check, not a target-model result."
    )
    return "\n".join(lines)
