"""Tests for step S3.3: the latency harness (cold rounds, steady state, percentiles, loads).

The model, the retrieval and the clock are all fakes: ``run_once`` returns canned readings, so the
harness's own work is what is tested (what it runs in which order, what it counts, what it prints).
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.agent.latency import (
    LOAD_NOTICE_SECONDS,
    LatencyReport,
    LatencyRun,
    count_loads,
    format_latency_report,
    measure_latency,
    spread,
)
from geekbot.domain import RunStats

MODEL = "fake-model:1b"


def run_of(
    question: str = "q1",
    *,
    first: float = 0.3,
    whole: float = 1.0,
    load: float | None = 0.01,
    prompt: int | None = 800,
    read: float | None = None,
    speed: float | None = 100.0,
    retrieval: float | None = 20.0,
    cold: bool = False,
) -> LatencyRun:
    stats = RunStats(
        seconds=whole,
        first_token_s=first,
        load_s=load,
        prompt_tokens=prompt,
        prompt_eval_s=read,
        tokens_per_second=speed,
    )
    return LatencyRun(question=question, stats=stats, retrieval_ms=retrieval, cold=cold)


# --- measure_latency: what runs, in which order ----------------------------------------------


class Recorder:
    """Records the order of cold starts and questions, and answers each question the same way."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def cold_start(self) -> None:
        self.calls.append("<cold start>")

    def run_once(self, question: str) -> LatencyRun:
        self.calls.append(question)
        return run_of(question)


def test_each_cold_round_restarts_the_model_then_asks_the_first_question_once() -> None:
    recorder = Recorder()
    report = measure_latency(
        recorder.run_once, recorder.cold_start, ["a", "b"], model=MODEL, cold_rounds=2, warm_runs=1
    )
    assert recorder.calls == ["<cold start>", "a", "<cold start>", "a", "a", "b"]
    assert [run.cold for run in report.runs] == [True, True, False, False]


def test_the_warm_passes_ask_every_question_in_turn() -> None:
    recorder = Recorder()
    report = measure_latency(
        recorder.run_once,
        recorder.cold_start,
        ["a", "b", "c"],
        model=MODEL,
        cold_rounds=0,
        warm_runs=2,
    )
    assert recorder.calls == ["a", "b", "c", "a", "b", "c"]
    assert report.cold == ()
    assert len(report.warm) == 6


def test_the_one_off_retrieval_time_is_carried_into_the_report() -> None:
    recorder = Recorder()
    report = measure_latency(
        recorder.run_once, recorder.cold_start, ["a"], model=MODEL, first_retrieval_ms=1452.0
    )
    assert report.first_retrieval_ms == 1452.0
    assert report.model == MODEL


@pytest.mark.parametrize(
    ("questions", "cold_rounds", "warm_runs"),
    [([], 1, 1), (["a"], -1, 1), (["a"], 0, 0)],
)
def test_nonsense_arguments_are_refused_before_anything_runs(
    questions: list[str], cold_rounds: int, warm_runs: int
) -> None:
    recorder = Recorder()
    with pytest.raises(ValueError, match="at least|negative"):
        measure_latency(
            recorder.run_once,
            recorder.cold_start,
            questions,
            model=MODEL,
            cold_rounds=cold_rounds,
            warm_runs=warm_runs,
        )
    assert recorder.calls == []


def test_a_failing_question_ends_the_measurement_with_its_own_error() -> None:
    def failing(question: str) -> LatencyRun:
        raise RuntimeError(f"no answer for {question}")

    with pytest.raises(RuntimeError, match="no answer for a"):
        measure_latency(failing, lambda: None, ["a"], model=MODEL, cold_rounds=0)


# --- spread and the load count ---------------------------------------------------------------


def test_spread_gives_the_count_median_95th_percentile_and_maximum() -> None:
    found = spread([1.0, 2.0, 3.0, 4.0, 5.0])
    assert found is not None
    assert (found.n, found.p50, found.maximum) == (5, 3.0, 5.0)
    assert found.p95 == pytest.approx(4.8)


def test_a_reading_that_never_existed_has_no_spread() -> None:
    assert spread([]) is None


@given(st.lists(st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=40))
def test_the_spread_is_ordered(values: list[float]) -> None:
    found = spread(values)
    assert found is not None
    assert min(values) <= found.p50 <= found.p95 <= found.maximum == max(values)


def test_runs_that_loaded_the_model_again_are_counted_at_the_threshold() -> None:
    runs = [
        run_of(load=0.0),
        run_of(load=LOAD_NOTICE_SECONDS - 0.01),
        run_of(load=LOAD_NOTICE_SECONDS),
        run_of(load=6.2),
        run_of(load=None),
    ]
    assert count_loads(runs) == 2


# --- the printed report ----------------------------------------------------------------------


def sample_report() -> LatencyReport:
    runs = (
        run_of("How long?", first=5.9, whole=6.5, load=0.02, prompt=812, read=0.31, cold=True),
        run_of("How long?", first=0.3, whole=1.0, prompt=812, read=0.12),
        run_of("Where?", first=0.4, whole=1.2, prompt=640, read=0.1),
        run_of("How long?", first=0.2, whole=0.9, prompt=812, read=0.02, speed=120.0),
        run_of("Where?", first=0.5, whole=1.4, prompt=640, load=3.1, retrieval=30.0),
    )
    return LatencyReport(model=MODEL, runs=runs, first_retrieval_ms=1452.0)


def test_the_report_names_the_model_and_says_what_the_numbers_are() -> None:
    shown = format_latency_report(sample_report(), setup="SETUP LINE")
    assert shown.splitlines()[0] == "SETUP LINE"
    assert f"for {MODEL}" in shown
    assert "close to the slowest one" in shown
    assert "not a target-model result" in shown


def test_the_one_off_retrieval_is_reported_apart() -> None:
    shown = format_latency_report(sample_report(), setup="s")
    assert "first retrieval in this process: 1452 ms" in shown
    assert "every later retrieval is warm" in shown


def test_the_cold_rounds_are_listed_one_by_one() -> None:
    shown = format_latency_report(sample_report(), setup="s")
    assert "first question after a fresh load of the model (1 round):" in shown
    assert (
        "  round 1: first token 5.90 s, whole answer 6.50 s, model load 0.02 s, prompt 812 tokens"
        in shown
    )


def test_the_steady_state_table_has_counts_and_percentiles() -> None:
    shown = format_latency_report(sample_report(), setup="s")
    assert "steady state, model warm (4 runs over 2 questions):" in shown
    first_row = next(line for line in shown.splitlines() if "time to first token" in line)
    assert "   4  p50" in first_row
    assert "0.35 s" in first_row  # the median of 0.2, 0.3, 0.4, 0.5
    assert any(
        "tokens per second" in line and "slowest 100.0" in line for line in shown.splitlines()
    )


def steady_report() -> str:
    runs = tuple(
        LatencyRun(
            question="q",
            stats=RunStats(seconds=2.0, first_token_s=0.5),
            retrieval_ms=dense + rerank,
            stages_ms={"dense": dense, "rerank": rerank},
        )
        for dense, rerank in [(20.0, 880.0), (30.0, 970.0), (25.0, 1075.0)]
    )
    return format_latency_report(LatencyReport(model=MODEL, runs=runs), setup="s")


def row_of(shown: str, label: str) -> str:
    return next(line for line in shown.splitlines() if label in line)


def test_a_slow_retrieval_is_traced_to_its_stages() -> None:
    shown = steady_report()
    assert "p50   1000 ms" in row_of(shown, "  retrieval ")  # 900, 1000, 1100
    assert "p50     25 ms" in row_of(shown, "of which dense")
    assert "p50    970 ms" in row_of(shown, "of which rerank")
    assert "max   1075 ms" in row_of(shown, "of which rerank")


def test_what_the_user_waits_for_is_retrieval_plus_the_model_run_by_run() -> None:
    shown = steady_report()
    assert "p50    1.50 s" in row_of(shown, "question to first token")  # 1.4, 1.5, 1.6
    assert "max    1.60 s" in row_of(shown, "question to first token")
    assert "p50    3.00 s" in row_of(shown, "question to whole answer")  # 2.9, 3.0, 3.1


def test_without_a_retrieval_time_there_is_no_question_to_answer_row() -> None:
    runs = (run_of(retrieval=None),)
    shown = format_latency_report(LatencyReport(model=MODEL, runs=runs), setup="s")
    assert "question to" not in shown


def test_model_loads_are_counted_over_every_run() -> None:
    shown = format_latency_report(sample_report(), setup="s")
    assert "model loads during questions: 1 of 5 runs spent 0.5 s or more" in shown


def test_the_prompt_readings_are_listed_per_question_in_the_order_they_happened() -> None:
    shown = format_latency_report(sample_report(), setup="s")
    assert "  'How long?': 812 (0.31 s), 812 (0.12 s), 812 (0.02 s)" in shown
    assert "  'Where?': 640 (0.10 s), 640" in shown  # the last run reported no read time
    assert "the read time should fall on a repeat" in shown


def test_a_long_question_is_cut_in_the_listing_and_a_missing_count_is_a_question_mark() -> None:
    long_question = "What is the policy on keeping chat history for staff members?"
    report = LatencyReport(model=MODEL, runs=(run_of(long_question, prompt=None),))
    shown = format_latency_report(report, setup="s")
    assert "'What is the policy on keepin…': ?" in shown
    assert "steady state, model warm (1 run over 1 question):" in shown


def test_a_report_without_cold_rounds_or_retrieval_figures_still_prints() -> None:
    runs = (run_of(retrieval=None, speed=None),)
    shown = format_latency_report(LatencyReport(model=MODEL, runs=runs), setup="s")
    assert "first question after a fresh load" not in shown
    assert "first retrieval in this process" not in shown
    assert not any(line.strip().startswith("retrieval") for line in shown.splitlines())
    assert "tokens per second" not in shown
