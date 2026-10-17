"""Tests for step S2.3: the percentile and the rerank-depth timing harness (this step's ✍️ piece).

The harness takes the scorer and the clock as arguments, so it is tested with a fake scorer that
advances a fake clock by a known amount per passage: every number below is exact, none is a
measurement of a real model.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.retrieval.rerank_timing import (
    DepthTiming,
    format_depth_timings,
    measure_rerank_depths,
    percentile,
)

# --- percentile ------------------------------------------------------------------------------


def test_percentiles_of_a_small_list_by_hand() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 25) == 2.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 75) == 4.0
    assert percentile(values, 100) == 5.0


def test_between_two_values_it_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([10.0, 20.0], 25) == pytest.approx(12.5)
    assert percentile([float(i) for i in range(1, 101)], 95) == pytest.approx(95.05)


def test_one_value_is_every_percentile_of_itself() -> None:
    assert percentile([7.5], 0) == percentile([7.5], 50) == percentile([7.5], 100) == 7.5


def test_the_order_of_the_input_does_not_matter() -> None:
    assert percentile([5.0, 1.0, 4.0, 2.0, 3.0], 50) == 3.0


def test_percentile_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)
    for bad in (-0.1, 100.1):
        with pytest.raises(ValueError, match="between 0 and 100"):
            percentile([1.0], bad)


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=60
    ),
    q=st.floats(min_value=0, max_value=100),
)
def test_percentile_agrees_with_numpy_and_stays_inside_the_data(
    values: list[float], q: float
) -> None:
    got = percentile(values, q)
    assert got == pytest.approx(float(np.percentile(values, q)), rel=1e-9, abs=1e-9)
    assert min(values) <= got <= max(values)


@given(
    values=st.lists(
        st.floats(min_value=0, max_value=1e6, allow_nan=False), min_size=1, max_size=40
    ),
    low=st.floats(min_value=0, max_value=100),
    high=st.floats(min_value=0, max_value=100),
)
def test_a_higher_percentile_is_never_smaller(values: list[float], low: float, high: float) -> None:
    low, high = sorted((low, high))
    assert percentile(values, low) <= percentile(values, high) + 1e-9


# --- the harness -----------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SlowScorer:
    """Takes ``per_passage`` seconds of fake time per passage, and records every call."""

    def __init__(self, clock: FakeClock, per_passage: float = 0.01) -> None:
        self._clock = clock
        self._per_passage = per_passage
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls.append((query, list(passages)))
        self._clock.now += self._per_passage * len(passages)
        return [0.0] * len(passages)


PASSAGES = [f"passage {i}" for i in range(20)]


def measure(**kwargs: object) -> tuple[list[DepthTiming], SlowScorer]:
    clock = FakeClock()
    scorer = SlowScorer(clock)
    options: dict[str, object] = {"repeats": 3, "warmup": 1, "clock": clock}
    options.update(kwargs)
    timings = measure_rerank_depths(scorer, ["q1", "q2"], PASSAGES, [5, 10, 20], **options)  # type: ignore[arg-type]
    return timings, scorer


def test_time_grows_with_depth_and_is_read_off_the_clock() -> None:
    timings, _ = measure()
    assert [t.depth for t in timings] == [5, 10, 20]
    # 10 ms per passage, every call identical, so every percentile is the same exact number
    assert [t.p50_ms for t in timings] == pytest.approx([50.0, 100.0, 200.0])
    assert [t.p95_ms for t in timings] == pytest.approx([50.0, 100.0, 200.0])
    assert [t.max_ms for t in timings] == pytest.approx([50.0, 100.0, 200.0])


def test_every_query_is_repeated_at_every_depth() -> None:
    timings, scorer = measure()
    assert [t.runs for t in timings] == [6, 6, 6]  # 2 queries x 3 repeats
    assert len(scorer.calls) == 1 + 3 * 6  # one warm-up, then 18 timed calls


def test_the_warmup_is_untimed_runs_first_and_uses_the_deepest_depth() -> None:
    _, scorer = measure(warmup=2)
    assert len(scorer.calls) == 2 + 18
    assert [len(p) for _, p in scorer.calls[:2]] == [20, 20]  # the largest depth, twice
    assert scorer.calls[0][0] == "q1"


def test_no_warmup_is_allowed() -> None:
    _, scorer = measure(warmup=0)
    assert len(scorer.calls) == 18


def test_every_depth_scores_the_same_leading_passages() -> None:
    _, scorer = measure(warmup=0)
    for _, passages in scorer.calls:
        assert passages == PASSAGES[: len(passages)]


def test_a_slow_call_shows_up_in_p95_and_max_but_barely_in_p50() -> None:
    clock = FakeClock()
    durations = iter([0.1] * 19 + [1.0])  # nineteen quick calls and one slow one

    class Spiky:
        def score(self, query: str, passages: list[str]) -> list[float]:
            clock.now += next(durations)
            return [0.0] * len(passages)

    [timing] = measure_rerank_depths(
        Spiky(), ["q"], PASSAGES, [5], repeats=20, warmup=0, clock=clock
    )
    assert timing.p50_ms == pytest.approx(100.0)
    assert timing.max_ms == pytest.approx(1000.0)
    assert 100.0 < timing.p95_ms < 1000.0  # between: the slow call pulls the 95th percentile up


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"queries": []}, "at least one query"),
        ({"depths": []}, "at least one depth"),
        ({"depths": [0]}, "at least 1"),
        ({"depths": [21]}, "more than the 20 passages"),
        ({"repeats": 0}, "repeats must be at least 1"),
        ({"warmup": -1}, "warmup must not be negative"),
    ],
)
def test_bad_arguments_are_refused_before_anything_is_scored(
    kwargs: dict[str, object], message: str
) -> None:
    clock = FakeClock()
    scorer = SlowScorer(clock)
    arguments: dict[str, object] = {"queries": ["q"], "depths": [5], "repeats": 1, "warmup": 1}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        measure_rerank_depths(
            scorer,
            arguments["queries"],  # type: ignore[arg-type]
            PASSAGES,
            arguments["depths"],  # type: ignore[arg-type]
            repeats=arguments["repeats"],  # type: ignore[arg-type]
            warmup=arguments["warmup"],  # type: ignore[arg-type]
            clock=clock,
        )
    assert scorer.calls == []


# --- the table -------------------------------------------------------------------------------


def test_the_table_names_the_model_and_lists_each_depth() -> None:
    table = format_depth_timings(
        "some/model",
        [DepthTiming(10, 15, 12.34, 20.5, 31.0), DepthTiming(30, 15, 40.0, 61.25, 70.0)],
    )
    lines = table.splitlines()
    assert lines[0] == "reranker some/model on the CPU"
    assert "p50 ms" in lines[1]
    assert lines[2].split() == ["10", "15", "12.3", "20.5", "31.0"]
    assert lines[3].split() == ["30", "15", "40.0", "61.2", "70.0"]
    assert "400 ms" in lines[-1]
