"""Tests for step S3.3: ``RunStats``, the plain record of what one model call cost."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.domain import RunStats

SECONDS = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)
COUNT = st.integers(min_value=0, max_value=10**6)


def test_only_the_readings_that_exist_are_in_the_dictionary() -> None:
    assert RunStats(seconds=1.5).as_dict() == {"seconds": 1.5}
    full = RunStats(
        seconds=3.0,
        first_token_s=0.5,
        load_s=0.01,
        prompt_tokens=800,
        prompt_eval_s=0.4,
        output_tokens=40,
        tokens_per_second=20.0,
        thinking_chars=12,
    )
    assert full.as_dict() == {
        "seconds": 3.0,
        "first_token_seconds": 0.5,
        "load_seconds": 0.01,
        "prompt_tokens": 800,
        "prompt_eval_seconds": 0.4,
        "output_tokens": 40,
        "tokens_per_second": 20.0,
        "thinking_chars": 12,
    }


def test_a_zero_is_a_reading_but_no_thinking_is_not() -> None:
    stats = RunStats(seconds=1.0, load_s=0.0, output_tokens=0)
    assert stats.as_dict() == {"seconds": 1.0, "load_seconds": 0.0, "output_tokens": 0}  # kept
    assert "thinking_chars" not in stats.as_dict()  # 0 characters means thinking was off


def test_from_mapping_needs_the_seconds() -> None:
    with pytest.raises(ValueError, match="no 'seconds'"):
        RunStats.from_mapping({"first_token_seconds": 0.2})


def test_from_mapping_reads_whole_numbers_back_as_integers() -> None:
    stats = RunStats.from_mapping({"seconds": 2.0, "prompt_tokens": 812.0, "thinking_chars": 7.0})
    assert stats.prompt_tokens == 812
    assert isinstance(stats.prompt_tokens, int)
    assert stats.thinking_chars == 7
    assert stats.output_tokens is None


@given(
    seconds=SECONDS,
    first=st.none() | SECONDS,
    load=st.none() | SECONDS,
    prompt=st.none() | COUNT,
    read=st.none() | SECONDS,
    out=st.none() | COUNT,
    speed=st.none() | SECONDS,
    thinking=COUNT,
)
def test_a_record_survives_the_trip_through_its_plain_data_form(
    seconds: float,
    first: float | None,
    load: float | None,
    prompt: int | None,
    read: float | None,
    out: int | None,
    speed: float | None,
    thinking: int,
) -> None:
    original = RunStats(
        seconds=seconds,
        first_token_s=first,
        load_s=load,
        prompt_tokens=prompt,
        prompt_eval_s=read,
        output_tokens=out,
        tokens_per_second=speed,
        thinking_chars=thinking,
    )
    assert RunStats.from_mapping(original.as_dict()) == original
