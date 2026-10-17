"""Tests for step S3.3: reading a chat stream (answer, reasoning, time to the first token).

``read_stream`` is this step's ✍️ piece. The chunks come from a generator that advances a fake clock
before it yields each one, so every time in these tests is exact: no sleeping, no real model.
"""

from collections.abc import Iterator, Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessageChunk, BaseMessage

from geekbot.llm.streaming import (
    REASONING_KEY,
    content_text,
    read_stream,
    stats_from_metadata,
)


class Clock:
    """A clock that only moves when the test (or the chunk generator) moves it."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def timed(clock: Clock, steps: Sequence[tuple[float, AIMessageChunk]]) -> Iterator[BaseMessage]:
    """Yield each chunk after the clock has moved on by its delay."""
    for delay, chunk in steps:
        clock.now += delay
        yield chunk


def text(piece: str) -> AIMessageChunk:
    return AIMessageChunk(content=piece)


def thought(piece: str) -> AIMessageChunk:
    return AIMessageChunk(content="", additional_kwargs={REASONING_KEY: piece})


# --- the time to the first token -------------------------------------------------------------


def test_the_time_to_first_token_is_when_the_first_piece_of_answer_text_arrives() -> None:
    clock = Clock()
    result = read_stream(
        timed(clock, [(0.5, text("Nin")), (0.1, text("ety")), (0.2, text("."))]), clock=clock
    )
    assert result.stats.first_token_s == pytest.approx(0.5)
    assert result.stats.seconds == pytest.approx(0.8)
    assert result.text == "Ninety."


def test_a_first_piece_of_only_whitespace_shows_the_user_nothing_so_it_does_not_count() -> None:
    clock = Clock()
    seen: list[str] = []
    steps = [(0.3, text("\n")), (0.2, text("  ")), (0.4, text("Hi"))]
    result = read_stream(timed(clock, steps), clock=clock, on_text=seen.append)
    assert result.stats.first_token_s == pytest.approx(0.9)
    assert seen == ["\n", "  ", "Hi"]  # but the pieces themselves are all passed on
    assert result.text == "Hi"


def test_reasoning_never_counts_as_the_first_token() -> None:
    clock = Clock()
    steps = [(1.0, thought("Hmm, ")), (2.0, thought("let me see.")), (0.5, text("Ninety days."))]
    result = read_stream(timed(clock, steps), clock=clock)
    assert result.stats.first_token_s == pytest.approx(3.5)
    assert result.thinking == "Hmm, let me see."
    assert result.stats.thinking_chars == len("Hmm, let me see.")
    assert result.text == "Ninety days."


def test_no_answer_text_means_no_first_token_and_an_empty_text() -> None:
    clock = Clock()
    result = read_stream(timed(clock, [(1.0, thought("still thinking"))]), clock=clock)
    assert result.stats.first_token_s is None
    assert result.text == ""
    assert result.thinking == "still thinking"
    assert result.stats.seconds == pytest.approx(1.0)


def test_an_empty_stream_is_an_empty_result_not_an_error() -> None:
    result = read_stream([], clock=Clock())
    assert (result.text, result.thinking) == ("", "")
    assert result.stats.first_token_s is None
    assert result.stats.seconds == 0.0


def test_the_clock_starts_when_the_stream_is_first_read_unless_a_start_is_given() -> None:
    clock = Clock(now=100.0)
    result = read_stream(timed(clock, [(0.5, text("a"))]), clock=clock)
    assert result.stats.first_token_s == pytest.approx(0.5)  # not 100.5

    clock = Clock(now=100.0)
    early = read_stream(timed(clock, [(0.5, text("a"))]), started=98.0, clock=clock)
    assert early.stats.first_token_s == pytest.approx(2.5)  # counted from the given start
    assert early.stats.seconds == pytest.approx(2.5)


# --- passing pieces on, and the text ---------------------------------------------------------


def test_pieces_are_passed_on_in_the_order_they_arrive_empty_ones_excepted() -> None:
    clock = Clock()
    events: list[tuple[str, str]] = []
    steps = [
        (0.1, thought("a")),
        (0.1, text("")),
        (0.1, text("Hello")),
        (0.1, thought("b")),
        (0.1, text(" world")),
    ]
    read_stream(
        timed(clock, steps),
        clock=clock,
        on_text=lambda piece: events.append(("text", piece)),
        on_thinking=lambda piece: events.append(("thinking", piece)),
    )
    assert events == [("thinking", "a"), ("text", "Hello"), ("thinking", "b"), ("text", " world")]


def test_the_text_is_the_pieces_joined_and_then_stripped() -> None:
    clock = Clock()
    result = read_stream(timed(clock, [(0, text("  Ninety")), (0, text(" days \n"))]), clock=clock)
    assert result.text == "Ninety days"


def test_without_callbacks_nothing_breaks() -> None:
    clock = Clock()
    assert (
        read_stream(timed(clock, [(0.1, text("x")), (0.1, thought("y"))]), clock=clock).text == "x"
    )


def test_a_stream_that_fails_half_way_raises_what_it_raised() -> None:
    boom = ConnectionError("dropped")

    def broken() -> Iterator[BaseMessage]:
        yield text("Nin")
        raise boom

    with pytest.raises(ConnectionError) as raised:
        read_stream(broken(), clock=Clock())
    assert raised.value is boom


# --- Ollama's numbers ------------------------------------------------------------------------


OLLAMA_DONE = {
    "model": "gemma3:4b",
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_000_000_000,
    "load_duration": 250_000_000,
    "prompt_eval_count": 800,
    "prompt_eval_duration": 400_000_000,
    "eval_count": 40,
    "eval_duration": 2_000_000_000,
}


def test_ollamas_numbers_survive_the_empty_marker_chunk_langchain_adds_after_the_last_one() -> None:
    clock = Clock()
    steps = [
        (0.1, text("Hi")),
        (0.1, AIMessageChunk(content="", response_metadata=dict(OLLAMA_DONE))),
        (0.0, AIMessageChunk(content="", chunk_position="last")),
    ]
    stats = read_stream(timed(clock, steps), clock=clock).stats
    assert stats.load_s == pytest.approx(0.25)
    assert stats.prompt_tokens == 800
    assert stats.prompt_eval_s == pytest.approx(0.4)
    assert stats.output_tokens == 40
    assert stats.tokens_per_second == pytest.approx(20.0)


def test_stats_from_metadata_reads_every_key() -> None:
    stats = stats_from_metadata(OLLAMA_DONE, seconds=3.1, first_token_s=0.7, thinking_chars=5)
    assert stats.seconds == 3.1
    assert stats.first_token_s == 0.7
    assert stats.load_s == pytest.approx(0.25)
    assert stats.prompt_tokens == 800
    assert stats.prompt_eval_s == pytest.approx(0.4)
    assert stats.output_tokens == 40
    assert stats.tokens_per_second == pytest.approx(20.0)
    assert stats.thinking_chars == 5


def test_missing_metadata_leaves_only_what_geekbot_measured() -> None:
    stats = stats_from_metadata({}, seconds=1.5, first_token_s=0.2)
    assert stats.as_dict() == {"seconds": 1.5, "first_token_seconds": 0.2}


def test_odd_metadata_is_ignored_not_trusted() -> None:
    odd = {"load_duration": "fast", "eval_count": True, "prompt_eval_count": None}
    assert stats_from_metadata(odd, seconds=1.0).as_dict() == {"seconds": 1.0}


def test_no_generation_time_means_no_tokens_per_second() -> None:
    zero = stats_from_metadata({"eval_count": 10, "eval_duration": 0}, seconds=1.0)
    assert zero.output_tokens == 10
    assert zero.tokens_per_second is None
    assert stats_from_metadata({"eval_duration": 5}, seconds=1.0).tokens_per_second is None


# --- content_text ----------------------------------------------------------------------------


def test_content_text_reads_a_string_or_content_blocks_and_never_strips() -> None:
    assert content_text("  plain  ") == "  plain  "
    blocks: list[str | dict[str, object]] = [
        {"type": "text", "text": "Hello, "},
        {"type": "image_url", "image_url": "ignored"},
        "world",
    ]
    assert content_text(blocks) == "Hello, world"
    assert content_text([]) == ""


# --- properties ------------------------------------------------------------------------------

PIECES = st.lists(st.text(max_size=12), max_size=14)


@given(PIECES)
def test_the_text_is_always_the_stripped_join_and_every_nonempty_piece_is_passed_on(
    pieces: list[str],
) -> None:
    clock = Clock()
    seen: list[str] = []
    steps = [(1.0, text(piece)) for piece in pieces]
    result = read_stream(timed(clock, steps), clock=clock, on_text=seen.append)
    assert result.text == "".join(pieces).strip()
    assert seen == [piece for piece in pieces if piece]


@given(PIECES)
def test_the_first_token_is_the_first_piece_with_something_to_read(pieces: list[str]) -> None:
    clock = Clock()
    steps = [(1.0, text(piece)) for piece in pieces]  # one tick per chunk, so times are exact
    stats = read_stream(timed(clock, steps), clock=clock).stats
    readable = [index for index, piece in enumerate(pieces) if piece.strip()]
    if readable:
        assert stats.first_token_s == float(readable[0] + 1)
        assert stats.first_token_s <= stats.seconds
    else:
        assert stats.first_token_s is None
    assert stats.seconds == float(len(pieces))
