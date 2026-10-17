"""Tests for step S3.3: printing a streamed answer, and the readings line with the first token.

``consume_stream`` and ``format_after_stream`` are pure: they take the graph's events and the final
state and call two small writers, so no terminal, graph or model is needed here.
"""

from typing import Any

import pytest

from geekbot.agent.render import (
    StreamedAnswer,
    consume_stream,
    format_after_stream,
    format_answer,
)

MODEL = "fake-model:1b"


def final_state(**changes: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": "How long are chats kept?",
        "answer": "Ninety days [1].",
        "raw_answer": "Ninety days [1].",
        "used": 1,
        "cited": [1],
        "dropped_citations": [],
        "answer_labels": ["KB", "USER"],
        "sources": [
            {
                "n": 1,
                "chunk_id": "c1",
                "title": "Handbook",
                "path": "/docs/handbook.md",
                "page": None,
                "section": "Retention",
                "text": "Chats are kept for 90 days.",
                "score": 1.0,
            }
        ],
        "retrieval_ms": {"total": 12.5},
        "generation": {"seconds": 1.25, "first_token_seconds": 0.31},
    }
    state.update(changes)
    return state


def text(piece: str) -> tuple[str, dict[str, str]]:
    return ("custom", {"event": "text", "text": piece})


def thought(piece: str) -> tuple[str, dict[str, str]]:
    return ("custom", {"event": "thinking", "text": piece})


def values(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return ("values", state)


class Sink:
    """Collects what ``consume_stream`` writes, in order, separately for the two channels."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.err: list[str] = []

    def run(self, events: list[Any], *, show_thinking: bool = False) -> StreamedAnswer:
        return consume_stream(
            events, write=self.out.append, note=self.err.append, show_thinking=show_thinking
        )


# --- consume_stream --------------------------------------------------------------------------


def test_answer_pieces_are_written_as_they_arrive_and_the_last_state_is_the_final_one() -> None:
    sink = Sink()
    final = final_state()
    streamed = sink.run(
        [values({"question": "q"}), text("Ninety "), text("days [1]."), values(final)]
    )
    assert sink.out == ["Ninety ", "days [1]."]
    assert streamed.streamed == "Ninety days [1]."
    assert streamed.final == final


def test_the_answer_does_not_start_with_the_whitespace_the_model_began_with() -> None:
    sink = Sink()
    sink.run([text("\n\n"), text("  Ninety"), text(" days"), values(final_state())])
    assert sink.out == ["Ninety", " days"]  # later whitespace is kept: it separates words


def test_without_show_thinking_the_reasoning_is_reduced_to_one_note() -> None:
    sink = Sink()
    sink.run([thought("Let me "), thought("check."), text("Ninety days."), values(final_state())])
    assert sink.err == ["thinking...\n"]
    assert sink.out == ["Ninety days."]


def test_with_show_thinking_the_reasoning_streams_to_the_note_channel_and_the_line_is_closed() -> (
    None
):
    sink = Sink()
    sink.run(
        [thought("Let me "), thought("check."), text("Ninety days."), values(final_state())],
        show_thinking=True,
    )
    assert "".join(sink.err) == "thinking: Let me check.\n"
    assert sink.out == ["Ninety days."]


def test_events_it_does_not_know_are_ignored() -> None:
    sink = Sink()
    streamed = sink.run(
        [
            ("updates", {"generate": {}}),
            ("custom", "not a mapping"),
            ("custom", {"event": "mystery", "text": "x"}),
            text("Hi"),
            values(final_state()),
        ]
    )
    assert sink.out == ["Hi"]
    assert streamed.streamed == "Hi"


def test_a_stream_without_a_final_state_is_an_error() -> None:
    with pytest.raises(ValueError, match="without a final state"):
        Sink().run([text("Hi")])


def test_nothing_streamed_when_the_model_was_never_called() -> None:
    sink = Sink()
    streamed = sink.run([values(final_state(answer="I could not find anything about that."))])
    assert sink.out == []
    assert streamed.streamed == ""


# --- format_after_stream ---------------------------------------------------------------------


def test_after_a_stream_the_answer_is_not_printed_again() -> None:
    streamed = StreamedAnswer(final=final_state(), streamed="Ninety days [1].")
    shown = format_after_stream(streamed, model=MODEL)
    assert "Ninety days" not in shown.split("Sources cited:")[0]
    assert "Sources cited:" in shown
    assert "[1] Handbook" in shown
    assert "labels: KB, USER (tainted)" in shown
    assert "corrected" not in shown


def test_when_the_citation_check_changed_the_answer_the_corrected_text_is_printed() -> None:
    final = final_state(
        answer="Ninety days [1].",
        raw_answer="Ninety days [1] and [7].",
        dropped_citations=[7],
    )
    shown = format_after_stream(
        StreamedAnswer(final=final, streamed="Ninety days [1] and [7]."), model=MODEL
    )
    assert shown.startswith("corrected answer (the citation check removed a mark")
    assert "\nNinety days [1].\n" in shown
    assert "removed from the answer (no such source in the prompt): [7]" in shown


def test_when_the_model_was_never_called_the_whole_answer_is_printed() -> None:
    final = final_state(
        answer="I could not find anything about that in the documents.",
        used=0,
        cited=[],
        answer_labels=[],
        generation={},
    )
    shown = format_after_stream(StreamedAnswer(final=final, streamed=""), model=MODEL)
    assert shown.startswith("I could not find anything about that in the documents.")


# --- format_answer ---------------------------------------------------------------------------


def test_the_readings_line_shows_the_first_token_and_the_prompt_the_model_read() -> None:
    final = final_state(
        generation={
            "seconds": 0.98,
            "first_token_seconds": 0.31,
            "load_seconds": 0.01,
            "prompt_tokens": 812,
            "prompt_eval_seconds": 0.12,
            "output_tokens": 38,
            "tokens_per_second": 131.0,
            "thinking_chars": 240,
        }
    )
    assert format_answer(final, model=MODEL).splitlines()[-1] == (
        f"{MODEL}: 0.98 s, first token 0.31 s, model load 0.01 s, prompt 812 tokens "
        "(read in 0.12 s), 38 tokens out, 131.0 tokens/s, thinking 240 characters"
    )


def test_a_reading_that_does_not_exist_is_not_printed() -> None:
    final = final_state(generation={"seconds": 2.0})
    assert format_answer(final, model=MODEL).splitlines()[-1] == f"{MODEL}: 2.00 s"


def test_the_answer_can_be_left_out_because_it_was_already_streamed() -> None:
    shown = format_answer(final_state(), model=MODEL, include_answer=False)
    assert "Ninety days" not in shown.split("Sources cited:")[0]
    assert shown.startswith("Sources cited:")


def csv_state(**changes: Any) -> dict[str, Any]:
    return final_state(
        mode="csv",
        sources=[],
        used=0,
        cited=[],
        answer="North has the highest total revenue, 520.",
        raw_answer="",
        answer_labels=["TOOL", "USER"],
        retrieval_ms={},
        generation={"seconds": 2.5, "load_seconds": 0.02},
        csv_status="answered",
        thread_taint=["TOOL", "USER"],
        history=[{}, {}],
        turn=1,
        **changes,
    )


def test_a_csv_answer_shows_its_status_labels_and_the_thread_in_csv_mode() -> None:
    shown = format_answer(csv_state(), model="qwen-fake:3b", thread="t1")
    assert shown.startswith("North has the highest total revenue, 520.")
    assert "csv status: answered" in shown
    assert "labels: TOOL, USER (tainted)" in shown
    assert "qwen-fake:3b: 2.50 s, model load 0.02 s" in shown
    assert shown.splitlines()[-1] == (
        "thread t1 (csv): turn 1, 2 messages remembered, taint TOOL, USER (tainted)"
    )
    assert "Sources" not in shown


def test_a_csv_follow_up_says_asked_as_not_searched_as() -> None:
    state = csv_state(
        question="And by month?",
        standalone_question="What is the revenue by month?",
        condense_seconds=0.4,
    )
    shown = format_answer(state, model="m")
    assert 'asked as: "What is the revenue by month?"' in shown
    assert "searched as" not in shown
    failed = format_answer(csv_state(condense_error="ValueError: x"), model="m")
    assert "rewrite failed, asked with your own words: ValueError: x" in failed


def test_a_chat_follow_up_still_says_searched_as() -> None:
    state = final_state(question="And logs?", standalone_question="How long are audit logs kept?")
    assert 'searched as: "How long are audit logs kept?"' in format_answer(state, model="m")


def test_the_reasoning_is_printed_only_on_request() -> None:
    final = final_state(thinking="Source 1 says 90.\nSo ninety.")
    assert "thinking (" not in format_answer(final, model=MODEL)
    shown = format_answer(final, model=MODEL, show_thinking=True)
    assert "thinking (28 characters):\n  Source 1 says 90.\n  So ninety.\n" in shown
    answer_at = shown.index("Ninety days")
    assert answer_at < shown.index("thinking (") < shown.index("Sources cited:")
