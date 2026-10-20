"""Tests for step S4.2: the translation of graph events into SSE frames, and of errors into sentences.

No server, no graph, no model: ``TurnTranslator`` is fed the pairs LangGraph would produce.
"""

from typing import Any

import pytest

from geekbot.agent.mode import ModeError
from geekbot.llm.chat import ChatError
from geekbot.llm.models import OllamaError
from geekbot.retrieval.rerank import RerankError
from geekbot.server.turns import TurnTranslator, describe_error
from geekbot.sse import parse_text


def pieces(frames: list[str]) -> list[tuple[str, Any]]:
    """The frames as (event, data) pairs."""
    return [(event.event, event.data) for event in parse_text("".join(frames))]


def text(piece: str) -> dict[str, str]:
    return {"event": "text", "text": piece}


def thinking(piece: str) -> dict[str, str]:
    return {"event": "thinking", "text": piece}


# --- the answer pieces -----------------------------------------------------------------------


def test_a_text_event_becomes_a_text_frame() -> None:
    translator = TurnTranslator()
    assert pieces(translator.feed("custom", text("Ninety days"))) == [
        ("text", {"text": "Ninety days"})
    ]


def test_the_first_piece_loses_its_leading_whitespace_and_later_pieces_keep_theirs() -> None:
    translator = TurnTranslator()
    frames = [
        *translator.feed("custom", text("\n  Ninety")),
        *translator.feed("custom", text(" days")),
        *translator.feed("custom", text(" [1].\n")),
    ]
    assert pieces(frames) == [
        ("text", {"text": "Ninety"}),
        ("text", {"text": " days"}),
        ("text", {"text": " [1].\n"}),
    ]


def test_pieces_that_are_only_whitespace_at_the_start_send_nothing() -> None:
    translator = TurnTranslator()
    assert translator.feed("custom", text("  ")) == []
    assert translator.feed("custom", text("\n")) == []
    assert pieces(translator.feed("custom", text(" Hi"))) == [("text", {"text": "Hi"})]
    assert pieces(translator.feed("custom", text(" "))) == [("text", {"text": " "})]


def test_an_empty_piece_sends_nothing() -> None:
    translator = TurnTranslator()
    assert translator.feed("custom", text("")) == []
    assert translator.feed("custom", thinking("")) == []


def test_thinking_pieces_are_their_own_event_and_do_not_start_the_answer() -> None:
    translator = TurnTranslator()
    assert pieces(translator.feed("custom", thinking("  hmm"))) == [("thinking", {"text": "  hmm"})]
    # the answer's first piece is still the first piece of the answer
    assert pieces(translator.feed("custom", text("  A"))) == [("text", {"text": "A"})]


def test_the_pieces_are_passed_on_without_looking_inside() -> None:
    """The text was cleaned by the graph (G3); this layer adds nothing and removes nothing."""
    translator = TurnTranslator()
    hostile = f"see ![x](http://evil.example/?q=1) and <b>bold</b> {chr(0x2028)} end"
    assert pieces(translator.feed("custom", text(hostile))) == [("text", {"text": hostile})]


# --- the state ---------------------------------------------------------------------------------


def test_the_last_state_is_kept_as_the_final_one() -> None:
    translator = TurnTranslator()
    assert translator.final is None
    assert translator.feed("values", {"question": "q"}) == []
    assert translator.feed("values", {"question": "q", "answer": "a"}) == []
    assert translator.final == {"question": "q", "answer": "a"}


def test_the_kept_state_is_a_copy() -> None:
    translator = TurnTranslator()
    state: dict[str, Any] = {"answer": "a"}
    translator.feed("values", state)
    state["answer"] = "changed later"
    assert translator.final == {"answer": "a"}


# --- what is ignored -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("updates", {"retrieve": {}}),  # a stream mode this layer does not use
        ("custom", "not a mapping"),
        ("custom", {"event": "progress", "text": "x"}),  # an event kind it does not know
        ("custom", {"text": "no event kind"}),
        ("custom", None),
    ],
)
def test_events_it_does_not_know_are_ignored(kind: str, payload: Any) -> None:
    translator = TurnTranslator()
    assert translator.feed(kind, payload) == []
    assert translator.final is None


# --- describe_error ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ChatError("the chat model 'm' failed (ConnectionError: down)"),
        OllamaError("Ollama POST /api/embed failed: boom"),
        RerankError("the reranker failed: x"),
        ModeError("this thread is a csv conversation and cannot become a chat one"),
    ],
)
def test_an_expected_failure_is_reported_in_its_own_words(error: Exception) -> None:
    message, expected = describe_error(error)
    assert message == str(error)
    assert expected is True


def test_a_message_with_line_breaks_is_put_on_one_line() -> None:
    message, _ = describe_error(ChatError("first line\n  second   line\n"))
    assert message == "first line second line"


def test_an_unexpected_failure_is_reported_by_type_only() -> None:
    message, expected = describe_error(ValueError("C:\\secret\\path leaked in a library message"))
    assert expected is False
    assert "ValueError" in message
    assert "secret" not in message and "leaked" not in message


def test_a_model_without_thinking_gets_the_hint_the_command_line_gives() -> None:
    error = ChatError('"gemma3:4b" does not support thinking (status code: 400)')
    plain, _ = describe_error(error, think=False)
    hinted, _ = describe_error(error, think=True)
    assert "choose a chat model with a thinking mode" not in plain
    assert hinted.startswith(plain)
    assert "Ask without thinking" in hinted


def test_the_hint_is_only_for_the_thinking_refusal() -> None:
    message, _ = describe_error(ChatError("the model is not there"), think=True)
    assert "Ask without thinking" not in message
