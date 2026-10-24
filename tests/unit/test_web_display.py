"""Tests for step S5.3: how the web search's question and results are turned into frames and lines.

``TurnTranslator`` (server side) turns a turn that stopped at the egress gate into an ``approval``
frame; ``TurnView`` and the ``display`` helpers (client side) fold that frame and the turn's decisions
into plain text. Neither touches the network or a model."""

from types import SimpleNamespace

from geekbot.agent.nodes import APPROVAL_KIND
from geekbot.agent.runner import INTERRUPT_KEY, interrupt_question
from geekbot.server.turns import TurnTranslator
from geekbot.sse import SseEvent, parse_text
from geekbot.ui.display import (
    SENTENCE_CHARS,
    STEP_CAPTIONS,
    TurnView,
    approval_lines,
    egress_lines,
    source_header,
    turn_details,
)

QUESTION = {
    "kind": APPROVAL_KIND,
    "tool": "web_search",
    "question": "And logs?",
    "query": "How long are audit logs kept?",
    "labels": ["KB", "USER"],
    "reason": "the arguments were built from text the user did not write",
}


def interrupt(value: object) -> object:
    return SimpleNamespace(value=value)


# --- the server: the stream ---------------------------------------------------------------------


def test_a_turn_that_stopped_becomes_one_approval_frame_and_no_final_interrupt_key() -> None:
    translator = TurnTranslator()
    frames = translator.feed("values", {"answer": "", INTERRUPT_KEY: (interrupt(QUESTION),)})
    (frame,) = frames
    (event,) = parse_text(frame)
    assert event.event == "approval"
    assert event.data == {
        "tool": "web_search",
        "question": "And logs?",
        "query": "How long are audit logs kept?",
        "labels": ["KB", "USER"],
        "reason": QUESTION["reason"],
    }
    assert translator.approval == QUESTION
    assert translator.final == {"answer": ""}  # the interrupt is not part of the state


def test_a_finished_turn_has_no_approval() -> None:
    translator = TurnTranslator()
    assert translator.feed("values", {"answer": "done"}) == []
    assert translator.approval is None and translator.final == {"answer": "done"}


def test_a_question_of_another_kind_is_not_ours_and_stops_nothing() -> None:
    translator = TurnTranslator()
    assert translator.feed("values", {INTERRUPT_KEY: (interrupt({"kind": "other"}),)}) == []
    assert translator.feed("values", {INTERRUPT_KEY: (interrupt("text"),)}) == []
    assert translator.approval is None


def test_interrupt_question_reads_only_this_programs_own_question() -> None:
    assert interrupt_question({INTERRUPT_KEY: (interrupt(QUESTION),)}) == QUESTION
    assert interrupt_question({}) is None
    assert interrupt_question({INTERRUPT_KEY: ()}) is None
    assert interrupt_question({INTERRUPT_KEY: (interrupt({"kind": "x"}),)}) is None


def test_the_web_step_is_a_status_frame_only_for_a_client_that_asked_for_steps() -> None:
    event = {"event": "step", "name": "web"}
    (frame,) = TurnTranslator(steps=True).feed("custom", event)
    assert parse_text(frame)[0].data == {"stage": "web"}
    assert TurnTranslator().feed("custom", event) == []


# --- the client ---------------------------------------------------------------------------------


def test_the_view_keeps_the_question_and_does_not_count_the_turn_as_finished() -> None:
    view = TurnView()
    view.feed(SseEvent("text", {"text": "partial"}))
    view.feed(SseEvent("approval", QUESTION))
    assert view.approval == QUESTION
    assert view.finished is False and view.done is None and view.status == ""


def test_the_web_step_has_a_caption() -> None:
    assert STEP_CAPTIONS["web"] == "Searching the web"
    view = TurnView()
    view.feed(SseEvent("status", {"stage": "web"}))
    assert view.status == "Searching the web"


def test_the_question_is_shown_as_what_will_be_sent_and_why() -> None:
    lines = approval_lines(QUESTION)
    assert lines[0] == "You asked: And logs?"
    assert lines[1] == (
        "The web search would send this text off this machine: How long are audit logs kept?"
    )
    assert "built from: KB, USER" in lines[2]
    assert len(approval_lines({"query": "q"})) == 2  # no question text: that line is left out


def test_a_web_source_says_so_and_a_document_does_not() -> None:
    web = {"n": 3, "title": "A page", "section": "pages.example", "page": None, "origin": "WEB"}
    assert source_header(web, cited=True) == "[3] web · A page · pages.example (cited)"
    doc = {"n": 1, "title": "Handbook", "section": "Retention", "page": 2}
    assert source_header(doc) == "[1] Handbook · Retention · page 2"
    assert source_header({**doc, "origin": "KB"}) == "[1] Handbook · Retention · page 2"


def test_the_details_box_lists_every_decision_and_a_failed_search() -> None:
    done = {
        "model": "m",
        "egress": [
            {
                "tool": "web_search",
                "query": "q1",
                "decision": "allowed",
                "tainted": False,
                "reason": "own",
            },
            {
                "tool": "web_search",
                "query": "q2",
                "decision": "denied",
                "tainted": True,
                "reason": "no",
            },
            {
                "tool": "web_search",
                "query": "q3",
                "decision": "mystery",
                "tainted": True,
                "reason": "?",
            },
        ],
        "web_error": "WebSearchError: the search service answered HTTP 500",
    }
    lines = egress_lines(done)
    assert lines[0] == "searched the web for: q1 (own)"
    assert lines[1].startswith("did not search the web (you said no or could not be asked) for: q2")
    assert lines[2].startswith(
        "web search:: q3"
    )  # a decision this client does not know is still shown
    assert (
        lines[3]
        == "the web search did not work: WebSearchError: the search service answered HTTP 500"
    )
    details = turn_details(done)
    assert all(line in details for line in lines)


def test_a_long_query_is_cut_to_one_line_in_the_details() -> None:
    done = {"egress": [{"query": "word " * 200, "decision": "approved", "reason": "x"}]}
    (line,) = egress_lines(done)
    assert "\n" not in line and len(line) < SENTENCE_CHARS * 2 + 80


def test_a_turn_without_the_web_adds_no_lines() -> None:
    assert egress_lines({}) == [] and egress_lines({"egress": [], "web_error": None}) == []
