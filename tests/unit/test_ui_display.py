"""Tests for step S4.4: the UI's text and folding helpers (``geekbot.ui.display``).

No Streamlit and no server: these are plain functions over the dicts the server sends. ``TurnView`` is
the one with real logic (a stream of events adds up to what the page shows), so it also gets a property
test.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from geekbot.sse import SseEvent
from geekbot.ui.display import (
    TurnView,
    assistant_message_id,
    document_rows,
    health_rows,
    human_bytes,
    job_outcome,
    one_line,
    passages_key,
    source_header,
    state_counts,
    thread_label,
    turn_details,
)

# --- small formatters ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "shown"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (5 * 1024 * 1024, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
        (5000 * 1024**3, "5000.0 GB"),  # no larger unit: it stays in GB
    ],
)
def test_human_bytes(size: int, shown: str) -> None:
    assert human_bytes(size) == shown


def test_one_line_flattens_white_space_and_cuts_with_dots() -> None:
    assert one_line("a\n\n  b\tc") == "a b c"
    assert one_line("x" * 10, limit=10) == "x" * 10
    assert one_line("x" * 11, limit=10) == "x" * 7 + "..."
    assert one_line("") == ""


def test_one_line_with_a_tiny_limit_never_raises() -> None:
    assert one_line("abcdef", limit=2) == "..."
    assert one_line("abcdef", limit=0) == "..."


# --- the knowledge base ----------------------------------------------------------------------

DOCUMENT = {
    "doc_id": "d1",
    "path": str(Path("notes") / "zebras.txt"),  # the path of this OS: the UI shows its file name
    "title": "Zebras",
    "kind": "text",
    "state": "ready",
    "error": None,
    "updated_at": "2026-10-20T10:00:00",
}


def test_document_rows_are_plain_strings_with_the_file_name_only() -> None:
    failed = {**DOCUMENT, "doc_id": "d2", "state": "failed", "error": "bad\nthing  happened"}
    rows = document_rows([DOCUMENT, failed])
    assert rows[0] == {
        "Title": "Zebras",
        "Type": "text",
        "State": "ready",
        "File": "zebras.txt",
        "Updated": "2026-10-20T10:00:00",
        "Problem": "",
    }
    assert rows[1]["Problem"] == "bad thing happened"
    assert all(isinstance(value, str) for row in rows for value in row.values())


def test_a_title_that_looks_like_markdown_is_passed_through_untouched() -> None:
    nasty = {**DOCUMENT, "title": "![x](http://evil.example/p.png) **bold**"}
    assert document_rows([nasty])[0]["Title"] == "![x](http://evil.example/p.png) **bold**"


def test_state_counts() -> None:
    assert state_counts([]) == "no documents"
    documents = [DOCUMENT, {**DOCUMENT, "state": "failed"}, DOCUMENT]
    assert state_counts(documents) == "2 ready, 1 failed"


def test_health_rows_say_what_the_server_reports() -> None:
    rows = health_rows(
        {
            "status": "ok",
            "documents": {"ready": 3, "failed": 1},
            "chunks": 40,
            "vectors": 40,
            "embedder": "nomic-embed-text",
            "jobs_active": 0,
            "chat_model": "gemma3:4b",
            "chat_ready": False,
            "answering": True,
        }
    )
    table = {row["Setting"]: row["Value"] for row in rows}
    assert table["Chat model"].startswith("gemma3:4b: not loaded yet")
    assert table["Answering a question now"] == "yes"
    assert table["Documents"] == "3 ready, 1 failed"
    assert table["Chunks"] == "40"
    assert all(set(row) == {"Setting", "Value"} for row in rows)


def test_health_rows_survive_a_missing_field() -> None:
    table = {row["Setting"]: row["Value"] for row in health_rows({})}
    assert table["Documents"] == "none"
    assert table["Embedder"] == "?"


def test_job_outcome_of_a_good_ingest() -> None:
    job = {
        "state": "done",
        "result": {"summary": "1 ingested, 0 unchanged, 0 failed, 0 skipped", "failed": False},
    }
    assert job_outcome(job) == (True, "1 ingested, 0 unchanged, 0 failed, 0 skipped")


def test_job_outcome_names_the_files_that_failed_and_the_ones_refused() -> None:
    job = {
        "state": "done",
        "result": {
            "summary": "1 ingested, 0 unchanged, 1 failed, 0 skipped",
            "failed": True,
            "refused_count": 2,
            "files": [
                {"path": str(Path("a") / "ok.txt"), "status": "ingested", "error": None},
                {
                    "path": str(Path("a") / "bad.pdf"),
                    "status": "failed",
                    "error": "not a PDF\nat all",
                },
            ],
        },
    }
    ok, message = job_outcome(job)
    assert ok is False
    assert "bad.pdf: not a PDF at all" in message
    assert "ok.txt" not in message
    assert "2 file(s) refused" in message


def test_job_outcome_of_a_job_that_raised() -> None:
    assert job_outcome({"state": "failed", "error": "disk full"}) == (False, "failed: disk full")
    assert job_outcome({"state": "failed", "error": None})[1].startswith("failed: ")


def test_job_outcome_of_a_delete_job_is_just_done() -> None:
    assert job_outcome({"state": "done", "result": {"doc_id": "d1", "removed": 3}}) == (
        True,
        "done",
    )
    assert job_outcome({"state": "done", "result": None}) == (True, "done")


# --- threads and sources ---------------------------------------------------------------------


def test_thread_label_uses_the_preview_then_the_id_and_marks_csv() -> None:
    assert thread_label({"thread_id": "t-1", "preview": "Where do zebras live?"}) == (
        "Where do zebras live?"
    )
    assert thread_label({"thread_id": "t-1", "preview": ""}) == "t-1"
    assert (
        thread_label({"thread_id": "t-2", "preview": "Totals?", "mode": "csv"}) == "Totals? [csv]"
    )
    long_label = thread_label({"thread_id": "t-3", "preview": "w" * 200})
    assert len(long_label) <= 48


def test_source_header() -> None:
    plain = {"n": 1, "title": "Handbook", "section": "", "page": None}
    assert source_header(plain) == "[1] Handbook"
    full = {"n": 2, "title": "Handbook", "section": "Retention", "page": 3}
    assert source_header(full) == "[2] Handbook · Retention · page 3"
    assert source_header(full, cited=True) == "[2] Handbook · Retention · page 3 (cited)"
    page_zero = {"n": 3, "title": "Slides", "section": "", "page": 0}
    assert source_header(page_zero) == "[3] Slides · page 0"  # page 0 is a page, not "no page"


def test_ids_the_page_uses_to_find_a_saved_answers_passages() -> None:
    assert assistant_message_id(1) == "t1-assistant"
    assert assistant_message_id(12) == "t12-assistant"
    assert passages_key("t-abc", "t1-assistant") == "t-abc/t1-assistant"


def test_turn_details_lists_what_was_changed_or_doubtful() -> None:
    lines = turn_details(
        {
            "model": "gemma3:4b",
            "generation": {"first_token_seconds": 0.4, "seconds": 1.25},
            "retrieval_ms": {"total": 312.0},
            "searched_as": "How long are logs kept?",
            "dropped_citations": [7],
            "citation_checks": [
                {"n": 1, "weak": False, "sentence": "fine"},
                {"n": 2, "weak": True, "sentence": "A shaky sentence."},
            ],
            "removed_markup": ["image", "link"],
            "labels": ["KB", "USER"],
            "tainted": True,
        }
    )
    text = "\n".join(lines)
    assert "model: gemma3:4b" in text
    assert "first token after 0.40 s" in text and "1.25 s in the model" in text
    assert "search 312 ms" in text
    assert "searched for: How long are logs kept?" in text
    assert "[7]" in text
    assert "citation [2] may not support: A shaky sentence." in text
    assert "citation [1]" not in text
    assert "removed from the answer: image, link" in text
    assert "built from: KB + USER (includes text from outside the chat)" in text


def test_turn_details_of_a_bare_turn_is_just_the_model() -> None:
    assert turn_details({"model": "m"}) == ["model: m"]


# --- the stream ------------------------------------------------------------------------------


def feed(view: TurnView, *events: tuple[str, object]) -> TurnView:
    for name, data in events:
        view.feed(SseEvent(name, data))
    return view


def test_pieces_add_up_and_the_done_event_has_the_last_word() -> None:
    view = feed(
        TurnView(),
        ("meta", {"thread_id": "t-1", "turn": 1}),
        ("text", {"text": "Ninety "}),
        ("text", {"text": "days [1] [7]."}),
    )
    assert view.text == "Ninety days [1] [7]."
    assert view.answer == "Ninety days [1] [7]."
    assert not view.finished
    feed(view, ("done", {"answer": "Ninety days [1].", "turn": 1}))
    assert view.finished
    assert view.answer == "Ninety days [1]."  # the server's checked answer, not the pieces
    assert view.meta == {"thread_id": "t-1", "turn": 1}


def test_an_error_discards_what_was_shown() -> None:
    """The S4.2 contract: the server forgets a failed turn, so the page must show nothing of it."""
    view = feed(
        TurnView(),
        ("text", {"text": "half an ans"}),
        ("thinking", {"text": "hmm"}),
        ("error", {"message": "Ollama went away"}),
    )
    assert view.error == "Ollama went away"
    assert (view.text, view.thinking, view.answer) == ("", "", "")
    assert view.finished
    assert view.done is None


def test_a_status_shows_until_the_first_piece() -> None:
    view = feed(TurnView(), ("status", {"stage": "loading_model", "model": "gemma3:4b"}))
    assert "gemma3:4b" in view.status
    feed(view, ("text", {"text": "Hi"}))
    assert view.status == ""


def test_thinking_is_kept_apart_from_the_answer() -> None:
    view = feed(TurnView(), ("thinking", {"text": "step 1. "}), ("thinking", {"text": "step 2."}))
    assert view.thinking == "step 1. step 2."
    assert view.text == ""


@pytest.mark.parametrize(
    "event",
    [
        SseEvent("text", "not a dict"),
        SseEvent("text", None),
        SseEvent("text", {"unexpected": 1}),
        SseEvent("from-the-future", {"x": 1}),
        SseEvent("error", {}),
    ],
)
def test_a_malformed_or_unknown_event_never_raises(event: SseEvent) -> None:
    view = TurnView()
    view.feed(event)
    assert view.text == ""
    assert view.done is None


def test_an_error_event_with_no_message_still_says_something() -> None:
    view = feed(TurnView(), ("error", {}))
    assert view.error


@given(st.lists(st.text()))
def test_any_pieces_add_up_to_their_concatenation(pieces: list[str]) -> None:
    view = TurnView()
    for piece in pieces:
        view.feed(SseEvent("text", {"text": piece}))
    assert view.text == "".join(pieces)
    assert view.answer == view.text
