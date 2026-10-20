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
    csv_upload_lines,
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


# --- CSV conversations (step C5) -------------------------------------------------------------


def test_a_csv_turn_shows_the_file_the_verdict_and_the_time_of_the_whole_question() -> None:
    lines = turn_details(
        {
            "mode": "csv",
            "model": "qwen2.5-coder:3b",
            "csv_file": "sales.csv",
            "csv_status": "abstained",
            "generation": {"seconds": 12.3, "load_seconds": 2.7},
            "searched_as": "What is the total revenue of the region that sells the most?",
            "labels": ["TOOL", "USER"],
            "tainted": True,
        }
    )
    assert lines == [
        "model: qwen2.5-coder:3b",
        "file: sales.csv",
        "verdict: abstained",
        "time: 12.30 s, model load 2.70 s",
        "asked as: What is the total revenue of the region that sells the most?",
        "built from: TOOL + USER (includes text from outside the chat)",
    ]


def test_a_csv_turn_without_a_model_load_does_not_mention_one() -> None:
    lines = turn_details({"mode": "csv", "model": "m", "generation": {"seconds": 1.0}})
    assert lines == ["model: m", "time: 1.00 s"]


def test_a_chat_turn_is_still_described_as_before_even_with_csv_fields_absent() -> None:
    lines = turn_details({"mode": "chat", "model": "m", "generation": {"seconds": 1.0}})
    assert lines == ["model: m", "time: 1.00 s in the model"]


def test_the_upload_summary_says_rows_columns_and_notes_in_plain_lines() -> None:
    info = {
        "file": "sales.csv",
        "rows": 12345,
        "column_count": 3,
        "columns": ["region", "revenue", "month"],
        "notes": ["2 malformed row(s) were skipped during parsing"],
        "existed": True,
        "memory_bytes": 3 * 1024 * 1024,
    }
    assert csv_upload_lines(info) == [
        "sales.csv: 12,345 rows, 3 columns",
        "columns: region, revenue, month",
        "2 malformed row(s) were skipped during parsing",
        "the server already had this exact file",
        "held in memory: 3.0 MB",
    ]


def test_the_upload_summary_cuts_long_names_and_counts_the_columns_it_does_not_list() -> None:
    info = {
        "file": "wide.csv",
        "rows": 1,
        "column_count": 60,
        "columns": ["c" * 100, "second"],
        "notes": ["n" * 500],
        "memory_bytes": 10,
    }
    lines = csv_upload_lines(info)
    assert lines[1] == "columns: " + "c" * 37 + "..., second and 58 more"
    assert len(lines[2]) == 120 and lines[2].endswith("...")
    assert lines[-1] == "held in memory: 10 B"


def test_the_upload_summary_survives_a_sparse_answer() -> None:
    assert csv_upload_lines({}) == ["?: 0 rows, 0 columns"]


def test_the_waits_of_a_csv_turn_have_their_own_status_lines() -> None:
    model = feed(TurnView(), ("status", {"stage": "loading_model", "model": "qwen2.5-coder:3b"}))
    assert "qwen2.5-coder:3b" in model.status
    file = feed(TurnView(), ("status", {"stage": "loading_file", "file": "sales.csv"}))
    assert "sales.csv" in file.status and "again" in file.status
    asking = feed(TurnView(), ("status", {"stage": "asking", "file": "sales.csv"}))
    assert asking.status.startswith("Asking sales.csv")


def test_the_asking_line_replaces_the_swap_line_and_goes_when_the_answer_is_there() -> None:
    view = feed(
        TurnView(),
        ("status", {"stage": "loading_model", "model": "qwen2.5-coder:3b"}),
        ("status", {"stage": "asking", "file": "sales.csv"}),
    )
    assert view.status.startswith("Asking")  # the swap is over: its line must not stay
    view = feed(view, ("done", {"answer": "North leads."}))
    assert (view.status, view.answer) == ("", "North leads.")


def test_a_file_name_in_a_status_line_is_cut_to_one_short_line() -> None:
    view = feed(TurnView(), ("status", {"stage": "asking", "file": "x" * 200 + "\n.csv"}))
    assert "\n" not in view.status and len(view.status) < 120


def test_the_health_table_names_the_csv_model_when_the_server_has_one() -> None:
    health = {"csv_model": "qwen2.5-coder:3b", "csv_ready": False, "csv_sessions": 2}
    table = {row["Setting"]: row["Value"] for row in health_rows(health)}
    assert table["CSV model"].startswith("qwen2.5-coder:3b: not loaded")
    assert table["CSV tables in memory"] == "2"
    loaded = {row["Setting"]: row["Value"] for row in health_rows({**health, "csv_ready": True})}
    assert loaded["CSV model"] == "qwen2.5-coder:3b: loaded"
    assert "CSV model" not in {row["Setting"] for row in health_rows({})}


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
