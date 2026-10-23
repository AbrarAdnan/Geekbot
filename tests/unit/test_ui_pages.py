"""Tests for step S4.4: the pages, run headless with Streamlit's own ``AppTest``.

Each test runs one page's ``render`` in a small script with a **fake client** (a plain object with the
methods the page calls), so no server, model or browser is involved. What is checked is what the page
does with what the client returns: what it shows, what it asks the client for, and what it never
shows (an answer that failed, a passage as Markdown). The client's contract with the real server is
``test_ui_client.py``; the folding of a stream into an answer is ``test_ui_display.py``.

This is the least proven part of the step: ``AppTest`` was never run in this project before, and
nothing here was run when it was written. A failure in this file is as likely to be a wrong guess
about ``AppTest`` as a wrong page; the failing line says which.

``conftest.py`` leaves this file out when Streamlit is not installed.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from geekbot.sse import SseEvent
from geekbot.ui.client import ApiError
from geekbot.ui.views import evals

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

SOURCE = {
    "n": 1,
    "chunk_id": "c1",
    "title": "Handbook",
    "path": "/docs/handbook.md",
    "page": None,
    "section": "Retention",
}
DONE = {
    "thread_id": "t-new",
    "turn": 1,
    "mode": "chat",
    "model": "fake:1b",
    "answer": "Ninety days [1].",
    "abstained": False,
    "cited": [1],
    "dropped_citations": [],
    "sources": [{**SOURCE, "score": 0.9, "text": "Chat history is kept for 90 days."}],
    "labels": ["KB", "USER"],
    "thread_taint": ["KB", "USER"],
    "tainted": True,
    "searched_as": None,
    "rewrite_error": None,
    "rewrite_seconds": 0.0,
    "citation_checks": [],
    "citation_check_error": None,
    "removed_markup": [],
    "retrieval_ms": {"total": 12.0},
    "generation": {"first_token_seconds": 0.3, "seconds": 0.9},
}
ANSWER_EVENTS = [
    SseEvent("meta", {"thread_id": "t-new", "turn": 1, "mode": "chat", "model": "fake:1b"}),
    SseEvent("text", {"text": "Ninety "}),
    SseEvent("text", {"text": "days [1]."}),
    SseEvent("done", DONE),
]
SAVED = {
    "thread_id": "t-old",
    "mode": "chat",
    "csv_file": None,
    "turns": 1,
    "message_count": 2,
    "preview": "How long?",
    "taint": ["KB", "USER"],
    "tainted": True,
    "updated_at": "2026-10-20T10:00:00",
    "messages": [
        {"id": "t1-user", "role": "user", "content": "How long?", "labels": ["USER"]},
        {
            "id": "t1-assistant",
            "role": "assistant",
            "content": "Ninety days [1].",
            "labels": ["KB", "USER"],
            "cited": [1],
            "sources": [SOURCE],
        },
    ],
}
JOB_DONE = {
    "id": "j1",
    "kind": "ingest",
    "state": "done",
    "progress": [],
    "result": {"summary": "1 ingested, 0 unchanged, 0 failed, 0 skipped", "failed": False},
}
# C5: what the server answers to a CSV upload (a column named like an image: it comes from the file),
# the finished turn of a CSV question, and a saved CSV conversation.
HOSTILE_COLUMN = "![x](http://evil.example/?q=1)"
CSV_INFO = {
    "thread_id": "t-new",
    "mode": "csv",
    "file": "sales.csv",
    "stored_as": "0123456789abcdef-sales.csv",
    "sha256": "0" * 64,
    "size_bytes": 35,
    "existed": False,
    "rows": 2,
    "column_count": 2,
    "columns": ["region", HOSTILE_COLUMN],
    "notes": [],
    "memory_bytes": 2048,
}
CSV_REPLY = (
    "North leads with 520.\n\n**How this was computed** (2 rows analyzed):\n```sql\nSELECT 1\n```"
)
CSV_DONE = {
    **DONE,
    "thread_id": "t-csv",
    "mode": "csv",
    "model": "fake-csv:1b",
    "answer": CSV_REPLY,
    "cited": [],
    "sources": [],
    "labels": ["TOOL", "USER"],
    "thread_taint": ["TOOL", "USER"],
    "csv_file": "sales.csv",
    "csv_status": "answered",
    "retrieval_ms": {},
    "generation": {"seconds": 0.5, "load_seconds": 0.0},
}
CSV_EVENTS = [
    SseEvent("meta", {"thread_id": "t-csv", "turn": 2, "mode": "csv", "model": "fake-csv:1b"}),
    SseEvent("status", {"stage": "asking", "file": "sales.csv"}),
    SseEvent("done", CSV_DONE),
]
CSV_SAVED = {
    **SAVED,
    "thread_id": "t-csv",
    "mode": "csv",
    "csv_file": "sales.csv",
    "preview": "Which region sells the most?",
    "taint": ["TOOL", "USER"],
    "messages": [
        {
            "id": "t1-user",
            "role": "user",
            "content": "Which region sells the most?",
            "labels": ["USER"],
        },
        {
            "id": "t1-assistant",
            "role": "assistant",
            "content": CSV_REPLY,
            "labels": ["TOOL", "USER"],
        },
    ],
}


class FakeClient:
    """The methods the pages call, answering from plain attributes a test sets."""

    base_url = "http://127.0.0.1:8000"
    key_origin = "a test"

    def __init__(self) -> None:
        self.documents_: list[dict[str, Any]] = []
        self.saved: dict[str, dict[str, Any]] = {}
        self.events: list[SseEvent] = list(ANSWER_EVENTS)
        self.ask_error: Exception | None = None
        self.documents_ready = 1
        self.asked: list[tuple[str, str, bool]] = []
        self.modes: list[str | None] = []  # the ``mode`` each question was sent with
        self.steps: list[bool] = []  # whether each question asked for step events (S3.6)
        self.ingested: list[list[str]] = []
        self.new_threads: list[str] = []  # the mode of each thread id asked for
        self.csv_uploads: list[tuple[str, str, bytes]] = []
        self.upload_error: Exception | None = None

    def health(self) -> dict[str, Any]:
        documents = {"ready": self.documents_ready} if self.documents_ready else {}
        return {
            "status": "ok",
            "documents": documents,
            "chunks": 3,
            "vectors": 3,
            "embedder": "fake-embedder",
            "jobs_active": 0,
            "chat_model": "fake:1b",
            "chat_ready": True,
            "answering": False,
        }

    def formats(self) -> list[str]:
        return [".md", ".pdf", ".txt"]

    def documents(self) -> list[dict[str, Any]]:
        return list(self.documents_)

    def threads(self, limit: int = 50) -> list[dict[str, Any]]:
        return [{k: v for k, v in t.items() if k != "messages"} for t in self.saved.values()]

    def thread(self, thread_id: str) -> dict[str, Any] | None:
        return self.saved.get(thread_id)

    def new_thread(self, mode: str = "chat") -> str:
        self.new_threads.append(mode)
        return "t-new"

    def upload_csv(self, thread_id: str, filename: str, data: bytes) -> dict[str, Any]:
        if self.upload_error is not None:
            raise self.upload_error
        self.csv_uploads.append((thread_id, filename, data))
        return CSV_INFO

    def delete_thread(self, thread_id: str) -> bool:
        return self.saved.pop(thread_id, None) is not None

    def ask(
        self,
        thread_id: str,
        question: str,
        *,
        think: bool = False,
        mode: str | None = None,
        steps: bool = False,
    ) -> Iterator[SseEvent]:
        self.asked.append((thread_id, question, think))
        self.modes.append(mode)
        self.steps.append(steps)
        if self.ask_error is not None:
            raise self.ask_error
        yield from self.events

    def ingest(self, paths: list[str]) -> dict[str, Any]:
        self.ingested.append(paths)
        return {"id": "j1"}

    def job(self, job_id: str) -> dict[str, Any]:
        return JOB_DONE


def run_page(module: str, fake: FakeClient, **state: Any) -> AppTest:
    """Run ``geekbot.ui.views.<module>.render`` once with ``fake`` and any session state given."""
    script = (
        "import streamlit as st\n"
        f"from geekbot.ui.views import {module}\n"
        f"{module}.render(st.session_state['client'])\n"
    )
    app = AppTest.from_string(script, default_timeout=10)
    app.session_state["client"] = fake
    for key, value in state.items():
        app.session_state[key] = value
    return app.run()


def shown(app: AppTest, kind: str) -> list[str]:
    """The text of every element of one kind (``markdown``, ``text``, ``caption``, ...)."""
    return [str(element.value) for element in getattr(app, kind)]


# --- chat ------------------------------------------------------------------------------------


def test_the_chat_page_starts_blank_with_a_prompt_and_nothing_asked() -> None:
    fake = FakeClient()
    app = run_page("chat", fake)
    assert not app.exception
    assert app.title[0].value == "Chat"
    assert len(app.chat_input) == 1
    assert len(app.chat_message) == 0
    assert fake.asked == []  # a plain run asks the server nothing


def test_a_question_is_asked_once_and_the_answer_shown_with_its_sources() -> None:
    fake = FakeClient()
    app = run_page("chat", fake)
    app.chat_input[0].set_value("How long are chats kept?").run()
    assert not app.exception
    assert fake.asked == [("t-new", "How long are chats kept?", False)]
    assert "Ninety days [1]." in shown(app, "markdown")  # the server's checked answer
    assert any("[1] Handbook · Retention (cited)" in text for text in shown(app, "text"))
    assert any("Chat history is kept for 90 days." in text for text in shown(app, "text"))
    labels = [expander.label for expander in app.expander]
    assert "Sources (1)" in labels and "Details" in labels


def test_a_plain_rerun_after_an_answer_does_not_ask_again() -> None:
    """Streamlit re-runs the script on every interaction; only the chat input may send a question."""
    fake = FakeClient()
    app = run_page("chat", fake)
    app.chat_input[0].set_value("How long?").run()
    app.run()
    app.run()
    assert len(fake.asked) == 1


def test_the_think_toggle_is_sent_with_the_question() -> None:
    fake = FakeClient()
    app = run_page("chat", fake)
    app.toggle[0].set_value(True).run()
    app.chat_input[0].set_value("Why?").run()
    assert fake.asked == [("t-new", "Why?", True)]


def test_a_document_question_asks_for_step_events_and_a_csv_question_does_not() -> None:
    """S3.6: the line under the question names the step ("Searching ...") for documents only; the CSV
    stream has its own progress line and no steps to report."""
    docs = FakeClient()
    run_page("chat", docs).chat_input[0].set_value("How long?").run()
    assert docs.steps == [True]
    sheet = FakeClient()
    sheet.saved["t-csv"] = CSV_SAVED
    sheet.events = CSV_EVENTS
    sheet.documents_ready = 0
    app = run_page("chat", sheet, geekbot_thread="t-csv")
    app.chat_input[0].set_value("And in March?").run()
    assert sheet.steps == [False]


def test_a_failed_answer_shows_the_error_and_none_of_the_pieces() -> None:
    fake = FakeClient()
    fake.events = [
        SseEvent("meta", {"thread_id": "t-new", "turn": 1}),
        SseEvent("text", {"text": "half an answer"}),
        SseEvent("error", {"message": "ConnectionError: down"}),
    ]
    app = run_page("chat", fake)
    app.chat_input[0].set_value("Q?").run()
    assert not app.exception
    assert [error.value for error in app.error] == ["ConnectionError: down"]
    assert not any("half an answer" in text for text in shown(app, "markdown"))


def test_a_refused_question_says_why() -> None:
    fake = FakeClient()
    fake.ask_error = ApiError(409, "a question is already being answered")
    app = run_page("chat", fake)
    app.chat_input[0].set_value("Q?").run()
    assert not app.exception
    assert len(app.error) == 1
    assert "already being answered" in app.error[0].value
    assert "wait for its answer" in app.error[0].value


def test_a_saved_conversation_comes_back_from_the_server_with_its_citations() -> None:
    """The reload case: the thread is in the address, everything else is read from the server."""
    fake = FakeClient()
    fake.saved["t-old"] = SAVED
    app = run_page("chat", fake, geekbot_thread="t-old")
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "Ninety days [1]." in shown(app, "markdown")
    assert any("How long?" in text for text in shown(app, "text"))
    assert any("[1] Handbook · Retention (cited)" in text for text in shown(app, "text"))
    assert fake.asked == []


def test_a_user_message_that_looks_like_markdown_is_shown_as_plain_text() -> None:
    fake = FakeClient()
    nasty = {
        **SAVED,
        "messages": [{**SAVED["messages"][0], "content": "![x](http://evil.example/p)"}],
    }
    fake.saved["t-old"] = nasty
    app = run_page("chat", fake, geekbot_thread="t-old")
    assert "![x](http://evil.example/p)" in shown(app, "text")
    assert not any("evil.example" in text for text in shown(app, "markdown"))


def test_a_saved_csv_conversation_is_continued_and_the_question_says_it_is_a_csv_one() -> None:
    fake = FakeClient()
    fake.saved["t-csv"] = CSV_SAVED
    fake.events = CSV_EVENTS
    fake.documents_ready = 0  # a CSV conversation does not care about the knowledge base
    app = run_page("chat", fake, geekbot_thread="t-csv")
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "A conversation about sales.csv." in shown(app, "caption")
    assert not any("no ready documents" in info.value for info in app.info)
    assert app.chat_input[0].proto.placeholder == "Ask about sales.csv"
    app.toggle[0].set_value(True).run()  # "think harder" is for documents: it is not sent here
    app.chat_input[0].set_value("And in March?").run()
    assert not app.exception
    assert fake.asked == [("t-csv", "And in March?", False)]
    assert fake.modes == ["csv"]
    answer = shown(app, "markdown")
    assert (
        answer.count(CSV_REPLY) >= 2
    )  # the saved reply and the fresh one, as Markdown with the SQL
    details = "\n".join(shown(app, "text"))
    assert "verdict: answered" in details and "file: sales.csv" in details
    assert not any("Asking" in caption for caption in shown(app, "caption"))  # gone with the answer


def test_a_blank_conversation_asks_what_it_is_about_and_starts_with_the_documents() -> None:
    fake = FakeClient()
    app = run_page("chat", fake)
    assert [radio.label for radio in app.radio] == ["This conversation is about"]
    assert app.radio[0].value == "My documents"
    assert len(app.chat_input) == 1
    assert app.chat_input[0].proto.placeholder == "Ask about your documents"


def test_choosing_a_csv_file_shows_a_chooser_and_nothing_is_sent_by_a_plain_rerun() -> None:
    fake = FakeClient()
    app = run_page("chat", fake)
    app.radio[0].set_value("A CSV file").run()
    assert not app.exception
    assert len(app.chat_input) == 0  # nothing to ask until a file is loaded
    assert [button.label for button in app.button if button.label == "Load the file"]
    app.run()
    app.run()
    assert (fake.new_threads, fake.csv_uploads) == ([], [])


# AppTest cannot choose a file, so the loading is driven the way the button drives it: the page's own
# function, once, in a script that remembers it has run (a rerun of the script must not repeat it).
LOAD_SCRIPT = (
    "import streamlit as st\n"
    "from geekbot.ui.views import chat\n"
    "if not st.session_state.get('loaded_once'):\n"
    "    st.session_state['loaded_once'] = True\n"
    "    chat._load_csv(st.session_state['client'], 'sales.csv', b'region,revenue\\nNorth,520\\n')\n"
    "chat.render(st.session_state['client'])\n"
)


def loaded_page(fake: FakeClient) -> AppTest:
    app = AppTest.from_string(LOAD_SCRIPT, default_timeout=10)
    app.session_state["client"] = fake
    return app.run()


def test_a_loaded_file_starts_a_csv_conversation_and_what_the_file_says_is_plain_text() -> None:
    fake = FakeClient()
    fake.events = [SseEvent("meta", {"thread_id": "t-new"}), SseEvent("done", CSV_DONE)]
    app = loaded_page(fake)
    assert not app.exception
    assert fake.new_threads == ["csv"]
    assert fake.csv_uploads == [("t-new", "sales.csv", b"region,revenue\nNorth,520\n")]
    texts = shown(app, "text")
    assert any("sales.csv: 2 rows, 2 columns" in text for text in texts)
    assert any(HOSTILE_COLUMN in text for text in texts)  # a column name is only a word
    assert not any("evil.example" in text for text in shown(app, "markdown"))
    assert app.chat_input[0].proto.placeholder == "Ask about sales.csv"
    app.chat_input[0].set_value("Which region sells the most?").run()
    assert not app.exception
    assert fake.asked == [("t-new", "Which region sells the most?", False)]
    assert fake.modes == ["csv"]
    app.run()
    app.run()
    assert len(fake.csv_uploads) == 1 and len(fake.asked) == 1  # reruns send nothing again


def test_a_file_the_server_cannot_use_is_reported_and_no_conversation_starts() -> None:
    fake = FakeClient()
    fake.upload_error = ApiError(422, "CSV has zero data rows (header only)")
    app = loaded_page(fake)
    assert not app.exception
    assert [error.value for error in app.error] == ["CSV has zero data rows (header only)"]
    assert fake.csv_uploads == []
    assert not any("rows" in text for text in shown(app, "text"))  # no upload summary


def test_an_empty_knowledge_base_is_said_to_the_user() -> None:
    fake = FakeClient()
    fake.documents_ready = 0
    app = run_page("chat", fake)
    assert any("no ready documents" in info.value for info in app.info)


# --- knowledge -------------------------------------------------------------------------------

DOCUMENT = {
    "doc_id": "d1",
    "path": str(Path("notes") / "zebras.txt"),
    "title": "Zebras",
    "kind": "text",
    "state": "ready",
    "error": None,
    "updated_at": "2026-10-20T10:00:00",
}


def test_the_knowledge_page_lists_documents_and_offers_to_remove_one() -> None:
    fake = FakeClient()
    fake.documents_ = [DOCUMENT]
    app = run_page("knowledge", fake)
    assert not app.exception
    assert app.title[0].value == "Knowledge base"
    assert "1 document(s): 1 ready" in shown(app, "caption")
    assert len(app.dataframe) == 1
    assert [box.label for box in app.selectbox] == ["Remove a document from the index"]


def test_an_empty_knowledge_page_says_nothing_is_indexed() -> None:
    app = run_page("knowledge", FakeClient())
    assert not app.exception
    assert "Nothing is indexed yet." in shown(app, "caption")
    assert len(app.dataframe) == 0


def test_adding_a_path_sends_it_once_and_shows_the_outcome() -> None:
    fake = FakeClient()
    app = run_page("knowledge", fake)
    app.text_area(key="kb-paths").input('"C:\\Docs\\notes"').run()
    adding = next(button for button in app.button if button.label == "Add these paths")
    adding.click().run()
    assert not app.exception
    assert fake.ingested == [["C:\\Docs\\notes"]]  # the pasted quotes were dropped
    assert any("OK" in text and "1 ingested" in text for text in shown(app, "text"))
    app.run()  # a later rerun neither sends it again nor shows the outcome again
    assert len(fake.ingested) == 1
    assert not any("1 ingested" in text for text in shown(app, "text"))


# --- settings --------------------------------------------------------------------------------


def test_the_settings_page_shows_the_server_and_never_the_key() -> None:
    app = run_page("settings", FakeClient())
    assert not app.exception
    assert app.title[0].value == "Settings"
    assert len(app.table) >= 2  # what the server reports, and how this UI reaches it
    assert len(app.json) == 1  # the settings this UI read, with the key masked
    assert "api_key" in app.json[0].value
    assert "********" not in app.json[0].value  # no key is configured in this test: it shows None
    flat = json.dumps([shown(app, kind) for kind in ("json", "caption")])
    assert "Read-only" in flat


# --- evals -----------------------------------------------------------------------------------


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "evals" / "results"
    (root / "run1").mkdir(parents=True)
    (root / "run1" / "report.md").write_text("# Report\n\nA finding.\n", encoding="utf-8")
    (root / "data.json").write_text('{"recall": 0.5}', encoding="utf-8")
    (root / "ignored.csv").write_text("a,b\n", encoding="utf-8")
    monkeypatch.setattr(evals, "RESULTS_DIR", root)
    return root


def test_the_evals_page_lists_result_files_and_shows_json_as_data(results: Path) -> None:
    app = run_page("evals", FakeClient())
    assert not app.exception
    assert app.title[0].value == "Evaluation results"
    assert len(app.selectbox) == 1
    assert len(app.json) == 1  # data.json comes first: "data.json" < "run1/report.md"


def test_a_markdown_report_is_rendered(results: Path) -> None:
    app = run_page("evals", FakeClient())
    app.selectbox[0].select_index(1).run()
    assert not app.exception
    assert any("# Report" in text for text in shown(app, "markdown"))


def test_with_no_results_the_page_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evals, "RESULTS_DIR", tmp_path / "nowhere")
    app = run_page("evals", FakeClient())
    assert not app.exception
    assert any("No result files" in info.value for info in app.info)


# --- the evals page's file helpers -----------------------------------------------------------


def test_list_results_is_sorted_relative_and_only_md_and_json(results: Path) -> None:
    assert [path.as_posix() for path in evals.list_results(results)] == [
        "data.json",
        "run1/report.md",
    ]
    assert evals.list_results(results / "missing") == []
    assert len(evals.list_results(results, limit=1)) == 1


def test_read_result_reads_a_file_inside_the_folder(results: Path) -> None:
    assert evals.read_result(results, Path("data.json")) == '{"recall": 0.5}'


def test_read_result_refuses_a_path_that_leaves_the_folder(results: Path) -> None:
    (results.parent / "secret.md").write_text("not for the page", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the results folder"):
        evals.read_result(results, Path("..") / "secret.md")


def test_read_result_refuses_a_file_over_the_size_limit(
    results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evals, "MAX_SHOWN_BYTES", 5)
    with pytest.raises(ValueError, match="too large"):
        evals.read_result(results, Path("data.json"))
