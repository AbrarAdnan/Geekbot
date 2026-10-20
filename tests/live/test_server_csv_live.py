"""Live test for step C5: CSV and document conversations through the server, with the real models.

Needs Ollama running with ``nomic-embed-text`` and the active profile's chat and CSV models pulled
(``gemma3:4b`` and ``qwen2.5-coder:3b`` on the default ``proto`` profile). It starts the real application
(its lifespan loads the chat model onto the **GPU** and the reranker onto the CPU), then moves the GPU
between the two models by asking questions, so run it when Gigalogy is not using the GPU:
``pytest -m live -s tests/live/test_server_csv_live.py``. Deselected by default.

What it checks is the part no unit test can: that the server's swap puts the right model alone on the
GPU, at its own context size, after each kind of question (the plan's Gate A items "after a CSV
conversation starts, the CSV model is 100% on the GPU; back in chat mode the chat model is"), that a
second CSV question does not swap again, that a follow-up is rewritten by the real CSV model, and that
the two kinds of unusable file are refused and leave nothing in the uploads folder. **It is a plumbing
check, not a quality result and not an SLO measurement**: the timings it prints are one sample each on
the proto profile, and whether the CSV answers are right is only shown.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from serverkit import wait_for

from geekbot.config import Settings
from geekbot.doctor import check_exclusive, check_residency
from geekbot.llm.vram import OLLAMA_PS_COMMAND, parse_ollama_ps, run_command
from geekbot.server import create_app
from geekbot.server.security import Perimeter
from geekbot.sse import SseEvent, parse_text

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SALES = FIXTURES / "sales_small.csv"
CORPUS = {
    "passwords.md": (
        "# Passwords\n\nRotate every password at least every ninety days and never reuse one "
        "across services."
    ),
    "sourdough.md": (
        "# Sourdough\n\nA sourdough starter is flour and water left to ferment until it bubbles. "
        "Feed it every day and keep it warm."
    ),
}
DOCUMENT_QUESTION = "How often should I rotate my passwords?"
CSV_QUESTION = "Which region sells the most?"
CSV_FOLLOW_UP = "And what is its total revenue?"
CSV_THIRD = "And which region sells the least?"
MAX_LOAD_SECONDS = 2.0  # R-2: the swap loaded the model; a question should not load it again


@pytest.fixture
def server(tmp_path: Path) -> Iterator[TestClient]:
    """The real application over a temporary data folder, started and with two notes ingested."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data", allowed_roots=str(docs))
    models = Path("data") / "models"
    if models.is_dir():
        shutil.copytree(models, settings.data_dir / "models")  # the reranker: nothing to download
    perimeter = Perimeter.from_settings(settings)
    headers = {"Authorization": f"Bearer {perimeter.api_key}"}
    app = create_app(settings, perimeter=perimeter)
    with TestClient(app, base_url="http://127.0.0.1:8000", headers=headers) as client:
        client.app.state.test_settings = settings  # type: ignore[attr-defined]
        accepted = client.post("/v1/kb/ingest", json={"paths": [str(docs)]})
        assert accepted.status_code == 202
        job = wait_for(client, accepted.json()["id"], timeout_s=120)
        assert job["state"] == "done" and not job["result"]["failed"], job
        yield client


def stages(events: list[SseEvent]) -> list[str]:
    return [event.data["stage"] for event in events if event.event == "status"]


def ask(client: TestClient, thread: str, question: str, **body: Any) -> list[SseEvent]:
    response = client.post(f"/v1/threads/{thread}/messages", json={"question": question, **body})
    assert response.status_code == 200, response.text
    return parse_text(response.text)


def assert_alone_on_the_gpu(active: str, other: str, context: int, when: str) -> None:
    """``ollama ps`` now: ``active`` fully on the GPU at ``context`` tokens, ``other`` not loaded."""
    loaded = parse_ollama_ps(run_command(OLLAMA_PS_COMMAND))
    print(
        f"after {when}: "
        + ", ".join(f"{m.name} {m.gpu_percent}% GPU, context {m.context}" for m in loaded)
    )
    residency = check_residency(active, loaded, expected_context=context)
    assert residency.status == "PASS", f"after {when}: {residency.detail}"
    exclusive = check_exclusive(active, other, loaded)
    assert exclusive.status == "PASS", f"after {when}: {exclusive.detail}"


@pytest.mark.live
def test_the_gpu_follows_the_conversation_and_bad_files_are_refused(
    server: TestClient, tmp_path: Path
) -> None:
    settings: Settings = server.app.state.test_settings  # type: ignore[attr-defined]
    chat, csv = settings.resolved_chat_model, settings.resolved_csv_model
    health = server.get("/v1/health").json()
    print(f"\nhealth: {health}")
    assert (health["chat_ready"], health["csv_ready"]) == (True, False)  # the warm-up loaded chat
    assert_alone_on_the_gpu(chat, csv, settings.chat_num_ctx, "the start-up warm-up")

    # --- a CSV conversation: the file, the swap, a question, a follow-up ----------------------
    thread = server.post("/v1/threads", json={"mode": "csv"}).json()["thread_id"]
    uploaded = server.post(
        "/v1/csv/upload",
        params={"thread": thread, "filename": SALES.name},
        content=SALES.read_bytes(),
        headers={"content-type": "application/octet-stream"},
    )
    assert uploaded.status_code == 200, uploaded.text
    info = uploaded.json()
    print(f"uploaded: {info['file']}, {info['rows']} rows, columns {info['columns']}")
    assert (info["rows"], info["column_count"]) == (12, 4)

    first = ask(server, thread, CSV_QUESTION, mode="csv")
    done = first[-1]
    print(f"first question: stages {stages(first)}, verdict {done.data.get('csv_status')}")
    print(f"answer: {done.data.get('answer')!r}, generation {done.data.get('generation')}")
    assert done.event == "done", done.data
    assert stages(first) == ["loading_model", "asking"]  # the first CSV question pays the swap
    assert done.data["model"] == csv
    assert done.data["generation"]["load_seconds"] < MAX_LOAD_SECONDS
    assert_alone_on_the_gpu(csv, chat, settings.csv_num_ctx, "the first CSV question")

    second = ask(server, thread, CSV_FOLLOW_UP, mode="csv")
    done = second[-1]
    print(f"follow-up: stages {stages(second)}, searched as {done.data.get('searched_as')!r}")
    print(f"answer: {done.data.get('answer')!r}")
    assert done.event == "done", done.data
    assert stages(second) == ["asking"]  # the CSV model is already there: no swap, no load
    assert (
        done.data["searched_as"] and not done.data["rewrite_error"]
    )  # the real CSV model rewrote it
    assert done.data["thread_taint"] == ["TOOL", "USER"]
    assert_alone_on_the_gpu(csv, chat, settings.csv_num_ctx, "the CSV follow-up")

    # --- a document conversation: the swap back --------------------------------------------------
    document = ask(server, "t-documents", DOCUMENT_QUESTION)
    done = document[-1]
    print(f"document question: stages {stages(document)}, cited {done.data.get('cited')}")
    print(f"answer: {done.data.get('answer')!r}")
    assert done.event == "done", done.data
    assert stages(document) == ["loading_model"]  # the swap back to the chat model
    assert done.data["model"] == chat and done.data["mode"] == "chat"
    assert done.data["sources"][0]["title"] == "passwords.md"
    assert_alone_on_the_gpu(chat, csv, settings.chat_num_ctx, "the document question")
    health = server.get("/v1/health").json()
    assert (health["chat_ready"], health["csv_ready"]) == (True, False)

    # --- and back to the CSV conversation: the table is still loaded, only the model swaps --------
    third = ask(server, thread, CSV_THIRD, mode="csv")
    done = third[-1]
    print(f"third CSV question: stages {stages(third)}\nanswer: {done.data.get('answer')!r}")
    assert done.event == "done", done.data
    assert stages(third) == ["loading_model", "asking"]
    assert_alone_on_the_gpu(csv, chat, settings.csv_num_ctx, "going back to the CSV conversation")

    # --- the thread reads back as a CSV conversation, with the user's file name ---------------------
    shown = server.get(f"/v1/threads/{thread}").json()
    assert (shown["mode"], shown["csv_file"], shown["turns"]) == ("csv", SALES.name, 3)

    # --- two files the server must refuse, leaving nothing behind -----------------------------------
    uploads = settings.data_dir / "uploads"
    before = sorted(path.name for path in uploads.iterdir())
    for bad in ("bad_header_only.csv", "bad_empty.csv"):
        refused = server.post(
            "/v1/csv/upload",
            params={"thread": "t-bad", "filename": bad},
            content=(FIXTURES / bad).read_bytes(),
            headers={"content-type": "application/octet-stream"},
        )
        print(f"{bad}: {refused.status_code} {refused.json()['detail']!r}")
        assert refused.status_code == 422
    assert sorted(path.name for path in uploads.iterdir()) == before
    assert server.get("/v1/health").json()["csv_sessions"] == 1  # the conversation's table only
