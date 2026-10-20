"""Live test for step S4.2: the server answers a streamed question with the real models.

Needs Ollama running with ``nomic-embed-text`` and the active profile's chat model pulled
(``gemma3:4b`` on the default ``proto`` profile). It starts the real application (its lifespan opens
the storage, loads the chat model onto the **GPU** and the reranker on the CPU), so run it when
Gigalogy is not using the GPU: ``pytest -m live -s tests/live/test_server_live.py``. Deselected by
default.

The knowledge base is three one-sentence documents in a temporary folder, ingested through the API
(the folder is the one allowed root, and every request carries the API key: step S4.3).
The reranker's weights are copied from your own ``data/models`` if that folder exists (so nothing is
downloaded); otherwise the first start-up downloads them once (about 87 MB). **This is a plumbing
check, not a quality result and not an SLO measurement:** it shows that the pieces fit together on the
real models (the stream arrives, the answer is grounded in the right document, a follow-up is
rewritten, the sources are still there when the thread is read back, the model is not loaded again
during a question), not how good or how fast the target model is.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from serverkit import wait_for

from geekbot.config import Settings
from geekbot.server import create_app
from geekbot.server.security import Perimeter
from geekbot.sse import parse_text

CORPUS = {
    "sourdough.md": (
        "# Sourdough\n\nA sourdough starter is flour and water left to ferment until it bubbles. "
        "Feed it every day and keep it warm."
    ),
    "rivers.md": (
        "# Rivers of Bangladesh\n\nThe Ganges and the Brahmaputra meet in Bangladesh and flow "
        "together into the Bay of Bengal."
    ),
    "passwords.md": (
        "# Passwords\n\nRotate every password at least every ninety days and never reuse one "
        "across services."
    ),
}
MAX_LOAD_SECONDS = 2.0  # R-2: the warm-up loaded the model; a question should not load it again
QUESTION = "How often should I rotate my passwords?"
FOLLOW_UP = "And can I reuse one across services?"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[TestClient]:
    """The real application over a temporary data folder, started and with the three notes ingested."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data", allowed_roots=str(docs))
    models = Path("data") / "models"
    if models.is_dir():
        shutil.copytree(models, settings.data_dir / "models")
    # S4.3: the real perimeter, as `geekbot serve` builds it (a key is required, made in the
    # temporary data folder), so the live test goes through the same checks as the UI will.
    perimeter = Perimeter.from_settings(settings)
    headers = {"Authorization": f"Bearer {perimeter.api_key}"}
    app = create_app(settings, perimeter=perimeter)
    with TestClient(app, base_url="http://127.0.0.1:8000", headers=headers) as client:
        accepted = client.post("/v1/kb/ingest", json={"paths": [str(docs)]})
        assert accepted.status_code == 202
        job = wait_for(client, accepted.json()["id"], timeout_s=120)
        assert job["state"] == "done", job
        assert not job["result"]["failed"], job
        yield client


@pytest.mark.live
def test_a_streamed_answer_a_follow_up_and_the_thread_read_back(server: TestClient) -> None:
    health = server.get("/v1/health").json()
    print(f"\nhealth: {health}")
    assert health["chat_ready"] is True  # the lifespan warmed the models up
    assert health["documents"] == {"ready": 3}

    thread_id = server.post("/v1/threads").json()["thread_id"]

    first = server.post(f"/v1/threads/{thread_id}/messages", json={"question": QUESTION})
    events = parse_text(first.text)
    names = [event.event for event in events]
    print(f"events: {names[:3]} ... {names[-1]} ({names.count('text')} text pieces)")
    assert first.status_code == 200
    assert names[0] == "meta" and names[-1] == "done"
    assert "status" not in names  # warmed up at start-up, so the question did not wait for a load
    assert names.count("text") >= 2  # it really streamed, in pieces
    done = events[-1].data
    print(f"answer: {done['answer']!r}\ncited {done['cited']}, generation {done['generation']}")
    assert done["sources"][0]["title"] == "passwords.md"
    answer = done["answer"].lower()
    assert "ninety" in answer or "90" in answer
    streamed = "".join(e.data["text"] for e in events if e.event == "text").strip()
    # What was shown live is the answer, unless the citation check removed a mark that named no
    # source (then `done` has the corrected text, as the command line prints it).
    assert done["dropped_citations"] or streamed == done["answer"]
    assert done["generation"].get("load_seconds", 0.0) < MAX_LOAD_SECONDS
    assert all(1 <= n <= len(done["sources"]) for n in done["cited"])

    second = server.post(f"/v1/threads/{thread_id}/messages", json={"question": FOLLOW_UP})
    follow_up = parse_text(second.text)[-1]
    print(f"follow-up searched as: {follow_up.data['searched_as']!r}\n{follow_up.data['answer']!r}")
    assert follow_up.event == "done"
    assert follow_up.data["turn"] == 2
    assert follow_up.data["searched_as"]  # a follow-up is rewritten into a question of its own

    shown = server.get(f"/v1/threads/{thread_id}").json()
    assert shown["turns"] == 2 and shown["message_count"] == 4
    answers = [m for m in shown["messages"] if m["role"] == "assistant"]
    assert answers[0]["sources"][0]["title"] == "passwords.md"  # citations persist
    assert [t["thread_id"] for t in server.get("/v1/threads").json()] == [thread_id]

    unanswerable = parse_text(
        server.post(
            f"/v1/threads/{server.post('/v1/threads').json()['thread_id']}/messages",
            json={"question": "What is the capital of France?"},
        ).text
    )[-1]
    print(f"unanswerable question: {unanswerable.event}, {unanswerable.data.get('answer')!r}")
    assert (
        unanswerable.event == "done"
    )  # what the model says is S3.5's concern; shown, not asserted
