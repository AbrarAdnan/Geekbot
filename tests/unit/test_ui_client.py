"""Tests for step S4.4: the UI's HTTP client (``geekbot.ui.client``).

Two kinds. Most run the client against the **real app** (real lifespan and middleware, real SQLite and
Chroma with a fake embedder, scripted models) through Starlette's test client, which is an
``httpx.Client`` the ``GeekbotClient`` is handed: that is the contract the UI depends on. The rest use
``httpx.MockTransport`` for what a healthy server never does (a stream that stops half way, a refused
connection, a frame that is not JSON), and a fake job source for ``wait_for_job``.
"""

import itertools
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from chatkit import ScriptedChat
from fastapi.testclient import TestClient
from serverkit import make_app, write_note

from geekbot.config import Settings
from geekbot.server.security import LOOPBACK_HOSTS
from geekbot.ui.client import (
    ApiError,
    GeekbotClient,
    JobTimeoutError,
    ServerUnavailableError,
    connect,
    wait_for_job,
)
from geekbot.ui.display import TurnView

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

KEY = "ui-client-test-key-" + "k" * 16
LOCAL = "http://127.0.0.1:8000"
AUTH = {"Authorization": f"Bearer {KEY}"}
QUESTION = "Where do zebras live?"


@contextmanager
def served(tmp_path: Path, **options: Any) -> Iterator[TestClient]:
    """The real app with the real perimeter (a key is required), as an ``httpx.Client``.

    The test client carries the right key by default; a ``GeekbotClient`` built on it sends its own
    key with every request, which wins, so a test can also give it a wrong one.
    """
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS, **options)
    with TestClient(app, base_url=LOCAL, headers=AUTH) as http:
        yield http


@contextmanager
def connected(tmp_path: Path, **options: Any) -> Iterator[GeekbotClient]:
    """A ``GeekbotClient`` holding the right key, over ``served``."""
    with served(tmp_path, **options) as http:
        yield GeekbotClient(LOCAL, KEY, http=http, key_origin="a test")


@pytest.fixture
def http(tmp_path: Path) -> Iterator[TestClient]:
    with served(tmp_path) as opened:
        yield opened


@pytest.fixture
def client(http: TestClient) -> GeekbotClient:
    return GeekbotClient(LOCAL, KEY, http=http, key_origin="a test")


def finished(client: GeekbotClient, job: dict[str, Any]) -> dict[str, Any]:
    return wait_for_job(client, job["id"], poll_seconds=0.02, timeout_seconds=30)


def ingest_the_note(client: GeekbotClient, tmp_path: Path) -> None:
    job = finished(client, client.ingest([str(write_note(tmp_path))]))
    assert job["state"] == "done", job


def settings_for_client(tmp_path: Path, key: str | None = KEY, **options: Any) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path / "data", api_key=key, **options)


def mocked(handler: Callable[[httpx.Request], httpx.Response]) -> GeekbotClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=LOCAL)
    return GeekbotClient(LOCAL, KEY, http=http)


def stream_of(body: str) -> httpx.Response:
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


# --- against the real server -----------------------------------------------------------------


def test_the_server_answers_ping_and_health(client: GeekbotClient) -> None:
    assert client.ping() is True
    health = client.health()
    assert health["status"] == "ok"
    assert health["chat_ready"] is True
    assert health["documents"] == {}


def test_a_wrong_key_is_a_401_with_the_servers_own_words(http: TestClient) -> None:
    wrong = GeekbotClient(LOCAL, "w" * 32, http=http)
    with pytest.raises(ApiError) as raised:
        wrong.health()
    assert raised.value.status == 401
    assert "API key" in raised.value.detail
    assert wrong.ping() is True  # /healthz asks for no key


def test_the_formats_the_server_reads(client: GeekbotClient) -> None:
    formats = client.formats()
    assert formats == sorted(formats)
    assert {".md", ".pdf", ".txt"} <= set(formats)
    assert all(re.fullmatch(r"\.[a-z0-9]+", extension) for extension in formats)


def test_an_upload_is_ingested_and_listed(client: GeekbotClient) -> None:
    accepted = client.upload("zebras.txt", b"Zebras live on the savanna.\n")
    assert accepted["stored_as"].endswith("-zebras.txt")
    assert accepted["existed"] is False
    assert accepted["size_bytes"] == len(b"Zebras live on the savanna.\n")
    assert finished(client, accepted["job"])["state"] == "done"
    (document,) = client.documents()
    assert document["state"] == "ready"
    assert Path(document["path"]).name.endswith("zebras.txt")


def test_the_same_upload_twice_is_one_file_and_costs_no_embedding(client: GeekbotClient) -> None:
    """The v1 bug #1 in the new setting: sending a file again must not re-index it."""
    data = b"Zebras live on the savanna.\n"
    first = finished(client, client.upload("zebras.txt", data)["job"])
    assert first["result"]["files"][0]["embedded"] > 0
    again = client.upload("zebras.txt", data)
    assert again["existed"] is True
    second = finished(client, again["job"])
    assert second["result"]["files"][0]["status"] == "unchanged"
    assert second["result"]["files"][0]["embedded"] == 0
    assert len(client.documents()) == 1


def test_an_unreadable_type_is_a_415_that_lists_what_is_accepted(client: GeekbotClient) -> None:
    with pytest.raises(ApiError) as raised:
        client.upload("evil.exe", b"MZ")
    assert raised.value.status == 415
    assert "not accepted" in raised.value.detail and ".txt" in raised.value.detail


def test_ingest_by_path_works_inside_an_allowed_folder(
    client: GeekbotClient, tmp_path: Path
) -> None:
    ingest_the_note(client, tmp_path)
    assert [Path(d["path"]).name for d in client.documents()] == ["zebras.txt"]


def test_ingest_outside_the_allowed_folders_is_a_403(client: GeekbotClient, tmp_path: Path) -> None:
    outside = Path(tmp_path.anchor) / "elsewhere"
    with pytest.raises(ApiError) as raised:
        client.ingest([str(outside)])
    assert raised.value.status == 403


def test_a_document_can_be_removed_and_a_missing_one_is_a_404(
    client: GeekbotClient, tmp_path: Path
) -> None:
    ingest_the_note(client, tmp_path)
    (document,) = client.documents()
    assert finished(client, client.delete_document(document["doc_id"]))["state"] == "done"
    assert client.documents() == []
    with pytest.raises(ApiError) as raised:
        client.delete_document("nope")
    assert raised.value.status == 404


def test_a_question_streams_and_the_thread_is_saved_and_can_be_forgotten(
    client: GeekbotClient, tmp_path: Path
) -> None:
    ingest_the_note(client, tmp_path)
    thread_id = client.new_thread()
    assert re.fullmatch(r"t-[0-9a-f]{12}", thread_id)
    assert client.thread(thread_id) is None  # nothing is stored before the first answer
    view = TurnView()
    names = []
    for event in client.ask(thread_id, QUESTION):
        names.append(event.event)
        view.feed(event)
    assert names[0] == "meta" and names[-1] == "done"
    assert view.answer == "Ninety days [1]."
    assert view.done is not None
    assert view.done["sources"][0]["path"].endswith("zebras.txt")
    saved = client.thread(thread_id)
    assert saved is not None and saved["message_count"] == 2
    assert [t["thread_id"] for t in client.threads()] == [thread_id]
    assert client.delete_thread(thread_id) is True
    assert client.delete_thread(thread_id) is False
    assert client.thread(thread_id) is None


def test_a_failed_turn_ends_in_an_error_event_not_in_an_exception(tmp_path: Path) -> None:
    with connected(tmp_path, chat=ScriptedChat(ConnectionError("down"))) as client:
        ingest_the_note(client, tmp_path)
        view = TurnView()
        events = list(client.ask("t-failing", QUESTION))
        for event in events:
            view.feed(event)
    assert [e.event for e in events] == ["meta", "error"]
    assert view.error is not None and "down" in view.error
    assert view.answer == ""


def test_the_refusals_before_the_stream_are_api_errors(client: GeekbotClient) -> None:
    with pytest.raises(ApiError) as raised:
        list(client.ask("t-x", ""))  # an empty question
    assert raised.value.status == 422
    assert raised.value.detail  # FastAPI's list of {"loc", "msg"} became one sentence
    with pytest.raises(ApiError) as raised:
        list(client.ask("bad name", "hello"))
    assert raised.value.status == 422
    assert "thread name" in raised.value.detail


def test_closing_the_stream_early_is_fine(client: GeekbotClient, tmp_path: Path) -> None:
    ingest_the_note(client, tmp_path)
    stream = client.ask("t-early", QUESTION)
    assert next(stream).event == "meta"
    stream.close()  # a user who navigates away: no exception, no half-read frame left over


# --- what a healthy server never does --------------------------------------------------------


def test_a_stream_that_stops_before_its_last_event_is_not_an_answer() -> None:
    body = 'event: meta\ndata: {"turn":1}\n\nevent: text\ndata: {"text":"Hi"}\n\n'
    client = mocked(lambda request: stream_of(body))
    seen = []
    with pytest.raises(ServerUnavailableError, match="before the answer was finished"):
        for event in client.ask("t-1", "q"):
            seen.append(event.event)
    assert seen == ["meta", "text"]  # what arrived was delivered; the missing end is the error


def test_a_frame_that_is_not_json_is_reported_not_crashed_on() -> None:
    client = mocked(lambda request: stream_of("event: done\ndata: {oops\n\n"))
    with pytest.raises(ServerUnavailableError, match="cannot read"):
        list(client.ask("t-1", "q"))


def test_a_refused_connection_says_where_and_what_to_run() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = mocked(refuse)
    assert client.ping() is False
    with pytest.raises(ServerUnavailableError) as raised:
        client.health()
    assert LOCAL in str(raised.value) and "geekbot up" in str(raised.value)
    with pytest.raises(ServerUnavailableError):
        list(client.ask("t-1", "q"))


def test_a_connection_lost_mid_answer_is_unavailable() -> None:
    class Dies(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'event: text\ndata: {"text":"Hi"}\n\n'
            raise httpx.ReadError("connection reset")

    client = mocked(lambda request: httpx.Response(200, stream=Dies()))
    seen = []
    with pytest.raises(ServerUnavailableError, match="connection to the server broke"):
        for event in client.ask("t-1", "q"):
            seen.append(event.event)
    assert seen == ["text"]


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(409, json={"detail": "a question is already being answered"}), "already"),
        (
            httpx.Response(422, json={"detail": [{"loc": ["body", "q"], "msg": "too short"}]}),
            "too short",
        ),
        (httpx.Response(500, content=b"<html>oops</html>"), "HTTP 500"),
        (httpx.Response(502, json={"detail": ""}), "HTTP 502"),
    ],
)
def test_an_error_status_becomes_one_sentence(response: httpx.Response, detail: str) -> None:
    client = mocked(lambda request: response)
    with pytest.raises(ApiError) as raised:
        client.health()
    assert raised.value.status == response.status_code
    assert detail in raised.value.detail
    assert str(raised.value) == raised.value.detail


def test_what_goes_over_the_wire() -> None:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/kb/upload":
            return httpx.Response(202, json={"job": {"id": "j1"}})
        if request.url.path.endswith("/messages"):
            return stream_of('event: done\ndata: {"answer":"x"}\n\n')
        return httpx.Response(200, json={"status": "ok"})

    client = mocked(record)
    client.ping()
    client.health()
    client.upload("my notes.txt", b"hello")
    list(client.ask("t-1", "why?", think=True))
    ping, health, upload, ask = seen
    assert "authorization" not in ping.headers  # /healthz is public
    assert health.headers["authorization"] == f"Bearer {KEY}"
    assert upload.url.params["filename"] == "my notes.txt"
    assert upload.read() == b"hello"
    assert upload.headers["content-length"] == "5"  # the server refuses a body of unknown size
    assert "transfer-encoding" not in upload.headers
    assert json.loads(ask.read()) == {"question": "why?", "think": True}
    assert ask.headers["authorization"] == f"Bearer {KEY}"


def test_the_key_is_not_in_what_a_client_shows_of_itself() -> None:
    client = mocked(lambda request: httpx.Response(200, json={}))
    assert KEY not in repr(client)


# --- connecting ------------------------------------------------------------------------------


def test_the_client_for_these_settings_has_their_port_and_key_origin(tmp_path: Path) -> None:
    client = GeekbotClient.from_settings(settings_for_client(tmp_path, server_port=9100))
    assert client.base_url == "http://127.0.0.1:9100"
    assert client.key_origin == "GEEKBOT_API_KEY"


def test_a_key_in_the_data_folder_is_found_and_named_by_its_file(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "api-key").write_text(KEY + "\n", encoding="utf-8")
    client = GeekbotClient.from_settings(settings_for_client(tmp_path, key=None))
    assert client.key_origin == str(tmp_path / "data" / "api-key")


def test_no_key_yet_means_the_server_never_ran_and_nothing_is_created(tmp_path: Path) -> None:
    with pytest.raises(ServerUnavailableError, match="no API key yet"):
        GeekbotClient.from_settings(settings_for_client(tmp_path, key=None))
    assert not (tmp_path / "data").exists()  # the UI never makes the key


def test_a_weak_key_is_explained(tmp_path: Path) -> None:
    with pytest.raises(ServerUnavailableError, match="at least 24 characters"):
        GeekbotClient.from_settings(settings_for_client(tmp_path, key="short"))


def test_connect_succeeds_against_a_server_that_accepts_the_key(tmp_path: Path) -> None:
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS)
    with TestClient(app, base_url=LOCAL) as http:
        client = connect(settings_for_client(tmp_path), http=http)
        assert client.health()["status"] == "ok"


def test_connect_says_when_the_server_is_not_there(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    http = httpx.Client(transport=httpx.MockTransport(refuse), base_url=LOCAL)
    with pytest.raises(ServerUnavailableError, match="not answering at http://127.0.0.1:8000"):
        connect(settings_for_client(tmp_path), http=http)


def test_connect_says_when_the_server_has_another_key(tmp_path: Path) -> None:
    app = make_app(tmp_path, api_key=KEY, hosts=LOOPBACK_HOSTS)
    with TestClient(app, base_url=LOCAL) as http:
        other = settings_for_client(tmp_path, key="another-key-" + "z" * 20)
        with pytest.raises(ServerUnavailableError) as raised:
            connect(other, http=http)
    assert "did not accept" in str(raised.value)
    assert "GEEKBOT_API_KEY" in str(raised.value)  # where the UI got its key; never the key itself
    assert "another-key" not in str(raised.value)


# --- waiting for a job -----------------------------------------------------------------------


class FakeJobs:
    """A job source that shows the states in order, the last one for ever."""

    def __init__(self, *states: str) -> None:
        self.states = list(states)

    def job(self, job_id: str, /) -> dict[str, Any]:
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"id": job_id, "state": state}


def test_wait_polls_until_the_job_is_done_and_reports_every_state() -> None:
    seen: list[str] = []
    slept: list[float] = []
    job = wait_for_job(
        FakeJobs("queued", "running", "done"),
        "j1",
        poll_seconds=0.25,
        on_update=lambda state: seen.append(state["state"]),
        sleep=slept.append,
    )
    assert job["state"] == "done"
    assert seen == ["queued", "running", "done"]
    assert slept == [0.25, 0.25]  # no sleep after the last poll


def test_a_failed_job_is_returned_not_raised() -> None:
    job = wait_for_job(FakeJobs("running", "failed"), "j1", sleep=lambda seconds: None)
    assert job["state"] == "failed"


def test_a_job_already_done_costs_one_call_and_no_sleep() -> None:
    slept: list[float] = []
    assert wait_for_job(FakeJobs("done"), "j1", sleep=slept.append)["state"] == "done"
    assert slept == []


def test_waiting_gives_up_and_says_the_job_keeps_running() -> None:
    ticks = itertools.count()
    with pytest.raises(JobTimeoutError, match="keeps running"):
        wait_for_job(
            FakeJobs("running"),
            "j9",
            timeout_seconds=3,
            sleep=lambda seconds: None,
            clock=lambda: float(next(ticks)),
        )
