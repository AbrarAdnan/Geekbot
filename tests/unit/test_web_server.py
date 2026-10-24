"""Tests for step S5.3: the web search through the server: the ``web`` flag, the ``approval`` event and
the route that answers it.

The app is the real one (lifespan, answer graph, SQLite thread database); the models, the retriever and
the search backend are fakes. What is checked is the wire contract of ``threads.py``: a question may
stop at the egress gate's question, the stream then ends with ``approval`` and no ``done``, the thread
shows what it waits for, nothing is sent before the answer, and an answer with no question is refused.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chatkit import REWRITE, ScriptedRetriever
from fastapi.testclient import TestClient
from serverkit import Asked, ask, make_app, upload_csv

from geekbot.sse import parse_text
from geekbot.tools.egress import EgressGate
from geekbot.tools.web import WebHit, WebSearch

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")

THREAD = "t-web"
Q1 = "How long does Lantern keep chat history?"
Q2 = "And what about audit logs?"
PAGE = WebHit("A web page", "https://pages.example/lantern", "Lantern keeps logs for a year.")


class Backend:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int) -> list[WebHit]:
        self.queries.append(query)
        return [PAGE]


@pytest.fixture
def backend() -> Backend:
    return Backend()


@pytest.fixture
def client(tmp_path: Path, backend: Backend) -> Iterator[TestClient]:
    web = EgressGate(enabled=True).wrap(WebSearch(backend, limit=5))
    with TestClient(make_app(tmp_path, retriever=ScriptedRetriever(), web=web)) as opened:
        yield opened


def approve(client: TestClient, thread: str, action: str, query: str = "", **extra: Any) -> Asked:
    response = client.post(
        f"/v1/threads/{thread}/approval", json={"action": action, "query": query, **extra}
    )
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return Asked(response.status_code, parse_text(response.text), None)
    return Asked(response.status_code, [], response.json())


def stopped_follow_up(client: TestClient, thread: str = THREAD) -> Asked:
    """Turn one searches the web; turn two is a follow-up the model rewrites, so the gate asks."""
    first = ask(client, thread, Q1, web=True)
    assert first.names[-1] == "done"
    stopped = ask(client, thread, Q2, web=True)
    assert stopped.names[-1] == "approval"
    return stopped


# --- the flag and the health report ---------------------------------------------------------


def test_health_says_whether_the_server_has_a_web_search(
    client: TestClient, tmp_path: Path
) -> None:
    assert client.get("/v1/health").json()["web_enabled"] is True
    (tmp_path / "off").mkdir()
    with TestClient(make_app(tmp_path / "off")) as off:
        assert off.get("/v1/health").json()["web_enabled"] is False


def test_asking_for_the_web_on_a_server_that_has_none_is_a_409_and_stores_nothing(
    tmp_path: Path,
) -> None:
    with TestClient(make_app(tmp_path)) as off:
        refused = ask(off, THREAD, Q1, web=True)
        assert refused.status == 409 and "switched off" in refused.body["detail"]
        assert off.get(f"/v1/threads/{THREAD}").status_code == 404
        assert ask(off, THREAD, Q1).names[-1] == "done"  # the same question without the flag works


def test_a_csv_conversation_cannot_ask_for_the_web(client: TestClient) -> None:
    assert upload_csv(client, "t-csv").status_code == 200
    refused = ask(client, "t-csv", "How many rows are there?", web=True, mode="csv")
    assert refused.status == 409 and "CSV" in refused.body["detail"]


# --- a question that goes through ------------------------------------------------------------


def test_the_users_own_question_searches_and_the_turn_says_so(
    client: TestClient, backend: Backend
) -> None:
    turn = ask(client, THREAD, Q1, web=True)
    assert turn.names[0] == "meta" and turn.names[-1] == "done"
    assert "approval" not in turn.names
    assert backend.queries == [Q1]
    done = turn.data("done")
    assert [(e["decision"], e["tainted"], e["query"]) for e in done["egress"]] == [
        ("allowed", False, Q1)
    ]
    web = [s for s in done["sources"] if s["origin"] == "WEB"]
    assert [(s["path"], s["section"]) for s in web] == [(PAGE.url, "pages.example")]
    assert [s["origin"] for s in done["sources"] if s["origin"] != "WEB"] == ["KB", "KB"]
    assert "WEB" in done["labels"] and done["tainted"] is True


def test_without_the_flag_nothing_is_sent_and_no_source_is_from_the_web(
    client: TestClient, backend: Backend
) -> None:
    done = ask(client, THREAD, Q1).data("done")
    assert backend.queries == [] and done["egress"] == []
    assert {s["origin"] for s in done["sources"]} == {"KB"}


# --- a question that stops -------------------------------------------------------------------


def test_a_rewritten_query_in_a_tainted_thread_ends_the_stream_with_the_question(
    client: TestClient, backend: Backend
) -> None:
    stopped = stopped_follow_up(client)
    assert "done" not in stopped.names and "error" not in stopped.names
    approval = stopped.data("approval")
    assert approval["query"] == REWRITE and approval["question"] == Q2
    assert approval["tool"] == "web_search"
    assert {"KB", "WEB"} <= set(approval["labels"])
    assert backend.queries == [Q1]  # the stopped turn sent nothing


def test_the_waiting_thread_shows_its_question_and_refuses_a_new_one(client: TestClient) -> None:
    stopped_follow_up(client)
    shown = client.get(f"/v1/threads/{THREAD}").json()
    assert shown["pending_approval"]["query"] == REWRITE
    assert shown["turns"] == 1  # the waiting turn is not remembered yet
    again = ask(client, THREAD, "Something else?")
    assert again.status == 409 and "waiting" in again.body["detail"]
    assert ask(client, "t-other", Q1).names[-1] == "done"  # the slot is free: other threads go on


def test_approving_sends_exactly_the_shown_text_and_finishes_the_turn(
    client: TestClient, backend: Backend
) -> None:
    stopped_follow_up(client)
    finished = approve(client, THREAD, "approve")
    assert finished.names[0] == "meta" and finished.names[-1] == "done"
    assert backend.queries == [Q1, REWRITE]
    done = finished.data("done")
    assert done["turn"] == 2 and done["answer"]
    assert [e["decision"] for e in done["egress"]] == ["approved"]  # this turn's decisions only
    shown = client.get(f"/v1/threads/{THREAD}").json()
    assert shown["pending_approval"] is None and shown["turns"] == 2


def test_editing_sends_the_users_text(client: TestClient, backend: Backend) -> None:
    stopped_follow_up(client)
    done = approve(client, THREAD, "edit", "my own words").data("done")
    assert backend.queries == [Q1, "my own words"]
    assert done["egress"][-1]["decision"] == "edited"


def test_denying_sends_nothing_and_the_turn_still_answers(
    client: TestClient, backend: Backend
) -> None:
    stopped_follow_up(client)
    finished = approve(client, THREAD, "deny")
    assert finished.names[-1] == "done"
    assert backend.queries == [Q1]
    done = finished.data("done")
    assert done["egress"][-1]["decision"] == "denied"
    assert {s["origin"] for s in done["sources"]} == {"KB"}


def test_an_answer_can_be_given_once_only(client: TestClient, backend: Backend) -> None:
    stopped_follow_up(client)
    assert approve(client, THREAD, "approve").names[-1] == "done"
    second = approve(client, THREAD, "approve")
    assert second.status == 404
    assert backend.queries == [Q1, REWRITE]  # not sent twice


# --- an answer with no question, and answers that are not answers -----------------------------


def test_an_answer_with_no_question_waiting_is_a_404_and_sends_nothing(
    client: TestClient, backend: Backend
) -> None:
    assert approve(client, "t-nothing", "approve").status == 404
    ask(client, THREAD, Q1)  # an answered thread with nothing waiting
    assert approve(client, THREAD, "approve").status == 404
    assert backend.queries == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"action": "yes"},
        {"action": "approve", "surprise": 1},
        {"action": "approve", "query": 5},
        {"query": "no action"},
    ],
)
def test_an_answer_the_server_cannot_read_is_a_422_and_the_question_stays_open(
    client: TestClient, backend: Backend, body: dict[str, Any]
) -> None:
    stopped_follow_up(client)
    assert client.post(f"/v1/threads/{THREAD}/approval", json=body).status_code == 422
    assert client.get(f"/v1/threads/{THREAD}").json()["pending_approval"] is not None
    assert backend.queries == [Q1]


def test_the_approval_route_needs_the_api_key(tmp_path: Path) -> None:
    key = "k" * 32
    with TestClient(make_app(tmp_path, api_key=key)) as secured:
        bare = secured.post(f"/v1/threads/{THREAD}/approval", json={"action": "approve"})
        assert bare.status_code == 401
        wrong = secured.post(
            f"/v1/threads/{THREAD}/approval",
            json={"action": "approve"},
            headers={"Authorization": "Bearer " + "x" * 32},
        )
        assert wrong.status_code == 401
