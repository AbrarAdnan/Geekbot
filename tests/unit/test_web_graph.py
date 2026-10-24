"""Tests for step S5.3: the web search inside the answer graph, with the egress gate and its question.

No model, no network: a scripted chat and rewrite call, a scripted retriever, and a backend that
records every query it is asked. What these check is the behaviour G2 promises: with the web off nothing
is sent; the user's own words go; a query a model rewrote in a tainted thread waits for the user's
answer and only the approved (or edited) text is ever sent; any answer that cannot be read is a denial.
"""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from chatkit import REWRITE, ScriptedChat, ScriptedCondenser, ScriptedRetriever

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.nodes import (
    APPROVAL_KIND,
    after_search,
    begin_turn,
    egress_gate,
    route_after_gate,
    web_search,
)
from geekbot.agent.runner import (
    INTERRUPT_KEY,
    THREADS_DB,
    interrupt_question,
    read_thread,
    read_thread_info,
    run_turn,
)
from geekbot.agent.runtime import AgentContext, UnavailableChat, UnavailableRetriever
from geekbot.domain import Label
from geekbot.tools.egress import EgressGate, GatedTool
from geekbot.tools.web import WebHit, WebSearch, WebSearchError

Q1 = "How long does Lantern keep chat history?"
Q2 = "And what about audit logs?"
PAGE = WebHit("A web page", "https://pages.example/lantern", "Lantern keeps logs for a year.")


class Backend:
    """A search backend that records every query and answers with ``hits`` (or fails)."""

    def __init__(self, hits: list[WebHit] | None = None, error: Exception | None = None) -> None:
        self.queries: list[str] = []
        self.hits = [PAGE] if hits is None else hits
        self.error = error

    def search(self, query: str, limit: int) -> list[WebHit]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.hits[:limit]


def gate(backend: Backend, *, enabled: bool = True) -> GatedTool:
    return EgressGate(enabled=enabled).wrap(WebSearch(backend, limit=5))


class Setup:
    """One conversation's fakes, its database file, and a way to ask and to answer the gate."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        backend: Backend | None = None,
        web: bool = True,
        chunks: tuple[Any, ...] | None = None,
    ) -> None:
        self.db = tmp_path / THREADS_DB
        self.backend = backend or Backend()
        self.retriever = ScriptedRetriever(*(chunks or ()))
        if chunks == ():  # ScriptedRetriever(): its default chunks; here there must be none
            self.retriever.chunks = ()
        self.chat = ScriptedChat()
        self.condenser = ScriptedCondenser()
        self.context = AgentContext(
            retriever=self.retriever,
            llm=self.chat,
            model="fake-model:1b",
            top_k=3,
            context_budget_tokens=3000,
            condenser=self.condenser,
            web=gate(self.backend) if web else None,
        )

    def ask(
        self,
        question: str,
        *,
        thread: str | None = "t1",
        web: bool = False,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        answered = asyncio.run(
            run_turn(
                question,
                context=self.context,
                thread=thread,
                threads_db=self.db,
                web=web,
                resume=resume,
            )
        )
        return answered.final

    def first_turn_then_a_tainted_follow_up(self) -> dict[str, Any]:
        """Turn one searches the web (so the thread carries ``WEB`` and ``KB``); turn two is a
        follow-up the model rewrites: the gate must stop at it. Returns turn two's stopped state."""
        first = self.ask(Q1, web=True)
        assert INTERRUPT_KEY not in first
        stopped = self.ask(Q2, web=True)
        assert INTERRUPT_KEY in stopped
        return stopped


# --- the web is off --------------------------------------------------------------------------


def test_a_turn_that_does_not_ask_for_the_web_never_touches_it(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    final = setup.ask(Q1)
    assert setup.backend.queries == []
    assert final["egress_log"] == []
    assert all(source.get("origin", "KB") == "KB" for source in final["sources"])


def test_with_no_web_tool_a_request_for_a_search_is_blocked_and_the_answer_goes_on(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path, web=False)
    final = setup.ask(Q1, web=True)
    (record,) = final["egress_log"]
    assert (record["decision"], record["tainted"]) == ("blocked", False)
    assert "off" in record["reason"]
    assert final["answer"] and final["web_query"] == ""
    assert Label.WEB.value not in final["answer_labels"]


def test_a_gate_switched_off_blocks_even_the_users_own_words(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.context = replace(setup.context, web=gate(setup.backend, enabled=False))
    final = setup.ask(Q1, web=True)
    assert setup.backend.queries == []
    assert [r["decision"] for r in final["egress_log"]] == ["blocked"]


# --- the user's own words --------------------------------------------------------------------


def test_the_users_own_question_is_searched_for_without_a_prompt(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    final = setup.ask(Q1, web=True)
    assert INTERRUPT_KEY not in final
    assert setup.backend.queries == [Q1]
    (record,) = final["egress_log"]
    assert (record["decision"], record["tainted"], record["query"]) == ("allowed", False, Q1)


def test_the_results_follow_the_documents_and_label_the_answer_web(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    final = setup.ask(Q1, web=True)
    kb_count = len(setup.retriever.chunks)
    web_sources = [s for s in final["sources"] if s.get("origin") == "WEB"]
    assert [s["n"] for s in final["sources"]] == list(range(1, kb_count + 2))
    (web,) = web_sources
    assert web["n"] == kb_count + 1
    assert (web["path"], web["section"], web["title"]) == (PAGE.url, "pages.example", PAGE.title)
    assert web["chunk_id"].startswith("web:")
    assert final["answer_labels"] == ["KB", "USER", "WEB"]
    assert "WEB" in final["thread_taint"] and "KB" in final["thread_taint"]
    prompt = str(setup.chat.calls[0][1].content)
    assert f"[{kb_count + 1}] web | A web page | pages.example" in prompt
    assert "> Lantern keeps logs for a year." in prompt  # data-marked like every source
    assert "retrieval_ms" in final and "web" in final["retrieval_ms"]


def test_a_remembered_web_source_keeps_its_origin_and_address(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1, web=True)
    saved = asyncio.run(read_thread(setup.db, "t1")) or {}
    sources = saved["history"][-1]["sources"]
    web = [s for s in sources if s.get("origin") == "WEB"]
    assert len(web) == 1 and web[0]["path"] == PAGE.url
    assert all(
        "origin" not in s for s in sources if s not in web
    )  # a document's record is as before
    assert "text" not in web[0]


def test_the_request_for_a_search_does_not_carry_over_to_the_next_turn(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1, web=True)
    setup.ask("Is there a mobile app?", web=False)
    assert setup.backend.queries == [Q1]


def test_begin_turn_moves_the_request_to_the_turns_own_switch_and_clears_it() -> None:
    update = begin_turn(
        {"question": "q", "requested_web": True, "web_query": "old", "egress_log": [{}]}
    )  # type: ignore[typeddict-item]
    assert update["use_web"] is True and update["requested_web"] is False
    assert (update["web_query"], update["web_labels"], update["web_approved"]) == ("", [], "")
    assert (update["egress_log"], update["web_error"]) == ([], "")
    assert begin_turn({"question": "q"})["use_web"] is False


def test_the_state_stays_plain_data_a_checkpointer_could_save(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1, web=True)
    saved = asyncio.run(read_thread(setup.db, "t1")) or {}
    assert json.loads(json.dumps(saved)) == saved


# --- the tainted follow-up: the gate asks -----------------------------------------------------


def test_a_rewritten_query_in_a_tainted_thread_stops_at_the_gate_and_sends_nothing(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    stopped = setup.first_turn_then_a_tainted_follow_up()
    assert setup.backend.queries == [Q1]  # turn two sent nothing yet
    question = interrupt_question(stopped)
    assert question is not None
    assert question["kind"] == APPROVAL_KIND and question["query"] == REWRITE
    assert question["question"] == Q2 and question["tool"] == "web_search"
    assert {"KB", "WEB"} <= set(question["labels"])
    assert "answer" not in stopped or stopped["answer"] == ""  # no answer was written


def test_the_stopped_thread_shows_its_question_and_resuming_clears_it(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    info = asyncio.run(read_thread_info(setup.db, "t1"))
    assert info is not None and info.pending is not None
    assert info.pending["query"] == REWRITE
    setup.ask("", resume={"action": "deny", "query": ""})
    after = asyncio.run(read_thread_info(setup.db, "t1"))
    assert after is not None and after.pending is None


def test_approving_sends_exactly_the_text_that_was_shown(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    searches_before = len(setup.retriever.calls)
    final = setup.ask("", resume={"action": "approve", "query": "ignored for approve"})
    assert setup.backend.queries == [Q1, REWRITE]
    assert (
        len(setup.retriever.calls) == searches_before
    )  # the knowledge base was not searched again
    record = final["egress_log"][-1]
    assert (record["decision"], record["tainted"], record["query"]) == ("approved", True, REWRITE)
    assert final["turn"] == 2 and final["answer"]
    assert any(s.get("origin") == "WEB" for s in final["sources"])


def test_editing_sends_the_users_own_text_and_not_the_rewrite(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    final = setup.ask("", resume={"action": "edit", "query": "  my   own\nwords  "})
    assert setup.backend.queries == [Q1, "my own words"]
    record = final["egress_log"][-1]
    assert (record["decision"], record["query"]) == ("edited", "my own words")
    assert final["web_labels"] == ["USER"]


def test_denying_sends_nothing_and_the_answer_comes_from_the_documents_alone(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    final = setup.ask("", resume={"action": "deny", "query": ""})
    assert setup.backend.queries == [Q1]
    assert final["egress_log"][-1]["decision"] == "denied"
    assert final["web_query"] == ""
    assert all(s.get("origin", "KB") == "KB" for s in final["sources"])
    assert final["answer"] and final["turn"] == 2


@pytest.mark.parametrize(
    "reply",
    [
        {"action": "yes"},
        {"action": "APPROVE"},
        {"action": None},
        {"action": "approve", "query": 5},
        {"query": REWRITE},
        {"action": "edit", "query": ""},
        {"action": "edit", "query": "x" * 400},
    ],
)
def test_an_answer_the_gate_cannot_read_is_a_denial(tmp_path: Path, reply: dict[str, Any]) -> None:
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    final = setup.ask("", resume=reply)
    assert setup.backend.queries == [Q1]
    assert final["egress_log"][-1]["decision"] == "denied"


def test_an_empty_reply_is_not_consent_the_question_just_stays_open(tmp_path: Path) -> None:
    """LangGraph does not count an empty resume as an answer: the turn keeps waiting, and sends nothing."""
    setup = Setup(tmp_path)
    setup.first_turn_then_a_tainted_follow_up()
    again = setup.ask("", resume={})
    assert interrupt_question(again) is not None
    assert setup.backend.queries == [Q1]


def test_a_follow_up_the_model_left_as_the_user_typed_it_is_the_users_own_words(
    tmp_path: Path,
) -> None:
    """The rewrite decides: the same text as the question carries no taint, whatever the thread did."""
    setup = Setup(tmp_path)
    setup.ask(Q1, web=True)
    setup.condenser.replies = [CondenseOutput(standalone_question=Q2)]
    final = setup.ask(Q2, web=True)
    assert INTERRUPT_KEY not in final
    assert setup.backend.queries == [Q1, Q2]
    assert final["egress_log"][-1]["decision"] == "allowed"


# --- the gate node on its own -----------------------------------------------------------------


def runtime(context: AgentContext) -> Any:
    return SimpleNamespace(context=context, stream_writer=lambda event: None)


def state(**extra: Any) -> Any:
    base = {
        "question": Q2,
        "standalone_question": REWRITE,
        "thread_taint": ["USER", "KB"],
        "sources": [],
        "egress_log": [],
    }
    base.update(extra)
    return base


def context(**extra: Any) -> AgentContext:
    base: dict[str, Any] = {
        "retriever": ScriptedRetriever(),
        "llm": ScriptedChat(),
        "model": "m",
        "top_k": 1,
        "context_budget_tokens": 100,
        "web": gate(Backend()),
    }
    base.update(extra)
    return AgentContext(**base)


def test_nobody_to_ask_is_a_denial_not_a_pass() -> None:
    update = egress_gate(state(), runtime(context(approvals=False)))
    assert update["web_query"] == ""
    (record,) = update["egress_log"]
    assert (record["decision"], record["tainted"]) == ("denied", True)
    assert "no one" in record["reason"]


def test_a_query_that_cannot_be_sent_is_blocked_with_the_reason() -> None:
    update = egress_gate(state(standalone_question="x" * 500), runtime(context()))
    assert update["web_query"] == ""
    (record,) = update["egress_log"]
    assert record["decision"] == "blocked" and "limit" in record["reason"]
    assert len(record["query"]) <= 300


def test_a_rewrite_in_a_thread_that_carried_nothing_from_outside_is_still_the_users_own() -> None:
    """The thread's taint is only ``USER``: the rewrite had nothing to be built from but their words."""
    update = egress_gate(state(thread_taint=["USER"]), runtime(context(approvals=False)))
    assert update["web_query"] == REWRITE
    assert update["egress_log"][0]["decision"] == "allowed"


def test_the_two_edges_after_the_search_and_after_the_gate() -> None:
    assert after_search({"use_web": True, "sources": []}) == "egress_gate"
    assert after_search({"use_web": False, "sources": []}) == "no_context"
    assert after_search({"sources": [{"n": 1}]}) == "generate"  # type: ignore[typeddict-item]
    assert route_after_gate({"web_query": "q", "sources": []}) == "web"
    assert route_after_gate({"web_query": "", "sources": []}) == "no_context"
    assert route_after_gate({"web_query": "", "sources": [{"n": 1}]}) == "generate"  # type: ignore[typeddict-item]


def test_the_web_step_asks_the_rule_again_and_refuses_a_call_the_gate_did_not_allow() -> None:
    """Defence in depth: a state that holds a tainted query and no approval sends nothing."""
    backend = Backend()
    ctx = context(web=gate(backend))
    update = web_search(
        state(web_query=REWRITE, web_labels=["KB", "USER"], web_approved=""), runtime(ctx)
    )
    assert backend.queries == []
    assert update["web_query"] == ""
    assert update["egress_log"][-1]["decision"] == "blocked"
    sent = web_search(
        state(web_query=REWRITE, web_labels=["KB", "USER"], web_approved=REWRITE), runtime(ctx)
    )
    assert backend.queries == [REWRITE] and len(sent["sources"]) == 1


def test_the_web_step_does_nothing_without_a_query_or_a_tool() -> None:
    assert web_search(state(web_query=""), runtime(context())) == {}
    assert web_search(state(web_query="q"), runtime(context(web=None))) == {}


# --- the search itself goes wrong -------------------------------------------------------------


def test_a_failed_search_is_reported_and_the_turn_still_answers_from_the_documents(
    tmp_path: Path,
) -> None:
    setup = Setup(
        tmp_path, backend=Backend(error=WebSearchError("the search service answered HTTP 500"))
    )
    final = setup.ask(Q1, web=True)
    assert "HTTP 500" in final["web_error"]
    assert [r["decision"] for r in final["egress_log"]] == ["allowed", "failed"]
    assert final["answer"] and all(s.get("origin", "KB") == "KB" for s in final["sources"])
    assert len(setup.backend.queries) == 1  # not retried: a retry would send the text twice


def test_a_search_with_no_results_changes_nothing(tmp_path: Path) -> None:
    setup = Setup(tmp_path, backend=Backend(hits=[]))
    final = setup.ask(Q1, web=True)
    assert all(s.get("origin", "KB") == "KB" for s in final["sources"])
    assert final["answer_labels"] == ["KB", "USER"]


def test_with_nothing_in_the_documents_the_web_results_alone_are_the_sources(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path, chunks=())
    final = setup.ask(Q1, web=True)
    assert [s["n"] for s in final["sources"]] == [1]
    assert final["sources"][0]["origin"] == "WEB"
    assert final["answer_labels"] == ["USER", "WEB"]
    assert final["thread_taint"] == ["USER", "WEB"]


def test_with_nothing_anywhere_the_application_says_so(tmp_path: Path) -> None:
    setup = Setup(tmp_path, chunks=(), backend=Backend(hits=[]))
    final = setup.ask(Q1, web=True)
    assert final["abstained"] is True and final["used"] == 0
    assert setup.chat.calls == []  # a model is never asked to answer without sources


# --- modes and no thread ----------------------------------------------------------------------


def test_a_csv_conversation_never_reaches_the_web_tool(tmp_path: Path) -> None:
    backend = Backend()
    csv = SimpleNamespace(
        ask=lambda question, /: SimpleNamespace(
            markdown="The total is 3.", status="answered", model="m", seconds=0.1, model_load_ms=0
        )
    )
    ctx = AgentContext(
        retriever=UnavailableRetriever(),
        llm=UnavailableChat(),
        model="m",
        top_k=1,
        context_budget_tokens=100,
        csv=csv,  # type: ignore[arg-type]
        web=gate(backend),
    )
    answered = asyncio.run(
        run_turn(
            "total?",
            context=ctx,
            thread="t1",
            threads_db=tmp_path / THREADS_DB,
            mode="csv",
            web=True,
        )
    )
    assert backend.queries == []
    assert answered.final["egress_log"] == []
    assert answered.final["answer"].startswith("The total")


def test_without_a_thread_nobody_can_be_asked_but_the_users_own_words_still_go(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    final = setup.ask(Q1, thread=None, web=True)
    assert INTERRUPT_KEY not in final
    assert setup.backend.queries == [Q1]
