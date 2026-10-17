"""Tests for step S3.4: conversation memory (threads), the rewrite of follow-ups, taint bookkeeping.

The graph is run for real, several turns in a row, with a real ``AsyncSqliteSaver`` on a temporary file
(that is the integration point: what is saved after a turn must be enough to start the next one) and
fakes for everything that would need a model: the retriever, the chat model and the rewrite call.
Each ``asyncio.run`` below is one ``geekbot ask --thread`` invocation: a new event loop, a new
database connection, a new graph, and only the file in common.
"""

import asyncio
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from langchain_core.messages import AIMessageChunk, BaseMessage

from geekbot.agent.condense import CondenseOutput
from geekbot.agent.nodes import NO_CONTEXT_ANSWER, begin_turn, needs_condense, remember
from geekbot.agent.packing import HISTORY_HEADING
from geekbot.agent.runner import (
    ThreadNameError,
    check_thread_name,
    delete_thread,
    read_thread,
    run_turn,
    thread_config,
)
from geekbot.agent.runtime import AgentContext
from geekbot.domain import Chunk, ScoredChunk
from geekbot.llm.chat import ChatError
from geekbot.retrieval import RetrievalResult

Q1 = "How long does Lantern keep chat history?"
Q2 = "And what about audit logs?"
REWRITE = "How long does Lantern keep audit logs?"


def scored(n: int, text: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"chunk{n}",
        doc_id="doc",
        path="/docs/handbook.md",
        title="Handbook",
        content_hash="h" * 64,
        occurrence=0,
        ordinal=n,
        char_start=0,
        char_end=len(text),
        page=None,
        section_path=(),
        header="",
        text=text,
    )
    return ScoredChunk(chunk=chunk, score=1.0 / n)


class ScriptedRetriever:
    """Returns ``chunks`` (change the attribute between turns) and remembers every query."""

    def __init__(self) -> None:
        self.chunks: tuple[ScoredChunk, ...] = (
            scored(1, "Chat history is kept for 90 days."),
            scored(2, "Audit logs are kept for a year."),
        )
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> RetrievalResult:
        self.calls.append((query, k))
        return RetrievalResult(chunks=self.chunks[:k], timings_ms={"total": 1.0})


class ScriptedChat:
    """Streams the replies in order (an exception is raised); the last one repeats."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies) or ["Ninety days [1]."]
        self.calls: list[list[BaseMessage]] = []

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        yield AIMessageChunk(content=reply)


class ScriptedCondenser:
    """Stands in for the structured rewrite call: replies in order, the last one repeats."""

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies) or [CondenseOutput(standalone_question=REWRITE)]
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: list[BaseMessage], /) -> Any:
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


class Setup:
    """One conversation's fakes, its database file, and a way to ask a question in a thread."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        chat: ScriptedChat | None = None,
        condenser: ScriptedCondenser | None = None,
        use_condenser: bool = True,
        keep: int = 40,
    ) -> None:
        self.db = tmp_path / "threads.db"
        self.retriever = ScriptedRetriever()
        self.chat = chat or ScriptedChat()
        self.condenser = condenser or ScriptedCondenser()
        self.context = AgentContext(
            retriever=self.retriever,
            llm=self.chat,
            model="fake-model:1b",
            top_k=3,
            context_budget_tokens=3000,
            condenser=self.condenser if use_condenser else None,
            history_budget_tokens=1000,
            history_keep_messages=keep,
        )

    def ask(self, question: str, thread: str | None = "t1") -> dict[str, Any]:
        answered = asyncio.run(
            run_turn(question, context=self.context, thread=thread, threads_db=self.db)
        )
        return answered.final


# --- the first turn and the follow-up --------------------------------------------------------


def test_the_first_question_of_a_thread_makes_no_model_call_before_retrieval(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    final = setup.ask(Q1)
    assert setup.condenser.calls == []
    assert setup.retriever.calls == [(Q1, 3)]
    assert final["standalone_question"] == Q1
    assert final["answer"] == "Ninety days [1]."


def test_after_a_turn_the_thread_remembers_the_question_the_answer_and_their_labels(
    tmp_path: Path,
) -> None:
    final = Setup(tmp_path).ask(Q1)
    assert final["turn"] == 1
    assert final["history"] == [
        {"id": "t1-user", "role": "user", "content": Q1, "labels": ["USER"]},
        {
            "id": "t1-assistant",
            "role": "assistant",
            "content": "Ninety days [1].",
            "labels": ["KB", "USER"],
        },
    ]
    assert final["thread_taint"] == ["KB", "USER"]


def test_a_follow_up_is_rewritten_and_retrieval_searches_for_the_rewrite(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1)
    final = setup.ask(Q2)
    assert len(setup.condenser.calls) == 1  # one rewrite call, on the follow-up only
    assert setup.retriever.calls == [(Q1, 3), (REWRITE, 3)]
    assert final["question"] == Q2
    assert final["standalone_question"] == REWRITE
    assert final["condense_seconds"] >= 0
    assert final["condense_error"] == ""
    assert final["turn"] == 2
    assert [m["id"] for m in final["history"]] == [
        "t1-user",
        "t1-assistant",
        "t2-user",
        "t2-assistant",
    ]


def test_the_rewrite_call_sees_the_earlier_turn_and_the_last_message(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1)
    setup.ask(Q2)
    (messages,) = setup.condenser.calls
    human = str(messages[1].content)
    assert f"> User: {Q1}" in human
    assert "> Assistant: Ninety days." in human  # the old [1] mark is gone
    assert human.endswith(f"Last message: {Q2}")


def test_the_answer_prompt_has_the_earlier_turn_but_the_question_as_the_user_typed_it(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1)
    setup.ask(Q2)
    first, second = (str(call[1].content) for call in setup.chat.calls)
    assert HISTORY_HEADING not in first
    assert second.startswith(HISTORY_HEADING)
    assert f"> User: {Q1}" in second
    assert second.index("> User:") < second.index("Sources:")  # earlier chat, then sources
    assert second.endswith(f"Question: {Q2}")  # not the rewrite: the history gives the context


def test_the_model_is_told_how_many_earlier_messages_went_into_the_prompt(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    assert setup.ask(Q1)["history_used"] == 0
    assert setup.ask(Q2)["history_used"] == 2


def test_a_second_thread_does_not_see_the_first(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1, thread="alpha")
    final = setup.ask(Q2, thread="beta")
    assert setup.condenser.calls == []  # beta's first turn
    assert final["turn"] == 1
    assert HISTORY_HEADING not in str(setup.chat.calls[-1][1].content)


def test_without_a_thread_nothing_is_saved_and_nothing_remembered(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1, thread=None)
    final = setup.ask(Q2, thread=None)
    assert not setup.db.exists()
    assert setup.condenser.calls == []
    assert HISTORY_HEADING not in str(setup.chat.calls[-1][1].content)
    assert final["turn"] == 1


# --- a turn must not inherit the last turn's results -----------------------------------------


def test_a_turn_that_finds_nothing_does_not_show_the_previous_turns_sources(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1)
    setup.retriever.chunks = ()
    final = setup.ask(Q2)
    assert final["sources"] == []
    assert final["answer"] == NO_CONTEXT_ANSWER
    assert final["raw_answer"] == ""
    assert (final["used"], final["history_used"]) == (0, 0)
    assert final["cited"] == []
    assert final["generation"] == {}
    assert final["answer_labels"] == []
    assert len(setup.chat.calls) == 1  # the model was not called for the second turn


def test_the_application_s_own_sentence_is_remembered_with_no_labels(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    setup.ask(Q1)
    setup.retriever.chunks = ()
    final = setup.ask(Q2)
    assert final["history"][-1] == {
        "id": "t2-assistant",
        "role": "assistant",
        "content": NO_CONTEXT_ANSWER,
        "labels": [],
    }


def test_the_threads_taint_never_goes_away(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    assert setup.ask(Q1)["thread_taint"] == ["KB", "USER"]
    setup.retriever.chunks = ()
    assert setup.ask(Q2)["thread_taint"] == ["KB", "USER"]  # a clean turn does not clean the thread


# --- a turn that fails ------------------------------------------------------------------------


def test_a_turn_that_fails_is_not_remembered_and_the_next_one_can_go_on(tmp_path: Path) -> None:
    chat = ScriptedChat("Ninety days [1].", ConnectionError("down"), "A year [2].")
    setup = Setup(tmp_path, chat=chat)
    setup.ask(Q1)
    with pytest.raises(ChatError, match="ConnectionError: down"):
        setup.ask(Q2)
    saved = asyncio.run(read_thread(setup.db, "t1"))
    assert saved is not None
    assert saved["turn"] == 1
    assert [m["id"] for m in saved["history"]] == ["t1-user", "t1-assistant"]
    final = setup.ask(Q2)  # asking again works, and the numbering has no gap
    assert final["answer"] == "A year [2]."
    assert [m["id"] for m in final["history"]][-2:] == ["t2-user", "t2-assistant"]


# --- the rewrite going wrong -----------------------------------------------------------------


def test_a_rewrite_that_fails_falls_back_to_the_users_words_and_says_why(tmp_path: Path) -> None:
    condenser = ScriptedCondenser(ValueError("Invalid json output:\n{broken\n  because"))
    setup = Setup(tmp_path, condenser=condenser)
    setup.ask(Q1)
    final = setup.ask(Q2)
    assert setup.retriever.calls[-1] == (Q2, 3)
    assert final["standalone_question"] == Q2
    assert final["condense_error"] == "ValueError: Invalid json output: {broken because"
    assert final["answer"]  # the turn still produced an answer


def test_a_rewrite_of_the_wrong_type_falls_back_too(tmp_path: Path) -> None:
    setup = Setup(tmp_path, condenser=ScriptedCondenser({"standalone_question": "a dict"}))
    setup.ask(Q1)
    final = setup.ask(Q2)
    assert final["standalone_question"] == Q2
    assert final["condense_error"].startswith("TypeError: expected a CondenseOutput")


def test_an_empty_rewrite_falls_back_without_an_error(tmp_path: Path) -> None:
    setup = Setup(tmp_path, condenser=ScriptedCondenser(CondenseOutput(standalone_question="  ")))
    setup.ask(Q1)
    final = setup.ask(Q2)
    assert final["standalone_question"] == Q2
    assert final["condense_error"] == ""


def test_with_no_condenser_a_follow_up_is_searched_as_typed(tmp_path: Path) -> None:
    setup = Setup(tmp_path, use_condenser=False)
    setup.ask(Q1)
    final = setup.ask(Q2)
    assert setup.retriever.calls[-1] == (Q2, 3)
    assert final["standalone_question"] == Q2
    assert final["condense_seconds"] == 0.0


# --- how much a thread keeps -----------------------------------------------------------------


def test_a_thread_keeps_only_whole_recent_turns_and_the_taint_outlives_the_forgotten_ones(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path, keep=4)
    for question in (Q1, Q2, "And backups?"):
        final = setup.ask(question)
    assert [m["id"] for m in final["history"]] == [
        "t2-user",
        "t2-assistant",
        "t3-user",
        "t3-assistant",
    ]
    assert final["turn"] == 3
    assert final["thread_taint"] == ["KB", "USER"]


def test_an_odd_limit_still_keeps_whole_turns(tmp_path: Path) -> None:
    setup = Setup(tmp_path, keep=5)
    for question in (Q1, Q2, "And backups?"):
        final = setup.ask(question)
    assert len(final["history"]) == 4


# --- what is saved is plain data -------------------------------------------------------------


def test_the_saved_state_is_plain_data(tmp_path: Path) -> None:
    final = Setup(tmp_path).ask(Q1)
    assert json.loads(json.dumps(final)) == final


# --- reading and deleting a thread -----------------------------------------------------------


def test_read_thread_returns_the_saved_state_and_none_for_a_stranger(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    assert asyncio.run(read_thread(setup.db, "t1")) is None
    assert not setup.db.exists()  # looking never creates the database
    setup.ask(Q1)
    saved = asyncio.run(read_thread(setup.db, "t1"))
    assert saved is not None
    assert saved["turn"] == 1
    assert saved["history"][0]["content"] == Q1
    assert asyncio.run(read_thread(setup.db, "other")) is None


def test_delete_thread_forgets_everything_and_says_whether_there_was_something(
    tmp_path: Path,
) -> None:
    setup = Setup(tmp_path)
    assert asyncio.run(delete_thread(setup.db, "t1")) is False  # no database yet
    setup.ask(Q1)
    setup.ask(Q1, thread="keep")
    assert asyncio.run(delete_thread(setup.db, "t1")) is True
    assert asyncio.run(read_thread(setup.db, "t1")) is None
    assert asyncio.run(delete_thread(setup.db, "t1")) is False
    assert asyncio.run(read_thread(setup.db, "keep")) is not None  # another thread is untouched
    final = setup.ask(Q2)  # a deleted thread starts from nothing
    assert final["turn"] == 1
    assert setup.condenser.calls == []


@pytest.mark.parametrize("name", ["t1", "my-thread_2.v3", "A", "9", "x" * 64])
def test_ordinary_thread_names_are_accepted(name: str) -> None:
    assert check_thread_name(name) == name
    assert thread_config(name)["configurable"]["thread_id"] == name


@pytest.mark.parametrize("name", ["", " ", "../x", "a/b", "-x", ".x", "x" * 65, "a b", "é", "a\n"])
def test_other_thread_names_are_refused(name: str) -> None:
    with pytest.raises(ThreadNameError, match="not a usable thread name"):
        check_thread_name(name)


def test_a_thread_needs_a_database_to_live_in(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    with pytest.raises(ValueError, match="threads database"):
        asyncio.run(run_turn(Q1, context=setup.context, thread="t1"))


# --- the nodes on their own ------------------------------------------------------------------


def runtime(keep: int = 40) -> Any:
    return SimpleNamespace(context=SimpleNamespace(history_keep_messages=keep))


def test_begin_turn_clears_what_belongs_to_one_turn_and_leaves_the_memory_alone() -> None:
    state: Any = {
        "question": Q2,
        "standalone_question": "old rewrite",
        "sources": [{"n": 1}],
        "answer": "old",
        "raw_answer": "old raw",
        "cited": [1],
        "thinking": "old thoughts",
        "generation": {"seconds": 9.0},
        "history": [{"id": "t1-user"}],
        "thread_taint": ["KB", "USER"],
        "turn": 1,
    }
    update = begin_turn(state)
    assert update["standalone_question"] == Q2  # the user's words, until a rewrite replaces them
    assert update["sources"] == []
    assert (update["answer"], update["raw_answer"], update["thinking"]) == ("", "", "")
    assert (update["cited"], update["generation"], update["used"]) == ([], {}, 0)
    assert {"question", "history", "thread_taint", "turn"}.isdisjoint(update)


def test_a_turn_is_a_follow_up_exactly_when_the_thread_has_history() -> None:
    assert needs_condense({"question": "q"}) == "retrieve"
    assert needs_condense({"question": "q", "history": []}) == "retrieve"
    assert needs_condense({"question": "q", "history": [{"id": "x"}]}) == "condense"  # type: ignore[typeddict-item]


LABEL_SETS = st.lists(
    st.sets(st.sampled_from(["KB", "WEB", "TOOL", "UNKNOWN"])).map(sorted), max_size=12
)


@given(LABEL_SETS, st.integers(min_value=2, max_value=10))
def test_the_taint_only_grows_and_the_memory_is_always_whole_turns(
    label_sets: list[list[str]], keep: int
) -> None:
    state: Any = {"question": "q", "answer": "a"}
    seen: set[str] = set()
    for names in label_sets:
        state["answer_labels"] = names
        update = remember(state, runtime(keep))
        assert set(update["thread_taint"]) >= set(state.get("thread_taint", []))
        state.update(update)
        seen |= set(names)
        assert set(state["thread_taint"]) == seen | {"USER"}
        history = state["history"]
        assert len(history) % 2 == 0
        assert len(history) <= max(2, keep)
        assert [m["role"] for m in history] == ["user", "assistant"] * (len(history) // 2)
    assert state.get("turn", 0) == len(label_sets)


def test_remember_numbers_the_turns_by_what_was_completed_not_by_what_was_started() -> None:
    state: Any = {"question": "q", "answer": "a", "answer_labels": ["KB"], "turn": 4}
    update = remember(state, runtime())
    assert update["turn"] == 5
    assert [m["id"] for m in update["history"]] == ["t5-user", "t5-assistant"]


def test_the_labels_of_an_answer_are_stored_sorted_and_without_repeats() -> None:
    state: Any = {"question": "q", "answer": "a", "answer_labels": ["USER", "KB", "KB"]}
    assert remember(state, runtime())["history"][1]["labels"] == ["KB", "USER"]


def test_a_stream_of_questions_through_one_event_loop_works_too(tmp_path: Path) -> None:
    """The server (S4.1) keeps one loop for many turns; two turns in one ``asyncio.run`` must work."""
    setup = Setup(tmp_path)

    async def two_turns() -> list[Sequence[str]]:
        seen = []
        for question in (Q1, Q2):
            answered = await run_turn(
                question, context=setup.context, thread="t1", threads_db=setup.db
            )
            seen.append([m["id"] for m in answered.final["history"]])
        return seen

    first, second = asyncio.run(two_turns())
    assert first == ["t1-user", "t1-assistant"]
    assert second == ["t1-user", "t1-assistant", "t2-user", "t2-assistant"]
