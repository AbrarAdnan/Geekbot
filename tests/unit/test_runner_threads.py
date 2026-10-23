"""Tests for step S4.2's additions to the runner: a streamed turn, reading a thread, listing threads.

The graph runs for real with a real ``AsyncSqliteSaver`` on a temporary file; the retriever, the chat
model and the rewrite call are the scripted fakes of ``chatkit``.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from chatkit import ScriptedChat, ScriptedCondenser, ScriptedRetriever

from geekbot.agent.runner import ThreadInfo, list_threads, read_thread_info, stream_turn
from geekbot.agent.runtime import AgentContext
from geekbot.llm.chat import ChatError

Q1 = "How long does Lantern keep chat history?"
Q2 = "And what about audit logs?"


def context(chat: ScriptedChat | None = None) -> AgentContext:
    return AgentContext(
        retriever=ScriptedRetriever(),
        llm=chat or ScriptedChat(),
        model="fake-model:1b",
        top_k=3,
        context_budget_tokens=3000,
        condenser=ScriptedCondenser(),
    )


async def events_of(
    question: str, ctx: AgentContext, db: Path, thread: str = "t1"
) -> list[tuple[str, Any]]:
    return [
        event async for event in stream_turn(question, context=ctx, thread=thread, threads_db=db)
    ]


# --- stream_turn -----------------------------------------------------------------------------


def test_a_streamed_turn_yields_the_pieces_and_ends_with_the_finished_state(tmp_path: Path) -> None:
    chat = ScriptedChat(["Ninety", " days", " [1]."])
    events = asyncio.run(events_of(Q1, context(chat), tmp_path / "t.db"))
    kinds = [kind for kind, _ in events]
    assert set(kinds) == {"custom", "values"}
    assert kinds[-1] == "values"
    custom = [payload for kind, payload in events if kind == "custom"]
    # Since S3.6 the turn also announces the steps it reaches (a plain retriever has no ranking step).
    assert {event["event"] for event in custom} == {"step", "text"}
    assert [e["name"] for e in custom if e["event"] == "step"] == ["searching", "writing"]
    pieces = [event for event in custom if event["event"] == "text"]
    # The pieces may be cut differently from the model's (the cleaning holds a "[" back until it is
    # decided), but together they are the answer.
    assert "".join(piece["text"] for piece in pieces) == "Ninety days [1]."
    final = events[-1][1]
    assert final["answer"] == "Ninety days [1]."
    assert final["turn"] == 1
    assert final["cited"] == [1]


def test_the_pieces_arrive_before_the_turn_is_over(tmp_path: Path) -> None:
    """Not after: the first piece comes before the state that holds the finished answer."""
    chat = ScriptedChat(["Ninety", " days [1]."])
    events = asyncio.run(events_of(Q1, context(chat), tmp_path / "t.db"))
    first_piece = next(
        i
        for i, (kind, payload) in enumerate(events)
        if kind == "custom" and payload["event"] == "text"
    )
    with_answer = next(
        i for i, (kind, state) in enumerate(events) if kind == "values" and state.get("answer")
    )
    assert first_piece < with_answer


def test_a_streamed_turn_is_saved_like_any_other(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    asyncio.run(events_of(Q1, context(), db))
    info = asyncio.run(read_thread_info(db, "t1"))
    assert isinstance(info, ThreadInfo)
    assert info.thread_id == "t1"
    assert info.values["turn"] == 1
    assert [m["role"] for m in info.values["history"]] == ["user", "assistant"]
    assert isinstance(info.updated_at, str) and info.updated_at


def test_a_second_streamed_turn_is_a_follow_up_in_the_same_thread(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    ctx = context()
    asyncio.run(events_of(Q1, ctx, db))
    events = asyncio.run(events_of(Q2, ctx, db))
    final = events[-1][1]
    assert final["turn"] == 2
    assert final["standalone_question"] != Q2  # the scripted rewrite was used
    assert len(final["history"]) == 4


def test_a_failing_model_raises_out_of_the_stream_and_the_turn_is_not_remembered(
    tmp_path: Path,
) -> None:
    db = tmp_path / "t.db"
    # Every try fails the same way: the refused connection is retried, then reported.
    ctx = context(ScriptedChat(ConnectionError("down")))
    with pytest.raises(ChatError, match="ConnectionError: down"):
        asyncio.run(events_of(Q1, ctx, db))
    info = asyncio.run(read_thread_info(db, "t1"))
    assert info is None or not info.values.get("history")


def test_stopping_early_closes_the_stream_and_the_next_question_still_works(
    tmp_path: Path,
) -> None:
    """A client that went away: the generator is closed after one event, nothing is remembered."""
    db = tmp_path / "t.db"
    ctx = context(ScriptedChat(["Ninety", " days [1]."]))

    async def walk_away() -> tuple[str, Any]:
        events = stream_turn(Q1, context=ctx, thread="t1", threads_db=db)
        first = await events.__anext__()
        await events.aclose()
        return first

    first = asyncio.run(walk_away())
    assert first[0] in {"custom", "values"}
    info = asyncio.run(read_thread_info(db, "t1"))
    assert info is None or not info.values.get("history")
    events = asyncio.run(events_of(Q1, ctx, db))
    assert events[-1][1]["turn"] == 1  # the interrupted turn left no gap and no half turn


# --- read_thread_info ------------------------------------------------------------------------


def test_reading_a_thread_that_is_not_there_gives_none_and_creates_nothing(tmp_path: Path) -> None:
    db = tmp_path / "none.db"
    assert asyncio.run(read_thread_info(db, "t1")) is None
    assert not db.exists()


def test_reading_a_thread_with_a_bad_name_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a usable thread name"):
        asyncio.run(read_thread_info(tmp_path / "t.db", "bad name"))


# --- list_threads ------------------------------------------------------------------------------


def test_listing_without_a_database_gives_nothing_and_creates_nothing(tmp_path: Path) -> None:
    db = tmp_path / "none.db"
    assert asyncio.run(list_threads(db)) == []
    assert not db.exists()


def test_threads_are_listed_newest_first_and_the_limit_applies(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    ctx = context()

    async def scenario() -> tuple[list[str], list[str], dict[str, int]]:
        for name in ("t-a", "t-b", "t-c", "t-a"):  # t-a is asked again last: it is the newest
            await events_of(Q1, ctx, db, thread=name)
        everything = await list_threads(db)
        two = await list_threads(db, limit=2)
        return (
            [info.thread_id for info in everything],
            [info.thread_id for info in two],
            {info.thread_id: info.values["turn"] for info in everything},
        )

    everything, two, turns = asyncio.run(scenario())
    assert everything == ["t-a", "t-c", "t-b"]
    assert two == ["t-a", "t-c"]
    assert turns == {"t-a": 2, "t-b": 1, "t-c": 1}


def test_rows_this_program_would_not_open_are_not_listed(tmp_path: Path) -> None:
    """A name that fails ``check_thread_name``, and a row of a subgraph, are somebody else's."""
    db = tmp_path / "t.db"
    asyncio.run(events_of(Q1, context(), db))
    with sqlite3.connect(db) as connection:  # the checkpointer's own table, as setup() makes it
        connection.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, ?, ?)",
            ("bad name!", "", "zzzz"),
        )
        connection.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id) VALUES (?, ?, ?)",
            ("t-sub", "child", "zzzz"),
        )
    connection.close()
    assert [info.thread_id for info in asyncio.run(list_threads(db))] == ["t1"]


def test_listing_never_changes_what_a_thread_remembers(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    asyncio.run(events_of(Q1, context(), db))
    before = asyncio.run(read_thread_info(db, "t1"))
    asyncio.run(list_threads(db))
    after = asyncio.run(read_thread_info(db, "t1"))
    assert before is not None and after is not None
    assert before.values["history"] == after.values["history"]
    assert before.updated_at == after.updated_at
