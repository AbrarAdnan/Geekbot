"""Running one turn of the answer graph, with or without a saved conversation (steps S3.4 and S4.2).

``geekbot ask`` is a synchronous command, but a conversation is saved by LangGraph's **async** SQLite
checkpointer (PLAN.md §6.3: ``AsyncSqliteSaver``), and an async checkpointer needs the graph to be run
with ``ainvoke`` / ``astream``. So the command hands one coroutine to ``asyncio.run`` and everything
async lives here: open the thread database, compile the graph with it, run the turn, close it again.
The nodes themselves stay plain synchronous functions; under ``astream`` LangGraph runs them in worker
threads, which keeps the event loop free (§6.3) and is what the server (S4.1) relies on too.

**The thread database** is one SQLite file under the data folder (``threads.db``), opened only while
the storage lock is held (the command opens the workspace first, the server's lifespan does), so the
rule "one process owns the storage" (§6.2) covers it. A thread is a name (``--thread t1``, or the id
the server hands out); it is the checkpointer's ``thread_id``, and everything the graph keeps in its
state for that thread (the messages, the labels, the taint) is saved after every step.

Without a thread there is no checkpointer and no file: the run starts from nothing, as before S3.4.

**For the server (S4.2)** there are three more functions. ``stream_turn`` runs a turn and hands every
event of the graph's stream to the caller as it happens (the server turns them into SSE frames).
``read_thread_info`` and ``list_threads`` read saved threads for the thread routes. Each opens the
database for the one call and closes it again, like the CLI does: a few milliseconds, against seconds
for the model, and it keeps this module free of any state that could outlive a request.
"""

import contextlib
import re
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from geekbot.agent.graph import build_graph
from geekbot.agent.nodes import APPROVAL_KIND
from geekbot.agent.render import StreamCollector, StreamedAnswer
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = [
    "INTERRUPT_KEY",
    "THREADS_DB",
    "ThreadInfo",
    "ThreadNameError",
    "check_thread_name",
    "delete_thread",
    "interrupt_question",
    "list_threads",
    "pending_approval",
    "read_thread",
    "read_thread_info",
    "run_turn",
    "stream_turn",
    "thread_config",
]

# The key LangGraph adds to the last ``values`` chunk of a turn that stopped at ``interrupt()``: the
# turn is not finished, it is waiting for an answer (the egress gate's question, S5.3).
INTERRUPT_KEY: Final = "__interrupt__"

THREADS_DB: Final = "threads.db"  # the file name under the data folder

# A thread name ends up in a file (as a row key) and in the terminal: letters, digits, dots, dashes
# and underscores, starting with a letter or digit, at most 64 characters.
_THREAD_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# The newest threads first. A checkpoint id is time-ordered (LangGraph builds it from a UUID v6), so the
# greatest id of a thread is its latest checkpoint. ``checkpoint_ns = ''`` is the graph itself, not a
# subgraph. The table and its columns are the checkpointer's own (``AsyncSqliteSaver.setup``); a test
# runs this against the installed version, so a change in a later release fails loudly.
_NEWEST_THREADS: Final = (
    "SELECT thread_id FROM checkpoints WHERE checkpoint_ns = '' "
    "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC LIMIT ?"
)

Graph = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


class ThreadNameError(ValueError):
    """The thread name is not one this program accepts."""


def check_thread_name(name: str) -> str:
    """``name`` if it is a usable thread name, else ``ThreadNameError`` saying what is allowed."""
    if not _THREAD_NAME.fullmatch(name):
        raise ThreadNameError(
            f"{name!r} is not a usable thread name: use letters, digits, dots, dashes and "
            "underscores, starting with a letter or digit, at most 64 characters"
        )
    return name


def thread_config(thread: str) -> RunnableConfig:
    """The config that tells the checkpointer which conversation a run belongs to."""
    return RunnableConfig(configurable={"thread_id": check_thread_name(thread)})


async def _open_graph(
    stack: contextlib.AsyncExitStack, thread: str | None, threads_db: Path | None
) -> tuple[Graph, RunnableConfig | None]:
    """The compiled graph for one run, and the config that names its thread (``None`` without one).

    With a thread the checkpoint database is opened on ``stack``, so it is closed when the caller's
    ``async with`` ends, whatever happens in between.
    """
    saver: AsyncSqliteSaver | None = None
    config: RunnableConfig | None = None
    if thread is not None:
        if threads_db is None:
            raise ValueError("a thread needs a threads database to be saved in")
        threads_db.parent.mkdir(parents=True, exist_ok=True)
        saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(threads_db)))
        config = thread_config(thread)
    return build_graph(saver), config


def _turn_input(
    question: str, mode: str | None, csv_file: str | None, web: bool = False
) -> AgentState:
    """What one turn starts from: the question, and (input only) the mode, file and search asked for.

    ``requested_web`` is always written, true or false: it is input only and ``begin_turn`` clears it,
    but a saved thread must never see last turn's request as this turn's.
    """
    turn: AgentState = {"question": question, "requested_web": web}
    if mode:
        turn["requested_mode"] = mode
    if csv_file:
        turn["csv_file"] = csv_file
    return turn


def _askable(context: AgentContext, thread: str | None) -> AgentContext:
    """``context`` with approvals on when the turn is saved in a thread (a question needs a checkpoint
    to wait in); a turn without a thread cannot be asked anything, and the gate then refuses."""
    return replace(context, approvals=thread is not None)


async def run_turn(
    question: str,
    *,
    context: AgentContext,
    thread: str | None = None,
    threads_db: Path | None = None,
    collector: StreamCollector | None = None,
    mode: str | None = None,
    csv_file: str | None = None,
    web: bool = False,
    resume: Mapping[str, Any] | None = None,
) -> StreamedAnswer:
    """Answer one question; with ``thread`` the conversation before it is remembered and extended.

    ``web`` asks for a web search on this turn (S5.3); the egress gate decides whether it may go. A
    turn that has to ask the user stops at the question: the result's final state then holds
    ``__interrupt__`` (``INTERRUPT_KEY``) and no answer, and the thread waits. ``resume`` answers that
    question (``{"action": "approve" | "edit" | "deny", "query": text}``) and continues the turn; the
    ``question`` is not read then.

    ``threads_db`` is the checkpoint file (required with ``thread``). With a ``collector`` the answer
    is streamed through it as it is written; without one the graph just runs to the end and
    ``streamed`` is empty. Either way the result carries the graph's final state. Errors the graph
    raises (``ChatError`` and the like) propagate unchanged, and a turn that raised is not saved as
    a remembered turn (``remember`` is the last step).

    ``mode`` is the mode the caller asks for, ``"chat"`` or ``"csv"`` (C4): a thread's first turn
    takes it, a later turn must agree with the thread's saved mode or ``ModeError`` is raised before
    anything runs (``agent.mode.choose_mode``). ``csv_file`` is the name of the CSV file in play,
    kept in the state for display.
    """
    context = _askable(context, thread)
    async with contextlib.AsyncExitStack() as stack:
        graph, config = await _open_graph(stack, thread, threads_db)
        turn: AgentState | Command[Any] = (
            Command(resume=dict(resume))
            if resume is not None
            else _turn_input(question, mode, csv_file, web)
        )
        if collector is None:
            final = await graph.ainvoke(turn, config, context=context)
            return StreamedAnswer(final=dict(final), streamed="")
        async for event_mode, payload in graph.astream(
            turn, config, context=context, stream_mode=["custom", "values"]
        ):
            collector.feed(event_mode, payload)
        return collector.result()


async def stream_turn(
    question: str,
    *,
    context: AgentContext,
    thread: str,
    threads_db: Path,
    mode: str = "chat",
    csv_file: str | None = None,
    web: bool = False,
    resume: Mapping[str, Any] | None = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Run one turn of a saved thread and yield the graph's stream events as they happen (S4.2).

    ``web`` asks for a web search on this turn and ``resume`` answers the egress gate's question of a
    turn that stopped at it (S5.3, see ``run_turn``). A turn that stops at the question ends its
    stream with a ``values`` payload holding ``INTERRUPT_KEY``: the server sends the question to the
    client and the turn continues when the answer comes in through a second call with ``resume``.

    Each item is the ``(kind, payload)`` pair that ``graph.astream(..., stream_mode=["custom",
    "values"])`` produces, passed on untouched: a ``"custom"`` payload is a ``StreamEvent`` (a piece
    of the answer or of the reasoning, already cleaned by G3), a ``"values"`` payload is the whole
    state after a step, and the last one is the finished turn. The caller decides what to do with
    them; the server turns them into SSE frames (``server/turns.py``).

    This is an async generator, so it only runs while somebody iterates it. If the caller stops early
    (a client that went away) the stream is closed first and the checkpoint database after it, in
    that order: closing the database under a graph that is still writing to it would fail inside
    LangGraph's own tasks. The step that is running in a worker thread cannot be interrupted and runs
    to its end; its result is dropped, and because ``remember`` is the last step an interrupted turn
    is never saved as a remembered one. Errors the graph raises propagate to the caller.
    """
    context = _askable(context, thread)
    async with contextlib.AsyncExitStack() as stack:
        graph, config = await _open_graph(stack, thread, threads_db)
        turn: AgentState | Command[Any] = (
            Command(resume=dict(resume))
            if resume is not None
            else _turn_input(question, mode, csv_file, web)
        )
        events = graph.astream(turn, config, context=context, stream_mode=["custom", "values"])
        try:
            async for kind, payload in events:
                yield kind, payload
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()


@dataclass(frozen=True)
class ThreadInfo:
    """A saved thread as the server lists and shows it: its state, and when it last changed.

    ``values`` is the graph state the checkpointer saved (``AgentState``: ``history``, ``thread_taint``,
    ``turn``, ``mode`` and the last turn's leftovers). ``updated_at`` is the checkpoint's timestamp
    (an ISO-8601 string), ``None`` if the checkpointer gave none.
    """

    thread_id: str
    values: dict[str, Any]
    updated_at: str | None
    pending: dict[str, Any] | None = (
        None  # the egress gate's open question, if the turn waits (S5.3)
    )


def interrupt_question(final: Mapping[str, Any]) -> dict[str, Any] | None:
    """The egress gate's question if a turn's final state stopped at it (``INTERRUPT_KEY``), else ``None``.

    ``final`` is the last ``values`` payload of a run. LangGraph puts a tuple of ``Interrupt`` objects
    under ``INTERRUPT_KEY`` when the turn is waiting; only a payload of this program's own kind counts.
    """
    for interrupt in final.get(INTERRUPT_KEY, ()):
        value = getattr(interrupt, "value", None)
        if isinstance(value, dict) and value.get("kind") == APPROVAL_KIND:
            return dict(value)
    return None


def pending_approval(snapshot: StateSnapshot) -> dict[str, Any] | None:
    """The egress gate's open question in a thread's snapshot, or ``None`` if nothing waits.

    LangGraph keeps the payload of an unanswered ``interrupt()`` on the task that raised it. Only a
    payload of the kind this program asks (``APPROVAL_KIND``) counts; anything else is ignored.
    """
    for task in snapshot.tasks:
        for interrupt in task.interrupts:
            value = interrupt.value
            if isinstance(value, dict) and value.get("kind") == APPROVAL_KIND:
                return dict(value)
    return None


async def read_thread(threads_db: Path, thread: str) -> dict[str, Any] | None:
    """The saved state of ``thread``, or ``None`` if there is none (or no database yet).

    Never creates the database: looking at a thread that does not exist must not leave a file behind.
    """
    config = thread_config(thread)
    if not threads_db.is_file():
        return None
    async with AsyncSqliteSaver.from_conn_string(str(threads_db)) as saver:
        snapshot = await build_graph(saver).aget_state(config)
    return dict(snapshot.values) if snapshot.values else None


async def read_thread_info(threads_db: Path, thread: str) -> ThreadInfo | None:
    """Like ``read_thread``, with the time of the thread's latest checkpoint (S4.2)."""
    config = thread_config(thread)
    if not threads_db.is_file():
        return None
    async with AsyncSqliteSaver.from_conn_string(str(threads_db)) as saver:
        snapshot = await build_graph(saver).aget_state(config)
    if not snapshot.values:
        return None
    return ThreadInfo(
        thread_id=thread,
        values=dict(snapshot.values),
        updated_at=snapshot.created_at,
        pending=pending_approval(snapshot),
    )


async def list_threads(threads_db: Path, *, limit: int = 50) -> list[ThreadInfo]:
    """The ``limit`` most recently changed threads, newest first (empty if there is no database).

    The names come from one query on the checkpointer's table (listing through its API would read and
    unpack every checkpoint of every thread: a long-lived thread has hundreds); each thread's state
    is then read with the graph, like ``read_thread_info``. A name this program would not accept
    (``check_thread_name``) is skipped: it is somebody else's row, and nothing here could open it.
    Never creates the database.
    """
    if not threads_db.is_file():
        return []
    found: list[ThreadInfo] = []
    async with AsyncSqliteSaver.from_conn_string(str(threads_db)) as saver:
        await (
            saver.setup()
        )  # creates the tables if the file is empty; takes the saver's lock itself
        async with saver.lock, saver.conn.execute(_NEWEST_THREADS, (limit,)) as cursor:
            names = [str(row[0]) for row in await cursor.fetchall()]
        graph = build_graph(saver)
        for name in names:
            try:
                config = thread_config(name)
            except ThreadNameError:
                continue
            snapshot = await graph.aget_state(config)
            if snapshot.values:
                found.append(ThreadInfo(name, dict(snapshot.values), snapshot.created_at))
    return found


async def delete_thread(threads_db: Path, thread: str) -> bool:
    """Forget ``thread`` completely; ``False`` if there was nothing to forget."""
    config = thread_config(thread)
    if not threads_db.is_file():
        return False
    async with AsyncSqliteSaver.from_conn_string(str(threads_db)) as saver:
        existed = await saver.aget_tuple(config) is not None
        await saver.adelete_thread(thread)
    return existed
