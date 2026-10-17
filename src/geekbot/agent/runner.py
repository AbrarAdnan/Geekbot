"""Running one turn of the answer graph, with or without a saved conversation (step S3.4).

``geekbot ask`` is a synchronous command, but a conversation is saved by LangGraph's **async** SQLite
checkpointer (PLAN.md §6.3: ``AsyncSqliteSaver``), and an async checkpointer needs the graph to be run
with ``ainvoke`` / ``astream``. So the command hands one coroutine to ``asyncio.run`` and everything
async lives here: open the thread database, compile the graph with it, run the turn, close it again.
The nodes themselves stay plain synchronous functions; under ``astream`` LangGraph runs them in worker
threads, which keeps the event loop free (§6.3) and is what the server (S4.1) will rely on too.

**The thread database** is one SQLite file under the data folder (``threads.db``), opened only while
the storage lock is held (the command opens the workspace first), so the rule "one process owns the
storage" (§6.2) covers it. A thread is a name the user picks (``--thread t1``); it is the checkpointer's
``thread_id``, and everything the graph keeps in its state for that thread (the messages, the labels,
the taint) is saved after every step.

Without a thread there is no checkpointer and no file: the run starts from nothing, as before S3.4.
"""

import contextlib
import re
from pathlib import Path
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from geekbot.agent.graph import build_graph
from geekbot.agent.render import StreamCollector, StreamedAnswer
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = [
    "THREADS_DB",
    "ThreadNameError",
    "check_thread_name",
    "delete_thread",
    "read_thread",
    "run_turn",
    "thread_config",
]

THREADS_DB: Final = "threads.db"  # the file name under the data folder

# A thread name ends up in a file (as a row key) and in the terminal: letters, digits, dots, dashes
# and underscores, starting with a letter or digit, at most 64 characters.
_THREAD_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


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


async def run_turn(
    question: str,
    *,
    context: AgentContext,
    thread: str | None = None,
    threads_db: Path | None = None,
    collector: StreamCollector | None = None,
    mode: str | None = None,
    csv_file: str | None = None,
) -> StreamedAnswer:
    """Answer one question; with ``thread`` the conversation before it is remembered and extended.

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
    async with contextlib.AsyncExitStack() as stack:
        saver: AsyncSqliteSaver | None = None
        config: RunnableConfig | None = None
        if thread is not None:
            if threads_db is None:
                raise ValueError("a thread needs a threads database to be saved in")
            threads_db.parent.mkdir(parents=True, exist_ok=True)
            saver = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(threads_db))
            )
            config = thread_config(thread)
        graph = build_graph(saver)
        turn: AgentState = {"question": question}
        if mode:
            turn["requested_mode"] = mode
        if csv_file:
            turn["csv_file"] = csv_file
        if collector is None:
            final = await graph.ainvoke(turn, config, context=context)
            return StreamedAnswer(final=dict(final), streamed="")
        async for event_mode, payload in graph.astream(
            turn, config, context=context, stream_mode=["custom", "values"]
        ):
            collector.feed(event_mode, payload)
        return collector.result()


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


async def delete_thread(threads_db: Path, thread: str) -> bool:
    """Forget ``thread`` completely; ``False`` if there was nothing to forget."""
    config = thread_config(thread)
    if not threads_db.is_file():
        return False
    async with AsyncSqliteSaver.from_conn_string(str(threads_db)) as saver:
        existed = await saver.aget_tuple(config) is not None
        await saver.adelete_thread(thread)
    return existed
