"""Threads and the streamed answer: the routes the chat UI is built on (step S4.2).

::

    POST   /v1/threads                      201  {"thread_id": "t-1a2b3c4d5e6f", "mode": "chat"}
    GET    /v1/threads?limit=50             the newest threads, each with a one-line preview
    GET    /v1/threads/{id}                 one thread with its messages and, per answer, its sources
    DELETE /v1/threads/{id}                 204; 404 if there is none; 409 while a question is running
    POST   /v1/threads/{id}/messages        {"question": "...", "think": false} -> an SSE stream

**The answer is a stream** (``text/event-stream``, written with ``sse.format_event``). Events, in the
order they come:

- ``meta``: ``{thread_id, turn, mode, model, think}``, sent at once, so the client knows the question
  was accepted and which model answers;
- ``status``: ``{stage: "loading_model", model}``, only if the models were not warmed up at start-up
  (``geekbot serve --no-warmup``); the first question then waits for the load;
- ``text``: ``{text}``, a piece of the answer, as the model writes it. Already cleaned by the output
  filter (G3): no images, no links to places the sources do not name;
- ``thinking``: ``{text}``, a piece of the model's reasoning (only with ``think``), cleaned the same way;
- ``done``: a ``TurnOut``, the finished turn: the checked answer (a citation that named no source is
  removed, so it can differ from the pieces), its sources with their passages, the taint, the
  readings. The last event of a good turn. The answer is always whole in ``done``: a turn in which the
  model was never called (nothing was found) streams no ``text`` at all, and the application's own
  sentence is only there;
- ``error``: ``{message}``, the turn failed. It is the last event and no ``done`` follows; **a client
  discards what the ``text`` events showed**, because the failed turn is not remembered.

Before the stream starts, a request that cannot be answered gets an ordinary HTTP error: 422 for a bad
thread name or an empty or too long question, 409 when another question is being answered (one at a
time, see ``answering.TurnGate``) or when the thread belongs to another mode. The thread is created
by its first answered turn; ``POST /v1/threads`` only reserves an id.

**Nothing blocks the event loop** (PLAN.md §6.3). The graph's steps are plain functions that
LangGraph runs in worker threads (retrieval, the reranker and the model's stream all block); the
route only awaits the events they produce and writes frames. A heartbeat test asks ``/healthz``
while an answer streams.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Final

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from geekbot.agent.mode import ModeError, choose_mode
from geekbot.agent.packing import QuestionError, check_question
from geekbot.agent.runner import (
    THREADS_DB,
    ThreadInfo,
    ThreadNameError,
    check_thread_name,
    delete_thread,
    list_threads,
    read_thread_info,
    stream_turn,
)
from geekbot.config import Settings
from geekbot.server.answering import Answering, TurnGate
from geekbot.server.routes import AnsweringDep, GateDep, SettingsDep
from geekbot.server.schemas import (
    AskRequest,
    ThreadCreated,
    ThreadOut,
    ThreadSummary,
    TurnOut,
)
from geekbot.server.turns import TurnTranslator, describe_error
from geekbot.sse import format_event

__all__ = ["router"]

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/threads", tags=["threads"])

BUSY: Final = "a question is already being answered; try again when it has finished"
SSE_HEADERS: Final = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


def thread_name(thread_id: str) -> str:
    """The path's thread id if it is a usable name, else HTTP 422 saying what is allowed."""
    try:
        return check_thread_name(thread_id)
    except ThreadNameError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


ThreadName = Annotated[str, Depends(thread_name)]


def _db(settings: Settings) -> Path:
    return settings.data_dir / THREADS_DB


# --- threads ---------------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_thread() -> ThreadCreated:
    """Hand out a fresh thread id (12 random hex digits). Stores nothing."""
    return ThreadCreated(thread_id=f"t-{uuid.uuid4().hex[:12]}")


def _remembers_something(info: ThreadInfo | None) -> bool:
    """Whether a saved thread has a turn in its memory.

    The checkpointer saves after every step, so a turn that failed or was abandoned half way (Ollama
    down, a client that went away) leaves a thread behind with a state and no history. Such a thread
    is not listed and is a 404: nothing was answered in it, and ``remember`` is the last step.
    """
    return info is not None and bool(info.values.get("history"))


@router.get("")
async def list_all(
    settings: SettingsDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[ThreadSummary]:
    """The ``limit`` most recently changed threads that have an answered turn, newest first."""
    infos = await list_threads(_db(settings), limit=limit)
    return [ThreadSummary.of(info) for info in infos if _remembers_something(info)]


@router.get("/{thread_id}")
async def show(thread_id: ThreadName, settings: SettingsDep) -> ThreadOut:
    """One thread with the messages it remembers. No answered turn yet means 404."""
    info = await read_thread_info(_db(settings), thread_id)
    if info is None or not _remembers_something(info):
        raise HTTPException(status_code=404, detail=f"no thread {thread_id!r}")
    return ThreadOut.of(info)


@router.delete("/{thread_id}", status_code=204)
async def delete(thread_id: ThreadName, settings: SettingsDep, gate: GateDep) -> Response:
    """Forget a thread completely: its messages, its sources, its taint. It cannot be undone.

    Takes the same one-at-a-time slot as a question, so a question cannot start in a thread while
    it is being deleted, and a thread cannot be deleted under a running question.
    """
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    try:
        existed = await delete_thread(_db(settings), thread_id)
    finally:
        gate.release(token)
    if not existed:
        raise HTTPException(status_code=404, detail=f"no thread {thread_id!r}")
    return Response(status_code=204)


# --- the streamed answer ---------------------------------------------------------------------


@router.post(
    "/{thread_id}/messages",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "A server-sent event stream: meta, status, text, thinking, done, error "
            "(see this module's docstring).",
            "content": {"text/event-stream": {}},
        }
    },
)
async def ask(
    thread_id: ThreadName,
    body: AskRequest,
    settings: SettingsDep,
    answering: AnsweringDep,
    gate: GateDep,
) -> StreamingResponse:
    """Ask a question in a thread and stream the answer. Creates the thread on its first question.

    Everything that can be refused is refused here, before the first byte: a bad question (422),
    a thread of another mode (409), a question already running (409). The slot of the gate is taken
    here, in the handler, with no ``await`` between the check and the take, so two requests cannot
    both pass; it is given back when the stream ends, and again after the response as a backstop
    (a stream that was never started would otherwise hold the slot for ever).
    """
    try:
        check_question(body.question)
    except QuestionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    saved = await read_thread_info(_db(settings), thread_id)
    try:
        choose_mode(saved.values.get("mode") if saved else None, "chat")
    except ModeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    turn = (int(saved.values.get("turn", 0)) if saved else 0) + 1
    return StreamingResponse(
        _answer_events(
            thread_id=thread_id,
            body=body,
            turn=turn,
            settings=settings,
            answering=answering,
            gate=gate,
            token=token,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(_release, gate, token),
    )


async def _release(gate: TurnGate, token: object) -> None:
    gate.release(token)


async def _answer_events(
    *,
    thread_id: str,
    body: AskRequest,
    turn: int,
    settings: Settings,
    answering: Answering,
    gate: TurnGate,
    token: object,
) -> AsyncIterator[str]:
    """The frames of one answer, from ``meta`` to ``done`` (or ``error``).

    Whatever goes wrong inside the stream becomes an ``error`` event, because the status line has
    been sent long ago. Only ``Exception`` is caught: a client that went away cancels this generator
    (``CancelledError`` / ``GeneratorExit``), and that must propagate. The ``finally`` gives the
    gate's slot back in every case.
    """
    translator = TurnTranslator()
    try:
        yield format_event(
            "meta",
            {
                "thread_id": thread_id,
                "turn": turn,
                "mode": "chat",
                "model": answering.model,
                "think": body.think,
            },
        )
        if not answering.ready:
            yield format_event("status", {"stage": "loading_model", "model": answering.model})
            await anyio.to_thread.run_sync(answering.ensure_ready)
        events = stream_turn(
            body.question,
            context=answering.context_for(think=body.think),
            thread=thread_id,
            threads_db=_db(settings),
            mode="chat",
        )
        try:
            async for kind, payload in events:
                for frame in translator.feed(kind, payload):
                    yield frame
        finally:
            await events.aclose()
        if translator.final is None:
            raise RuntimeError("the graph ended without a final state")
        done = TurnOut.of(translator.final, thread_id=thread_id, model=answering.model)
        _log_turn(done)
        yield format_event("done", done.model_dump(mode="json"))
    except Exception as error:  # the stream is open: report as an event, never raise
        message, expected = describe_error(error, think=body.think)
        if expected:
            log.warning("thread %s: the turn failed: %s", thread_id, message)
        else:
            log.exception("thread %s: the turn failed unexpectedly", thread_id)
        yield format_event("error", {"message": message})
    finally:
        gate.release(token)


def _log_turn(done: TurnOut) -> None:
    """One line per answered turn: where the time went."""
    first = done.generation.get("first_token_seconds")
    log.info(
        "thread %s turn %d answered: %d sources, %d cited, first token %s, %.2f s in the model",
        done.thread_id,
        done.turn,
        len(done.sources),
        len(done.cited),
        f"{first:.2f} s" if first is not None else "n/a",
        done.generation.get("seconds", 0.0),
    )
