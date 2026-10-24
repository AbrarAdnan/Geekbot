"""Threads and the streamed answer: the routes the chat UI is built on (step S4.2).

::

    POST   /v1/threads                      201  {"thread_id": "t-1a2b3c4d5e6f", "mode": "chat"}
    GET    /v1/threads?limit=50             the newest threads, each with a one-line preview
    GET    /v1/threads/{id}                 one thread with its messages and, per answer, its sources
    DELETE /v1/threads/{id}                 204; 404 if there is none; 409 while a question is running
    POST   /v1/threads/{id}/messages        {"question": "...", "think": false} -> an SSE stream
    POST   /v1/threads/{id}/approval        {"action": "approve|edit|deny", "query": "..."} -> an SSE stream

**Web search (S5.3).** ``{"web": true}`` on a question asks for a web search on that turn (409 if the
server has the web off, which is the default, or the conversation is a CSV one). Most turns just
search: the user's own words are theirs to send. A query a model rewrote in a conversation that has
carried documents, web text or tool output stops the turn at the egress gate: the stream ends with an
``approval`` event ``{tool, query, labels, reason}`` and **no** ``done``; ``GET /v1/threads/{id}`` then
shows ``pending_approval``, a new question in the thread is a 409, and ``/approval`` answers the
question and streams the rest of the turn (same events as ``/messages``). Nothing is sent before the
answer, and an answer without a waiting question is a 404.

**The answer is a stream** (``text/event-stream``, written with ``sse.format_event``). Events, in the
order they come:

- ``meta``: ``{thread_id, turn, mode, model, think}``, sent at once, so the client knows the question
  was accepted and which model answers;
- ``status``: ``{stage: "loading_model", model}``, only if the models were not warmed up at start-up
  (``geekbot serve --no-warmup``); the first question then waits for the load;
- ``status`` again, **only when the request has** ``"steps": true`` (S3.6): ``{stage: "searching"}``
  when the search starts, ``{stage: "ranking"}`` when the candidates are being ordered and
  ``{stage: "writing"}`` when the model is about to answer. They are progress for a person watching
  and carry nothing else; a client that does not ask never sees them;
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
time, see ``answering.TurnGate``) or when the conversation's CSV file is not on the server. The
thread is created by its first answered turn; ``POST /v1/threads`` only reserves an id.

**Chat and CSV conversations (C5).** The route serves both; the thread decides. A thread that has
been answered keeps the mode it started with; one that has not is a CSV conversation if a CSV file
was uploaded to it (``csvroutes.py``: the file is then a loaded session in the registry) and a chat
conversation otherwise. A CSV turn streams ``meta`` (``mode: "csv"``, the CSV model), ``status``
(``loading_model`` when the GPU has to be swapped to the CSV model, ``loading_file`` when the table
has to be loaded again after a restart or an eviction, and always, last, ``asking``: the question
has reached the table) and then ``done`` or ``error``. There are no ``text`` events: the package
answers in one piece, so ``asking`` is the only sign of progress while it works (a question takes
seconds, up to the time cap), and the whole reply (the answer, the SQL that was run, a preview of
the result) is in ``done.answer``, already through the output filter. A CSV question
takes the same one-at-a-time slot as a chat question, so the swap of models never happens under a
running turn.

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
from geekbot.csv_mode.sessions import CsvSessions, Lease
from geekbot.server.answering import Answering, CsvSetup, TurnGate
from geekbot.server.csvmode import CsvSession, stored_file
from geekbot.server.routes import AnsweringDep, CsvSessionsDep, GateDep, SettingsDep
from geekbot.server.schemas import (
    ApprovalRequest,
    AskRequest,
    ThreadCreated,
    ThreadOut,
    ThreadRequest,
    ThreadSummary,
    TurnOut,
)
from geekbot.server.turns import TurnTranslator, describe_error
from geekbot.server.uploads import display_name
from geekbot.sse import format_event

__all__ = ["router"]

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/threads", tags=["threads"])

BUSY: Final = "a question is already being answered; try again when it has finished"
WAITING: Final = (
    "a web search is waiting for your answer in this conversation: approve, edit or deny it first"
)
RESUMING: Final = (
    "(the answer to the web search's question)"  # the question is not read on a resume
)
NO_WEB: Final = "web search is switched off on this server (GEEKBOT_WEB_ENABLED)"
NO_WEB_IN_CSV: Final = (
    "web search is not available in a CSV conversation: its data stays on this machine"
)
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
async def create_thread(body: ThreadRequest | None = None) -> ThreadCreated:
    """Hand out a fresh thread id (12 random hex digits). Stores nothing.

    The body is optional (``{"mode": "csv"}`` says the conversation is meant to be about a CSV file);
    an unknown mode is a 422. The mode is only echoed: it becomes a fact when a CSV file is uploaded
    to the thread and the first turn is answered (see ``ThreadCreated``).
    """
    mode = body.mode if body is not None else "chat"
    return ThreadCreated(thread_id=f"t-{uuid.uuid4().hex[:12]}", mode=mode)


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
async def delete(
    thread_id: ThreadName, settings: SettingsDep, gate: GateDep, sessions: CsvSessionsDep
) -> Response:
    """Forget a thread completely: its messages, its sources, its taint. It cannot be undone.

    Takes the same one-at-a-time slot as a question, so a question cannot start in a thread while
    it is being deleted, and a thread cannot be deleted under a running question. A CSV
    conversation's loaded table is released too (its file stays in the uploads folder, where another
    upload of the same bytes would find it). A thread that was only a CSV upload with no answered
    turn is deleted by this as well.
    """
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    try:
        existed = await delete_thread(_db(settings), thread_id)
        dropped = sessions.drop(thread_id)
    finally:
        gate.release(token)
    if not existed and not dropped:
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
    sessions: CsvSessionsDep,
) -> StreamingResponse:
    """Ask a question in a thread and stream the answer. Creates the thread on its first question.

    Everything that can be refused is refused here, before the first byte: a bad question (422),
    a CSV conversation whose file is gone (409), a question already running (409). The slot of the
    gate is taken here, in the handler, with no ``await`` between the check and the take, so two
    requests cannot both pass; it is given back when the stream ends, and again after the response as
    a backstop (a stream that was never started would otherwise hold the slot for ever).

    The thread decides the mode (see the module docstring): a saved thread keeps its own, and an
    unsaved one with a CSV file uploaded to it is a CSV conversation.
    """
    try:
        check_question(body.question)
    except QuestionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    saved = await read_thread_info(_db(settings), thread_id)
    if saved is not None and saved.pending is not None:
        raise HTTPException(status_code=409, detail=WAITING)
    uploaded = saved is None and sessions.has(thread_id)
    try:
        mode = choose_mode(
            saved.values.get("mode") if saved else None, body.mode or ("csv" if uploaded else None)
        )
    except ModeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if body.web and mode == "csv":
        raise HTTPException(status_code=409, detail=NO_WEB_IN_CSV)
    if body.web and not answering.web_enabled:
        raise HTTPException(status_code=409, detail=NO_WEB)
    if mode == "csv":
        return await _ask_csv(
            thread_id=thread_id,
            body=body,
            saved=saved,
            settings=settings,
            answering=answering,
            gate=gate,
            sessions=sessions,
        )
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


@router.post(
    "/{thread_id}/approval",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "The rest of the turn that stopped at the web search's question, as the "
            "same event stream as the question's: meta, status, text, thinking, done, error.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def approve(
    thread_id: ThreadName,
    body: ApprovalRequest,
    settings: SettingsDep,
    answering: AnsweringDep,
    gate: GateDep,
) -> StreamingResponse:
    """Answer the question a turn stopped at (S5.3) and stream the rest of the turn.

    404 when nothing is waiting in the thread (an answer without a question is never accepted, so a
    stray or repeated request cannot send anything), 409 when another question is being answered.
    The turn continues from the step that asked: the gate sends exactly the text that was shown
    (``approve``), the user's own text (``edit``) or nothing (``deny``), then the turn goes on to the
    answer. A turn can stop at most once, so the stream ends with ``done`` or ``error``.
    """
    saved = await read_thread_info(_db(settings), thread_id)
    if saved is None or saved.pending is None:
        raise HTTPException(
            status_code=404, detail=f"nothing is waiting for an answer in {thread_id!r}"
        )
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    turn = int(saved.values.get("turn", 0)) + 1
    request = AskRequest(question=RESUMING, think=body.think, steps=body.steps, web=True)
    return StreamingResponse(
        _answer_events(
            thread_id=thread_id,
            body=request,
            turn=turn,
            settings=settings,
            answering=answering,
            gate=gate,
            token=token,
            resume=body,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(_release, gate, token),
    )


async def _release(gate: TurnGate, token: object) -> None:
    gate.release(token)


NO_CSV_FILE: Final = (
    "the server does not hold this conversation's CSV file (the conversation was started with the "
    "command line, or the file was deleted, or its table was closed before its first question "
    "was answered): start a new conversation and upload the file again"
)


async def _ask_csv(
    *,
    thread_id: str,
    body: AskRequest,
    saved: ThreadInfo | None,
    settings: Settings,
    answering: Answering,
    gate: TurnGate,
    sessions: CsvSessions[CsvSession],
) -> StreamingResponse:
    """The CSV half of ``ask``: find the conversation's file, take the slot, stream the turn.

    The file is the one the loaded session came from or, if the session is not loaded (the server
    was restarted, or the registry evicted it), the one the thread's saved state names. A name that
    does not point at a stored upload is a 409 that says what to do; nothing is loaded or swapped
    before that is known.
    """
    csv = answering.csv
    if csv is None:
        raise HTTPException(status_code=409, detail="this server has no CSV model set up")
    path = sessions.path_of(thread_id)
    if path is None:
        named = str(saved.values.get("csv_file") or "") if saved else ""
        path = await stored_file(settings, named)
    if path is None:
        raise HTTPException(status_code=409, detail=NO_CSV_FILE)
    token = gate.try_acquire(thread_id)
    if token is None:
        raise HTTPException(status_code=409, detail=BUSY)
    turn = (int(saved.values.get("turn", 0)) if saved else 0) + 1
    return StreamingResponse(
        _csv_events(
            thread_id=thread_id,
            body=body,
            turn=turn,
            path=path,
            settings=settings,
            answering=answering,
            csv=csv,
            sessions=sessions,
            gate=gate,
            token=token,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(_release, gate, token),
    )


async def _csv_events(
    *,
    thread_id: str,
    body: AskRequest,
    turn: int,
    path: Path,
    settings: Settings,
    answering: Answering,
    csv: CsvSetup,
    sessions: CsvSessions[CsvSession],
    gate: TurnGate,
    token: object,
) -> AsyncIterator[str]:
    """The frames of one CSV answer: ``meta``, ``status`` frames (the last is always ``asking``), then
    ``done`` (or ``error``).

    In order: the model is swapped on the GPU if the chat model holds it; the table is loaded if its
    session is not in the registry; the session is pinned for the length of the question, so no
    upload for another conversation can close it underneath (``csv_mode.sessions``); the graph runs
    its CSV branch (a follow-up is rewritten with the CSV model first, then the package answers). Both
    waits happen in worker threads, because they take seconds and the event loop must stay free
    (PLAN.md §6.3). Like ``_answer_events``, a failure becomes an ``error`` event, and the ``finally``
    gives back the session and the slot in every case, a client that went away included.
    """
    translator = TurnTranslator()
    lease: Lease[CsvSession] | None = None
    try:
        yield format_event(
            "meta",
            {
                "thread_id": thread_id,
                "turn": turn,
                "mode": "csv",
                "model": csv.model,
                "think": False,  # the CSV model is not asked to reason
            },
        )
        if not answering.csv_ready:
            yield format_event("status", {"stage": "loading_model", "model": csv.model})
            await anyio.to_thread.run_sync(answering.ensure_csv_ready)
        if not sessions.has(thread_id):
            yield format_event("status", {"stage": "loading_file", "file": display_name(path.name)})
        lease = await anyio.to_thread.run_sync(sessions.acquire, thread_id, path)
        # The package answers whole, so nothing streams while it works; this is what the page shows.
        yield format_event("status", {"stage": "asking", "file": display_name(path.name)})
        events = stream_turn(
            body.question,
            context=answering.csv_context(lease.session),
            thread=thread_id,
            threads_db=_db(settings),
            mode="csv",
            csv_file=path.name,
        )
        try:
            async for kind, payload in events:
                for frame in translator.feed(kind, payload):
                    yield frame
        finally:
            await events.aclose()
        if translator.final is None:
            raise RuntimeError("the graph ended without a final state")
        done = TurnOut.of(translator.final, thread_id=thread_id, model=csv.model)
        _log_csv_turn(done)
        yield format_event("done", done.model_dump(mode="json"))
    except Exception as error:  # the stream is open: report as an event, never raise
        message, expected = describe_error(error)
        if expected:
            log.warning("thread %s: the CSV turn failed: %s", thread_id, message)
        else:
            log.exception("thread %s: the CSV turn failed unexpectedly", thread_id)
        yield format_event("error", {"message": message})
    finally:
        if lease is not None:
            sessions.release(lease)
        gate.release(token)


def _log_csv_turn(done: TurnOut) -> None:
    """One line per answered CSV turn: the verdict and where the time went."""
    log.info(
        "thread %s turn %d answered (%s): %.2f s, model load %.2f s",
        done.thread_id,
        done.turn,
        done.csv_status or "no verdict",
        done.generation.get("seconds", 0.0),
        done.generation.get("load_seconds", 0.0),
    )


async def _answer_events(
    *,
    thread_id: str,
    body: AskRequest,
    turn: int,
    settings: Settings,
    answering: Answering,
    gate: TurnGate,
    token: object,
    resume: ApprovalRequest | None = None,
) -> AsyncIterator[str]:
    """The frames of one answer, from ``meta`` to ``done`` (or ``error``).

    Whatever goes wrong inside the stream becomes an ``error`` event, because the status line has
    been sent long ago. Only ``Exception`` is caught: a client that went away cancels this generator
    (``CancelledError`` / ``GeneratorExit``), and that must propagate. The ``finally`` gives the
    gate's slot back in every case.

    A turn that asked for a web search can stop at the egress gate's question (S5.3): the stream then
    ends with an ``approval`` frame and **no** ``done``: the turn is waiting, the slot is given back
    (a waiting turn holds no model and no thread of work), and the user's answer arrives through
    ``approve`` below as ``resume``, which continues the same turn from where it stopped.
    """
    translator = TurnTranslator(steps=body.steps)
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
            web=body.web,
            resume={"action": resume.action, "query": resume.query} if resume is not None else None,
        )
        try:
            async for kind, payload in events:
                for frame in translator.feed(kind, payload):
                    yield frame
        finally:
            await events.aclose()
        if translator.final is None:
            raise RuntimeError("the graph ended without a final state")
        if translator.approval is not None:
            log.info(
                "thread %s turn %d is waiting for the user's answer to the web search",
                thread_id,
                turn,
            )
            return
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
