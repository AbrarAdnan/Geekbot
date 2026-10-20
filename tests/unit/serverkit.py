"""Helpers shared by the server tests (steps S4.1, S4.2 and C5).

A fake embedder with a fixed dimension, so an ingest runs the real pipeline (loaders, chunker, SQLite,
Chroma) without Ollama; fake answering tools (a scripted chat model, rewrite call and, unless a test
passes its own, the real hybrid retriever over that store); since C5 a fake CSV side as well (the real
registry of sessions over ``csvkit.FakeCsv`` tables, a scripted rewrite call for the CSV model, a fake
swap to the CSV model); an app built from them; and a few helpers for the HTTP side (wait for a job,
ask a question and read the stream, upload a CSV, time the heartbeat).
"""

import asyncio
import gc
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from chatkit import ScriptedChat, ScriptedCondenser
from csvkit import CSV_MODEL, SALES, CsvFakes
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.csv_mode.sessions import CsvSessions
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.workspace import Workspace
from geekbot.llm.chat import ChatModel
from geekbot.retrieval import Retriever, SearchMode
from geekbot.retrieval.factory import build_retriever
from geekbot.server import create_app
from geekbot.server.answering import Answering, CsvSetup
from geekbot.server.security import (
    LOOPBACK_HOSTS,
    MAX_BODY_BYTES,
    MAX_UPLOAD_BYTES,
    PathPolicy,
    Perimeter,
)
from geekbot.sse import SseEvent, parse_text

MODEL = "fake-model:1b"
# The test client sends "Host: testserver" and the async one in the heartbeat tests "Host: test".
TEST_HOSTS = (*LOOPBACK_HOSTS, "testserver", "test")
DIM = 4
SPEC = EmbedderSpec(
    model="fake",
    digest="d1",
    document_prefix="doc: ",
    query_prefix="q: ",
    dimension=DIM,
    with_header=True,
)


class FixedEmbeddings:
    """Four numbers per text, derived from its length: enough for a real index, no model."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        n = float(len(text) % 7 + 1)
        return [n, 1.0, 0.5, 0.25]


def open_fake_workspace(settings: Settings) -> Workspace:
    embeddings = PrefixedEmbeddings(FixedEmbeddings(), document_prefix="doc: ", query_prefix="q: ")
    return Workspace.open(settings, embeddings=embeddings, spec=SPEC)


def settings_for(root: Path) -> Settings:
    """Settings with the data folder under ``root`` and ``root`` itself an allowed folder, so the
    notes the tests write there can be ingested by path (the data folder inside stays forbidden)."""
    return Settings(_env_file=None, data_dir=root / "data", allowed_roots=str(root))


@dataclass
class Fakes:
    """What the app was given instead of Ollama, kept on ``app.state.fakes`` for the assertions.

    ``thinks`` is the ``think`` flag of every chat model the route asked for besides the shared one;
    ``loads`` has one entry per warm-up that ran. C5: ``csv`` is the fake CSV side (every table that
    was opened, how it answers), ``csv_condenser`` the rewrite call of the CSV model and ``csv_loads``
    one entry per swap to the CSV model.
    """

    chat: ScriptedChat
    condenser: ScriptedCondenser
    retriever: Retriever | None
    csv: CsvFakes
    csv_condenser: ScriptedCondenser
    thinks: list[bool] = field(default_factory=list)
    loads: list[str] = field(default_factory=list)
    csv_loads: list[str] = field(default_factory=list)


def make_app(
    root: Path,
    *,
    chat: ScriptedChat | None = None,
    condenser: ScriptedCondenser | None = None,
    retriever: Retriever | None = None,
    csv: CsvFakes | None = None,
    csv_condenser: ScriptedCondenser | None = None,
    max_csv_sessions: int = 4,
    max_csv_bytes: int = 500_000_000,
    warm: bool = True,
    api_key: str | None = None,
    hosts: tuple[str, ...] = TEST_HOSTS,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> FastAPI:
    """An app over a fake-embedder workspace with scripted models.

    ``retriever`` replaces the real hybrid retriever (which searches whatever was ingested through
    the API). ``warm=False`` starts the answering tools as ``--no-warmup`` would: not ready, until the
    first question runs the (fake, instant) load. The perimeter (S4.3) is the real middleware with no
    API key unless ``api_key`` is given, the test host names allowed unless ``hosts`` says otherwise,
    and ``root`` as the one allowed folder (``settings_for``). The CSV side (C5) uses the real
    registry of sessions with the given bounds, over the fake tables of ``csv``.
    """
    fakes = Fakes(
        chat=chat or ScriptedChat(),
        condenser=condenser or ScriptedCondenser(),
        retriever=retriever,
        csv=csv or CsvFakes(),
        csv_condenser=csv_condenser or ScriptedCondenser(),
    )

    def open_answering(settings: Settings, workspace: Workspace) -> Answering:
        context = AgentContext(
            retriever=fakes.retriever or build_retriever(SearchMode.HYBRID, workspace, settings),
            llm=fakes.chat,
            model=MODEL,
            top_k=settings.answer_top_k,
            context_budget_tokens=settings.context_budget_tokens,
            condenser=fakes.condenser,
        )

        def make_llm(think: bool) -> ChatModel:
            fakes.thinks.append(think)
            return fakes.chat

        def load() -> float:
            fakes.loads.append("load")
            return 0.0

        def load_csv() -> float:
            fakes.csv_loads.append("load")
            return 0.0

        answering = Answering(
            context,
            make_llm=make_llm,
            load=load,
            csv=CsvSetup(model=CSV_MODEL, condenser=fakes.csv_condenser, load=load_csv),
        )
        answering.ready = warm
        return answering

    def open_csv_sessions(settings: Settings) -> CsvSessions[Any]:
        del settings
        return CsvSessions(fakes.csv.open, max_sessions=max_csv_sessions, max_bytes=max_csv_bytes)

    settings = settings_for(root)
    perimeter = Perimeter(
        paths=PathPolicy.from_settings(settings),
        api_key=api_key,
        hosts=hosts,
        max_body_bytes=max_body_bytes,
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app(
        settings,
        open_workspace=open_fake_workspace,
        open_answering=open_answering,
        open_csv_sessions=open_csv_sessions,
        perimeter=perimeter,
    )
    app.state.fakes = fakes
    return app


def wait_for(client: TestClient, job_id: str, timeout_s: float = 20.0) -> dict[str, Any]:
    """Poll a job until it has finished."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job: dict[str, Any] = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in {timeout_s} s")


def write_note(root: Path, name: str = "zebras.txt") -> Path:
    path = root / name
    path.write_text("Zebras live on the savanna.\n\nTheir stripes are unique.\n", encoding="utf-8")
    return path


def ingest_note(client: TestClient, root: Path, name: str = "zebras.txt") -> Path:
    """Write a small note and ingest it through the API; the job must succeed."""
    note = write_note(root, name)
    accepted = client.post("/v1/kb/ingest", json={"paths": [str(note)]})
    assert accepted.status_code == 202
    assert wait_for(client, accepted.json()["id"])["state"] == "done"
    return note


@dataclass
class Asked:
    """One answered request: the HTTP status, the SSE events (if it was a stream) or the JSON error."""

    status: int
    events: list[SseEvent]
    body: Any

    @property
    def names(self) -> list[str]:
        return [event.event for event in self.events]

    def data(self, name: str) -> Any:
        """The data of the first event called ``name``."""
        return next(event.data for event in self.events if event.event == name)

    @property
    def text(self) -> str:
        """The answer pieces joined, as a client would show them."""
        return "".join(event.data["text"] for event in self.events if event.event == "text")


def ask(client: TestClient, thread_id: str, question: str, **body: Any) -> Asked:
    """POST a question and read the whole response (the test client returns it when it is complete)."""
    response = client.post(f"/v1/threads/{thread_id}/messages", json={"question": question, **body})
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        return Asked(response.status_code, parse_text(response.text), None)
    return Asked(response.status_code, [], response.json())


def upload_csv(
    client: TestClient, thread_id: str, name: str = "sales.csv", data: bytes = SALES
) -> httpx.Response:
    """POST a CSV to a thread as the raw body, the way the UI's client does."""
    return client.post(
        "/v1/csv/upload",
        params={"thread": thread_id, "filename": name},
        content=data,
        headers={"content-type": "application/octet-stream"},
    )


# --- the heartbeat (PLAN.md section 6.3), shared by the tests that measure it ------------------
#
# The check is two signals, and they catch different things:
#
# 1. **asyncio's debug mode** with ``slow_callback_duration = SLOW_CALLBACK_S`` logs "Executing ...
#    took N seconds" for any callback that held the event loop for that long. That is the detector
#    for the thing the test is about (blocking work on the loop), and ``assert_heartbeat`` fails on
#    it. It is **not** independent of the machine: it also fires for anything that stops the whole
#    process for a moment. Seen in C5's full run: a step of 0.125 s with 3,000 pings of garbage
#    behind it and a heap grown by 2,500 tests, which is the garbage collector (so ``run_timed``
#    keeps it off while it measures). The threshold is 0.15 s: above the stalls a loaded machine
#    shows, and below the shortest block the tests put in the way of the loop (0.2 s), so a call
#    that blocks the loop in any of the three tests is still caught.
# 2. **``/healthz`` latency**: the p99 of many pings must stay under 200 ms (PLAN.md section 6.3).
#    The first version asked about 60 times, where the "p99" is simply the slowest ping, so one
#    stall of a shared CI runner failed the test (seen on ``ubuntu / fresh install (3.12)``, S4.4,
#    and once on the Windows job of ``0cb4172``). Now the window is about two seconds of back-to-back
#    pings (``ping_until_done``), at least ``MIN_PINGS`` of them: the p99 is then rank 297 of 300 or
#    higher, so up to two stray slow pings are let through and a *sustained* slowdown still fails.
#    One loop block of 0.4 s delays only one ping, so it is signal 1 that catches it, not this one.
#    The limit stays 200 ms.

HEARTBEAT_P99_S = 0.2
SLOW_CALLBACK_S = 0.15
MIN_PINGS = 300


def run_timed[T](main: Callable[[], Coroutine[Any, Any, T]], caplog: pytest.LogCaptureFixture) -> T:
    """Run ``main`` on a fresh event loop in asyncio's debug mode, ready for ``assert_heartbeat``.

    The loop reports any callback slower than ``SLOW_CALLBACK_S`` to the ``asyncio`` logger, which
    ``caplog`` collects. The garbage collector is run once and then switched off for the length of
    the run, and switched on again afterwards: a collection of a big heap takes longer than the
    detector's threshold (seen: 0.125 s in a full run, never when the test ran alone) and says
    nothing about the server. A run lasts a few seconds, so what it allocates meanwhile does not
    matter.
    """

    async def detect() -> T:
        asyncio.get_running_loop().slow_callback_duration = SLOW_CALLBACK_S
        return await main()

    gc.collect()
    gc.disable()
    try:
        with caplog.at_level(logging.WARNING, logger="asyncio"):
            return asyncio.run(detect(), debug=True)
    finally:
        gc.enable()


def p99(values: list[float]) -> float:
    """The 99th percentile by rank. With fewer than 100 samples it is the slowest one."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]


async def ping(http: httpx.AsyncClient) -> float:
    """How long ``/healthz`` took to answer, in seconds."""
    started = time.perf_counter()
    response = await http.get("/healthz")
    assert response.status_code == 200
    return time.perf_counter() - started


async def ping_until_done(
    http: httpx.AsyncClient, work: asyncio.Task[Any], *, deadline_s: float = 60.0
) -> list[float]:
    """Ask ``/healthz`` again and again until ``work`` has finished; every answer time in seconds.

    The pause between two pings is ``sleep(0)``, a yield and not a timer: on Windows a timer is
    rounded up to about 15 ms, which would cut the number of pings by a factor of ten. The deadline
    makes a hang fail the test and not stop the suite.
    """
    latencies: list[float] = []
    deadline = time.monotonic() + deadline_s
    while not work.done():
        assert time.monotonic() < deadline, "the work being timed did not finish"
        latencies.append(await ping(http))
        await asyncio.sleep(0)
    return latencies


def assert_heartbeat(latencies: list[float], caplog: pytest.LogCaptureFixture) -> None:
    """The two signals of the heartbeat check (see the comment above), with the numbers in the message.

    ``latencies`` are what ``ping_until_done`` measured and ``caplog`` is what ``run_timed`` was
    given: the run must have been made with ``run_timed``, or the second signal sees nothing.
    """
    assert len(latencies) >= MIN_PINGS, (
        f"only {len(latencies)} pings: the window was too short to say anything about a p99"
    )
    slowest = ", ".join(f"{value * 1000:.0f} ms" for value in sorted(latencies)[-3:])
    assert p99(latencies) < HEARTBEAT_P99_S, (
        f"p99 {p99(latencies) * 1000:.0f} ms over {len(latencies)} pings (slowest three: {slowest})"
    )
    blocked = [r.getMessage() for r in caplog.records if "took" in r.getMessage()]
    assert not blocked, f"something held the event loop: {blocked[:3]}"
