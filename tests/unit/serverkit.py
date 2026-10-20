"""Helpers shared by the server tests (steps S4.1 and S4.2).

A fake embedder with a fixed dimension, so an ingest runs the real pipeline (loaders, chunker, SQLite,
Chroma) without Ollama; fake answering tools (a scripted chat model, rewrite call and, unless a test
passes its own, the real hybrid retriever over that store); an app built from them; and a few helpers
for the HTTP side (wait for a job, ask a question and read the stream).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatkit import ScriptedChat, ScriptedCondenser
from fastapi import FastAPI
from fastapi.testclient import TestClient

from geekbot.agent.runtime import AgentContext
from geekbot.config import Settings
from geekbot.index.embedder import EmbedderSpec, PrefixedEmbeddings
from geekbot.index.workspace import Workspace
from geekbot.llm.chat import ChatModel
from geekbot.retrieval import Retriever, SearchMode
from geekbot.retrieval.factory import build_retriever
from geekbot.server import create_app
from geekbot.server.answering import Answering
from geekbot.sse import SseEvent, parse_text

MODEL = "fake-model:1b"
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
    return Settings(_env_file=None, data_dir=root / "data")


@dataclass
class Fakes:
    """What the app was given instead of Ollama, kept on ``app.state.fakes`` for the assertions.

    ``thinks`` is the ``think`` flag of every chat model the route asked for besides the shared one;
    ``loads`` has one entry per warm-up that ran.
    """

    chat: ScriptedChat
    condenser: ScriptedCondenser
    retriever: Retriever | None
    thinks: list[bool] = field(default_factory=list)
    loads: list[str] = field(default_factory=list)


def make_app(
    root: Path,
    *,
    chat: ScriptedChat | None = None,
    condenser: ScriptedCondenser | None = None,
    retriever: Retriever | None = None,
    warm: bool = True,
) -> FastAPI:
    """An app over a fake-embedder workspace with scripted models.

    ``retriever`` replaces the real hybrid retriever (which searches whatever was ingested through
    the API). ``warm=False`` starts the answering tools as ``--no-warmup`` would: not ready, until the
    first question runs the (fake, instant) load.
    """
    fakes = Fakes(
        chat=chat or ScriptedChat(),
        condenser=condenser or ScriptedCondenser(),
        retriever=retriever,
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

        answering = Answering(context, make_llm=make_llm, load=load)
        answering.ready = warm
        return answering

    app = create_app(
        settings_for(root), open_workspace=open_fake_workspace, open_answering=open_answering
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
