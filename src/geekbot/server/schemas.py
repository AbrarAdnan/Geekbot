"""The shapes that go over the wire (steps S4.1 and S4.2).

Plain pydantic models, so FastAPI validates what comes in and documents what goes out. They are
copies of the domain records (``Document``, ``Job``) and of the graph's state, not the records
themselves: the wire format is a promise to the UI, and a field added to a domain record or to the
state must not leak into it by accident.

**Untrusted text.** ``SourceOut.text`` is a passage of an ingested document, ``MessageOut.content`` of
an assistant message is a model's text built from such passages: neither is Geekbot's own. The answer
has been through the output cleaning (G3: no images, no links to places the sources do not name), the
passage has not. A client must show a passage as plain text, never render it as Markdown or HTML.
"""

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from geekbot.agent.runner import ThreadInfo
from geekbot.domain import Document, Label, is_tainted
from geekbot.server.jobs import Job, JobState

__all__ = [
    "AskRequest",
    "CheckOut",
    "DocumentOut",
    "HealthOut",
    "IngestRequest",
    "JobOut",
    "MessageOut",
    "ReadyOut",
    "SourceOut",
    "SourceRefOut",
    "ThreadCreated",
    "ThreadOut",
    "ThreadSummary",
    "TurnOut",
]

MAX_INGEST_PATHS = 200  # one request; a bigger library is several requests (or a folder path)
MAX_WORKERS = 8
# Far above what a question may be (about 1,000 estimated tokens, ``agent.packing.check_question``),
# so the length check that explains itself runs first; this only stops a huge body being read in full.
MAX_QUESTION_CHARS = 20_000
PREVIEW_CHARS = 80  # how much of a thread's first question the thread list shows


class HealthOut(BaseModel):
    """``/healthz``: the process is up and its event loop answers. Touches no storage."""

    status: str = "ok"


class ReadyOut(BaseModel):
    """``/v1/health``: what the storage holds, and what the server is busy with.

    ``chat_ready`` is false until the chat model is loaded (``geekbot serve --no-warmup`` leaves that to
    the first question); ``answering`` is true while a question is being answered.
    """

    status: str
    documents: dict[str, int]
    chunks: int
    vectors: int
    embedder: str
    jobs_active: int
    chat_model: str
    chat_ready: bool
    answering: bool


class DocumentOut(BaseModel):
    """One document in the knowledge base."""

    doc_id: str
    path: str
    title: str
    kind: str
    state: str
    error: str | None
    updated_at: str

    @classmethod
    def of(cls, document: Document) -> Self:
        return cls(
            doc_id=document.doc_id,
            path=document.path,
            title=document.title,
            kind=document.kind,
            state=document.state,
            error=document.error,
            updated_at=document.updated_at,
        )


class IngestRequest(BaseModel):
    """Files or folders to index. ``extra=forbid``: a misspelt option is an error, not ignored."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(min_length=1, max_length=MAX_INGEST_PATHS)
    workers: int = Field(default=1, ge=1, le=MAX_WORKERS)


class JobOut(BaseModel):
    """A job and where it stands; ``result`` is set when it is done."""

    id: str
    kind: str
    state: JobState
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress: list[dict[str, Any]]
    result: dict[str, Any] | None
    error: str | None

    @classmethod
    def of(cls, job: Job) -> Self:
        return cls(
            id=job.id,
            kind=job.kind,
            state=job.state,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            progress=list(job.progress),
            result=job.result,
            error=job.error,
        )


# --- threads and answers (step S4.2) ----------------------------------------------------------


class ThreadCreated(BaseModel):
    """``POST /v1/threads``: a fresh thread id.

    Nothing is stored until the thread's first question has been answered: a thread is what the
    checkpointer saved after a turn, so an id that was never used is not listed and is a 404.
    """

    thread_id: str
    mode: str = "chat"


class AskRequest(BaseModel):
    """A question for a thread. ``think`` asks the model to reason first (slower; only for a model
    with a thinking mode). ``extra=forbid``: a misspelt option is an error, not ignored."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    think: bool = False


class SourceRefOut(BaseModel):
    """Where an answer's ``[n]`` mark points: the document, and the place in it. No passage text."""

    n: int
    chunk_id: str
    title: str
    path: str
    page: int | None
    section: str


class SourceOut(SourceRefOut):
    """A source of the turn that was just answered: the reference, the retrieval score, the passage.

    ``text`` is the stored passage: untrusted, show it as plain text (see the module docstring).
    """

    score: float
    text: str


class CheckOut(BaseModel):
    """How well a cited source fits the sentence that cites it (the relevance check, S3.5)."""

    n: int
    score: float
    weak: bool
    sentence: str


class TurnOut(BaseModel):
    """The ``done`` event: one finished turn.

    ``cited`` are the ``[n]`` marks the answer still carries, ``dropped_citations`` the ones the check
    removed because no such source was in the prompt. ``sources`` are the ones the prompt held, in the
    order the model saw them. ``labels`` and ``thread_taint`` are the taint bookkeeping of PLAN.md
    section 8.2. ``searched_as`` is what retrieval looked for when it was not the user's own words (a
    rewritten follow-up). ``generation`` holds the readings of the model call (seconds, time to the
    first token, model load, tokens); a reading the model did not give is absent.
    """

    thread_id: str
    turn: int
    mode: str
    model: str
    answer: str
    abstained: bool
    cited: list[int]
    dropped_citations: list[int]
    sources: list[SourceOut]
    labels: list[str]
    thread_taint: list[str]
    tainted: bool
    searched_as: str | None
    rewrite_error: str | None
    rewrite_seconds: float
    citation_checks: list[CheckOut]
    citation_check_error: str | None
    removed_markup: list[str]
    retrieval_ms: dict[str, float]
    generation: dict[str, float]

    @classmethod
    def of(cls, final: Mapping[str, Any], *, thread_id: str, model: str) -> Self:
        """The turn out of the graph's final state."""
        used = int(final.get("used", 0))
        sources = [
            SourceOut(
                n=source["n"],
                chunk_id=source["chunk_id"],
                title=source["title"],
                path=source["path"],
                page=source["page"],
                section=source["section"],
                score=source["score"],
                text=source["text"],
            )
            for source in list(final.get("sources", []))[:used]
        ]
        question = str(final.get("question", ""))
        searched = str(final.get("standalone_question", ""))
        taint = [str(name) for name in final.get("thread_taint", [])]
        return cls(
            thread_id=thread_id,
            turn=int(final.get("turn", 0)),
            mode=str(final.get("mode") or "chat"),
            model=model,
            answer=str(final.get("answer", "")),
            abstained=bool(final.get("abstained", False)),
            cited=[int(n) for n in final.get("cited", [])],
            dropped_citations=[int(n) for n in final.get("dropped_citations", [])],
            sources=sources,
            labels=[str(name) for name in final.get("answer_labels", [])],
            thread_taint=taint,
            tainted=is_tainted(Label(name) for name in taint),
            searched_as=searched if searched and searched != question else None,
            rewrite_error=str(final.get("condense_error", "")) or None,
            rewrite_seconds=float(final.get("condense_seconds", 0.0)),
            citation_checks=[
                CheckOut(
                    n=check["n"],
                    score=check["score"],
                    weak=check["weak"],
                    sentence=check["sentence"],
                )
                for check in final.get("citation_checks", [])
            ],
            citation_check_error=str(final.get("citation_check_error", "")) or None,
            removed_markup=[str(item) for item in final.get("removed_markup", [])],
            retrieval_ms={name: float(ms) for name, ms in final.get("retrieval_ms", {}).items()},
            generation={name: float(v) for name, v in final.get("generation", {}).items()},
        )


class MessageOut(BaseModel):
    """One remembered message of a thread.

    An assistant message that was built from sources carries them (``sources``, numbered as the model
    saw them) and the marks it cites (``cited``), so a reloaded thread can still show what each answer
    rested on. A user message, a "nothing found" answer and a CSV answer have neither.
    """

    id: str
    role: Literal["user", "assistant"]
    content: str
    labels: list[str]
    cited: list[int] = Field(default_factory=list)
    sources: list[SourceRefOut] = Field(default_factory=list)

    @classmethod
    def of(cls, message: Mapping[str, Any]) -> Self:
        return cls(
            id=message["id"],
            role=message["role"],
            content=message["content"],
            labels=[str(name) for name in message.get("labels", [])],
            cited=[int(n) for n in message.get("cited", [])],
            sources=[SourceRefOut(**source) for source in message.get("sources", [])],
        )


class ThreadSummary(BaseModel):
    """A thread in the list: what it is, how long, and how it starts.

    ``preview`` is the oldest question the thread still remembers (a thread keeps its newest messages
    only), on one line and cut to ``PREVIEW_CHARS``. ``taint`` is every label the thread ever
    carried (it only grows); ``tainted`` is whether any of them is an outsider's.
    """

    thread_id: str
    mode: str
    csv_file: str | None
    turns: int
    message_count: int
    preview: str
    taint: list[str]
    tainted: bool
    updated_at: str | None

    @classmethod
    def of(cls, info: ThreadInfo) -> Self:
        values = info.values
        history = list(values.get("history", []))
        first = next((m["content"] for m in history if m.get("role") == "user"), "")
        preview = " ".join(str(first).split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[: PREVIEW_CHARS - 3].rstrip() + "..."
        taint = [str(name) for name in values.get("thread_taint", [])]
        return cls(
            thread_id=info.thread_id,
            mode=str(values.get("mode") or "chat"),
            csv_file=str(values.get("csv_file") or "") or None,
            turns=int(values.get("turn", 0)),
            message_count=len(history),
            preview=preview,
            taint=taint,
            tainted=is_tainted(Label(name) for name in taint),
            updated_at=info.updated_at,
        )


class ThreadOut(ThreadSummary):
    """A thread with its remembered messages, oldest first."""

    messages: list[MessageOut]

    @classmethod
    def of(cls, info: ThreadInfo) -> Self:
        summary = ThreadSummary.of(info)
        return cls(
            **summary.model_dump(),
            messages=[MessageOut.of(message) for message in info.values.get("history", [])],
        )
