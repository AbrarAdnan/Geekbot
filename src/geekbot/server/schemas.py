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
from geekbot.server.uploads import display_name

__all__ = [
    "ApprovalOut",
    "ApprovalRequest",
    "AskRequest",
    "CheckOut",
    "CsvUploadOut",
    "DocumentOut",
    "EgressOut",
    "HealthOut",
    "IngestRequest",
    "JobOut",
    "MessageOut",
    "ReadyOut",
    "SourceOut",
    "SourceRefOut",
    "ThreadCreated",
    "ThreadOut",
    "ThreadRequest",
    "ThreadSummary",
    "TurnOut",
    "UploadOut",
]

MAX_INGEST_PATHS = 200  # one request; a bigger library is several requests (or a folder path)
MAX_WORKERS = 8
# Far above what a question may be (about 1,000 estimated tokens, ``agent.packing.check_question``),
# so the length check that explains itself runs first; this only stops a huge body being read in full.
MAX_QUESTION_CHARS = 20_000
PREVIEW_CHARS = 80  # how much of a thread's first question the thread list shows


def _shown_csv_file(stored: object) -> str | None:
    """The name of a conversation's CSV file as the user knows it (the stored name without its
    ``<sha16>-`` prefix), or ``None`` if the thread has none. The thread keeps the stored name so
    that the server can open the same file again (``csvmode.stored_file``)."""
    name = str(stored or "")
    return display_name(name) if name else None


class HealthOut(BaseModel):
    """``/healthz``: the process is up and its event loop answers. Touches no storage."""

    status: str = "ok"


class ReadyOut(BaseModel):
    """``/v1/health``: what the storage holds, and what the server is busy with.

    ``chat_ready`` is false until the chat model is loaded (``geekbot serve --no-warmup`` leaves that to
    the first question); ``answering`` is true while a question is being answered. Since C5 the CSV
    model is reported too (``csv_model`` is empty when none is set up; ``csv_ready`` is true while it,
    and not the chat model, is the one on the GPU) and so is the number of loaded CSV sessions.
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
    csv_model: str = ""
    csv_ready: bool = False
    csv_sessions: int = 0
    web_enabled: bool = False  # S5.3: the web search tool exists on this server (off by default)


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


class UploadOut(BaseModel):
    """``POST /v1/kb/upload``: the file was saved and its ingest is queued as ``job``.

    ``stored_as`` is the file's name in the server's uploads folder (``<sha16>-<safe name>``, never a
    path), ``sha256`` and ``size_bytes`` describe what arrived, and ``existed`` says the same bytes
    had already been uploaded under that name (the job then finds the document unchanged).
    """

    job: JobOut
    stored_as: str
    sha256: str
    size_bytes: int
    existed: bool


class CsvUploadOut(BaseModel):
    """``POST /v1/csv/upload``: the CSV was saved and loaded as the thread's session.

    Unlike a document upload there is no job: loading and profiling the table is done when the
    answer arrives. ``file`` is the name the user gave it, ``stored_as`` its name in the uploads
    folder, ``existed`` that the same bytes had been uploaded before. ``columns`` are the first
    ``csvmode.MAX_COLUMNS_LISTED`` names (``column_count`` is how many there are) and ``notes`` what
    the loader changed or skipped. **Column names and notes come from the file: untrusted text**, to
    be shown as plain text and never as Markdown. ``memory_bytes`` is what the table occupies in
    memory, which is what the server's bound on CSV sessions counts.
    """

    thread_id: str
    mode: Literal["csv"] = "csv"
    file: str
    stored_as: str
    sha256: str
    size_bytes: int
    existed: bool
    rows: int
    column_count: int
    columns: list[str]
    notes: list[str]
    memory_bytes: int


# --- threads and answers (step S4.2) ----------------------------------------------------------


class ThreadRequest(BaseModel):
    """The optional body of ``POST /v1/threads``: the mode the conversation is meant to have.

    ``extra=forbid`` and a closed set of names: a misspelt mode is a 422, not a chat conversation.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["chat", "csv"] = "chat"


class ThreadCreated(BaseModel):
    """``POST /v1/threads``: a fresh thread id, and the mode it was asked for.

    Nothing is stored until the thread's first question has been answered: a thread is what the
    checkpointer saved after a turn, so an id that was never used is not listed and is a 404. The
    mode is therefore a statement of intent, not a record: what makes a thread a CSV conversation
    is the CSV file uploaded to it (``POST /v1/csv/upload``), and the first answered turn fixes the
    mode for good.
    """

    thread_id: str
    mode: str = "chat"


class AskRequest(BaseModel):
    """A question for a thread. ``think`` asks the model to reason first (slower; only for a model
    with a thinking mode; a CSV conversation ignores it). ``extra=forbid``: a misspelt option is an
    error, not ignored.

    ``mode`` is what the client believes the conversation is (C5). Left out, the thread decides:
    a saved thread keeps its mode, and an unsaved one with a CSV file uploaded to it is a CSV
    conversation. Given, it must agree with the thread (409 otherwise), and ``"csv"`` on a thread
    the server has no file for is a 409 that says to upload it again, instead of a question quietly
    answered from the documents. The UI sends it for every CSV conversation.

    ``steps`` (S3.6) asks for progress events: while a chat question runs, ``status`` frames name the
    step it has reached (``searching``, ``ranking``, ``writing``). Off by default, so a client that
    only wants the answer receives only the answer's events.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    think: bool = False
    mode: Literal["chat", "csv"] | None = None
    steps: bool = False
    web: bool = False


class ApprovalRequest(BaseModel):
    """``POST /v1/threads/{id}/approval``: the user's answer to the egress gate's question (S5.3).

    ``approve`` sends the exact query that was shown; ``edit`` sends ``query`` instead (the user's own
    text); ``deny`` sends nothing and the turn goes on without the web. ``query`` is only read for
    ``edit``. Closed fields and a closed set of actions: an answer the server cannot read is a 422 and
    never counts as consent.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "edit", "deny"]
    query: str = Field(default="", max_length=MAX_QUESTION_CHARS)
    think: bool = False  # as in the question that stopped: the continued turn writes its answer
    steps: bool = False  # with the same chat model and the same progress events


class ApprovalOut(BaseModel):
    """The egress gate's question: the exact text that would leave the machine, and why it is asked.

    ``labels`` are what the text was built from (``KB``, ``WEB``, ``TOOL``, ``UNKNOWN`` mean text the
    user did not write). Sent as the ``approval`` event of a stream, and as ``pending_approval`` of a
    thread that is waiting. ``query`` is built from untrusted text: show it as plain text.
    """

    tool: str
    query: str
    labels: list[str]
    reason: str
    question: str = ""  # what the user typed: the waiting turn is not in the thread's history yet

    @classmethod
    def of(cls, question: Mapping[str, Any]) -> Self:
        return cls(
            tool=str(question.get("tool", "")),
            question=str(question.get("question", "")),
            query=str(question.get("query", "")),
            labels=[str(name) for name in question.get("labels", [])],
            reason=str(question.get("reason", "")),
        )


class EgressOut(BaseModel):
    """One decision of the egress gate during a turn (see ``agent.state.EgressRecord``)."""

    tool: str
    query: str
    decision: str
    tainted: bool
    reason: str


class SourceRefOut(BaseModel):
    """Where an answer's ``[n]`` mark points: the document, and the place in it. No passage text.

    ``origin`` is ``KB`` for a document and ``WEB`` for a search result (then ``path`` is its address,
    to be shown as text, and ``section`` its host).
    """

    n: int
    chunk_id: str
    title: str
    path: str
    page: int | None
    section: str
    origin: str = "KB"


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

    In a CSV conversation (C5) ``mode`` is ``csv``, there are no sources, ``csv_file`` is the name of
    the file asked about and ``csv_status`` is the package's verdict (``answered``, ``abstained``,
    ``needs_clarification`` or ``failed``); ``answer`` is the package's reply, which carries the SQL
    that was run and a preview of its result. ``generation`` then has ``seconds`` (the whole
    question) and ``load_seconds``. Both CSV fields are ``None`` in chat mode.
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
    csv_file: str | None = None
    csv_status: str | None = None
    egress: list[EgressOut] = Field(default_factory=list)
    web_error: str | None = None

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
                origin=str(source.get("origin") or "KB"),
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
            csv_file=_shown_csv_file(final.get("csv_file")),
            csv_status=str(final.get("csv_status") or "") or None,
            egress=[
                EgressOut(
                    tool=record["tool"],
                    query=record["query"],
                    decision=record["decision"],
                    tainted=bool(record["tainted"]),
                    reason=record["reason"],
                )
                for record in final.get("egress_log", [])
            ],
            web_error=str(final.get("web_error") or "") or None,
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
            csv_file=_shown_csv_file(values.get("csv_file")),
            turns=int(values.get("turn", 0)),
            message_count=len(history),
            preview=preview,
            taint=taint,
            tainted=is_tainted(Label(name) for name in taint),
            updated_at=info.updated_at,
        )


class ThreadOut(ThreadSummary):
    """A thread with its remembered messages, oldest first.

    ``pending_approval`` is the egress gate's open question when the thread's last turn is waiting for
    the user (S5.3); ``None`` otherwise.
    """

    messages: list[MessageOut]
    pending_approval: ApprovalOut | None = None

    @classmethod
    def of(cls, info: ThreadInfo) -> Self:
        summary = ThreadSummary.of(info)
        return cls(
            **summary.model_dump(),
            messages=[MessageOut.of(message) for message in info.values.get("history", [])],
            pending_approval=ApprovalOut.of(info.pending) if info.pending else None,
        )
