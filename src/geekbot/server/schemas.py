"""The shapes that go over the wire (step S4.1).

Plain pydantic models, so FastAPI validates what comes in and documents what goes out. They are
copies of the domain records (``Document``, ``Job``), not the records themselves: the wire format
is a promise to the UI, and a field added to a domain record must not leak into it by accident.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from geekbot.domain import Document
from geekbot.server.jobs import Job, JobState

__all__ = [
    "DocumentOut",
    "HealthOut",
    "IngestRequest",
    "JobOut",
    "ReadyOut",
]

MAX_INGEST_PATHS = 200  # one request; a bigger library is several requests (or a folder path)
MAX_WORKERS = 8


class HealthOut(BaseModel):
    """``/healthz``: the process is up and its event loop answers. Touches no storage."""

    status: str = "ok"


class ReadyOut(BaseModel):
    """``/v1/health``: what the storage holds, and what the server is busy with."""

    status: str
    documents: dict[str, int]
    chunks: int
    vectors: int
    embedder: str
    jobs_active: int


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
