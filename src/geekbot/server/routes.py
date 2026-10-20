"""The HTTP routes of step S4.1: health, the knowledge base, and jobs.

Every route that touches SQLite, Chroma or the file system hands the blocking call to a worker
thread (``anyio.to_thread.run_sync``); the handlers themselves only await and build responses
(PLAN.md §6.3). Work that *changes* the storage (ingest, delete) is not done in the request at all:
it becomes a job (``jobs.py``), so there is exactly one writer at a time and the request returns at
once with HTTP 202.

Not here yet: threads and streamed answers (S4.2), the path allowlist, upload limits and the API
key (S4.3). Until S4.3 the ingest route accepts any path that exists on this machine; the server
listens on 127.0.0.1 only (``app.HOST``), so only a program on this PC can ask.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Request

from geekbot.domain import DocState
from geekbot.index.workspace import Workspace
from geekbot.ingest.pipeline import FileOutcome, Ingestor, collect_files
from geekbot.server.jobs import Job, JobQueue, QueueFullError
from geekbot.server.schemas import (
    DocumentOut,
    HealthOut,
    IngestRequest,
    JobOut,
    ReadyOut,
)

__all__ = ["router"]

router = APIRouter()


def workspace_of(request: Request) -> Workspace:
    """The opened storage, kept on the application by the lifespan."""
    workspace: Workspace = request.app.state.workspace
    return workspace


def jobs_of(request: Request) -> JobQueue:
    """The job queue, kept on the application by the lifespan."""
    jobs: JobQueue = request.app.state.jobs
    return jobs


WorkspaceDep = Annotated[Workspace, Depends(workspace_of)]
JobsDep = Annotated[JobQueue, Depends(jobs_of)]


# --- health ----------------------------------------------------------------------------------


@router.get("/healthz")
async def healthz() -> HealthOut:
    """Liveness: the process is up and its event loop answers.

    ✍️ written by Claude (prototyping mode, D-047). It deliberately does *nothing* else: no
    storage, no model, no ``await`` on anything slow. That is the whole point of the route: the
    heartbeat test (§6.3) calls it while an ingest and an answer run, and if any blocking call
    sneaks onto the event loop, this is the route whose latency shows it. A readiness check that
    reads the storage is ``/v1/health``.
    """
    return HealthOut()


@router.get("/v1/health")
async def readiness(workspace: WorkspaceDep, jobs: JobsDep) -> ReadyOut:
    """Readiness: what the storage holds and how busy the server is."""

    def read() -> tuple[Counter[str], int, int]:
        states = Counter[str](document.state for document in workspace.store.list_documents())
        chunks = sum(len(ids) for ids in workspace.store.chunk_ids_by_document().values())
        return states, chunks, workspace.dense.count()

    states, chunks, vectors = await anyio.to_thread.run_sync(read)
    return ReadyOut(
        status="ok",
        documents=dict(states),
        chunks=chunks,
        vectors=vectors,
        embedder=workspace.embedder.spec.model,
        jobs_active=jobs.active,
    )


# --- the knowledge base ----------------------------------------------------------------------


@router.get("/v1/kb/documents")
async def list_documents(
    workspace: WorkspaceDep, state: DocState | None = None
) -> list[DocumentOut]:
    """Every stored document (optionally only those in one outbox ``state``)."""
    documents = await anyio.to_thread.run_sync(workspace.store.list_documents, state)
    return [DocumentOut.of(document) for document in documents]


@router.post("/v1/kb/ingest", status_code=202)
async def ingest(body: IngestRequest, workspace: WorkspaceDep, jobs: JobsDep) -> JobOut:
    """Queue an ingest of these files or folders; poll ``/v1/jobs/{id}`` for the outcome."""
    paths = [Path(text) for text in body.paths]
    missing = await anyio.to_thread.run_sync(_missing, paths)
    if missing:
        raise HTTPException(status_code=422, detail=f"not found on this machine: {missing}")

    def work(job: Job) -> dict[str, Any]:
        files, unsupported = collect_files(paths)
        ingestor = Ingestor(
            workspace.store, workspace.dense, workspace.embedder, workers=body.workers
        )
        report = ingestor.ingest(files, on_file=lambda outcome: job.progress.append(_row(outcome)))
        return {
            "summary": report.summary(),
            "failed": report.has_failures,
            "unsupported": [str(path) for path in unsupported],
            "files": [_row(outcome) for outcome in report.outcomes],
        }

    return _submit(jobs, "ingest", work)


@router.delete("/v1/kb/documents/{doc_id}", status_code=202)
async def delete_document(doc_id: str, workspace: WorkspaceDep, jobs: JobsDep) -> JobOut:
    """Queue the removal of one document: its rows first, then its vectors."""
    if await anyio.to_thread.run_sync(workspace.store.get_document, doc_id) is None:
        raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")

    def work(job: Job) -> dict[str, Any]:
        removed = workspace.store.delete_document(doc_id)
        workspace.dense.delete_document(doc_id)
        return {"doc_id": doc_id, "removed": removed}

    return _submit(jobs, "delete", work)


# --- jobs ------------------------------------------------------------------------------------


@router.get("/v1/jobs")
async def list_jobs(jobs: JobsDep) -> list[JobOut]:
    """The kept jobs, newest first."""
    return [JobOut.of(job) for job in jobs.all()]


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, jobs: JobsDep) -> JobOut:
    """One job, or 404 (never existed, or forgotten after a restart)."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
    return JobOut.of(job)


# --- helpers ---------------------------------------------------------------------------------


def _submit(jobs: JobQueue, kind: str, work: Callable[[Job], dict[str, Any]]) -> JobOut:
    try:
        return JobOut.of(jobs.submit(kind, work))
    except QueueFullError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error


def _missing(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _row(outcome: FileOutcome) -> dict[str, Any]:
    return {
        "path": str(outcome.path),
        "status": outcome.status,
        "chunks": outcome.chunks,
        "embedded": outcome.embedded,
        "cache_hits": outcome.cache_hits,
        "error": outcome.error,
    }
