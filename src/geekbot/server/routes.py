"""The HTTP routes of steps S4.1, S4.3 and S4.4: health, the knowledge base, and jobs.

Every route that touches SQLite, Chroma or the file system hands the blocking call to a worker
thread (``anyio.to_thread.run_sync``); the handlers themselves only await and build responses
(PLAN.md §6.3). Work that *changes* the storage (ingest, delete) is not done in the request at all:
it becomes a job (``jobs.py``), so there is exactly one writer at a time and the request returns at
once with HTTP 202.

Threads and streamed answers are in ``threads.py`` (S4.2); the dependency helpers below are shared by
both modules.

**What the knowledge-base routes may read (S4.3).** A client never gets a path to read from the
server's disk unless the user opened that folder: ``/v1/kb/ingest`` takes full paths that must pass
``security.PathPolicy`` (inside ``GEEKBOT_ALLOWED_ROOTS`` or the uploads folder, never the data
folder, no network paths, no escape through a link), and every file a folder walk finds is checked
again. ``/v1/kb/upload`` is the way in that needs no folder: the client sends the bytes. Who may call
at all (the API key, the Host check, the body-size caps) is the middleware in ``security.py``, in
front of every route here.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from geekbot.config import Settings
from geekbot.csv_mode.sessions import CsvSessions
from geekbot.domain import DocState
from geekbot.index.workspace import Workspace
from geekbot.ingest.loaders import SUPPORTED_EXTENSIONS
from geekbot.ingest.pipeline import FileOutcome, Ingestor, collect_files
from geekbot.server.answering import Answering, TurnGate
from geekbot.server.csvmode import CsvSession
from geekbot.server.jobs import Job, JobQueue, QueueFullError
from geekbot.server.schemas import (
    DocumentOut,
    HealthOut,
    IngestRequest,
    JobOut,
    ReadyOut,
    UploadOut,
)
from geekbot.server.security import (
    MAX_REFUSED_LISTED,
    UPLOADS_DIR,
    PathNotAllowedError,
    PathPolicy,
    Perimeter,
)
from geekbot.server.uploads import UploadError, save_upload

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


def settings_of(request: Request) -> Settings:
    """The settings the server was started with, kept on the application by the lifespan."""
    settings: Settings = request.app.state.settings
    return settings


def answering_of(request: Request) -> Answering:
    """The answering tools built at start-up (S4.2), kept on the application by the lifespan."""
    answering: Answering = request.app.state.answering
    return answering


def gate_of(request: Request) -> TurnGate:
    """The one-question-at-a-time gate (S4.2), kept on the application by the lifespan."""
    gate: TurnGate = request.app.state.gate
    return gate


def perimeter_of(request: Request) -> Perimeter:
    """The perimeter (S4.3): the path policy and the upload cap, kept on the application."""
    perimeter: Perimeter = request.app.state.perimeter
    return perimeter


def csv_sessions_of(request: Request) -> CsvSessions[CsvSession]:
    """The registry of loaded CSV sessions (C5), kept on the application by the lifespan."""
    sessions: CsvSessions[CsvSession] = request.app.state.csv_sessions
    return sessions


WorkspaceDep = Annotated[Workspace, Depends(workspace_of)]
JobsDep = Annotated[JobQueue, Depends(jobs_of)]
SettingsDep = Annotated[Settings, Depends(settings_of)]
AnsweringDep = Annotated[Answering, Depends(answering_of)]
GateDep = Annotated[TurnGate, Depends(gate_of)]
PerimeterDep = Annotated[Perimeter, Depends(perimeter_of)]
CsvSessionsDep = Annotated[CsvSessions[CsvSession], Depends(csv_sessions_of)]


# --- health ----------------------------------------------------------------------------------


@router.get("/healthz")
async def healthz() -> HealthOut:
    """Liveness: the process is up and its event loop answers.

    ✍️ written by Claude (prototyping mode, D-047). It deliberately does *nothing* else: no
    storage, no model, no ``await`` on anything slow. That is the whole point of the route: the
    heartbeat test (§6.3) calls it while an ingest and an answer run, and if any blocking call
    sneaks onto the event loop, this is the route whose latency shows it. A readiness check that
    reads the storage is ``/v1/health``. It is also the one route that asks for no API key
    (``security.PUBLIC_PATHS``), so the UI can tell that the server is up before it sends anything.
    """
    return HealthOut()


@router.get("/v1/health")
async def readiness(
    workspace: WorkspaceDep,
    jobs: JobsDep,
    answering: AnsweringDep,
    gate: GateDep,
    sessions: CsvSessionsDep,
) -> ReadyOut:
    """Readiness: what the storage holds, which model is loaded, what is running."""

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
        chat_model=answering.model,
        chat_ready=answering.ready,
        answering=gate.busy,
        csv_model=answering.csv.model if answering.csv is not None else "",
        csv_ready=answering.csv_ready,
        csv_sessions=sessions.count,
        web_enabled=answering.web_enabled,
    )


# --- the knowledge base ----------------------------------------------------------------------


@router.get("/v1/kb/documents")
async def list_documents(
    workspace: WorkspaceDep, state: DocState | None = None
) -> list[DocumentOut]:
    """Every stored document (optionally only those in one outbox ``state``)."""
    documents = await anyio.to_thread.run_sync(workspace.store.list_documents, state)
    return [DocumentOut.of(document) for document in documents]


@router.get("/v1/kb/formats")
async def formats() -> list[str]:
    """The file extensions the loaders read, with the dot, sorted: what ``/v1/kb/upload`` accepts.

    The UI (S4.4) asks instead of keeping a list of its own, so a loader added later shows up in its
    file chooser without a change there.
    """
    return sorted(SUPPORTED_EXTENSIONS)


@router.post("/v1/kb/ingest", status_code=202)
async def ingest(
    body: IngestRequest, workspace: WorkspaceDep, jobs: JobsDep, perimeter: PerimeterDep
) -> JobOut:
    """Queue an ingest of these files or folders; poll ``/v1/jobs/{id}`` for the outcome.

    Each path must be a full path inside a folder the server may read (see the module docstring):
    403 for a path that is not allowed, 422 for text that is not a path or a path that does not exist.
    Nothing is queued unless every path passes. The documents are stored under their *real* paths
    (links resolved), which is what was checked and so what is read.
    """
    paths = await anyio.to_thread.run_sync(_admit, perimeter.paths, body.paths)
    return _submit(jobs, "ingest", _ingest_work(workspace, perimeter.paths, paths, body.workers))


@router.post("/v1/kb/upload", status_code=202)
async def upload(
    request: Request,
    filename: Annotated[str, Query(min_length=1, max_length=255)],
    settings: SettingsDep,
    workspace: WorkspaceDep,
    jobs: JobsDep,
    perimeter: PerimeterDep,
) -> UploadOut:
    """Receive one file as the raw request body, save it, and queue its ingest.

    ``filename`` says what the file is called and what type it is; only the name's last part and its
    extension are used (``uploads.safe_upload_name``). The file is kept in the uploads folder as
    ``<sha16>-<safe name>``. 415 for a type the loaders do not read, 413 over the size cap, 422 for an
    empty body. If the job queue is full the file stays saved and the answer is 429: send it again.
    """
    try:
        stored = await save_upload(
            request.stream(),
            filename=filename,
            directory=settings.data_dir / UPLOADS_DIR,
            max_bytes=perimeter.max_upload_bytes,
            suffixes=SUPPORTED_EXTENSIONS,
        )
    except UploadError as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
    job = _submit(jobs, "ingest", _ingest_work(workspace, perimeter.paths, [stored.path], 1))
    return UploadOut(
        job=job,
        stored_as=stored.path.name,
        sha256=stored.sha256,
        size_bytes=stored.size,
        existed=stored.existed,
    )


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


def _admit(policy: PathPolicy, raw_paths: list[str]) -> list[Path]:
    """The real path of every requested path, or the HTTP error that refuses the whole request.

    All paths are checked against the policy before any is looked for, so a request that names a
    forbidden path is answered 403 (422 for a malformed one) even when another path in it is missing,
    and a missing path says nothing about the disk outside the allowed folders. Blocking: it resolves
    links and looks at the disk, so the route runs it in a worker thread.
    """
    admitted: list[Path] = []
    for raw in raw_paths:
        try:
            real = policy.check(raw)
        except PathNotAllowedError as error:
            status = 422 if error.malformed else 403
            raise HTTPException(status_code=status, detail=str(error)) from error
        if real not in admitted:
            admitted.append(real)
    missing = [str(path) for path in admitted if not path.exists()]
    if missing:
        raise HTTPException(status_code=422, detail=f"not found on this machine: {missing}")
    return admitted


def _ingest_work(
    workspace: Workspace, policy: PathPolicy, paths: list[Path], workers: int
) -> Callable[[Job], dict[str, Any]]:
    """The job that ingests ``paths``: walk them, drop what the policy refuses, ingest the rest.

    The paths were admitted when the request came in; the files *inside* a folder are checked here,
    one by one, because a link in an allowed folder can lead outside it (``PathPolicy.split``).
    Refused files are not an error: they are listed in the result (``refused``, the first
    ``MAX_REFUSED_LISTED``, and ``refused_count``) and the rest of the folder is ingested.
    """

    def work(job: Job) -> dict[str, Any]:
        found, unsupported = collect_files(paths)
        files, refused = policy.split(found)
        left_out, hidden = policy.split(unsupported)
        ingestor = Ingestor(workspace.store, workspace.dense, workspace.embedder, workers=workers)
        report = ingestor.ingest(files, on_file=lambda outcome: job.progress.append(_row(outcome)))
        everything_refused = [*refused, *hidden]
        return {
            "summary": report.summary(),
            "failed": report.has_failures,
            "unsupported": [str(path) for path in left_out],
            "refused": [str(path) for path in everything_refused[:MAX_REFUSED_LISTED]],
            "refused_count": len(everything_refused),
            "files": [_row(outcome) for outcome in report.outcomes],
        }

    return work


def _row(outcome: FileOutcome) -> dict[str, Any]:
    return {
        "path": str(outcome.path),
        "status": outcome.status,
        "chunks": outcome.chunks,
        "embedded": outcome.embedded,
        "cache_hits": outcome.cache_hits,
        "error": outcome.error,
    }
