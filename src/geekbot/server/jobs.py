"""A queue that runs long storage work one job at a time, off the event loop (step S4.1).

Why a queue and not "run it in the request": an ingest can take minutes, and ``Ingestor`` is not
thread-safe (it says so), so two ingests at once would be a bug. One worker task takes jobs in the
order they came; each job's blocking work runs in a worker thread via ``anyio.to_thread`` so the
event loop stays free to answer ``/healthz`` and, later, stream answers (PLAN.md §6.3). The caller
gets a job id at once (HTTP 202) and polls it.

A job that raises is a *failed job*, never a dead worker: the next job still runs. The queue is
bounded; when it is full ``submit`` raises ``QueueFullError`` (HTTP 429) instead of piling up work
nobody is waiting for. Finished jobs are kept in memory only, the last ``keep`` of them; a restart
forgets them (the documents' own states in SQLite are the durable record).
"""

import asyncio
import contextlib
import logging
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Literal

import anyio.to_thread

__all__ = ["Job", "JobQueue", "JobState", "QueueFullError"]

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "done", "failed"]
DEFAULT_MAX_PENDING: Final = 16
DEFAULT_KEEP: Final = 100


class QueueFullError(RuntimeError):
    """Too many jobs are waiting; try again later."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    """One unit of work and where it stands.

    ``progress`` is appended to from the worker thread (a list ``append`` is atomic in CPython) and
    only read by the routes, so it needs no lock.
    """

    id: str
    kind: str
    state: JobState = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


# The work gets its own job (to report progress on) and returns the result to store.
Work = Callable[[Job], dict[str, Any]]


class JobQueue:
    """Runs submitted jobs one after another. Create it, ``start`` it inside the running loop."""

    def __init__(self, *, max_pending: int = DEFAULT_MAX_PENDING, keep: int = DEFAULT_KEEP) -> None:
        self._max_pending = max_pending
        self._keep = keep
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._queue: asyncio.Queue[tuple[Job, Work]] | None = None
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Create the queue and the worker task. Call from the event loop that will run the jobs."""
        if self._worker is not None:
            raise RuntimeError("the job queue is already started")
        self._queue = asyncio.Queue(maxsize=self._max_pending)
        self._worker = asyncio.create_task(self._run(), name="geekbot-job-worker")

    async def stop(self) -> None:
        """Stop taking jobs. A job already running finishes first (a thread cannot be cancelled)."""
        worker, self._worker = self._worker, None
        if worker is None:
            return
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    def submit(self, kind: str, work: Work) -> Job:
        """Queue ``work`` and return its ``Job`` (state ``queued``). Never waits."""
        if self._queue is None:
            raise RuntimeError("the job queue is not started")
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        try:
            self._queue.put_nowait((job, work))
        except asyncio.QueueFull:
            raise QueueFullError(
                f"{self._max_pending} jobs are already waiting; try again when some have finished"
            ) from None
        self._jobs[job.id] = job
        self._forget_old()
        return job

    def get(self, job_id: str) -> Job | None:
        """The job with this id, or ``None`` (never existed, or forgotten after a restart)."""
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        """Every kept job, newest first."""
        return list(reversed(self._jobs.values()))

    @property
    def active(self) -> int:
        """How many jobs are queued or running."""
        return sum(1 for job in self._jobs.values() if job.state in ("queued", "running"))

    def _forget_old(self) -> None:
        # Only finished jobs are dropped; a queued or running one is always kept.
        for job_id in [j.id for j in self._jobs.values() if j.state in ("done", "failed")]:
            if len(self._jobs) <= self._keep:
                break
            del self._jobs[job_id]

    async def _run(self) -> None:
        queue = self._queue
        if queue is None:
            raise RuntimeError("the job queue is not started")
        while True:
            job, work = await queue.get()
            job.state = "running"
            job.started_at = _now()
            try:
                job.result = await anyio.to_thread.run_sync(work, job)
            except Exception as error:  # a failed job must not kill the worker
                log.exception("job %s (%s) failed", job.id, job.kind)
                job.error = f"{type(error).__name__}: {error}"
                job.state = "failed"
            else:
                job.state = "done"
            finally:
                job.finished_at = _now()
                queue.task_done()
