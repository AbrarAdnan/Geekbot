"""Tests for step S4.1: the FastAPI core (no model, no network, real SQLite and Chroma).

The embedder is a fake with a fixed dimension, so an ingest runs the real pipeline end to end
without Ollama. Layout: the job queue on its own, then the routes through ``TestClient``, then the
two properties the step exists for: a second storage owner is refused, and the event loop stays free
while blocking work runs (the heartbeat test, PLAN.md §6.3; ``/healthz`` is its ✍️ piece, written by
Claude in prototyping mode, D-047).
"""

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from serverkit import make_app, wait_for, write_note

from geekbot.index.lock import StorageLockedError
from geekbot.server import HOST
from geekbot.server.jobs import Job, JobQueue, QueueFullError

pytestmark = pytest.mark.usefixtures("clean_geekbot_env")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(make_app(tmp_path)) as opened:
        yield opened


# --- the job queue ---------------------------------------------------------------------------


def run_queue(scenario: Any, **options: Any) -> Any:
    async def main() -> Any:
        queue = JobQueue(**options)
        await queue.start()
        try:
            return await scenario(queue)
        finally:
            await queue.stop()

    return asyncio.run(main())


async def finished(queue: JobQueue, job: Job) -> None:
    while job.state in ("queued", "running"):
        await asyncio.sleep(0.005)


def test_jobs_run_one_at_a_time_in_order() -> None:
    running = 0
    peak = 0
    order: list[str] = []
    lock = threading.Lock()

    def work_named(name: str) -> Any:
        def work(job: Job) -> dict[str, Any]:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.03)
            order.append(name)
            with lock:
                running -= 1
            return {"name": name}

        return work

    async def scenario(queue: JobQueue) -> list[Job]:
        jobs = [queue.submit("t", work_named(n)) for n in ("a", "b", "c")]
        for job in jobs:
            await finished(queue, job)
        return jobs

    jobs = run_queue(scenario)
    assert order == ["a", "b", "c"]
    assert peak == 1
    assert [job.state for job in jobs] == ["done"] * 3
    assert jobs[0].result == {"name": "a"}


def test_a_failing_job_does_not_stop_the_worker() -> None:
    def boom(job: Job) -> dict[str, Any]:
        raise ValueError("bad file")

    async def scenario(queue: JobQueue) -> tuple[Job, Job]:
        first = queue.submit("t", boom)
        second = queue.submit("t", lambda job: {"ok": True})
        await finished(queue, second)
        return first, second

    first, second = run_queue(scenario)
    assert first.state == "failed"
    assert first.error == "ValueError: bad file"
    assert second.state == "done"


def test_a_full_queue_refuses_instead_of_piling_up() -> None:
    gate = threading.Event()

    def slow(job: Job) -> dict[str, Any]:
        gate.wait(5)
        return {}

    async def scenario(queue: JobQueue) -> None:
        queue.submit("t", slow)  # taken by the worker at once (or soon)
        await asyncio.sleep(0.05)
        queue.submit("t", slow)  # waits in the queue (1 of 1)
        with pytest.raises(QueueFullError):
            queue.submit("t", slow)
        gate.set()

    run_queue(scenario, max_pending=1)


def test_finished_jobs_are_forgotten_but_active_ones_are_kept() -> None:
    async def scenario(queue: JobQueue) -> list[str]:
        jobs = [queue.submit("t", lambda job: {}) for _ in range(5)]
        for job in jobs:
            await finished(queue, job)
        queue.submit("t", lambda job: {})  # triggers the cleanup
        return [job.id for job in queue.all()]

    kept = run_queue(scenario, keep=3)
    assert len(kept) <= 4


# --- the routes ------------------------------------------------------------------------------


def test_healthz_is_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_on_an_empty_store(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body == {
        "status": "ok",
        "documents": {},
        "chunks": 0,
        "vectors": 0,
        "embedder": "fake",
        "jobs_active": 0,
        "chat_model": "fake-model:1b",  # S4.2: the answering tools are part of readiness
        "chat_ready": True,
        "answering": False,
        "csv_model": "fake-csv:1b",  # C5: so is the CSV side
        "csv_ready": False,
        "csv_sessions": 0,
        "web_enabled": False,  # S5.3: off by default, and off means no tool at all
    }


def test_ingest_a_file_then_again_embeds_nothing(client: TestClient, tmp_path: Path) -> None:
    note = write_note(tmp_path)
    accepted = client.post("/v1/kb/ingest", json={"paths": [str(note)]})
    assert accepted.status_code == 202
    first = wait_for(client, accepted.json()["id"])
    assert first["state"] == "done", first
    assert first["result"]["files"][0]["status"] == "ingested"
    assert first["result"]["files"][0]["embedded"] > 0
    assert len(first["progress"]) == 1  # one line per finished file

    documents = client.get("/v1/kb/documents").json()
    assert [d["state"] for d in documents] == ["ready"]
    assert documents[0]["path"] == str(note.resolve())  # stored under its real path (S4.3)
    assert client.get("/v1/kb/documents", params={"state": "failed"}).json() == []
    health = client.get("/v1/health").json()
    assert health["documents"] == {"ready": 1}
    assert health["vectors"] == health["chunks"] > 0

    again = wait_for(client, client.post("/v1/kb/ingest", json={"paths": [str(note)]}).json()["id"])
    assert again["result"]["files"][0]["status"] == "unchanged"
    assert "0 embeddings computed" in again["result"]["summary"]


def test_ingest_unsupported_and_bad_files_are_reported_not_fatal(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    write_note(folder)
    (folder / "bin.xyz").write_bytes(b"\x00\x01")  # in a folder walk: left out as unsupported
    given = tmp_path / "given.xyz"
    given.write_bytes(b"\x00\x01")  # named explicitly: kept, the loader skips it
    body = {"paths": [str(folder), str(given)]}
    job = wait_for(client, client.post("/v1/kb/ingest", json=body).json()["id"])
    assert job["state"] == "done"
    assert job["result"]["unsupported"] == [str((folder / "bin.xyz").resolve())]
    statuses = {Path(f["path"]).name: f["status"] for f in job["result"]["files"]}
    assert statuses == {"zebras.txt": "ingested", "given.xyz": "skipped"}


@pytest.mark.parametrize(
    "body",
    [
        {"paths": []},
        {"paths": ["x"], "workers": 0},
        {"paths": ["x"], "workers": 99},
        {"paths": ["x"], "unknown": 1},
        {},
    ],
)
def test_ingest_rejects_a_bad_request(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/v1/kb/ingest", json=body).status_code == 422


def test_ingest_of_a_missing_path_is_422_and_queues_nothing(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post("/v1/kb/ingest", json={"paths": [str(tmp_path / "nope.txt")]})
    assert response.status_code == 422
    assert client.get("/v1/jobs").json() == []


def test_delete_a_document(client: TestClient, tmp_path: Path) -> None:
    note = write_note(tmp_path)
    wait_for(client, client.post("/v1/kb/ingest", json={"paths": [str(note)]}).json()["id"])
    doc_id = client.get("/v1/kb/documents").json()[0]["doc_id"]

    accepted = client.delete(f"/v1/kb/documents/{doc_id}")
    assert accepted.status_code == 202
    assert wait_for(client, accepted.json()["id"])["result"] == {"doc_id": doc_id, "removed": True}
    assert client.get("/v1/kb/documents").json() == []
    assert client.get("/v1/health").json()["vectors"] == 0


def test_delete_of_an_unknown_document_is_404(client: TestClient) -> None:
    assert client.delete("/v1/kb/documents/nope").status_code == 404


def test_an_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/v1/jobs/nope").status_code == 404


def test_jobs_are_listed_newest_first(client: TestClient, tmp_path: Path) -> None:
    note = write_note(tmp_path)
    first = client.post("/v1/kb/ingest", json={"paths": [str(note)]}).json()["id"]
    wait_for(client, first)
    second = client.post("/v1/kb/ingest", json={"paths": [str(note)]}).json()["id"]
    wait_for(client, second)
    assert [job["id"] for job in client.get("/v1/jobs").json()] == [second, first]


def test_a_failing_ingest_job_is_reported_as_failed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("store closed")

    monkeypatch.setattr("geekbot.server.routes.Ingestor.ingest", broken)
    note = write_note(tmp_path)
    job = wait_for(client, client.post("/v1/kb/ingest", json={"paths": [str(note)]}).json()["id"])
    assert job["state"] == "failed"
    assert job["error"] == "RuntimeError: store closed"
    assert client.get("/healthz").status_code == 200  # the server is fine


# --- one process owns the storage (PLAN.md §6.2) -----------------------------------------------


def test_a_second_server_on_the_same_data_folder_is_refused(tmp_path: Path) -> None:
    with (
        TestClient(make_app(tmp_path)),
        pytest.raises(StorageLockedError),
        TestClient(make_app(tmp_path)),
    ):
        pass


def test_the_lock_is_released_when_the_server_stops(tmp_path: Path) -> None:
    with TestClient(make_app(tmp_path)):
        pass
    with TestClient(make_app(tmp_path)) as again:  # no StorageLockedError
        assert again.get("/healthz").status_code == 200


def test_the_server_listens_on_loopback_only() -> None:
    assert HOST == "127.0.0.1"


# --- the heartbeat (PLAN.md §6.3) --------------------------------------------------------------

HEARTBEAT_P99_S = 0.2


def p99(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]


async def ping(client: httpx.AsyncClient) -> float:
    started = time.perf_counter()
    response = await client.get("/healthz")
    assert response.status_code == 200
    return time.perf_counter() - started


def test_healthz_stays_fast_while_a_job_blocks_a_thread(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Blocking work in a job runs in a worker thread, so the loop keeps answering.

    asyncio's debug mode logs "Executing <callback> took N seconds" for any callback that holds the
    loop longer than ``slow_callback_duration``; none may appear.
    """

    async def main() -> list[float]:
        asyncio.get_running_loop().slow_callback_duration = 0.1
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                jobs = app.state.jobs
                jobs.submit("blocking", lambda job: (time.sleep(1.0), {})[1])
                latencies: list[float] = []
                while jobs.active:
                    latencies.append(await ping(http))
                    await asyncio.sleep(0.01)
                return latencies

    with caplog.at_level(logging.WARNING, logger="asyncio"):
        latencies = asyncio.run(main(), debug=True)
    assert len(latencies) >= 50
    assert p99(latencies) < HEARTBEAT_P99_S
    assert not [r for r in caplog.records if "took" in r.getMessage()]


def test_the_heartbeat_would_notice_a_blocked_loop(tmp_path: Path) -> None:
    """A control for the test above: blocking *on* the loop does push the latency over the limit."""

    async def main() -> float:
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:

                async def block() -> None:
                    time.sleep(0.4)  # noqa: ASYNC251 - the mistake the rule forbids, on purpose

                # A request that arrives while the loop is held waits for the whole hold: measure
                # from the moment the blocker is scheduled until the ping has been answered.
                started = time.perf_counter()
                task = asyncio.create_task(block())
                await asyncio.sleep(0)  # the blocker runs now; nothing else can until it is done
                await ping(http)
                await task
                return time.perf_counter() - started

    assert asyncio.run(main()) > HEARTBEAT_P99_S


def test_healthz_stays_fast_during_a_real_ingest(tmp_path: Path) -> None:
    """The real ingest (parse, chunk, embed, SQLite, Chroma) off the loop, with many files."""
    folder = tmp_path / "docs"
    folder.mkdir()
    for i in range(30):
        (folder / f"d{i}.txt").write_text(
            "\n\n".join(f"Paragraph {j} of document {i} about zebras." * 8 for j in range(12)),
            encoding="utf-8",
        )

    async def main() -> tuple[list[float], dict[str, Any]]:
        asyncio.get_running_loop().slow_callback_duration = 0.1
        app = make_app(tmp_path)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                accepted = await http.post("/v1/kb/ingest", json={"paths": [str(folder)]})
                job_id = accepted.json()["id"]
                latencies: list[float] = []
                while True:
                    latencies.append(await ping(http))
                    job = (await http.get(f"/v1/jobs/{job_id}")).json()
                    if job["state"] in ("done", "failed"):
                        return latencies, job
                    await asyncio.sleep(0.005)

    latencies, job = asyncio.run(main(), debug=True)
    assert job["state"] == "done", job
    assert p99(latencies) < HEARTBEAT_P99_S
