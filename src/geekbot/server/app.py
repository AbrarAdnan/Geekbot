"""The FastAPI application and its lifespan (steps S4.1 and S4.2).

**One process owns the storage** (PLAN.md §6.2). The lifespan is where that becomes true: before
the server accepts a single request it opens the ``Workspace``, whose first act is to take the
exclusive storage lock. A second server on the same data folder therefore fails at start-up, with
the lock's own message, and never listens. On shutdown the job queue stops (a job already running
finishes first) and the workspace closes, which releases the lock.

Opening the workspace talks to Ollama (a probe embedding) and Chroma, so it runs in a worker thread,
not on the event loop (§6.3). ``create_app`` takes an ``open_workspace`` function so a test can open
one with a fake embedder.

**S4.2 adds the answering tools** (``answering.py``): once the workspace is open, the lifespan builds
the retriever and the chat models over it and, unless ``warm_up`` is off, loads the chat model onto
the GPU and the reranker onto the CPU, so that no question waits for a load (R-2). That is also a
worker thread: loading a model takes seconds. ``open_answering`` is replaceable for the same reason as
``open_workspace``. If anything after the workspace fails to start, the workspace is closed again, so
the lock is never left behind by a server that did not come up.

``HOST`` is a constant on purpose: the server listens on the loopback address only, and no setting
or option changes that. Making it reachable from another machine would need a design of its own
(TLS, rate limits); D-084 left that out of the prototype.

**S4.3 puts the perimeter in front of everything** (``security.py``): ``create_app`` takes a
``Perimeter`` (the API key, the allowed Host names, the path policy, the size caps) and installs
``PerimeterMiddleware``, which answers a request that fails the Host check, lacks the key or is too
large before FastAPI sees it. Without a ``perimeter`` argument ``create_app`` builds the one
``geekbot serve`` uses from the settings, key included, so forgetting to pass one gives the safe
configuration, not an open server. While a key is required the ``/docs``, ``/redoc`` and
``/openapi.json`` pages are not served: a browser cannot send the key to open them.

**C5 adds the registry of CSV sessions** (``csv_mode.sessions``) to the application state. Building it
loads nothing (a session is made when a CSV is uploaded or asked about), so it needs no worker
thread; closing it at shutdown does, because it closes the sessions' model clients.
``open_csv_sessions`` is replaceable like the other two openers, so a test can give it fakes.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Final

import anyio.to_thread
from fastapi import FastAPI

from geekbot import __version__
from geekbot.config import Settings
from geekbot.index.lock import StorageLockedError
from geekbot.index.workspace import Workspace
from geekbot.server.answering import Answering, TurnGate, open_chat_answering
from geekbot.server.csvmode import SessionsOpener, open_sessions
from geekbot.server.csvroutes import router as csv_router
from geekbot.server.jobs import JobQueue
from geekbot.server.routes import router
from geekbot.server.security import Perimeter, PerimeterMiddleware
from geekbot.server.threads import router as threads_router

__all__ = ["HOST", "create_app"]

log = logging.getLogger(__name__)

HOST: Final = "127.0.0.1"

WorkspaceOpener = Callable[[Settings], Workspace]
AnsweringOpener = Callable[[Settings, Workspace], Answering]


def create_app(
    settings: Settings | None = None,
    *,
    open_workspace: WorkspaceOpener | None = None,
    open_answering: AnsweringOpener | None = None,
    open_csv_sessions: SessionsOpener | None = None,
    warm_up: bool = True,
    perimeter: Perimeter | None = None,
) -> FastAPI:
    """Build the application. Nothing is opened until the lifespan starts (server start-up).

    ``warm_up`` is passed to the real answering tools (``geekbot serve --no-warmup`` turns it off); an
    ``open_answering`` function replaces them altogether (tests pass one with fake models), and an
    ``open_csv_sessions`` function replaces the registry of CSV sessions (tests pass one that loads
    fakes).
    ``perimeter`` is what the middleware enforces; left out, it is built from the settings with an
    API key (created and saved on first use), which can raise ``ApiKeyError`` or ``PolicyError``.
    """
    settings = settings or Settings()
    if perimeter is None:
        perimeter = Perimeter.from_settings(settings)
    opener: WorkspaceOpener = open_workspace or Workspace.open

    def real_answering(settings: Settings, workspace: Workspace) -> Answering:
        return open_chat_answering(settings, workspace, warm_up=warm_up)

    answering_opener: AnsweringOpener = open_answering or real_answering
    sessions_opener: SessionsOpener = open_csv_sessions or open_sessions

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            workspace = await anyio.to_thread.run_sync(opener, settings)
        except StorageLockedError:
            log.error("another Geekbot process owns %s: not starting", settings.data_dir)
            raise
        try:
            answering = await anyio.to_thread.run_sync(answering_opener, settings, workspace)
            jobs = JobQueue()
            await jobs.start()
            csv_sessions = sessions_opener(settings)
            app.state.settings = settings
            app.state.workspace = workspace
            app.state.answering = answering
            app.state.gate = TurnGate()
            app.state.jobs = jobs
            app.state.csv_sessions = csv_sessions
            try:
                yield
            finally:
                await jobs.stop()
                await anyio.to_thread.run_sync(csv_sessions.close_all)
        finally:
            await anyio.to_thread.run_sync(workspace.close)

    # With a key required the pages would only ever answer 401: a browser cannot send the key.
    show_docs = perimeter.api_key is None
    app = FastAPI(
        title="Geekbot",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if show_docs else None,
        redoc_url="/redoc" if show_docs else None,
        openapi_url="/openapi.json" if show_docs else None,
    )
    app.state.perimeter = perimeter
    app.add_middleware(
        PerimeterMiddleware,
        api_key=perimeter.api_key,
        hosts=perimeter.hosts,
        max_body_bytes=perimeter.max_body_bytes,
        max_upload_bytes=perimeter.max_upload_bytes,
    )
    app.include_router(router)
    app.include_router(threads_router)
    app.include_router(csv_router)
    return app
