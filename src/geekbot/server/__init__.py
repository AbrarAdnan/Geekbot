"""The API server (Phase 4): the one process that owns the storage (PLAN.md §6.2).

``app.py`` builds the FastAPI application and its lifespan, ``routes.py`` holds the health and
knowledge-base routes, ``jobs.py`` the queue that runs long storage work (ingest, delete) one job at
a time, and ``schemas.py`` the shapes that go over the wire. Step S4.2 added ``threads.py`` (the
thread routes and the streamed answer), ``answering.py`` (the models and retriever built at start-up,
and the gate that lets one question through at a time) and ``turns.py`` (graph events to SSE frames).
Step S4.3 added ``security.py`` (the perimeter: API key, Host check, size caps, the path allowlist)
and ``uploads.py`` (receiving a file safely).
"""

from geekbot.server.app import HOST, create_app

__all__ = ["HOST", "create_app"]
