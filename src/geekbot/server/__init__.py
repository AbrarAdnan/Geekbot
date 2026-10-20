"""The API server (Phase 4): the one process that owns the storage (PLAN.md §6.2).

``app.py`` builds the FastAPI application and its lifespan, ``routes.py`` holds the HTTP routes,
``jobs.py`` the queue that runs long storage work (ingest, delete) one job at a time, and
``schemas.py`` the shapes that go over the wire.
"""

from geekbot.server.app import HOST, create_app

__all__ = ["HOST", "create_app"]
