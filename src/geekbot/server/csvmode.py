"""CSV conversations in the server: what a loaded session offers, the real opener, the file rules (step C5).

The server never imports the vendored ``csv_analyst`` package (PLAN.md §6.7: only ``geekbot.csv_mode``
does). It sees a loaded CSV through ``CsvSession``, the few things a route needs from one, and gets
the real thing from ``open_sessions``, which imports the adapter only when it first loads a file:
pandas is slow to import and a server that never sees a CSV should not pay for it.

**Where a conversation's file lives.** An uploaded CSV is stored as ``<sha16>-<safe name>`` in
``<data>/uploads`` (``uploads.save_upload``, the same rules as a document upload). That stored name is
what the thread keeps in its saved state as ``csv_file``, so after a restart, or after the registry
evicted the session, the next question can open the same file again. ``stored_file`` turns a name from
the saved state back into a path, and only for names of the shape ``save_upload`` produces.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Final, Protocol

import anyio.to_thread

from geekbot.agent.runtime import CsvReply
from geekbot.config import Settings
from geekbot.csv_mode.sessions import CsvSessions
from geekbot.server.security import UPLOADS_DIR
from geekbot.server.uploads import stored_name_ok

__all__ = [
    "CSV_SUFFIXES",
    "MAX_COLUMNS_LISTED",
    "MAX_NOTES_LISTED",
    "CsvSession",
    "SessionsOpener",
    "open_sessions",
    "stored_file",
]

# What the vendored loader accepts (``csv_analyst.ingestion.validators.ALLOWED_EXTENSIONS``).
CSV_SUFFIXES: Final = frozenset({".csv", ".tsv"})
MAX_COLUMNS_LISTED: Final = 40  # the upload answer names this many columns, and says how many more
MAX_NOTES_LISTED: Final = 10


class CsvSession(Protocol):
    """A loaded CSV, as the routes use it (``csv_mode.adapter.CsvAnalyst`` is the real one)."""

    @property
    def rows(self) -> int: ...

    @property
    def column_names(self) -> list[str]: ...

    @property
    def load_notes(self) -> list[str]: ...

    @property
    def memory_bytes(self) -> int: ...

    def ask(self, question: str, /) -> CsvReply:
        """Answer one question from the table. Blocks for seconds: a worker thread's job."""
        ...

    def close(self) -> None: ...


def open_sessions(settings: Settings) -> CsvSessions[CsvSession]:
    """The registry the server runs with: sessions of the real adapter, bounded by the settings."""

    def open_session(path: Path) -> CsvSession:
        from geekbot.csv_mode.adapter import CsvAnalyst  # imports pandas: only once a CSV is used

        return CsvAnalyst.open(path, settings)

    return CsvSessions(
        open_session,
        max_sessions=settings.csv_max_sessions,
        max_bytes=settings.csv_session_budget_bytes,
    )


async def stored_file(settings: Settings, stored_as: str | None) -> Path | None:
    """The path of a stored upload named by a thread's saved ``csv_file``, or ``None``.

    ``None`` when the name is empty, is not of the shape ``save_upload`` produces (a conversation
    started with ``geekbot csv ask`` holds the plain name of a file the server never saw), or the file
    is gone. The folder is fixed, so nothing in ``stored_as`` can lead out of it.
    """
    if not stored_as or not stored_name_ok(stored_as):
        return None
    path = settings.data_dir / UPLOADS_DIR / stored_as
    return path if await anyio.to_thread.run_sync(path.is_file) else None


# A type for the opener ``create_app`` accepts, so a test can give it a registry of fakes.
SessionsOpener = Callable[[Settings], CsvSessions[CsvSession]]
