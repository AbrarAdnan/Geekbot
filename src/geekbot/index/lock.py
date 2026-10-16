"""The storage lock: exactly one process owns the stores (step S1.4, PLAN.md §6.2, risk 6).

SQLite in WAL mode lets many readers and one writer share a database, but two *processes* that
both believe they own it can still step on each other: two writers racing on the outbox, or one
process rebuilding the Chroma index while another embeds into it. The plan's answer is a rule,
not a hope: before any store is opened, the process takes an exclusive lock on
``<data_dir>/.storage.lock``. A second process that tries to take it fails at once with a message
that says why, instead of waiting or, worse, carrying on.

The lock is an operating-system file lock (``filelock``). The system drops it when the process
dies, however it dies, so a crash leaves no stale lock behind that a person has to delete.
"""

from pathlib import Path
from types import TracebackType
from typing import Final, Self

from filelock import FileLock, Timeout

__all__ = ["LOCK_FILE_NAME", "StorageLock", "StorageLockedError"]

LOCK_FILE_NAME: Final = ".storage.lock"


class StorageLockedError(RuntimeError):
    """Another process (or another store in this one) already owns the storage."""


class StorageLock:
    """An exclusive, fail-fast lock on one data directory.

    ``acquire`` never waits: if the lock is taken it raises ``StorageLockedError`` straight away.
    Use it as a context manager, or call ``acquire`` and ``release`` yourself (``Store`` does).

    The lock is shared by every thread of the process that holds it (``thread_local=False``): the
    server takes it in the main thread at start-up and releases it from whichever thread shuts
    down. Taking the same ``StorageLock`` object twice is an error, not a re-entry, because a
    "second" owner inside one process is the very thing the lock exists to refuse.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._path = Path(data_dir) / LOCK_FILE_NAME
        self._lock = FileLock(self._path, thread_local=False)

    @property
    def path(self) -> Path:
        """The lock file. It is created on first use and may stay behind; it holds nothing."""
        return self._path

    @property
    def is_held(self) -> bool:
        """Whether this object currently holds the lock."""
        return self._lock.is_locked

    def acquire(self) -> None:
        """Take the lock or raise ``StorageLockedError`` at once. Creates the directory."""
        if self._lock.is_locked:
            raise StorageLockedError(f"this process already holds the storage lock {self._path}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout as exc:
            raise StorageLockedError(
                f"another Geekbot process already owns the storage in {self._path.parent} "
                f"(lock file {self._path}). Stop it first, or use a different data directory."
            ) from exc

    def release(self) -> None:
        """Give the lock up. Releasing a lock that is not held does nothing."""
        self._lock.release()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
