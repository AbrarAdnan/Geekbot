"""The registry of loaded CSV sessions: one per thread, bounded, never evicted while in use (step C5).

A loaded CSV session holds the **whole table in memory** (the package reads the file into a pandas
table and profiles it), and a server answers many conversations. Without a bound, ten uploads of 80 MB
files would be several gigabytes of resident tables on the PC that also runs the user's other projects
(PLAN.md risk 20). So the server keeps sessions here, and this registry decides what stays:

- **One session per thread** (PLAN.md §6.7). A new upload for a thread *replaces* its session.
- **Two bounds**, both settings, both unmeasured starting points: at most ``max_sessions`` sessions,
  and at most ``max_bytes`` of table memory in all. The size of a session is what it reports as
  ``memory_bytes`` (for the real one, pandas' own deep memory count of the table), not the size of the
  file on disk: a CSV of strings is several times bigger in memory than in a file.
- **Least recently used goes first.** Every ``acquire`` and ``open`` makes a session the newest. When a
  bound is exceeded the oldest sessions are closed, until the bounds hold again.
- **A session in use is never closed under its user.** ``acquire`` hands out a ``Lease`` and pins the
  session; ``release`` unpins it. Eviction, replacement and ``drop`` skip or postpone a pinned session
  (it is closed at its last ``release``). Without this, an upload for conversation B could close the
  model client that conversation A's question is using, half way through the question.
- **A session that alone is over the byte bound is refused** (``SessionTooLargeError``) and closed
  again: keeping it would mean evicting everything else and still being over.
- **An evicted session is not lost.** The file is still in the uploads folder; the next question of
  that thread opens it again (``acquire`` does it), which costs the seconds it takes to load it.

Opening a session reads and profiles a file: it blocks for seconds. ``open`` and ``acquire`` therefore
must not be called on the event loop (PLAN.md §6.3); the server calls them in a worker thread. The
registry's lock guards the table of sessions only and is never held while a file is loaded or a
session is closed, so one slow load does not stop a question that is using another session.

✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md.
"""

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = ["CsvSessions", "Lease", "Session", "SessionTooLargeError"]

log = logging.getLogger(__name__)


class Session(Protocol):
    """What the registry needs from a session: how much memory it holds, and a way to let it go."""

    @property
    def memory_bytes(self) -> int: ...

    def close(self) -> None: ...


class SessionTooLargeError(ValueError):
    """One session alone holds more than the registry's byte bound. The message is for the user."""


@dataclass
class _Entry[S: Session]:
    thread_id: str
    path: Path
    session: S
    size: int
    pins: int = 0
    retired: bool = False  # taken out of the table; to be closed when the last lease is released


@dataclass
class Lease[S: Session]:
    """A pinned session. Give it back with ``CsvSessions.release``; ``session`` is what to ask.

    A lease can be given back once: a second ``release`` of the same lease does nothing, so it can
    never free the pin of another question that is using the same session.
    """

    thread_id: str
    session: S
    entry: _Entry[S]
    released: bool = False


class CsvSessions[S: Session]:
    """The sessions of the server's CSV conversations (see the module docstring for the rules).

    ``open_session`` loads the file at a path and returns a session; it is the only thing the registry
    knows about the package behind it, which is also what lets the tests use a fake.
    """

    def __init__(
        self, open_session: Callable[[Path], S], *, max_sessions: int, max_bytes: int
    ) -> None:
        if max_sessions < 1 or max_bytes < 1:
            raise ValueError("a registry needs room for at least one session and one byte")
        self._open = open_session
        self._max_sessions = max_sessions
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        # Least recently used first; ``move_to_end`` makes a session the newest.
        self._entries: OrderedDict[str, _Entry[S]] = OrderedDict()

    # --- looking ---------------------------------------------------------------------------

    def has(self, thread_id: str) -> bool:
        """Whether the thread has a loaded session right now (it may have a file and no session)."""
        with self._lock:
            return thread_id in self._entries

    def path_of(self, thread_id: str) -> Path | None:
        """The file the thread's loaded session was made from, or ``None``."""
        with self._lock:
            entry = self._entries.get(thread_id)
            return entry.path if entry is not None else None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(entry.size for entry in self._entries.values())

    @property
    def thread_ids(self) -> list[str]:
        """The threads that have a session, least recently used first."""
        with self._lock:
            return list(self._entries)

    # --- taking and giving back -------------------------------------------------------------

    def open(self, thread_id: str, path: Path) -> S:
        """Load ``path`` as the thread's session, replacing the one it had. Blocking.

        If the file cannot be loaded the error propagates and the thread keeps the session it had: a
        bad replacement must not destroy a good conversation. ``SessionTooLargeError`` if the table
        alone does not fit the byte bound.
        """
        return self._install(thread_id, path, pin=False).session

    def acquire(self, thread_id: str, path: Path) -> Lease[S]:
        """The thread's session for ``path``, pinned: loaded if it is not (or was loaded from another
        file). Blocking when it has to load. Give the lease back with ``release``."""
        with self._lock:
            entry = self._entries.get(thread_id)
            if entry is not None and entry.path == path:
                entry.pins += 1
                self._entries.move_to_end(thread_id)
                return Lease(thread_id, entry.session, entry)
        return self._install(thread_id, path, pin=True)

    def release(self, lease: Lease[S]) -> None:
        """Unpin. A session that was replaced or dropped while it was pinned is closed now."""
        entry = lease.entry
        with self._lock:
            if lease.released:  # given back already: nothing to give back
                return
            lease.released = True
            entry.pins -= 1
            close_now = entry.retired and entry.pins == 0
        if close_now:
            self._close(entry)

    def drop(self, thread_id: str) -> bool:
        """Forget the thread's session (the conversation was deleted). ``False`` if it had none."""
        with self._lock:
            entry = self._entries.pop(thread_id, None)
            if entry is None:
                return False
            close_now = self._retire(entry)
        if close_now:
            self._close(entry)
        return True

    def close_all(self) -> None:
        """Close every session (the server is stopping). Pinned ones close at their release."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            doomed = [entry for entry in entries if self._retire(entry)]
        for entry in doomed:
            self._close(entry)

    # --- inside ----------------------------------------------------------------------------

    def _install(self, thread_id: str, path: Path, *, pin: bool) -> Lease[S]:
        session = self._open(path)  # blocking, and outside the lock on purpose
        try:
            size = int(session.memory_bytes)
            if size > self._max_bytes:
                raise SessionTooLargeError(
                    f"this table needs about {size:,} bytes of memory, more than the "
                    f"{self._max_bytes:,} this server keeps for CSV conversations "
                    "(GEEKBOT_CSV_SESSION_BUDGET_BYTES)"
                )
        except BaseException:
            session.close()
            raise
        entry = _Entry(thread_id=thread_id, path=path, session=session, size=size)
        if pin:
            entry.pins = 1
        with self._lock:
            previous = self._entries.pop(thread_id, None)
            self._entries[thread_id] = entry
            doomed = [previous] if previous is not None else []
            doomed += self._evict(keep=thread_id)
            to_close = [item for item in doomed if self._retire(item)]
        for item in to_close:
            self._close(item)
        return Lease(thread_id, session, entry)

    def _evict(self, *, keep: str) -> list[_Entry[S]]:
        """Remove least recently used, unpinned sessions until both bounds hold. Lock held.

        ``keep`` (the session just installed) is never a candidate. If everything left is pinned the
        bounds stay exceeded for now and a warning says so: closing a session under its question
        would be worse, and the next installation tries again.
        """
        removed: list[_Entry[S]] = []
        while len(self._entries) > self._max_sessions or self._bytes() > self._max_bytes:
            victim = next(
                (
                    entry
                    for entry in self._entries.values()
                    if entry.thread_id != keep and entry.pins == 0
                ),
                None,
            )
            if victim is None:
                log.warning(
                    "CSV sessions are over their bounds (%d sessions, %d bytes) but all the "
                    "others are in use",
                    len(self._entries),
                    self._bytes(),
                )
                break
            del self._entries[victim.thread_id]
            removed.append(victim)
            log.info("closed the CSV session of thread %s (least recently used)", victim.thread_id)
        return removed

    def _bytes(self) -> int:
        return sum(entry.size for entry in self._entries.values())

    def _retire(self, entry: _Entry[S]) -> bool:
        """Mark an entry as out of the table; ``True`` if it can be closed right now. Lock held."""
        del self  # no state needed; a method so that it can use the class's type parameter
        entry.retired = True
        return entry.pins == 0

    def _close(self, entry: _Entry[S]) -> None:
        del self
        try:
            entry.session.close()
        except Exception:  # a session that cannot close must not break the one being installed
            log.exception("closing the CSV session of thread %s failed", entry.thread_id)
