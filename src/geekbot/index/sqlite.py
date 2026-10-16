"""The SQLite store: documents, chunks and full-text search (step S1.4, PLAN.md §6.2-§6.4).

SQLite is the source of truth; Chroma (S1.5) is a derived index that can be rebuilt from it. Four
rules of the plan shape everything here.

- **One process owns the storage** (§6.2). ``Store`` takes the exclusive ``StorageLock`` before it
  opens the database, so a second process fails at once.
- **No blocking I/O on the event loop** (§6.3). ``Store`` is synchronous on purpose; the server
  (S4.1) calls it through ``anyio.to_thread.run_sync``. Every write goes through *one* writer
  thread with a queue, so writes are serialized without any locking of our own. Reads borrow a
  connection from a small pool. This works because the database is in WAL (write-ahead log) mode:
  a reader sees the last committed state and neither blocks the writer nor is blocked by it.
- **Retrieval only returns ``ready`` documents** (§6.4). A document is ``pending`` while its rows
  are written and ``embedding`` while its vectors are, and is searchable only after the last step
  has flipped it to ``ready``. A crash in between leaves rows nobody can retrieve, which
  ``doctor --repair`` (S1.6) can finish or remove. The price: an edited document is invisible
  while it is re-processed.
- **Re-ingest is a set difference on ``chunk_id``** (§6.4). ``sync_chunks`` inserts the new IDs,
  deletes the vanished ones and rewrites the position columns of the kept ones in place.

**Schema and migrations.** The schema is a list of numbered migrations. The database remembers how
many it has applied in ``PRAGMA user_version``, and opening it applies the rest, each in one
transaction together with the version bump, so a failed migration leaves the database as it was.
A database written by a *newer* Geekbot is refused rather than half-understood. Later steps add
their tables (jobs, threads, labels) as further migrations.

**Full-text search.** ``chunks_fts`` is an FTS5 *external-content* table: it stores the inverted
index but not a second copy of the text, and reads the text back from ``chunks``. The price is that
we keep it in step ourselves, with the three triggers in migration 1; the 'delete' command needs
the *old* values, which is why the triggers pass ``old.text``. Only ``text`` is indexed, not the
contextual header: the ablation's header rung (R5, PLAN.md §9) must change what the embedder sees
and nothing else, and a header that BM25 always saw would be a second variable.
"""

import json
import queue
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from geekbot.domain import (
    DOC_STATES,
    Chunk,
    DocState,
    Document,
    NewDocument,
    ScoredChunk,
)
from geekbot.index.lock import StorageLock
from geekbot.ingest.ids import IdentifiedChunk

__all__ = [
    "DB_FILE_NAME",
    "MAX_QUERY_TERMS",
    "MIGRATIONS",
    "ChunkSync",
    "Migration",
    "Store",
    "StoreClosedError",
    "StoreError",
    "UnknownDocumentError",
    "fts_match_query",
    "migrate",
]

DB_FILE_NAME: Final = "geekbot.db"
MIN_SQLITE_VERSION: Final = (3, 35, 0)
BUSY_TIMEOUT_S: Final = 5.0  # how long a connection waits for a lock before "database is locked"
READ_WAIT_S: Final = 30.0  # how long a reader waits for a free pooled connection
DEFAULT_READ_POOL_SIZE: Final = 4
MAX_QUERY_TERMS: Final = 64


class StoreError(Exception):
    """Something the store cannot do: a database from the future, a closed store, a bad call."""


class StoreClosedError(StoreError):
    """The store was used after ``close()``."""


class UnknownDocumentError(StoreError):
    """A document ID that the store does not have."""


# --- migrations ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Migration:
    """One step of the schema's history: a number, a few words, and the SQL that does it."""

    version: int
    description: str
    sql: str


# Version 1. Notes on the choices (the reasons are in D-055):
# - ``chunks.row_id`` is an explicit INTEGER PRIMARY KEY. FTS5's external-content link needs an
#   integer that never changes, and the hidden rowid of a table keyed by text may be renumbered
#   by VACUUM. ``chunk_id`` stays the identity (UNIQUE), ``row_id`` is only the FTS5 handle.
# - ``ordinal`` and the offsets are position metadata, so they are plain columns without a UNIQUE
#   constraint: re-ingest rewrites them in place, and two rows may swap places in one statement.
# - ``documents.state`` is the outbox (PLAN.md §6.4); the CHECK keeps a typo from becoming a state.
# - The update trigger fires on ``UPDATE OF text`` only. A move (new offsets, same text) must not
#   touch the full-text index; ``sync_chunks`` leaves ``text`` out of the SET list for a move.
_SCHEMA_V1 = """
CREATE TABLE documents (
    doc_id           TEXT PRIMARY KEY CHECK (length(doc_id) = 16),
    path             TEXT NOT NULL,
    title            TEXT NOT NULL,
    kind             TEXT NOT NULL,
    file_hash        TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    state            TEXT NOT NULL DEFAULT 'pending'
                     CHECK (state IN ('pending', 'embedding', 'ready', 'failed')),
    error            TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX documents_by_state ON documents (state);

CREATE TABLE chunks (
    row_id       INTEGER PRIMARY KEY,
    chunk_id     TEXT NOT NULL UNIQUE CHECK (length(chunk_id) = 24),
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
    occurrence   INTEGER NOT NULL CHECK (occurrence >= 0),
    ordinal      INTEGER NOT NULL CHECK (ordinal >= 0),
    char_start   INTEGER NOT NULL CHECK (char_start >= 0),
    char_end     INTEGER NOT NULL CHECK (char_end >= char_start),
    page         INTEGER CHECK (page IS NULL OR page >= 1),
    section_path TEXT NOT NULL DEFAULT '[]',
    header       TEXT NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX chunks_by_doc ON chunks (doc_id, ordinal);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content = 'chunks',
    content_rowid = 'row_id',
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts (rowid, text) VALUES (new.row_id, new.text);
END;
CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.row_id, old.text);
END;
CREATE TRIGGER chunks_fts_update AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', old.row_id, old.text);
    INSERT INTO chunks_fts (rowid, text) VALUES (new.row_id, new.text);
END;
"""

MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(1, "documents, chunks and the full-text index", _SCHEMA_V1),
)


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> int:
    """Bring the database up to date and return the version it is at.

    Each pending migration runs as one script inside ``BEGIN IMMEDIATE ... COMMIT`` together with
    the ``user_version`` bump (SQLite's DDL is transactional), so a migration that fails half-way
    changes nothing and the error names it. The connection must be in autocommit mode
    (``isolation_level=None``), as the store's own are.

    ``ValueError`` if the migrations are not numbered 1, 2, 3, ... in order; ``StoreError`` if the
    database is at a version this code does not know (a newer Geekbot wrote it).
    """
    if [migration.version for migration in migrations] != list(range(1, len(migrations) + 1)):
        raise ValueError("migrations must be numbered 1, 2, 3, ... in order, with no gaps")
    current = _user_version(conn)
    if current > len(migrations):
        raise StoreError(
            f"the database is at schema version {current}, newer than the {len(migrations)} this "
            "Geekbot knows: it was written by a newer version, so it is not touched"
        )
    for migration in migrations[current:]:
        script = (
            "BEGIN IMMEDIATE;\n"
            + migration.sql
            + f"\nPRAGMA user_version = {migration.version};\nCOMMIT;"
        )
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise StoreError(
                f"migration {migration.version} ({migration.description}) failed: {exc}"
            ) from exc
    return len(migrations)


# --- connections -----------------------------------------------------------------------------


def _check_sqlite() -> None:
    """Fail with a clear message if this SQLite build cannot run the store.

    A build without FTS5 would otherwise fail inside migration 1 with "no such module: fts5".
    ``json_each`` (the JSON functions) is used to pass a list of IDs as one parameter.
    """
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        needed = ".".join(str(part) for part in MIN_SQLITE_VERSION)
        raise StoreError(
            f"SQLite {needed} or newer is required, this Python has {sqlite3.sqlite_version}"
        )
    probe = sqlite3.connect(":memory:")
    try:
        probe.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        probe.execute("SELECT value FROM json_each('[1]')").fetchall()
    except sqlite3.OperationalError as exc:
        raise StoreError(f"this SQLite build lacks FTS5 or the JSON functions: {exc}") from exc
    finally:
        probe.close()


def _connect(db_path: Path, *, read_only: bool) -> sqlite3.Connection:
    """Open one connection in autocommit mode: transactions are ours to begin and end.

    ``check_same_thread=False`` because a pooled reader is used by whichever thread borrows it,
    one at a time, and the writer is opened by the thread that builds the store but used only by
    the writer thread afterwards. A reader also gets ``query_only``: whatever it runs, it cannot
    change the database.
    """
    conn = sqlite3.connect(
        db_path, timeout=BUSY_TIMEOUT_S, isolation_level=None, check_same_thread=False
    )
    conn.execute("PRAGMA foreign_keys = ON")
    if read_only:
        conn.execute("PRAGMA query_only = ON")
    return conn


def _enable_wal(conn: sqlite3.Connection) -> None:
    # synchronous=NORMAL is the usual pairing with WAL: a power cut may lose the last few commits
    # but cannot corrupt the file, and the outbox plus ``doctor --repair`` cover lost work.
    mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
    if mode != "wal":
        raise StoreError(
            f"the database could not be put in WAL mode (it answered {mode!r}); "
            "is the data directory on a network drive?"
        )
    conn.execute("PRAGMA synchronous = NORMAL")


def _utc_now() -> datetime:
    return datetime.now(UTC)


# --- search queries --------------------------------------------------------------------------

_QUERY_WORD: Final = re.compile(r"[^\W_]+")  # runs of letters and digits, like FTS5's tokenizer


def fts_match_query(text: str) -> str | None:
    """Turn what a person typed into an FTS5 ``MATCH`` expression that cannot be a syntax error.

    This and ``Store.search_bm25`` are step S1.4's ✍️ piece (written by Claude in prototyping mode,
    D-047).

    The text is cut into words (runs of letters and digits), each word is put in double quotes,
    and the words are joined with ``OR``: ``refund policy?`` becomes ``"refund" OR "policy"``.

    - *Quoting is the safety.* FTS5 has its own query language: ``AND``, ``OR``, ``NOT``,
      ``NEAR(...)``, ``col:term``, ``^term``, ``term*``, unbalanced quotes and parentheses. Raw
      user text containing any of them either changes the meaning or raises "fts5: syntax error".
      Inside quotes none of that is special, and a quoted word contains no quote character,
      because only letters and digits are kept. A property test feeds arbitrary text through this.
    - *OR, not AND.* A question such as "what is the refund policy" has words the right chunk may
      not contain. With OR, BM25 does the work: a chunk matching more of the rarer words scores
      higher, and a word found in most chunks adds almost nothing (its IDF is near zero).
    - Words are not lowercased here (the tokenizer folds case and accents). Repeats are dropped,
      case-insensitively, because the same word twice would count twice in the score. Text is
      composed to NFC first, so an accent typed as a separate mark is not split off its letter.
    - At most ``MAX_QUERY_TERMS`` words are kept, the first ones, so a pasted page cannot make a
      query that scans the whole index.

    Returns ``None`` when the text has no words (empty, spaces, only punctuation).
    """
    seen: set[str] = set()
    terms: list[str] = []
    for match in _QUERY_WORD.finditer(unicodedata.normalize("NFC", text)):
        word = match.group()
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(f'"{word}"')
        if len(terms) == MAX_QUERY_TERMS:
            break
    return " OR ".join(terms) if terms else None


_CHUNK_COLUMNS: Final = (
    "c.chunk_id, c.doc_id, d.path, d.title, c.content_hash, c.occurrence, c.ordinal, "
    "c.char_start, c.char_end, c.page, c.section_path, c.header, c.text"
)

# ``bm25()`` returns a number where *lower is better* (a more negative number is a better match),
# and the score is the sum over the query's words, so we sort ascending and negate it for the
# caller. The state filter lives in the same statement as MATCH and LIMIT, so LIMIT counts ready
# chunks only: a query never comes back short because the best hits belonged to a pending file.
# The final ORDER BY term breaks ties (identical chunks tie exactly) by ID, so a ranking is the
# same on every run, which a benchmark needs.
_SEARCH_SQL: Final = (
    "SELECT " + _CHUNK_COLUMNS + ", bm25(chunks_fts) AS score "
    "FROM chunks_fts "
    "JOIN chunks AS c ON c.row_id = chunks_fts.rowid "
    "JOIN documents AS d ON d.doc_id = c.doc_id "
    "WHERE chunks_fts MATCH ? AND d.state = 'ready' "
    "ORDER BY score, c.chunk_id "
    "LIMIT ?"
)

_GET_CHUNKS_SQL: Final = (
    "SELECT " + _CHUNK_COLUMNS + " "
    "FROM chunks AS c JOIN documents AS d ON d.doc_id = c.doc_id "
    "WHERE c.chunk_id IN (SELECT value FROM json_each(:ids)) "
    "AND (:include_unready OR d.state = 'ready')"
)

_DOCUMENT_SQL: Final = (
    "SELECT doc_id, path, title, kind, file_hash, pipeline_version, state, error, "
    "created_at, updated_at FROM documents WHERE doc_id = ?"
)

_STORED_CHUNKS_SQL: Final = (
    "SELECT chunk_id, ordinal, char_start, char_end, page, section_path, header, text "
    "FROM chunks WHERE doc_id = ?"
)


def _chunk_from_row(row: Sequence[Any]) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        doc_id=row[1],
        path=row[2],
        title=row[3],
        content_hash=row[4],
        occurrence=row[5],
        ordinal=row[6],
        char_start=row[7],
        char_end=row[8],
        page=row[9],
        section_path=tuple(json.loads(row[10])),
        header=row[11],
        text=row[12],
    )


def _doc_state(value: str) -> DocState:
    for state in DOC_STATES:
        if state == value:
            return state
    raise StoreError(f"the database holds a document state this code does not know: {value!r}")


def _document_from_row(row: Sequence[Any]) -> Document:
    return Document(
        doc_id=row[0],
        path=row[1],
        title=row[2],
        kind=row[3],
        file_hash=row[4],
        pipeline_version=row[5],
        state=_doc_state(row[6]),
        error=row[7],
        created_at=row[8],
        updated_at=row[9],
    )


# --- re-ingest -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkSync:
    """What ``sync_chunks`` did to one document's chunks, as chunk IDs.

    - ``added``: new to the store, in document order. These need vectors (S1.5, S1.6).
    - ``removed``: gone from the document, sorted. Their vectors must be dropped.
    - ``updated``: kept, but their text or position changed (the offsets of every chunk below an
      insertion move). The vector is still right when the text is the same.
    - ``unchanged``: kept exactly as stored, in document order.
    """

    added: tuple[str, ...]
    removed: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]


def _mutable_values(item: IdentifiedChunk) -> tuple[int, int, int, int | None, str, str, str]:
    """The columns of a chunk that may change while its ``chunk_id`` stays: position and text.

    The text can differ for one ID because the ID is taken from the *normalized* text (line
    endings and trailing whitespace are ignored): a file saved with Windows line endings is the
    same chunk, but the stored text must still be the exact slice of the file the offsets point
    into.
    """
    chunk = item.chunk
    return (
        chunk.ordinal,
        chunk.char_start,
        chunk.char_end,
        chunk.page,
        json.dumps(list(chunk.section_path), ensure_ascii=False),
        chunk.header,
        chunk.text,
    )


def _sync_chunks(
    conn: sqlite3.Connection, doc_id: str, chunks: Sequence[IdentifiedChunk], now: str
) -> ChunkSync:
    """The body of ``Store.sync_chunks``; runs on the writer thread inside one transaction."""
    if conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is None:
        raise UnknownDocumentError(doc_id)
    stored = {row[0]: tuple(row[1:]) for row in conn.execute(_STORED_CHUNKS_SQL, (doc_id,))}
    wanted = {item.chunk_id for item in chunks}
    removed = sorted(chunk_id for chunk_id in stored if chunk_id not in wanted)

    added: list[str] = []
    updated: list[str] = []
    unchanged: list[str] = []
    inserts: list[tuple[Any, ...]] = []
    moves: list[tuple[Any, ...]] = []  # same text, new position: the full-text index is untouched
    rewrites: list[tuple[Any, ...]] = []  # new text too: the update trigger re-indexes it
    for item in chunks:
        values = _mutable_values(item)
        old = stored.get(item.chunk_id)
        if old is None:
            added.append(item.chunk_id)
            inserts.append((item.chunk_id, doc_id, item.content_hash, item.occurrence, *values))
        elif old == values:
            unchanged.append(item.chunk_id)
        else:
            updated.append(item.chunk_id)
            if old[6] == values[6]:
                moves.append((*values[:6], item.chunk_id))
            else:
                rewrites.append((*values, item.chunk_id))

    if removed:
        conn.execute(
            "DELETE FROM chunks WHERE chunk_id IN (SELECT value FROM json_each(?))",
            (json.dumps(removed),),
        )
    conn.executemany(
        "UPDATE chunks SET ordinal = ?, char_start = ?, char_end = ?, page = ?, "
        "section_path = ?, header = ? WHERE chunk_id = ?",
        moves,
    )
    conn.executemany(
        "UPDATE chunks SET ordinal = ?, char_start = ?, char_end = ?, page = ?, "
        "section_path = ?, header = ?, text = ? WHERE chunk_id = ?",
        rewrites,
    )
    conn.executemany(
        "INSERT INTO chunks (chunk_id, doc_id, content_hash, occurrence, ordinal, char_start, "
        "char_end, page, section_path, header, text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        inserts,
    )
    conn.execute("UPDATE documents SET updated_at = ? WHERE doc_id = ?", (now, doc_id))
    return ChunkSync(tuple(added), tuple(removed), tuple(updated), tuple(unchanged))


# --- the store -------------------------------------------------------------------------------


class Store:
    """Documents and chunks in SQLite, with BM25 search. Synchronous and thread-safe.

    ``Store(data_dir)`` takes the storage lock (``StorageLockedError`` if another process or
    store owns the directory), opens ``<data_dir>/geekbot.db`` in WAL mode and applies the
    migrations. ``close()`` (or leaving a ``with`` block) releases everything.

    **Threads.** Any thread may call any method. Writes are queued to one writer thread and run
    one at a time, each in its own transaction (``write``); reads borrow a read-only connection
    (``read``) and run against a consistent snapshot. A read never waits for a write, and a write
    in progress is invisible to readers until it commits. Do not call a store method from inside
    the function you pass to ``write``: it runs on the writer thread, so a nested write would
    wait for itself, and ``write`` raises ``StoreError`` instead of hanging.

    ``clock`` exists for tests: it supplies the time stamped on rows.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        read_pool_size: int = DEFAULT_READ_POOL_SIZE,
        migrations: Sequence[Migration] = MIGRATIONS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if read_pool_size < 1:
            raise ValueError(f"read_pool_size must be at least 1, got {read_pool_size}")
        directory = Path(data_dir)
        lock = StorageLock(directory)
        lock.acquire()
        opened: list[sqlite3.Connection] = []
        readers: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        try:
            _check_sqlite()
            db_path = directory / DB_FILE_NAME
            writer = _connect(db_path, read_only=False)
            opened.append(writer)
            _enable_wal(writer)
            migrate(writer, migrations)
            for _ in range(read_pool_size):
                reader = _connect(db_path, read_only=True)
                opened.append(reader)
                readers.put(reader)
        except BaseException:
            for conn in opened:
                conn.close()
            lock.release()
            raise
        self._db_path = db_path
        self._lock = lock
        self._writer = writer
        self._readers = readers
        self._pool_size = read_pool_size
        self._clock = clock
        self._closed = False
        self._writer_thread_id: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="geekbot-sqlite")

    @property
    def db_path(self) -> Path:
        """The database file."""
        return self._db_path

    # --- lifecycle ---

    def close(self) -> None:
        """Finish the queued writes, close every connection and release the storage lock.

        Safe to call twice. After it, every method raises ``StoreClosedError``.
        """
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)
        while True:
            try:
                self._readers.get_nowait().close()
            except queue.Empty:
                break
        # The writer closes last: the last connection to close checkpoints the WAL file into the
        # database and removes it, and the writer is the one connection that may write.
        self._writer.close()
        self._lock.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # --- the two doors: one writer, many readers ---

    def write[T](self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``work(connection)`` on the writer thread in one transaction and return its result.

        The transaction is ``BEGIN IMMEDIATE`` (the write lock is taken up front, so a conflict
        shows at the start and not at the commit). If ``work`` raises, everything it did is rolled
        back and the exception reaches the caller; otherwise it commits.
        """
        if self._closed:
            raise StoreClosedError("the store is closed")
        if threading.get_ident() == self._writer_thread_id:
            raise StoreError("write() was called from inside a write: it would wait for itself")
        try:
            future = self._executor.submit(self._run_write, work)
        except RuntimeError as exc:  # the executor was shut down by a concurrent close()
            raise StoreClosedError("the store is closed") from exc
        return future.result()

    def _run_write[T](self, work: Callable[[sqlite3.Connection], T]) -> T:
        self._writer_thread_id = threading.get_ident()
        conn = self._writer
        conn.execute("BEGIN IMMEDIATE")
        try:
            result = work(conn)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return result

    def read[T](self, work: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``work(connection)`` on a pooled read-only connection, in one snapshot.

        The connection cannot write. Everything ``work`` reads comes from the same committed
        state, even if the writer commits meanwhile. Waits up to ``READ_WAIT_S`` for a free
        connection (``StoreError`` after that).
        """
        if self._closed:
            raise StoreClosedError("the store is closed")
        try:
            conn = self._readers.get(timeout=READ_WAIT_S)
        except queue.Empty as exc:
            raise StoreError(
                f"no read connection became free within {READ_WAIT_S:.0f} s "
                f"(the pool has {self._pool_size})"
            ) from exc
        try:
            conn.execute("BEGIN")
            try:
                return work(conn)
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
        finally:
            # Asked through a method on purpose: another thread may have called close() while we
            # worked, but mypy, having seen the check at the top of this method, would otherwise
            # treat the flag as still False and call this branch unreachable.
            if self._is_closed():
                conn.close()
            else:
                self._readers.put(conn)

    def _is_closed(self) -> bool:
        return self._closed

    def _now(self) -> str:
        return self._clock().astimezone(UTC).isoformat(timespec="seconds")

    # --- documents ---

    def upsert_document(self, document: NewDocument) -> None:
        """Record that a document is being (re)ingested: insert it or update it, as ``pending``.

        A document the store already has keeps its ``created_at`` and gets the new path, title,
        hashes and pipeline version; its state goes back to ``pending`` and any error is cleared,
        because this is the first step of a new run through the outbox (PLAN.md §6.4). Its old
        chunks stay until ``sync_chunks`` replaces them, but they are not searchable meanwhile.
        """
        now = self._now()

        def work(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO documents (doc_id, path, title, kind, file_hash, pipeline_version, "
                "state, error, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?) "
                "ON CONFLICT (doc_id) DO UPDATE SET path = excluded.path, title = excluded.title, "
                "kind = excluded.kind, file_hash = excluded.file_hash, "
                "pipeline_version = excluded.pipeline_version, state = 'pending', error = NULL, "
                "updated_at = excluded.updated_at",
                (
                    document.doc_id,
                    document.path,
                    document.title,
                    document.kind,
                    document.file_hash,
                    document.pipeline_version,
                    now,
                    now,
                ),
            )

        self.write(work)

    def set_document_state(self, doc_id: str, state: DocState, *, error: str | None = None) -> None:
        """Move a document to ``state``. ``error`` is only for ``failed`` and is cleared otherwise.

        This does not check that the move makes sense (``pending`` straight to ``ready``): the
        pipeline's state machine (S1.6) owns that rule. ``ValueError`` for an unknown state or an
        ``error`` with any state but ``failed``; ``UnknownDocumentError`` if there is no such
        document.
        """
        if state not in DOC_STATES:
            raise ValueError(f"unknown document state {state!r}; it must be one of {DOC_STATES}")
        if error is not None and state != "failed":
            raise ValueError("an error message only goes with the state 'failed'")
        now = self._now()

        def work(conn: sqlite3.Connection) -> None:
            cursor = conn.execute(
                "UPDATE documents SET state = ?, error = ?, updated_at = ? WHERE doc_id = ?",
                (state, error, now, doc_id),
            )
            if cursor.rowcount == 0:
                raise UnknownDocumentError(doc_id)

        self.write(work)

    def get_document(self, doc_id: str) -> Document | None:
        """The stored document, or ``None``."""

        def work(conn: sqlite3.Connection) -> Document | None:
            row = conn.execute(_DOCUMENT_SQL, (doc_id,)).fetchone()
            return None if row is None else _document_from_row(row)

        return self.read(work)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks (and their index entries). ``False`` if unknown.

        The chunks are deleted first and explicitly, so the full-text triggers run for each of
        them, instead of relying on the foreign key's cascade to do it.
        """

        def work(conn: sqlite3.Connection) -> bool:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

        return self.write(work)

    # --- chunks ---

    def sync_chunks(self, doc_id: str, chunks: Sequence[IdentifiedChunk]) -> ChunkSync:
        """Make the document's stored chunks exactly ``chunks``: a set difference on ``chunk_id``.

        ``chunks`` is all of one document's chunks, from ``identify_chunks``. In one transaction
        the chunks whose ID is new are inserted, those whose ID is gone are deleted, and those
        whose ID is kept are rewritten in place if their text or position changed (S1.3: position
        is metadata, never identity). The full-text index follows through the triggers, and a
        pure move does not touch it. The result says which IDs were added, removed, updated or
        left alone, which is what the pipeline needs to know what to embed and what to forget.

        ``ValueError`` if a chunk belongs to another document or two chunks share an ID;
        ``UnknownDocumentError`` if the document was not recorded with ``upsert_document`` first.
        Any failure rolls the whole sync back.
        """
        for item in chunks:
            if item.doc_id != doc_id:
                raise ValueError(f"chunk {item.chunk_id} is not part of document {doc_id}")
        if len({item.chunk_id for item in chunks}) != len(chunks):
            raise ValueError("two chunks share a chunk_id: pass one document's chunks, once each")
        now = self._now()
        return self.write(lambda conn: _sync_chunks(conn, doc_id, chunks, now))

    def get_chunks(self, chunk_ids: Sequence[str], *, ready_only: bool = True) -> list[Chunk]:
        """The stored chunks with these IDs, in the order asked for; unknown IDs are skipped.

        By default only chunks of ``ready`` documents come back (what retrieval may see). Pass
        ``ready_only=False`` to see a document's chunks while it is still being processed.
        """
        if not chunk_ids:
            return []
        parameters = {"ids": json.dumps(list(chunk_ids)), "include_unready": not ready_only}

        def work(conn: sqlite3.Connection) -> list[Chunk]:
            rows = conn.execute(_GET_CHUNKS_SQL, parameters).fetchall()
            return [_chunk_from_row(row) for row in rows]

        found = {chunk.chunk_id: chunk for chunk in self.read(work)}
        return [found[chunk_id] for chunk_id in chunk_ids if chunk_id in found]

    # --- lexical search ---

    def search_bm25(self, text: str, limit: int = 10) -> list[ScoredChunk]:
        """The ``limit`` best BM25 matches for what a person typed, best first, ready chunks only.

        Together with ``fts_match_query`` this is step S1.4's ✍️ piece (written by Claude in
        prototyping mode, D-047).

        ``text`` is plain text, not an FTS5 expression: ``fts_match_query`` turns it into a safe
        one, so any input is accepted and one with no words returns ``[]``. FTS5 scores with
        BM25 (k1 = 1.2 and b = 0.75, fixed by SQLite): a chunk scores higher for containing more
        of the query's words, for containing a word more often, for the word being rare across
        the index, and for being short. The score is returned as a positive number, higher is
        better; it is only comparable with scores from the same query.

        Only chunks of documents in the ``ready`` state are searched (PLAN.md §6.4). Ties are
        broken by ``chunk_id``, so the order never changes between runs. ``ValueError`` if
        ``limit`` is below 1.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        match = fts_match_query(text)
        if match is None:
            return []

        def work(conn: sqlite3.Connection) -> list[ScoredChunk]:
            rows = conn.execute(_SEARCH_SQL, (match, limit)).fetchall()
            return [ScoredChunk(chunk=_chunk_from_row(row), score=-float(row[13])) for row in rows]

        return self.read(work)

    def rebuild_fts(self) -> None:
        """Throw the full-text index away and build it again from the ``chunks`` table.

        The repair for an index that no longer matches the rows (``doctor --repair``, S1.6), and
        the second half of any migration that changes the tokenizer: the index is derived data,
        so this is always safe, only slow for a large store.
        """

        def work(conn: sqlite3.Connection) -> None:
            conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")

        self.write(work)
