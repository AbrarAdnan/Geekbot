"""The embedding cache: never embed the same text twice (step S1.5, PLAN.md §6.4, risk 8).

Embedding is the slow part of ingest on a CPU, and most re-ingests change little. The cache keeps
every vector it has computed, so a document that is processed again, a chunk that moved, or a file
that was deleted and added back costs no embedding at all (Gate A: "re-ingesting unchanged data
costs 0 embeddings").

**The key** is ``(embedder fingerprint, SHA-256 of the exact string that was embedded)``.

- The fingerprint (``EmbedderSpec``) covers the model, its digest, the prefixes, the vector size
  and the header mode, so a vector computed under one setup is never served for another.
- The text hash is of the *embedding text* (the chunk, with its header in front when the header
  mode is on), not of the chunk alone. The plan names the chunk's ``content_hash``, which is enough
  while the header is off. With the header on, the same chunk text under two different headings is
  two different strings and two different vectors, so a key built from the chunk alone would
  serve one heading's vector for the other (D-058). With the header off the two keys are
  equivalent.

The cache is its own SQLite file, not a table of ``geekbot.db``: it holds derived data that can
be deleted at any time without losing anything but time, and keeping it apart means a cache write
never queues behind the store's single writer. Vectors are stored as 32-bit floats, which is what
Chroma keeps; ``to_float32`` rounds fresh vectors the same way so a cached vector and a freshly
computed one are identical, not merely close.
"""

import hashlib
import json
import sqlite3
import threading
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Self

from geekbot.index.embedder import EmbedderSpec, EmbeddingError, EmbeddingsClient

__all__ = [
    "CACHE_FILE_NAME",
    "CachedEmbedder",
    "EmbedResult",
    "EmbeddingCache",
    "text_key",
    "to_float32",
]

CACHE_FILE_NAME: Final = "embed_cache.db"
_SCHEMA_VERSION: Final = 1

_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS embeddings (
    fingerprint TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (fingerprint, text_hash)
) WITHOUT ROWID;
"""


def text_key(text: str) -> str:
    """The cache key of an embedding text: its SHA-256 as 64 hex characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_float32(vector: Sequence[float]) -> list[float]:
    """Round a vector to 32-bit floats and back, as the cache and Chroma store it."""
    return list(array("f", vector))


def _encode(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def _decode(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


class EmbeddingCache:
    """A persistent ``(fingerprint, text hash) -> vector`` map in one SQLite file.

    Safe to share between threads: one connection, guarded by a lock. The server calls it from
    worker threads (PLAN.md §6.3), and a cache lookup is far shorter than the embedding it saves.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        try:
            self._prepare()
        except BaseException:
            self._conn.close()
            raise

    def _prepare(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        row = self._conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0])
        if version not in (0, _SCHEMA_VERSION):
            # A file written by another version of this code. It is only a cache: start over.
            self._conn.execute("DROP TABLE IF EXISTS embeddings")
        with self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the file. Safe to call twice."""
        with self._lock:
            self._conn.close()

    def lookup(self, fingerprint: str, text_hashes: Sequence[str]) -> dict[str, list[float]]:
        """The cached vectors for ``text_hashes`` under ``fingerprint``, as ``{text_hash: vector}``.

        ✍️ Written by Claude (prototyping mode, D-047). A hash with no entry under this
        fingerprint is simply absent from the result, so the caller embeds exactly the missing
        ones. Entries under any other fingerprint are invisible: that is what keeps a vector from
        one embedder from being served as another's. The hashes go in as one JSON array
        (``json_each``), so a batch of any size is one statement and no hand-built SQL.
        """
        if not text_hashes:
            return {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT text_hash, vector FROM embeddings "
                "WHERE fingerprint = ? AND text_hash IN (SELECT value FROM json_each(?))",
                (fingerprint, json.dumps(list(text_hashes))),
            ).fetchall()
        return {str(text_hash): _decode(bytes(blob)) for text_hash, blob in rows}

    def store(self, fingerprint: str, vectors: Mapping[str, Sequence[float]]) -> None:
        """Remember ``{text_hash: vector}`` under ``fingerprint`` (an existing entry is replaced)."""
        if not vectors:
            return
        rows = [
            (fingerprint, text_hash, len(vector), _encode(vector))
            for text_hash, vector in vectors.items()
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (fingerprint, text_hash, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def count(self, fingerprint: str | None = None) -> int:
        """How many vectors are cached, in total or under one fingerprint."""
        with self._lock:
            if fingerprint is None:
                row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE fingerprint = ?", (fingerprint,)
                ).fetchone()
        return int(row[0])


@dataclass(frozen=True)
class EmbedResult:
    """The vectors for a list of texts, and what it cost to get them.

    ``embedded`` is the number of *distinct* texts that went to the embedder; ``cache_hits`` the
    number of distinct texts found in the cache. Their sum is the number of distinct texts, which
    is at most ``len(vectors)`` when the same text appears twice. A second pass over unchanged
    data must report ``embedded == 0``.
    """

    vectors: list[list[float]]
    cache_hits: int
    embedded: int


class CachedEmbedder:
    """Embeds documents through the cache and queries directly.

    A query is not cached: queries rarely repeat exactly, and caching them would only grow the
    file. A document text is looked up first, and only the missing ones are embedded (each distinct
    text once, even if it occurs in several chunks), stored, and returned in the input's order.
    """

    def __init__(
        self, embeddings: EmbeddingsClient, cache: EmbeddingCache, spec: EmbedderSpec
    ) -> None:
        self._embeddings = embeddings
        self._cache = cache
        self._spec = spec
        self._fingerprint = spec.fingerprint()

    @property
    def spec(self) -> EmbedderSpec:
        """The embedder this object stands for."""
        return self._spec

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        """One vector per text, in order. Texts already in the cache are not embedded again."""
        keys = [text_key(text) for text in texts]
        distinct = dict.fromkeys(keys)  # keeps first-seen order
        known = self._cache.lookup(self._fingerprint, list(distinct))
        missing = [key for key in distinct if key not in known]
        if missing:
            first_text = {key: text for key, text in zip(keys, texts, strict=True)}
            fresh = self._embeddings.embed_documents([first_text[key] for key in missing])
            if len(fresh) != len(missing):
                raise EmbeddingError(f"asked for {len(missing)} embeddings, got {len(fresh)}")
            rounded = {key: to_float32(vector) for key, vector in zip(missing, fresh, strict=True)}
            for key, vector in rounded.items():
                if len(vector) != self._spec.dimension:
                    raise EmbeddingError(
                        f"the embedder returned {len(vector)} numbers, expected "
                        f"{self._spec.dimension} (the dimension in its fingerprint)"
                    )
                known[key] = vector
            self._cache.store(self._fingerprint, rounded)
        return EmbedResult(
            vectors=[known[key] for key in keys],
            cache_hits=len(distinct) - len(missing),
            embedded=len(missing),
        )

    def embed_query(self, text: str) -> list[float]:
        """The vector of a query, rounded like every stored vector."""
        vector = to_float32(self._embeddings.embed_query(text))
        if len(vector) != self._spec.dimension:
            raise EmbeddingError(
                f"the embedder returned {len(vector)} numbers for a query, expected "
                f"{self._spec.dimension}"
            )
        return vector
