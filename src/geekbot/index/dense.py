"""The dense index: Chroma as a derived store of chunk vectors (step S1.5, PLAN.md §6.2, §6.4).

SQLite is the source of truth for chunks (``index/sqlite.py``); this file holds only their vectors,
keyed by ``chunk_id``, and can always be rebuilt from SQLite and the embedding cache. Chroma never
embeds anything here: Geekbot hands it vectors it computed itself (so they are cached, prefixed and
on the CPU), and the collection's embedding function is switched off so that Chroma can never go
and download a default model behind our back.

**Cosine, not L2.** The collection is created with the cosine space and ``query`` returns
``1 - distance``, a similarity where higher is better and 1.0 is identical. Chroma's default is L2.
For unit-length vectors the two rank the same (v1 bug #14), but nothing guarantees an Ollama
embedder returns unit length, and the live test in ``tests/live`` checks it (the R1 expectation).
The one reason to ask for ``space="l2"`` is the ablation's rung R0 (S2.6), which replays v1's
default; the score is then minus the (squared) distance, so higher is still better.

**The fingerprint guard (PLAN.md §6.4).** The collection's metadata records the embedder
fingerprint. Opening an index with a different fingerprint raises ``FingerprintMismatchError``
instead of quietly returning neighbours from a different geometry; ``reset=True`` is the explicit
"drop the vectors and start over" that a re-index begins with.

**Ownership.** The one storage-owning process opens this (PLAN.md §6.2); ``Store`` already holds
the storage lock, so this class does not take a second one. Chroma's client may be called from
several threads, and the server runs every call in a worker thread (PLAN.md §6.3).
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Self

import chromadb
import numpy as np
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import ChromaError, NotFoundError

from geekbot.index.embedder import EmbedderSpec

__all__ = [
    "CHROMA_DIR_NAME",
    "DenseEntry",
    "DenseHit",
    "DenseIndex",
    "DenseIndexError",
    "FingerprintMismatchError",
    "Space",
]

CHROMA_DIR_NAME: Final = "chroma"
COLLECTION_NAME: Final = "chunks"
_FINGERPRINT_KEY: Final = "embedder_fingerprint"
_SPEC_KEY: Final = "embedder_spec"
_SPACE_KEY: Final = "space"

# The geometry Chroma's HNSW graph measures distance in. Cosine is the product; L2 exists for R0.
Space = Literal["cosine", "l2"]
# Chroma's own limit on one call is larger (get_max_batch_size, 5,461 here); a smaller batch keeps
# the memory of one call small and is plenty fast.
_BATCH: Final = 1000


class DenseIndexError(RuntimeError):
    """The dense index could not do what was asked (wrong vector size, Chroma failure)."""


class FingerprintMismatchError(DenseIndexError):
    """The stored vectors were made by a different embedder than the one now configured."""

    def __init__(self, stored: str, expected: str, stored_spec: str, expected_spec: str) -> None:
        super().__init__(
            f"the dense index was built with embedder {stored} but this run uses {expected}; "
            f"stored {stored_spec}; configured {expected_spec}. Vectors from different embedders "
            "are not comparable. Re-index (open with reset=True) to rebuild it."
        )
        self.stored = stored
        self.expected = expected


@dataclass(frozen=True)
class DenseEntry:
    """One vector to store: the chunk it belongs to, its document, and the numbers."""

    chunk_id: str
    doc_id: str
    vector: Sequence[float]


@dataclass(frozen=True)
class DenseHit:
    """One neighbour of a query: a chunk and its cosine similarity (higher is closer)."""

    chunk_id: str
    score: float


def _close_client(client: ClientAPI) -> None:
    """Close Chroma's client. The method exists at run time (1.5) but not in its type stubs."""
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _batches[T](items: Sequence[T], size: int = _BATCH) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class DenseIndex:
    """Chroma's persistent client with one cosine collection of chunk vectors."""

    def __init__(
        self, data_dir: Path, spec: EmbedderSpec, *, reset: bool = False, space: Space = "cosine"
    ) -> None:
        self._spec = spec
        self._space: Space = space
        self._fingerprint = spec.fingerprint()
        path = data_dir / CHROMA_DIR_NAME
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        try:
            self._collection = self._open_collection(reset=reset)
        except BaseException:
            _close_client(self._client)
            raise

    def _open_collection(self, *, reset: bool) -> Collection:
        if reset:
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except NotFoundError:
                pass
        try:
            collection = self._client.get_collection(COLLECTION_NAME, embedding_function=None)
        except NotFoundError:
            return self._client.create_collection(
                COLLECTION_NAME,
                embedding_function=None,
                configuration={"hnsw": {"space": self._space}},
                metadata={
                    _FINGERPRINT_KEY: self._fingerprint,
                    _SPEC_KEY: self._spec.describe(),
                    _SPACE_KEY: self._space,
                },
            )
        stored = collection.metadata or {}
        # A collection made before the space was recorded is cosine, the only one there was.
        stored_space = str(stored.get(_SPACE_KEY, "cosine"))
        if stored_space != self._space:
            raise DenseIndexError(
                f"the dense index uses the {stored_space} space but this run asked for "
                f"{self._space}; open it with reset=True to rebuild it"
            )
        stored_fingerprint = str(stored.get(_FINGERPRINT_KEY, ""))
        if stored_fingerprint != self._fingerprint:
            raise FingerprintMismatchError(
                stored_fingerprint,
                self._fingerprint,
                str(stored.get(_SPEC_KEY, "unknown")),
                self._spec.describe(),
            )
        return collection

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
        """Release Chroma's files. Needed on Windows before a data folder can be deleted."""
        _close_client(self._client)

    @property
    def spec(self) -> EmbedderSpec:
        """The embedder this index is bound to."""
        return self._spec

    def count(self) -> int:
        """How many vectors are stored."""
        return self._collection.count()

    def _check_size(self, vector: Sequence[float]) -> None:
        if len(vector) != self._spec.dimension:
            raise DenseIndexError(
                f"a vector of {len(vector)} numbers cannot go in an index of "
                f"{self._spec.dimension}-number vectors"
            )

    def upsert(self, entries: Sequence[DenseEntry]) -> None:
        """Store these vectors; a ``chunk_id`` already present is overwritten, so this is idempotent."""
        for entry in entries:
            self._check_size(entry.vector)
        try:
            for batch in _batches(entries):
                self._collection.upsert(
                    ids=[entry.chunk_id for entry in batch],
                    embeddings=np.array([list(entry.vector) for entry in batch], dtype=np.float32),
                    metadatas=[{"doc_id": entry.doc_id} for entry in batch],
                )
        except ChromaError as error:
            raise DenseIndexError(f"Chroma upsert failed: {error}") from error

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove these vectors. An ID that is not there is ignored, so a retry is harmless."""
        for batch in _batches(chunk_ids):
            self._collection.delete(ids=list(batch))

    def delete_document(self, doc_id: str) -> None:
        """Remove every vector of one document."""
        self._collection.delete(where={"doc_id": doc_id})

    def query(self, vector: Sequence[float], k: int) -> list[DenseHit]:
        """The ``k`` nearest chunks to ``vector``, most similar first.

        Returns fewer than ``k`` when the index holds fewer. Whether a chunk may be shown (its
        document is ``ready``) is not known here; the retriever filters through SQLite (S2.1).
        """
        if k < 1:
            return []
        self._check_size(vector)
        stored = self._collection.count()
        if stored == 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=np.array([list(vector)], dtype=np.float32),
                n_results=min(k, stored),
                include=["distances"],
            )
        except ChromaError as error:
            raise DenseIndexError(f"Chroma query failed: {error}") from error
        ids = result["ids"][0]
        distances = (result["distances"] or [[]])[0]
        # Cosine distance is 1 - similarity. An L2 distance has no such upper bound, so its score
        # is just the negated distance: still "higher is closer", only not on a 0-1 scale.
        base = 0.0 if self._space == "l2" else 1.0
        return [
            DenseHit(chunk_id=chunk_id, score=base - float(distance))
            for chunk_id, distance in zip(ids, distances, strict=True)
        ]

    def ids(self) -> set[str]:
        """Every stored ``chunk_id`` (what ``doctor --repair`` compares with SQLite, S1.6)."""
        found: set[str] = set()
        offset = 0
        while True:
            page = self._collection.get(limit=_BATCH, offset=offset, include=[])
            found.update(page["ids"])
            if len(page["ids"]) < _BATCH:
                return found
            offset += _BATCH

    def ids_of_document(self, doc_id: str) -> set[str]:
        """The stored ``chunk_id`` values of one document."""
        page = self._collection.get(where={"doc_id": doc_id}, include=[])
        return set(page["ids"])

    def vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """The stored vectors of these chunks (those that exist), as 32-bit-rounded floats."""
        found: dict[str, list[float]] = {}
        for batch in _batches(chunk_ids):
            page = self._collection.get(ids=list(batch), include=["embeddings"])
            embeddings = page["embeddings"]
            if embeddings is None:
                continue
            for chunk_id, embedding in zip(page["ids"], embeddings, strict=True):
                found[chunk_id] = [float(x) for x in embedding]
        return found
