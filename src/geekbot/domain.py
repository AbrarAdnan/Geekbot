"""Plain records that cross layers (PLAN.md §6.6).

A record here is data and nothing else: no behaviour that touches a store, a model or the disk, so
the ingest, index, retrieval and agent packages can all import it without importing one another.
S1.4 adds the ones the store hands out; the plan's other records (``Citation``, ``RunStats``,
``Label``) arrive with the steps that first need them.
"""

from dataclasses import dataclass
from typing import Final, Literal

# The outbox states of a document (PLAN.md §6.4): its chunk rows are written while it is
# ``pending``, its vectors while it is ``embedding``, and retrieval may only see it once it is
# ``ready``. ``failed`` keeps the reason in ``Document.error``. Which moves between states are
# allowed is the pipeline's rule (S1.6); the store only checks that the state is one of these.
DocState = Literal["pending", "embedding", "ready", "failed"]
DOC_STATES: Final[tuple[DocState, ...]] = ("pending", "embedding", "ready", "failed")


@dataclass(frozen=True)
class NewDocument:
    """What the pipeline knows about a file when it hands it to the store.

    ``doc_id`` comes from ``ids.doc_id_of``, ``file_hash`` from ``ids.file_hash_of``. The path is
    kept for display and citations, so it keeps its spelling (the ID uses a normalized one).
    ``pipeline_version`` names the loader and chunker that produced the chunks: an unchanged file
    still has to be processed again when that changes, which ``file_hash`` alone cannot say.
    """

    doc_id: str
    path: str
    title: str
    kind: str
    file_hash: str
    pipeline_version: str


@dataclass(frozen=True)
class Document:
    """A stored document: a ``NewDocument`` plus where it is in the outbox."""

    doc_id: str
    path: str
    title: str
    kind: str
    file_hash: str
    pipeline_version: str
    state: DocState
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Chunk:
    """A stored chunk, with the path and title of its document so a citation needs no second query.

    The fields are those of ``IdentifiedChunk`` (S1.3) plus ``path`` and ``title``. ``char_start``
    and ``char_end`` are offsets into the document's loaded text and ``page`` is 1-based, as in
    S1.1 and S1.2; ``section_path`` is the trail of headings above the chunk.
    """

    chunk_id: str
    doc_id: str
    path: str
    title: str
    content_hash: str
    occurrence: int
    ordinal: int
    char_start: int
    char_end: int
    page: int | None
    section_path: tuple[str, ...]
    header: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk and how well a retriever scored it. A higher ``score`` is better.

    What the score means depends on the retriever (a BM25 value, a cosine similarity), so two
    scores from different retrievers are never compared; fusion (S2.2) works on ranks.
    """

    chunk: Chunk
    score: float
