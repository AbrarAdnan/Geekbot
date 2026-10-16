"""Put a document's vectors into Chroma, and nothing else (step S1.6, PLAN.md §6.4).

The one routine both the ingest pipeline and the repair use for the middle step of the outbox:
"the rows are final; now make the vectors match". Having a single copy matters because the repair
is only trustworthy if it does exactly what the pipeline does.

It embeds *every* chunk of the document through the cache and upserts all of them, instead of
only the chunks that are new. That looks wasteful and is not: a chunk whose text and header are
unchanged is a cache hit (no embedding), and the upsert of an identical vector is harmless. In
return it is correct in every case the "only the new ones" shortcut gets wrong: a crash that left
some vectors unwritten, a heading renamed above a chunk whose text did not change (its embedding
text changed with the header), and a vector store that was wiped.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from geekbot.index.dense import DenseEntry, DenseIndex
from geekbot.index.embed_cache import CachedEmbedder

__all__ = ["VectorWrite", "write_document_vectors"]


@dataclass(frozen=True)
class VectorWrite:
    """What writing one document's vectors cost.

    ``embedded`` is the number of texts the embedder had to compute; ``cache_hits`` the number it
    did not; ``removed`` the number of stale vectors of this document that were deleted.
    """

    embedded: int
    cache_hits: int
    removed: int


def write_document_vectors(
    dense: DenseIndex,
    embedder: CachedEmbedder,
    doc_id: str,
    chunks: Sequence[tuple[str, str]],
) -> VectorWrite:
    """Make the document's vectors in ``dense`` exactly those of ``chunks``.

    ``chunks`` is ``(chunk_id, embedding_text)`` for every chunk of the document, in order. After
    the call the index holds one vector per chunk and no vector of this document that is not in
    ``chunks``. Embeds only what the cache lacks. Raises whatever the embedder or Chroma raises;
    the caller moves the document to ``failed``.
    """
    result = embedder.embed_documents([text for _, text in chunks])
    dense.upsert(
        [
            DenseEntry(chunk_id, doc_id, vector)
            for (chunk_id, _), vector in zip(chunks, result.vectors, strict=True)
        ]
    )
    wanted = {chunk_id for chunk_id, _ in chunks}
    stale = sorted(dense.ids_of_document(doc_id) - wanted)
    dense.delete(stale)
    return VectorWrite(embedded=result.embedded, cache_hits=result.cache_hits, removed=len(stale))
