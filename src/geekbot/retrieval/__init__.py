"""Retrieval: a question in, ranked chunks out (steps S2.1 to S2.3, PLAN.md §6.6, §9).

Every way of finding chunks (dense, lexical, fused, reranked) has the same shape, so the answer
graph (Phase 3) and the ablation (S2.6, rungs R0-R7) can swap one for another by changing a name:
``search(query, k)`` returns a ``RetrievalResult``, the chunks best first and how long each stage
took. The timings are part of the result, not an extra, because the plan sets a latency budget
for retrieval (400 ms p95, §6.5) and a budget nobody can read off a result is not enforced.

Retrieval only returns chunks of ``ready`` documents (PLAN.md §6.4). Where that filter sits matters
and is each retriever's own business; ``Store.get_chunks`` and ``Store.search_bm25`` already apply
it inside SQLite.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, TypedDict

from geekbot.domain import ScoredChunk

__all__ = [
    "EXCERPT_CHARS",
    "RetrievalResult",
    "Retriever",
    "SearchHit",
    "SearchMode",
    "format_result",
]

EXCERPT_CHARS: Final = 200


class SearchHit(TypedDict):
    """One candidate as the answer graph keeps it between search steps (S3.6): an ID and a score.

    Plain data on purpose. The graph's state is saved after every step, so it holds the *references*
    to the candidates (``chunk_id`` and the score the search that found it gave) and the text is read
    back from the store only for the few chunks that survive fusion (``retrieval.staged``).
    """

    chunk_id: str
    score: float


class SearchMode(StrEnum):
    """The ways of finding chunks, by name: ``geekbot search --mode`` and ``geekbot ask --mode``.

    They are rungs of the ablation ladder (PLAN.md §9): each adds one thing to the one before.
    ``retrieval.factory.build_retriever`` turns a name into a retriever. The enum lives here, in
    a module that imports almost nothing, so the CLI can use it as an option type without paying
    for Chroma or fastembed at start-up.
    """

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class RetrievalResult:
    """The chunks a retriever found, best first, and where the time went.

    ``timings_ms`` maps a stage name to the milliseconds it took, plus ``total``. The stage names
    are the retriever's own (a dense search has ``embed_query``, ``vector_search`` and
    ``resolve``). A higher ``score`` is better, but scores of different retrievers are never
    compared (``ScoredChunk``); fusion works on ranks.
    """

    chunks: tuple[ScoredChunk, ...]
    timings_ms: Mapping[str, float] = field(default_factory=dict)


class Retriever(Protocol):
    """Anything that can find chunks for a question."""

    def search(self, query: str, k: int) -> RetrievalResult:
        """The ``k`` best chunks for ``query`` (fewer if there are fewer). ``ValueError`` if ``k < 1``."""
        ...


def format_result(result: RetrievalResult, *, excerpt_chars: int = EXCERPT_CHARS) -> str:
    """The result as text for a terminal: one numbered entry per chunk, then the timings."""
    lines: list[str] = []
    for rank, scored in enumerate(result.chunks, start=1):
        chunk = scored.chunk
        where = [chunk.title]
        if chunk.page is not None:
            where.append(f"p.{chunk.page}")
        if chunk.section_path:
            where.append(" > ".join(chunk.section_path))
        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[: excerpt_chars - 1].rstrip() + "…"
        lines.append(f"{rank}. {scored.score:.3f}  {' · '.join(where)}  [{chunk.chunk_id[:8]}]")
        lines.append(f"   {excerpt}")
    if not result.chunks:
        lines.append("no results")
    if result.timings_ms:
        lines.append(" · ".join(f"{name} {ms:.1f} ms" for name, ms in result.timings_ms.items()))
    return "\n".join(lines)
