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
from typing import Final, Protocol

from geekbot.domain import ScoredChunk

__all__ = ["EXCERPT_CHARS", "RetrievalResult", "Retriever", "format_result"]

EXCERPT_CHARS: Final = 200


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
