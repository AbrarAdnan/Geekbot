"""Build a retriever from its name (step S3.1, PLAN.md §6.6, §9).

``geekbot search`` (S2.1 to S2.3) and ``geekbot ask`` (S3.1) both need "the retriever called
``hybrid_rerank``" made from an opened workspace and the settings. Written twice, the two would
drift: one gets a new default and the other does not. So there is one place that says what each name
means. S2.6 extends it into the ablation's rungs R0 to R7.

Everything heavy is imported inside the function, at the moment of use. Two reasons: Chroma and
fastembed are slow to import, and the tests replace ``FastembedScorer`` by patching it on its own
module, which only works if this function looks the name up when it runs.
"""

from typing import TYPE_CHECKING

from geekbot.retrieval import Retriever, SearchMode
from geekbot.retrieval.hybrid import DEFAULT_CANDIDATES  # what each side contributes before fusing

if TYPE_CHECKING:
    from geekbot.config import Settings
    from geekbot.index.workspace import Workspace
    from geekbot.retrieval.rerank import PassageScorer

__all__ = ["DEFAULT_CANDIDATES", "build_retriever", "citation_scorer_of"]


def citation_scorer_of(retriever: Retriever) -> "PassageScorer | None":
    """The cross-encoder inside ``retriever``, if it has one (``hybrid_rerank``), else ``None``.

    The answer graph's ``cite_check`` (S3.5) scores citations with the model that already reranks,
    so no second copy is loaded; a retriever without a reranker means the check is skipped.
    """
    from geekbot.retrieval.rerank import Reranker
    from geekbot.retrieval.staged import StagedSearch

    return retriever.scorer if isinstance(retriever, Reranker | StagedSearch) else None


def build_retriever(
    mode: SearchMode,
    workspace: "Workspace",
    settings: "Settings",
    *,
    candidates: int = DEFAULT_CANDIDATES,
    staged: bool = False,
) -> Retriever:
    """The retriever for ``mode``, reading from ``workspace``.

    - ``dense``: Chroma finds candidates by meaning; SQLite decides which are ready.
    - ``lexical``: BM25 over SQLite (the workspace's embedder is not used).
    - ``hybrid``: both, each asked for ``candidates`` chunks, fused by rank (RRF).
    - ``hybrid_rerank``: the hybrid result's best ``settings.rerank_depth`` chunks, re-scored by a
      cross-encoder on the CPU. Its model is downloaded on first use.

    ``staged`` (S3.6) returns the two hybrid modes as a ``StagedSearch``: the same retrieval with
    its stages exposed, which the answer graph runs side by side. Its own ``search`` gives the same
    result as the sequential classes. The other two modes have one stage and ignore it.
    """
    from geekbot.retrieval.dense import DenseRetriever
    from geekbot.retrieval.hybrid import HybridRetriever
    from geekbot.retrieval.lexical import LexicalRetriever
    from geekbot.retrieval.rerank import FastembedScorer, Reranker
    from geekbot.retrieval.staged import StagedSearch

    lexical = LexicalRetriever(workspace.store)
    if mode is SearchMode.LEXICAL:
        return lexical
    dense = DenseRetriever(workspace.store, workspace.dense, workspace.embedder)
    if mode is SearchMode.DENSE:
        return dense
    scorer = (
        None
        if mode is SearchMode.HYBRID
        else FastembedScorer(
            settings.rerank_model,
            cache_dir=settings.data_dir / "models",
            threads=settings.rerank_threads,
        )
    )
    if staged:
        return StagedSearch(
            dense,
            lexical,
            workspace.store.get_chunks,
            scorer,
            candidates=candidates,
            depth=settings.rerank_depth,
        )
    hybrid = HybridRetriever(dense, lexical, candidates=candidates)
    if scorer is None:
        return hybrid
    return Reranker(hybrid, scorer, depth=settings.rerank_depth)
