"""Open everything a run needs, in the right order, and close it in the reverse (step S1.6).

The storage-owning process works with four things at once: the SQLite store (which also holds the
exclusive storage lock), the Chroma dense index, the embedding cache and the embedder. They depend
on each other (the dense index needs the embedder's fingerprint, which needs a call to Ollama), and
a half-opened set must not leave a lock held or a file open. ``Workspace.open`` does the ordering
once, so the ingest command, ``doctor --repair`` and later the server (S4.1) and ``geekbot search``
(S2.1) cannot each get it slightly wrong.

Order, and why:

1. ``Store``: takes the storage lock first, so a second process fails at once and nothing else is
   touched (PLAN.md §6.2).
2. The embedder: one call to Ollama for the model's digest and one probe embedding. If Ollama is
   down or the model is not pulled, this fails before any vector file is opened.
3. ``DenseIndex``: needs the embedder's fingerprint; raises ``FingerprintMismatchError`` if the
   stored vectors came from another embedder, unless ``reset_dense`` asks for a re-index.
4. ``EmbeddingCache``.
"""

from contextlib import ExitStack
from types import TracebackType
from typing import Self

import httpx

from geekbot.config import Settings
from geekbot.index.dense import DenseIndex
from geekbot.index.embed_cache import CACHE_FILE_NAME, CachedEmbedder, EmbeddingCache
from geekbot.index.embedder import EmbedderSpec, EmbeddingsClient, open_ollama_embedder
from geekbot.index.sqlite import Store

__all__ = ["Workspace"]


class Workspace:
    """The opened store, dense index, cache and embedder. Use as a context manager."""

    def __init__(
        self,
        store: Store,
        dense: DenseIndex,
        cache: EmbeddingCache,
        embedder: CachedEmbedder,
        stack: ExitStack,
    ) -> None:
        self.store = store
        self.dense = dense
        self.cache = cache
        self.embedder = embedder
        self._stack = stack

    @classmethod
    def open(
        cls,
        settings: Settings,
        *,
        reset_dense: bool = False,
        embeddings: EmbeddingsClient | None = None,
        spec: EmbedderSpec | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> Self:
        """Open everything under ``settings.data_dir``.

        ``reset_dense`` discards the stored vectors (a re-index; run ``repair`` afterwards to
        rebuild them from the stored chunks). ``embeddings`` and ``spec`` replace the Ollama
        embedder, for tests; give both or neither. ``transport`` is for tests of the Ollama path.
        """
        if (embeddings is None) != (spec is None):
            raise ValueError("pass both `embeddings` and `spec`, or neither")
        with ExitStack() as stack:
            store = stack.enter_context(Store(settings.data_dir))
            if embeddings is None or spec is None:
                prefixed, spec, client = open_ollama_embedder(settings, transport=transport)
                stack.callback(client.close)
                embeddings = prefixed
            dense = stack.enter_context(DenseIndex(settings.data_dir, spec, reset=reset_dense))
            cache = stack.enter_context(EmbeddingCache(settings.data_dir / CACHE_FILE_NAME))
            embedder = CachedEmbedder(embeddings, cache, spec)
            return cls(store, dense, cache, embedder, stack.pop_all())

    def close(self) -> None:
        """Close everything, last opened first, and release the storage lock."""
        self._stack.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
