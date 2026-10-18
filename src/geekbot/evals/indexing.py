"""Build the indexes the ablation searches (step S2.6, PLAN.md §9).

Each rung of the ladder searches one *index variant*: how the text was cut, whether the embedder got
its task prefixes, whether the contextual header was embedded, and in which geometry Chroma measures
distance. Rungs that share a variant share one index, and variants share one embedding cache, so a
text that is embedded under the same fingerprint twice (R0 and R1 differ only in the geometry) is
embedded once.

Documents come in as ``LoadedDocument``, the loaders' output, not as files. That lets BEIR's
5,183 abstracts go through the same chunker, store, embedder and retrievers as a user's files
without writing 5,183 files, and it lets the golden corpus use the real loaders. What this skips is
only the loading (which is not what the ablation varies) and the file-hash skip logic (every build
here starts from an empty folder).

Everything lives in a scratch folder the caller names. The user's ``data/`` is never opened.
"""

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal, Self

from geekbot.config import Settings
from geekbot.domain import NewDocument
from geekbot.evals.beir import BeirDataset
from geekbot.evals.legacy import legacy_chunks
from geekbot.index.dense import DenseEntry, DenseIndex, Space
from geekbot.index.embed_cache import CachedEmbedder, EmbeddingCache
from geekbot.index.embedder import (
    EmbedderSpec,
    EmbeddingsClient,
    PrefixedEmbeddings,
    open_ollama_embedder,
)
from geekbot.index.sqlite import Store
from geekbot.ingest.chunking import ChunkDraft, chunk_document
from geekbot.ingest.ids import doc_id_of, identify_chunks
from geekbot.ingest.loaders.model import LoadedDocument, Segment
from geekbot.ingest.outbox import Outbox

__all__ = [
    "BATCH_DOCUMENTS",
    "BuildStats",
    "BuiltIndex",
    "Chunking",
    "EmbedderChoice",
    "IndexVariant",
    "beir_doc_path",
    "beir_documents",
    "build_index",
    "open_embedder",
]

Chunking = Literal["legacy", "v2"]

# Documents whose chunks are embedded in one call. Big enough that the embedder works on full
# batches (an Ollama request per document would spend more time on HTTP than on the model), small
# enough that a crash loses little.
BATCH_DOCUMENTS: Final = 64
_MARKER: Final = "variant.json"


@dataclass(frozen=True)
class IndexVariant:
    """Everything about an index that a rung can change (PLAN.md §9)."""

    chunking: Chunking
    prefixes: bool
    header: bool
    space: Space

    @property
    def key(self) -> str:
        """A folder-safe name: ``legacy-raw-nohdr-l2``, ``v2-pfx-hdr-cosine``."""
        return (
            f"{self.chunking}-{'pfx' if self.prefixes else 'raw'}-"
            f"{'hdr' if self.header else 'nohdr'}-{self.space}"
        )


@dataclass(frozen=True)
class EmbedderChoice:
    """One embedder and the way it wants to be spoken to (a bake-off candidate or the held one).

    ``document_prefix`` and ``query_prefix`` are used only by variants with ``prefixes=True``; R0 to
    R2 are defined by having none.
    """

    label: str
    model: str
    document_prefix: str = ""
    query_prefix: str = ""


def open_embedder(
    settings: Settings, choice: EmbedderChoice, variant: IndexVariant
) -> tuple[PrefixedEmbeddings, EmbedderSpec, Callable[[], None]]:
    """Connect to Ollama for ``choice`` under ``variant``; returns the embedder, its spec, a closer.

    The spec (and so the cache key and the index fingerprint) records the prefixes actually used
    and whether the header is embedded, so two variants can never serve each other's vectors.
    """
    adjusted = settings.model_copy(
        update={
            "embed_model": choice.model,
            "embed_document_prefix": choice.document_prefix if variant.prefixes else "",
            "embed_query_prefix": choice.query_prefix if variant.prefixes else "",
            "embed_with_header": variant.header,
        }
    )
    prefixed, spec, client = open_ollama_embedder(adjusted)
    return prefixed, spec, client.close


# --- the documents ---------------------------------------------------------------------------


def beir_doc_path(dataset: str, doc_id: str) -> Path:
    """A made-up but stable path for a BEIR document, so ``doc_id_of`` can name it like any other."""
    return Path("/beir") / dataset / doc_id


def beir_documents(dataset: BeirDataset) -> tuple[list[LoadedDocument], dict[str, str]]:
    """BEIR's corpus as loaded documents, and ``{doc_id: BEIR id}`` for translating back.

    The title is the document's title (what the header rung embeds) and is *not* part of the text,
    so only R5 onwards sees it. A document with no text has nothing to chunk and is left out; it
    cannot be found by any rung.
    """
    documents: list[LoadedDocument] = []
    names: dict[str, str] = {}
    for beir_id, record in dataset.corpus.items():
        path = beir_doc_path(dataset.name, beir_id)
        names[doc_id_of(path)] = beir_id
        if not record.text.strip():
            continue
        documents.append(
            LoadedDocument(
                path=path,
                kind="text",
                title=record.title,
                segments=(Segment(text=record.text),),
            )
        )
    return documents, names


# --- building --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildStats:
    """What building an index cost. ``chunks_per_s`` counts only chunks the embedder computed."""

    documents: int
    chunks: int
    embedded: int
    cache_hits: int
    embed_seconds: float
    total_seconds: float
    reused: bool = False

    @property
    def chunks_per_s(self) -> float | None:
        return self.embedded / self.embed_seconds if self.embedded and self.embed_seconds else None


class BuiltIndex:
    """An opened variant index: the store, the dense index and the embedders, closed together."""

    def __init__(
        self,
        variant: IndexVariant,
        choice: EmbedderChoice,
        directory: Path,
        store: Store,
        dense: DenseIndex,
        cached: CachedEmbedder,
        cache: EmbeddingCache,
        embeddings: EmbeddingsClient,
        spec: EmbedderSpec,
        stats: BuildStats,
        closers: Sequence[Callable[[], None]],
    ) -> None:
        self.variant = variant
        self.choice = choice
        self.directory = directory
        self.store = store
        self.dense = dense
        self.cached = cached
        self.cache = cache
        self.embeddings = embeddings
        self.spec = spec
        self.stats = stats
        self._closers = list(closers)

    def close(self) -> None:
        """Close everything, last opened first. Safe to call twice."""
        while self._closers:
            self._closers.pop()()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _drafts(doc: LoadedDocument, variant: IndexVariant) -> list[ChunkDraft]:
    return legacy_chunks(doc) if variant.chunking == "legacy" else chunk_document(doc)


def _file_hash(doc: LoadedDocument) -> str:
    return hashlib.sha256(doc.text.encode("utf-8", "surrogatepass")).hexdigest()


def _batches[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_marker(directory: Path, spec: EmbedderSpec, documents: int) -> BuildStats | None:
    """The stats of a finished build, if this folder holds one for the same embedder and corpus."""
    try:
        data = json.loads((directory / _MARKER).read_text(encoding="utf-8"))
        if data["fingerprint"] != spec.fingerprint() or data["documents"] != documents:
            return None
        return BuildStats(**{**data["stats"], "reused": True})
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_index(
    documents: Sequence[LoadedDocument],
    variant: IndexVariant,
    choice: EmbedderChoice,
    directory: Path,
    cache_path: Path,
    *,
    embeddings: EmbeddingsClient,
    spec: EmbedderSpec,
    closer: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    on_progress: Callable[[int, int], None] | None = None,
) -> BuiltIndex:
    """Open ``directory`` as the index of ``documents`` under ``variant``, building it if needed.

    A folder holding a finished build for the same embedder fingerprint and the same number of
    documents is reused as it is (its recorded stats say so with ``reused=True``); anything else is
    wiped and built again. The embedding cache at ``cache_path`` is shared by every variant of one
    embedder, which is why a rebuild after a crash costs little.

    The build embeds a batch of documents at a time and marks each document ``ready`` only after
    its vectors are stored, the same order as the ingest pipeline (PLAN.md §6.4).
    """
    directory = directory.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stats = _read_marker(directory, spec, len(documents))
    if stats is None and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    closers: list[Callable[[], None]] = []
    if closer is not None:
        closers.append(closer)
    try:
        store = Store(directory)
        closers.append(store.close)
        dense = DenseIndex(directory, spec, reset=stats is None, space=variant.space)
        closers.append(dense.close)
        cache = EmbeddingCache(cache_path)
        closers.append(cache.close)
        cached = CachedEmbedder(embeddings, cache, spec)
        if stats is None:
            stats = _populate(
                documents, variant, store, dense, cached, spec, directory, clock, on_progress
            )
    except BaseException:
        while closers:
            closers.pop()()
        raise
    return BuiltIndex(
        variant, choice, directory, store, dense, cached, cache, embeddings, spec, stats, closers
    )


def _populate(
    documents: Sequence[LoadedDocument],
    variant: IndexVariant,
    store: Store,
    dense: DenseIndex,
    cached: CachedEmbedder,
    spec: EmbedderSpec,
    directory: Path,
    clock: Callable[[], float],
    on_progress: Callable[[int, int], None] | None,
) -> BuildStats:
    outbox = Outbox(store)
    version = f"ablation-{variant.key}"
    started = clock()
    embed_seconds = 0.0
    embedded = cache_hits = chunk_total = 0
    done = 0
    for group in _batches(documents, BATCH_DOCUMENTS):
        pending: list[tuple[str, list[tuple[str, str]]]] = []
        for doc in group:
            items = identify_chunks(doc.path, _drafts(doc, variant))
            doc_id = doc_id_of(doc.path)
            outbox.begin(
                NewDocument(
                    doc_id=doc_id,
                    path=str(doc.path),
                    title=doc.title,
                    kind=doc.kind,
                    file_hash=_file_hash(doc),
                    pipeline_version=version,
                )
            )
            store.sync_chunks(doc_id, items)
            outbox.move(doc_id, "embedding")
            texts = [
                (item.chunk_id, item.chunk.embedding_text(with_header=spec.with_header))
                for item in items
            ]
            pending.append((doc_id, texts))
        flat = [text for _, texts in pending for _, text in texts]
        before = clock()
        result = cached.embed_documents(flat)
        embed_seconds += clock() - before
        embedded += result.embedded
        cache_hits += result.cache_hits
        chunk_total += len(flat)
        vectors = iter(result.vectors)
        entries = [
            DenseEntry(chunk_id, doc_id, next(vectors))
            for doc_id, texts in pending
            for chunk_id, _ in texts
        ]
        dense.upsert(entries)
        for doc_id, _ in pending:
            outbox.move(doc_id, "ready")
        done += len(group)
        if on_progress is not None:
            on_progress(done, len(documents))
    stats = BuildStats(
        documents=len(documents),
        chunks=chunk_total,
        embedded=embedded,
        cache_hits=cache_hits,
        embed_seconds=embed_seconds,
        total_seconds=clock() - started,
    )
    marker = {
        "fingerprint": spec.fingerprint(),
        "documents": len(documents),
        "stats": {k: v for k, v in asdict(stats).items() if k != "reused"},
    }
    (directory / _MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return stats
