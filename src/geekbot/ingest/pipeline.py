"""The ingest pipeline: files in, searchable documents out (step S1.6, PLAN.md §6.4, §11).

One file goes through the outbox (``ingest/outbox.py``) in this order, and the order is the whole
point:

1. **Skip if nothing changed.** A file whose content hash and pipeline version match a ``ready``
   document is not read, parsed or embedded again. This is what makes re-ingesting an unchanged
   folder free, and it fixes v1's worst habit of re-embedding everything on every run (PLAN.md §5,
   finding #1).
2. **Parse** (load, chunk), in a process pool if asked (``ingest/jobs.py``).
3. **Record** the document as ``pending`` and write its chunk rows in one SQLite transaction
   (a set difference on ``chunk_id``, so unchanged chunks are not touched).
4. **Embed** (``embedding``): vectors for every chunk through the embedding cache, upserted to
   Chroma, stale vectors deleted (``index/vectors.py``).
5. **Publish** (``ready``). Only now can a search see the document.

If anything raises in 3-5 the document is moved to ``failed`` with the reason and the run goes on
with the next file; if the *process* dies, the document is left in whatever state it had reached
and ``doctor --repair`` finishes or fails it (``index/consistency.py``). One bad file never stops a
folder (v1 finding #4).

A file that was ingested before and now fails to load is marked ``failed`` too, so a search does not
keep returning the old text as if it were current; a file that never loaded is only reported.
"""

import contextlib
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from geekbot.domain import NewDocument
from geekbot.index.dense import DenseIndex
from geekbot.index.embed_cache import CachedEmbedder
from geekbot.index.sqlite import Store
from geekbot.index.vectors import write_document_vectors
from geekbot.ingest.chunking import DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS
from geekbot.ingest.ids import doc_id_of, file_hash_of, identify_chunks
from geekbot.ingest.jobs import ParsedFile, parse_files
from geekbot.ingest.loaders import SUPPORTED_EXTENSIONS
from geekbot.ingest.outbox import InvalidTransitionError, Outbox

__all__ = [
    "IGNORED_DIRECTORIES",
    "PIPELINE_REVISION",
    "FileOutcome",
    "FileStatus",
    "IngestReport",
    "Ingestor",
    "collect_files",
    "format_outcome",
    "pipeline_version",
]

# Bump when the loaders or the chunker change what they produce for the same file, so that files
# already stored are processed again. The chunk budgets are added to it automatically.
PIPELINE_REVISION: Final = 1

# Folders a walk never enters: version control, caches, environments (v1 finding #13: one click
# once tried to index a whole virtual environment). Any folder whose name starts with "." is skipped too.
IGNORED_DIRECTORIES: Final = frozenset(
    {"__pycache__", "node_modules", "venv", "env", "site-packages", "dist", "build"}
)

FileStatus = Literal["ingested", "unchanged", "failed", "skipped"]


def pipeline_version(
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS, overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
) -> str:
    """What produced the stored chunks: the code revision and the chunk budgets.

    An unchanged file is skipped only if this matches too, so changing the chunk size (or the
    revision above) re-processes everything on the next run without a flag.
    """
    return f"r{PIPELINE_REVISION}-chunk{chunk_tokens}-overlap{overlap_tokens}"


# --- finding the files -----------------------------------------------------------------------


def collect_files(paths: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """Expand files and folders into ``(files, unsupported)``, each in a stable order, no repeats.

    A path that is a file is kept as given (an unsupported type is then reported by the loader as
    *skipped*). A folder is walked recursively; inside it only files with a supported extension are
    returned and the others are counted as ``unsupported`` (so "indexed 40 files" can say that 12
    others were left out), hidden folders and the ones in ``IGNORED_DIRECTORIES`` are not entered.
    ``FileNotFoundError`` for a path that does not exist: a typo must not mean an empty ingest.
    """
    files: list[Path] = []
    unsupported: list[Path] = []
    seen: set[Path] = set()

    def add(target: list[Path], path: Path) -> None:
        absolute = path.absolute()
        if absolute not in seen:
            seen.add(absolute)
            target.append(absolute)

    for given in paths:
        if given.is_file():
            add(files, given)
        elif given.is_dir():
            for root, directories, names in os.walk(given):
                directories[:] = sorted(
                    d for d in directories if not d.startswith(".") and d not in IGNORED_DIRECTORIES
                )
                for name in sorted(names):
                    candidate = Path(root) / name
                    if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                        add(files, candidate)
                    else:
                        add(unsupported, candidate)
        else:
            raise FileNotFoundError(f"{given} does not exist")
    return files, unsupported


# --- results ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FileOutcome:
    """What happened to one file."""

    path: Path
    status: FileStatus
    chunks: int = 0  # chunks the document has now
    embedded: int = 0  # embeddings computed for it (cache misses)
    cache_hits: int = 0  # embeddings taken from the cache
    removed_vectors: int = 0  # stale vectors deleted
    error: str | None = None  # why it failed or was skipped


@dataclass(frozen=True)
class IngestReport:
    """The outcome of one ``ingest`` call: every file, in order."""

    outcomes: tuple[FileOutcome, ...]

    def count(self, status: FileStatus) -> int:
        """How many files ended in ``status``."""
        return sum(1 for outcome in self.outcomes if outcome.status == status)

    @property
    def embedded(self) -> int:
        """Embeddings computed in this run. A second run over unchanged files must report 0."""
        return sum(outcome.embedded for outcome in self.outcomes)

    @property
    def cache_hits(self) -> int:
        """Embeddings that came from the cache instead of the model."""
        return sum(outcome.cache_hits for outcome in self.outcomes)

    @property
    def has_failures(self) -> bool:
        """True if any file failed (a skipped file is not a failure)."""
        return self.count("failed") > 0

    def summary(self) -> str:
        """``2 ingested, 1 unchanged, 0 failed, 0 skipped; 14 embeddings computed, 3 from the cache``."""
        return (
            f"{self.count('ingested')} ingested, {self.count('unchanged')} unchanged, "
            f"{self.count('failed')} failed, {self.count('skipped')} skipped; "
            f"{self.embedded} embeddings computed, {self.cache_hits} from the cache"
        )


def format_outcome(outcome: FileOutcome) -> str:
    """One line for the console: the status, the path and the detail."""
    if outcome.status == "ingested":
        detail = (
            f"{outcome.chunks} chunks; {outcome.embedded} embedded, {outcome.cache_hits} cached"
        )
        return f"ingested   {outcome.path} ({detail})"
    if outcome.status == "unchanged":
        return f"unchanged  {outcome.path} ({outcome.chunks} chunks)"
    label = "FAILED    " if outcome.status == "failed" else "skipped   "
    return f"{label} {outcome.path}: {outcome.error}"


# --- the pipeline ----------------------------------------------------------------------------


class Ingestor:
    """Ingests files into a store and a dense index. Used by the CLI now, by the server (S4.1) later.

    Not thread-safe: one ``Ingestor`` runs one ingest at a time (the server's job queue, S4.1,
    serializes them). The store, the dense index and the cached embedder are the caller's to
    open and close; this class only uses them.
    """

    def __init__(
        self,
        store: Store,
        dense: DenseIndex,
        embedder: CachedEmbedder,
        *,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        workers: int = 1,
    ) -> None:
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self._store = store
        self._dense = dense
        self._embedder = embedder
        self._outbox = Outbox(store)
        self._chunk_tokens = chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._workers = workers
        self._version = pipeline_version(chunk_tokens, overlap_tokens)

    def ingest(
        self,
        files: Sequence[Path],
        *,
        on_file: Callable[[FileOutcome], None] | None = None,
    ) -> IngestReport:
        """Ingest these files (already expanded by ``collect_files``); one outcome per file, in order.

        ``on_file`` is called as each file finishes, for progress output. Raises only for things
        that make the whole run meaningless (a closed store); a bad file is an outcome.
        """
        outcomes: dict[Path, FileOutcome] = {}

        def finish(outcome: FileOutcome) -> None:
            outcomes[outcome.path] = outcome
            if on_file is not None:
                on_file(outcome)

        # One query for every document's chunk count, so an unchanged file costs a hash and a row
        # lookup, and no chunk text is read just to be counted.
        counts = {doc_id: len(ids) for doc_id, ids in self._store.chunk_ids_by_document().items()}
        todo: list[tuple[Path, str]] = []
        for path in files:
            checked = self._check_unchanged(path, counts)
            if isinstance(checked, FileOutcome):
                finish(checked)
            else:
                todo.append((path, checked))

        hashes = dict(todo)
        parsed_files = parse_files(
            [path for path, _ in todo],
            workers=self._workers,
            chunk_tokens=self._chunk_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        for parsed in parsed_files:
            finish(self._store_parsed(parsed, hashes[parsed.path]))
        return IngestReport(tuple(outcomes[path] for path in files))

    def remove(self, path: Path) -> bool:
        """Forget a file: its rows, then its vectors. ``False`` if it was never ingested.

        The rows go first, so the document is gone from search at once; a crash before the second
        step leaves vectors nobody owns, which ``doctor --repair`` sweeps (and which search cannot
        return anyway, because it goes through SQLite).
        """
        doc_id = doc_id_of(path)
        existed = self._store.delete_document(doc_id)
        self._dense.delete_document(doc_id)
        return existed

    # --- one file ---

    def _check_unchanged(self, path: Path, chunk_counts: dict[str, int]) -> FileOutcome | str:
        """A ``FileOutcome`` if the file needs no work (or cannot be read), else its content hash."""
        try:
            file_hash = file_hash_of(path)
        except OSError as error:
            return self._failure(path, None, f"could not be read ({error})")
        existing = self._store.get_document(doc_id_of(path))
        if (
            existing is not None
            and existing.state == "ready"
            and existing.file_hash == file_hash
            and existing.pipeline_version == self._version
        ):
            return FileOutcome(path, "unchanged", chunks=chunk_counts.get(existing.doc_id, 0))
        return file_hash

    def _store_parsed(self, parsed: ParsedFile, file_hash: str) -> FileOutcome:
        path = parsed.path
        if parsed.skipped:
            return FileOutcome(path, "skipped", error=parsed.error)
        document = parsed.document
        if document is None:
            return self._failure(path, file_hash, parsed.error or "could not be loaded")
        doc_id = doc_id_of(path)
        try:
            items = identify_chunks(path, parsed.chunks)
            texts = [
                (
                    item.chunk_id,
                    item.chunk.embedding_text(with_header=self._embedder.spec.with_header),
                )
                for item in items
            ]
        except Exception as error:  # noqa: BLE001 - e.g. a lone surrogate cannot be hashed
            return self._failure(path, file_hash, f"{type(error).__name__}: {error}")
        self._outbox.begin(
            NewDocument(
                doc_id=doc_id,
                path=str(path.absolute()),
                title=document.title,
                kind=document.kind,
                file_hash=file_hash,
                pipeline_version=self._version,
            )
        )
        try:
            self._store.sync_chunks(doc_id, items)
            self._outbox.move(doc_id, "embedding")
            written = write_document_vectors(self._dense, self._embedder, doc_id, texts)
            self._outbox.move(doc_id, "ready")
        except Exception as error:  # noqa: BLE001 - any failure fails this file, not the run
            reason = f"{type(error).__name__}: {error}"
            with contextlib.suppress(InvalidTransitionError):
                self._outbox.fail(doc_id, reason)
            return FileOutcome(path, "failed", error=reason)
        return FileOutcome(
            path,
            "ingested",
            chunks=len(items),
            embedded=written.embedded,
            cache_hits=written.cache_hits,
            removed_vectors=written.removed,
        )

    def _failure(self, path: Path, file_hash: str | None, reason: str) -> FileOutcome:
        """Report a file that could not be processed; hide its old content if it had some.

        A file that was ingested before and now fails would otherwise keep serving the text of its
        previous version. So it is recorded as ``failed`` (search only returns ``ready``
        documents). A file that was never stored has no old content to hide and is only reported.
        """
        existing = self._store.get_document(doc_id_of(path))
        if existing is not None:
            self._outbox.begin(
                NewDocument(
                    doc_id=existing.doc_id,
                    path=str(path.absolute()),
                    title=existing.title,
                    kind=existing.kind,
                    file_hash=file_hash or existing.file_hash,
                    pipeline_version=self._version,
                )
            )
            self._outbox.fail(existing.doc_id, reason)
        return FileOutcome(path, "failed", error=reason)
