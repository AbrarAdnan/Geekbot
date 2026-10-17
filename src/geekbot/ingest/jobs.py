"""Parsing many files, in parallel when it pays (step S1.6, PLAN.md §6.3).

Parsing is the CPU-bound part of ingest (PDF text, Word and PowerPoint through MarkItDown, the
chunker), so with several files it runs in a **process pool**. Three rules keep that safe:

1. **Workers never touch storage.** A worker gets a path and returns plain data
   (``ParsedFile``): the loaded document and its chunk drafts, or an error in words. Only the
   process that owns the store and the vectors writes to them (PLAN.md §6.2), so there is no lock
   to share and no second writer.
2. **Spawn, not fork.** On Windows a new process starts from scratch and re-imports this module,
   so the worker function is a plain top-level function with picklable arguments and results. The
   pool is created with the ``spawn`` method everywhere so Linux and Windows behave the same, and
   so a worker never inherits an open SQLite connection or a Chroma client by accident.
3. **Bounded.** Only a few files are in flight at once, so a folder of thousands of PDFs is not
   all parsed into memory before the first one is stored.

With one worker (the default) there is no pool at all: files are parsed in the calling process, in
order, which is also what the tests and a small ingest want.
"""

import multiprocessing
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geekbot.ingest.chunking import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    ChunkDraft,
    chunk_document,
)
from geekbot.ingest.loaders import LoadedDocument, LoadError, UnsupportedFormatError, load_file

__all__ = ["ParsedFile", "parse_file", "parse_files"]

_MAX_REASON_CHARS: Final = 300
_IN_FLIGHT_PER_WORKER: Final = 2


@dataclass(frozen=True)
class ParsedFile:
    """What parsing one file produced: a document and its chunks, or the reason it did not.

    Exactly one of two shapes: ``error is None`` and ``document`` is set (``chunks`` may be empty
    for a file with no text), or ``error`` says why and ``document`` is ``None``. ``skipped`` is
    true when no loader exists for the type, which the report counts apart from real failures.
    """

    path: Path
    document: LoadedDocument | None
    chunks: tuple[ChunkDraft, ...]
    error: str | None = None
    skipped: bool = False


def parse_file(
    path: Path,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> ParsedFile:
    """Load and chunk one file. Never raises for a bad file: the problem comes back in the result.

    A top-level function with simple arguments, because a process pool has to pickle it.
    """
    try:
        document = load_file(path)
    except UnsupportedFormatError as error:
        return ParsedFile(path, None, (), error=str(error), skipped=True)
    except LoadError as error:
        return ParsedFile(path, None, (), error=str(error))
    try:
        chunks = chunk_document(document, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    except Exception as error:  # noqa: BLE001 - a bad file is a result, never a crash
        detail = f"{type(error).__name__}: {error}"[:_MAX_REASON_CHARS]
        return ParsedFile(path, None, (), error=f"could not be split into chunks ({detail})")
    return ParsedFile(path, document, tuple(chunks))


def parse_files(
    paths: Iterable[Path],
    *,
    workers: int = 1,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Iterator[ParsedFile]:
    """Parse the files and yield one ``ParsedFile`` per path, in the order the paths were given.

    ``workers`` is the number of processes; 1 means no pool. The generator keeps at most
    ``2 * workers`` files in flight, and closes the pool when it is exhausted or abandoned.
    ``ValueError`` if ``workers`` is below 1.
    """
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if workers == 1:
        for path in paths:
            yield parse_file(path, chunk_tokens, overlap_tokens)
        return
    context = multiprocessing.get_context("spawn")
    window = workers * _IN_FLIGHT_PER_WORKER
    pending: deque[Future[ParsedFile]] = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        try:
            for path in paths:
                pending.append(pool.submit(parse_file, path, chunk_tokens, overlap_tokens))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
