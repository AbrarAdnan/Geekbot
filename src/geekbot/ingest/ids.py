"""Identity and fingerprints: who a chunk is, and did it change (step S1.3, PLAN.md §6.4).

Two questions get two answers, on purpose.

- **Who is it?** ``doc_id`` (which source) and ``chunk_id`` (which piece of it). An ID has to
  survive an edit elsewhere in the document. If a chunk's ID contained its position, inserting one
  paragraph near the top would rename every chunk below it, and the whole tail would be embedded
  again (v1 re-embedded everything on every run, PLAN.md §5).
- **What does it say?** ``file_hash`` (the file's bytes) and ``content_hash`` (a chunk's text).
  Fingerprints: they change when the content changes and for no other reason. ``content_hash`` is
  also the embedding cache's key (S1.5). Moving a file changes every ``chunk_id``, because the path
  is part of it, but no ``content_hash``, so the cache answers and nothing is embedded again.

**Chunk identity** is ``doc_id`` + ``content_hash`` + ``occurrence``. Position (``ordinal``,
``char_start``, ``page``, ...) is metadata, never identity. ``occurrence`` exists for repeats: a
footer that says the same thing on forty pages has one ``content_hash``, and without a counter the
forty chunks would share one ID and the store would keep one row (the rev-2 collision, D-021). The
repeats are numbered 0, 1, 2, ... in document order.

That numbering has a useful consequence: whatever the order, a text that appears ``k`` times owns
exactly the IDs ``0 .. k-1``. Adding a copy adds exactly one ID and removing a copy removes exactly
one, so "re-ingest = set difference on ``chunk_id``" (PLAN.md §6.4) does the least possible work.
The price: when copies move, the numbers can move between them, so the pipeline (S1.6) must
refresh the position metadata of the IDs it keeps, not only add and delete rows.

Everything is SHA-256 from ``hashlib``. Python's built-in ``hash()`` is salted per process and
would give different IDs on every run.
"""

import hashlib
import os
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geekbot.ingest.chunking import ChunkDraft

__all__ = [
    "CHUNK_ID_LENGTH",
    "CONTENT_HASH_LENGTH",
    "DOC_ID_LENGTH",
    "IdentifiedChunk",
    "assign_occurrences",
    "chunk_id_of",
    "content_hash_of",
    "doc_id_of",
    "file_hash_of",
    "identify_chunks",
    "normalize_path",
    "normalize_text",
]

# Lengths in hex characters. 16 hex characters are 64 bits of the digest and 24 are 96 bits: far
# beyond the number of documents or chunks one machine holds (by the birthday bound, a million
# chunks collide with a probability around 1e-17), and short enough to read in a log.
DOC_ID_LENGTH: Final = 16
CHUNK_ID_LENGTH: Final = 24
CONTENT_HASH_LENGTH: Final = 64  # a full SHA-256: it is a cache key, not something to read

type StrPath = str | os.PathLike[str]

_LINE_END: Final = re.compile(r"\r\n?")  # CRLF, or a lone CR (old Mac files)


def _sha256_hex(text: str) -> str:
    # "surrogatepass": a file name or a PDF's text can hold a lone surrogate. Encoding it strictly
    # would crash the whole ingest on one odd character; this gives it a stable digest instead.
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _require_hex(name: str, value: str, length: int) -> None:
    """Fail loudly when a caller passes the wrong thing (a chunk's text where its hash belongs)."""
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{name} must be {length} lowercase hex characters, got {value!r}")


# --- documents -------------------------------------------------------------------------------


def normalize_path(path: StrPath) -> str:
    """One spelling for one file: absolute, ``..`` collapsed, case-folded on Windows, NFC.

    ``docs/a.txt``, ``./docs/../docs/a.txt`` and ``C:\\Work\\docs\\a.txt`` must be the same source,
    or the same file ingested twice would become two documents. On Windows, ``C:\\Docs\\A.pdf``
    and ``c:/docs/a.pdf`` are the same file too (``normcase`` lowercases and unifies the
    slashes); on Linux and macOS case counts and is kept. NFC makes a name typed with a composed
    letter equal to the same name with a combining accent (macOS file systems store the latter).

    The path is not looked up on disk: no ``resolve()``, no symlinks followed. That keeps the
    function pure and the same for a file that was deleted since. The cost is that two spellings
    that reach one file through a symlink are two documents. A relative path means relative to
    the current directory *now*, so the pipeline should pass absolute paths.
    """
    absolute = os.path.normpath(Path(path).absolute())
    return unicodedata.normalize("NFC", os.path.normcase(absolute))


def doc_id_of(path: StrPath) -> str:
    """``sha256(normalized_abs_path)[:16]``: which source (PLAN.md §6.4)."""
    return _sha256_hex(normalize_path(path))[:DOC_ID_LENGTH]


def file_hash_of(path: StrPath) -> str:
    """SHA-256 of the file's bytes, to skip a file that has not changed.

    Read in blocks by ``hashlib.file_digest``, so a 500 MB PDF costs no memory. The name and the
    modification time are not part of it: touching a file or renaming it does not make it
    "changed", rewriting one byte does. A missing file raises ``FileNotFoundError``.

    The hash covers the *file*, not how it is processed. If the loader or the chunker changes,
    an unchanged file still needs re-processing; the pipeline (S1.6) keys that on its own version.
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# --- chunks ----------------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """The form of a chunk's text that its ``content_hash`` is taken from.

    Three things that do not change what a chunk says are removed: the Unicode composition (NFC,
    so a letter with an accent as one character equals the letter plus a combining accent), the
    line-ending style (CRLF and CR become LF, so a file checked out on Windows matches the same
    file on Linux), and whitespace at the end of the chunk. Everything else counts, indentation
    and the spaces inside a line included (a space before a line break too), because in code and
    tables they carry meaning.

    The rule of thumb is the cheaper mistake: calling an unchanged chunk "changed" costs one
    re-embedding, while calling a changed chunk "unchanged" serves a wrong vector from the cache.
    So this errs on keeping differences.
    """
    unified = _LINE_END.sub("\n", unicodedata.normalize("NFC", text))
    return unified.rstrip()


def content_hash_of(text: str) -> str:
    """``sha256(normalized_chunk_text)``: did this chunk's content change (PLAN.md §6.4)."""
    return _sha256_hex(normalize_text(text))


def assign_occurrences(content_hashes: Sequence[str]) -> list[int]:
    """Number the repeats: the nth time a hash appears in the list gets ``n``, counting from 0.

    ``["a", "b", "a", "a", "b"]`` becomes ``[0, 0, 1, 2, 1]``. The input is one document's chunk
    hashes in document order; the count restarts for every document because the caller passes
    one document at a time. Two chunks of different documents may have the same hash and the same
    occurrence; their ``doc_id`` tells them apart.

    The number depends only on how many equal hashes came *before*, so a chunk with new text
    anywhere in the document leaves every other chunk's number alone.
    """
    seen: dict[str, int] = {}
    occurrences: list[int] = []
    for content_hash in content_hashes:
        occurrence = seen.get(content_hash, 0)
        occurrences.append(occurrence)
        seen[content_hash] = occurrence + 1
    return occurrences


def chunk_id_of(doc_id: str, content_hash: str, occurrence: int) -> str:
    """``sha256(doc_id : content_hash : occurrence)[:24]``: stable identity (PLAN.md §6.4).

    The parts are joined with ``:``, which cannot occur in a hex string, so two different
    triples never produce the same input. Arguments of the wrong shape raise ``ValueError``
    instead of quietly producing an ID that matches nothing.
    """
    _require_hex("doc_id", doc_id, DOC_ID_LENGTH)
    _require_hex("content_hash", content_hash, CONTENT_HASH_LENGTH)
    if occurrence < 0:
        raise ValueError(f"occurrence counts from 0, got {occurrence}")
    return _sha256_hex(f"{doc_id}:{content_hash}:{occurrence}")[:CHUNK_ID_LENGTH]


@dataclass(frozen=True)
class IdentifiedChunk:
    """A chunk with its identity: what the store (S1.4) writes as one row.

    ``chunk`` keeps all the position metadata (``ordinal``, offsets, ``page``, ``section_path``,
    ``header``); none of it is part of ``chunk_id``.
    """

    chunk: ChunkDraft
    doc_id: str
    content_hash: str
    occurrence: int
    chunk_id: str


def identify_chunks(path: StrPath, chunks: Sequence[ChunkDraft]) -> list[IdentifiedChunk]:
    """Give every chunk of one document its hashes, its occurrence number and its ID.

    ``chunks`` must be all of one document's chunks in document order (what ``chunk_document``
    returns): the occurrence numbers are counted over the whole list.
    """
    document = doc_id_of(path)
    hashes = [content_hash_of(chunk.text) for chunk in chunks]
    occurrences = assign_occurrences(hashes)
    return [
        IdentifiedChunk(
            chunk=chunk,
            doc_id=document,
            content_hash=content_hash,
            occurrence=occurrence,
            chunk_id=chunk_id_of(document, content_hash, occurrence),
        )
        for chunk, content_hash, occurrence in zip(chunks, hashes, occurrences, strict=True)
    ]
