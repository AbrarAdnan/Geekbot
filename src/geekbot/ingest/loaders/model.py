"""What a loader returns, and how it fails (step S1.1, PLAN.md §6.6).

A loader's only job is *file in, text out*, with just enough position information (the page) for
a citation later. It does not split the text into chunks (S1.2), hash it (S1.3) or store it
(S1.4); keeping it that small is what lets one loader be replaced without touching the rest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

DocKind = Literal["pdf", "docx", "pptx", "html", "markdown", "text", "code", "notebook", "table"]

# What ``LoadedDocument.text`` puts between two segments. The chunker (S1.2) counts with the same
# constant, so a chunk's offsets point into that text exactly.
SEGMENT_SEPARATOR: Final = "\n\n"


class LoadError(Exception):
    """One file could not be turned into text.

    The message says why, in words for the person who ingested the file ("password-protected",
    "no text found"), because it ends up in the per-file report and not in a traceback.
    """


@dataclass(frozen=True)
class Segment:
    """A run of text from one place in a document.

    ``page`` is **1-based** (a PDF page, a slide number) and ``None`` when the format has no
    pages (Markdown, Word, code). v1 printed PDF pages 0-based (PLAN.md §5 finding #11), so a
    citation said "page 0" for the first page; the rule here is that ``page`` is the number a
    person sees in their reader.
    """

    text: str
    page: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    """One file's text, ready for the chunker.

    ``segments`` keep the page boundaries (a PDF is one segment per page) so a chunk can later
    say which page it came from. ``language`` is the programming language of code and notebooks;
    S1.2's code splitter picks its separators from it. ``title`` is the file name unless the
    format carries a better one (an HTML ``<title>``); S1.2 can use it as a contextual header.
    """

    path: Path
    kind: DocKind
    title: str
    segments: tuple[Segment, ...]
    language: str | None = None

    @property
    def text(self) -> str:
        """All segments joined by a blank line (for callers that don't care about pages)."""
        return SEGMENT_SEPARATOR.join(segment.text for segment in self.segments)
