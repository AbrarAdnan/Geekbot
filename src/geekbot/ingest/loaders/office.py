"""Word, PowerPoint and HTML loader, through MarkItDown (step S1.1).

Why MarkItDown and not ``unstructured`` (v1's choice, PLAN.md §5 finding #4): ``unstructured`` was
never in v1's requirements, so these formats crashed. MarkItDown is a small, pure-Python
converter whose output is Markdown, which keeps headings, lists and tables as text a chunker can
split on (S1.2 cuts at the ``#`` headers).

Plugins stay off: a plugin is third-party code MarkItDown would load and run on our files.
"""

import functools
import re
from pathlib import Path
from typing import Any, Final

from geekbot.ingest.loaders.common import clean_text
from geekbot.ingest.loaders.model import DocKind, LoadedDocument, LoadError, Segment

KINDS: Final[dict[str, DocKind]] = {
    ".docx": "docx",
    ".pptx": "pptx",
    ".html": "html",
    ".htm": "html",
}

# MarkItDown writes ``<!-- Slide number: 3 -->`` before each slide's text. Splitting there gives a
# page number per slide, so a citation can say "slide 3". If a future release changes the marker,
# the regex finds nothing and the whole deck becomes one segment (no page numbers, no lost text).
_SLIDE_MARK: Final = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->")


@functools.cache
def _converter() -> Any:
    """One MarkItDown per process: building it registers every converter and is not free.

    Imported here, not at the top of the module, because MarkItDown pulls in an ONNX model
    loader (for file-type detection) and ``import geekbot`` should not pay for that.
    """
    from markitdown import MarkItDown  # type: ignore[import-untyped, unused-ignore]

    return MarkItDown(enable_plugins=False)


def load_office(path: Path) -> LoadedDocument:
    """Convert a ``.docx``, ``.pptx`` or ``.html`` file to Markdown text.

    A deck becomes one segment per slide (``page`` = the slide number). Word and HTML have no
    pages, so they are one segment. An HTML ``<title>`` becomes the document title when there is
    one. Images are not read: only the text a file carries is indexed (multimodal ingestion is on
    the Parked list).
    """
    kind = KINDS[path.suffix.lower()]
    result = _converter().convert_local(path)
    text = clean_text(str(result.text_content))
    if not text:
        raise LoadError("no text found in the document")
    title = str(getattr(result, "title", None) or "").strip() or path.name
    segments = _split_slides(text) if kind == "pptx" else (Segment(text),)
    if not segments:
        raise LoadError("no text found on any slide")
    return LoadedDocument(path=path, kind=kind, title=title, segments=segments)


def _split_slides(text: str) -> tuple[Segment, ...]:
    """Cut a converted deck at its slide markers. Text before the first marker is kept as is.

    Slides without text (a picture only) are dropped, so a deck of pictures gives no segments.
    Text with no marker at all is returned whole.
    """
    marks = list(_SLIDE_MARK.finditer(text))
    if not marks:
        return (Segment(text),)
    segments: list[Segment] = []
    preamble = clean_text(text[: marks[0].start()])
    if preamble:
        segments.append(Segment(preamble))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = clean_text(text[mark.end() : end])
        if body:
            segments.append(Segment(body, page=int(mark.group(1))))
    return tuple(segments)
