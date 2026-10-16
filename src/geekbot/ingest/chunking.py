"""Chunking: loaded text in, citable pieces out (step S1.2, PLAN.md §6.4, §6.6).

A chunk is the unit that is embedded, searched and cited, so its size is a trade-off. Too big and
one vector averages several topics while the prompt fills with text the question did not need; too
small and a chunk loses the sentence that explains it. PLAN.md starts at 512 tokens with 64 tokens
of overlap; the ablation (S2.6, rung R4) is where that choice gets tested.

One document is chunked segment by segment (a segment is a PDF page, a slide, or the whole file),
in three passes:

1. **Sections.** Markdown-like text is cut at its ``#`` headings, so a chunk never mixes two
   sections and always knows its ``section_path``. Plain text, code and PDF pages are one section.
2. **Pieces.** A section is cut recursively at the coarsest boundary that makes the parts fit:
   paragraph, line, sentence, word and, as a last resort, a fixed run of characters. Code is cut at
   its declarations (``def``, ``class``, ``func``, ...) before any of those.
3. **Windows.** Pieces are merged greedily into chunks of at most the token budget. The last
   pieces of one chunk (up to the overlap budget) open the next one, so a sentence cut at a
   boundary is whole in at least one of the two chunks.

**Why hand-written (D-053).** LangChain's splitters return text without positions; recovering them
with ``str.find`` is ambiguous when a passage repeats. Here every chunk is an exact slice of
``doc.text``, so ``char_start`` and ``char_end`` are facts, and "no text is lost" is a property that
can be tested exactly (``tests/unit/test_chunking.py``).

A chunk never crosses a section or a page. That keeps its citation to one page and one heading, at
the price of small chunks where a section is short.
"""

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

from geekbot.ingest.loaders.model import SEGMENT_SEPARATOR, DocKind, LoadedDocument

__all__ = [
    "DEFAULT_CHUNK_TOKENS",
    "DEFAULT_OVERLAP_TOKENS",
    "ChunkDraft",
    "chunk_document",
    "contextual_header",
    "estimate_tokens",
]

DEFAULT_CHUNK_TOKENS: Final = 512
DEFAULT_OVERLAP_TOKENS: Final = 64
HEADER_SEPARATOR: Final = " > "

# Kinds whose text is Markdown (MarkItDown writes Word, PowerPoint and HTML as Markdown; a table
# card and a notebook are Markdown too). Plain text and code can contain "# ..." lines that are
# comments, so they are never read for headings.
_HEADED_KINDS: Final[frozenset[DocKind]] = frozenset(
    {"markdown", "docx", "pptx", "html", "notebook", "table"}
)


# --- tokens ----------------------------------------------------------------------------------

# Hiragana, katakana, CJK ideographs and hangul, as \uXXXX ranges for the regex below (written
# with doubled backslashes, so this file stays plain ASCII). They have no spaces between them, and
# a tokenizer cuts them into single characters or very short runs: each one counts as one token.
_CJK: Final = "\\u3040-\\u30ff\\u3400-\\u4dbf\\u4e00-\\u9fff\\uac00-\\ud7af\\uf900-\\ufaff"
_TOKEN: Final = re.compile(rf"[{_CJK}]|(?:(?![{_CJK}])\w)+|[^\w\s]")
_LETTERS_PER_TOKEN: Final = 5


def estimate_tokens(text: str) -> int:
    """A cheap, dependency-free token count that errs on the high side.

    A run of letters or digits counts one token per five characters (rounded up), a CJK character
    one, a punctuation mark one, whitespace none. English prose averages about four characters
    per word, and a WordPiece or BPE tokenizer splits rarer words into two or three pieces, so
    this tends to over-count a little: chunks end up a bit under their budget instead of over it.

    Two properties the chunker relies on, both true by construction:

    - no more than one token per character, so a run of ``n`` characters always fits a budget of
      ``n`` tokens (the last-resort split needs that);
    - cutting a text in two never counts fewer tokens in total (``count(a + b) <= count(a) +
      count(b)``), so the sum of the pieces' counts is a safe upper bound for a merged chunk.

    This is an estimate, not the embedder's tokenizer. The embedder arrives in S1.5, where a live
    test can compare this against the real count (Ollama's ``/api/embed`` returns it).
    """
    total = 0
    for match in _TOKEN.finditer(text):
        width = match.end() - match.start()
        total += (width + _LETTERS_PER_TOKEN - 1) // _LETTERS_PER_TOKEN
    return total


# --- the result ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkDraft:
    """One chunk of a document, before it has an identity (S1.3 adds the ID and hashes).

    ``text`` is exactly ``doc.text[char_start:char_end]``: the offsets are positions in the
    document's joined text (``LoadedDocument.text``), and surrounding blank lines are not part of
    the chunk. ``page`` is the page or slide the chunk came from (1-based, ``None`` if the format
    has none). ``section_path`` is the trail of headings above it, outermost first.

    ``header`` is the contextual header: the document title followed by the section path. It is
    kept apart from ``text`` on purpose. The ablation's header rung (R5) switches it on and off
    for the embedder, and neither the chunk nor its future ID may change when it does.
    """

    text: str
    char_start: int
    char_end: int
    page: int | None
    section_path: tuple[str, ...]
    header: str
    ordinal: int

    def embedding_text(self, *, with_header: bool) -> str:
        """The text to embed: the chunk, with its contextual header in front when asked for."""
        if with_header and self.header:
            return f"{self.header}\n\n{self.text}"
        return self.text


def contextual_header(title: str, section_path: Sequence[str]) -> str:
    """``title > heading > subheading``: where in which document a chunk comes from."""
    return HEADER_SEPARATOR.join([title, *section_path])


# --- the public entry point ------------------------------------------------------------------


def chunk_document(
    doc: LoadedDocument,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    count_tokens: Callable[[str], int] = estimate_tokens,
) -> list[ChunkDraft]:
    """Split a loaded document into chunks of at most ``chunk_tokens``, in document order.

    Guarantees, all checked by property tests with arbitrary text:

    - **Exact:** ``doc.text[c.char_start:c.char_end] == c.text`` for every chunk ``c``.
    - **No loss:** every character that is not whitespace is inside at least one chunk. Only
      blank lines around a chunk and trailing whitespace are trimmed.
    - **Bounded:** ``count_tokens(c.text) <= chunk_tokens``, and two neighbouring chunks share at
      most ``overlap_tokens`` tokens.
    - **One place:** a chunk lies inside one segment, so it has one ``page``, and inside one
      section, so it has one ``section_path``.
    - **Ordered:** chunk starts never decrease, chunk ends strictly increase, ``ordinal`` counts
      0, 1, 2, ... A chunk that would lie wholly inside its predecessor is not emitted.

    ``count_tokens`` must count at most one token per character and never count fewer tokens for
    a longer text, as ``estimate_tokens`` does. ``ValueError`` if the budgets make no sense.
    """
    budget = _Budget(chunk_tokens, overlap_tokens, count_tokens)
    levels = _code_levels(doc.language) if doc.kind == "code" else _PROSE_LEVELS
    chunks: list[ChunkDraft] = []
    base = 0  # where this segment starts inside doc.text
    for segment in doc.segments:
        if doc.kind in _HEADED_KINDS:
            sections = _sections(segment.text)
        else:
            sections = [_Section(0, len(segment.text), ())]
        for start, end, path in _spans(segment.text, sections, levels, budget):
            if chunks and base + end <= chunks[-1].char_end:
                continue  # wholly inside the previous chunk: nothing new to index
            chunks.append(
                ChunkDraft(
                    text=segment.text[start:end],
                    char_start=base + start,
                    char_end=base + end,
                    page=segment.page,
                    section_path=path,
                    header=contextual_header(doc.title, path),
                    ordinal=len(chunks),
                )
            )
        base += len(segment.text) + len(SEGMENT_SEPARATOR)
    return chunks


@dataclass(frozen=True)
class _Budget:
    """The token limits and the counter, passed around together."""

    size: int
    overlap: int
    count: Callable[[str], int]

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"chunk_tokens must be at least 1, got {self.size}")
        if not 0 <= self.overlap < self.size:
            raise ValueError(
                f"overlap_tokens must be from 0 to chunk_tokens - 1, got {self.overlap}"
            )


# --- pass 1: sections ------------------------------------------------------------------------

# An ATX heading: up to three spaces, 1-6 "#", a space, the title, and optionally a closing run of
# "#" that must be separated from the title by a space ("# C#" is the title "C#").
_HEADING: Final = re.compile(r"[ ]{0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t\r]*$")
_FENCE: Final = re.compile(r"[ ]{0,3}(`{3,}|~{3,})")
# A line longer than this is a paragraph that happens to start with "#", not a heading. The limit
# also keeps the lazy heading pattern cheap on a hostile line: documents are untrusted input.
_MAX_HEADING_CHARS: Final = 300


@dataclass(frozen=True)
class _Section:
    start: int
    end: int
    path: tuple[str, ...]
    bare: bool = False  # a heading with nothing under it


def _headings(text: str) -> list[tuple[int, int, str]]:
    """``(offset of the line, level, title)`` of every heading that is not inside a code fence.

    A ``#`` line in a fenced block is a comment or a shell prompt, not a heading. Splitting there
    would cut the code block in two and give its second half a section path that does not exist.
    A fence left open runs to the end of the text, which is how Markdown renderers read it.
    """
    found: list[tuple[int, int, str]] = []
    fence = ""  # the opening marker while inside a fence
    offset = 0
    for line in text.split("\n"):
        stripped = line.strip()
        if fence:
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                fence = ""
        elif opener := _FENCE.match(line):
            fence = opener.group(1)
        elif len(line) <= _MAX_HEADING_CHARS and (heading := _HEADING.match(line)):
            title = heading.group(2).strip()
            if title:
                found.append((offset, len(heading.group(1)), title))
        offset += len(line) + 1
    return found


def _sections(text: str) -> list[_Section]:
    """Cut a Markdown text at its headings; each section keeps its heading line."""
    headings = _headings(text)
    if not headings:
        return [_Section(0, len(text), ())]
    raw: list[_Section] = []
    if headings[0][0] > 0:
        raw.append(_Section(0, headings[0][0], ()))  # the text before the first heading
    trail: list[tuple[int, str]] = []  # (level, title) of the headings above the current one
    offsets = [offset for offset, _, _ in headings]
    ends = [*offsets[1:], len(text)]  # each section runs to the next heading
    for (start, level, title), end in zip(headings, ends, strict=True):
        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, title))
        bare = not text[start:end].partition("\n")[2].strip()
        raw.append(_Section(start, end, tuple(name for _, name in trail), bare))
    return _absorb_bare_headings(raw)


def _absorb_bare_headings(sections: list[_Section]) -> list[_Section]:
    """Glue a heading that has nothing under it onto the section after it.

    ``# Title`` straight followed by ``## Part`` is common (a document title, then its first
    part). As a section of its own, the title would be a chunk of two tokens that ranks badly and
    takes a slot in the answer's context. Its line stays in the text, in front of the next
    section, whose path is the deeper one.
    """
    merged: list[_Section] = []
    carried_start: int | None = None
    for index, section in enumerate(sections):
        start = section.start if carried_start is None else carried_start
        if section.bare and index < len(sections) - 1:
            carried_start = start
            continue
        carried_start = None
        merged.append(_Section(start, section.end, section.path))
    return merged


# --- pass 2: pieces --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Level:
    """One kind of boundary to cut at, tried from the coarsest to the finest.

    ``cut_at_end`` is True for separators (cut after the blank lines, after the sentence's
    whitespace, so the separator stays with the text before it) and False for declarations (cut
    where ``def`` begins, so a decorator or the keyword stays with its body).
    """

    pattern: re.Pattern[str]
    cut_at_end: bool = True

    def cuts(self, span: str) -> list[int]:
        """Positions inside ``span`` (never at either end) where it may be cut."""
        found = {(m.end() if self.cut_at_end else m.start()) for m in self.pattern.finditer(span)}
        return sorted(cut for cut in found if 0 < cut < len(span))


def _separator(pattern: str) -> _Level:
    return _Level(re.compile(pattern))


def _declaration(pattern: str) -> _Level:
    return _Level(re.compile(pattern, re.MULTILINE), cut_at_end=False)


_PARAGRAPH: Final = _separator(r"\n[ \t\r]*\n(?:[ \t\r]*\n)*")
_LINE: Final = _separator(r"\n")
# After ". ", "! " or "? ", and after the CJK full stop, exclamation and question marks (which
# are not followed by a space). Written as escapes, like _CJK, to keep this file ASCII.
_SENTENCE: Final = _separator("(?<=[.!?])\\s+|(?<=[\\u3002\\uff01\\uff1f])")
_WORD: Final = _separator(r"\s+")

_PROSE_LEVELS: Final = (_PARAGRAPH, _LINE, _SENTENCE, _WORD)

_JS_DECLARATIONS: Final = (
    r"export |import |class |function |async function |const |let |var |interface |type |enum |@"
)

# Where a top-level declaration begins, by language (the names S1.1's CODE_LANGUAGES gives). A
# comment above a declaration stays with the code before it. Languages without an entry use the
# generic rule: any line that starts in column 0 with something other than a closing bracket or
# the middle of a block comment.
_TOP_LEVEL: Final[dict[str, _Level]] = {
    # Decorators are consumed by the pattern, so the cut falls above them, not between them and
    # their ``def``.
    "python": _declaration(r"^(?:@[^\n]*\n)*(?:class |def |async def )"),
    "ruby": _declaration(r"^(?=class |module |def )"),
    "js": _declaration(rf"^(?={_JS_DECLARATIONS})"),
    "ts": _declaration(rf"^(?={_JS_DECLARATIONS})"),
    "go": _declaration(r"^(?=func |type |var |const |import |package )"),
    "rust": _declaration(
        r"^(?=pub |fn |impl |struct |enum |trait |mod |use |const |static |type |#\[)"
    ),
}
_GENERIC_TOP_LEVEL: Final = _declaration(r"^(?=[^\s})\]/*])")

# Methods inside a class, for the languages where indentation shows them.
_MEMBER_LEVEL: Final[dict[str, _Level]] = {
    "python": _declaration(r"^[ \t]+(?:@[^\n]*\n[ \t]*)*(?:async def |def )"),
    "ruby": _declaration(r"^[ \t]+def "),
}


def _code_levels(language: str | None) -> tuple[_Level, ...]:
    key = language or ""
    levels = [_TOP_LEVEL.get(key, _GENERIC_TOP_LEVEL)]
    if key in _MEMBER_LEVEL:
        levels.append(_MEMBER_LEVEL[key])
    return (*levels, _PARAGRAPH, _LINE, _WORD)


@dataclass(frozen=True)
class _Piece:
    """An indivisible stretch of the text and its token count. Pieces tile a section exactly."""

    start: int
    end: int
    tokens: int


def _atomize(
    text: str, lo: int, hi: int, levels: Sequence[_Level], budget: _Budget
) -> list[_Piece]:
    """Cut ``text[lo:hi]`` into pieces that each fit the budget, preferring the coarsest cuts.

    A span that fits is one piece. Otherwise it is cut at the first level that has a boundary
    inside it, and each part is cut again with the finer levels it has left. Pieces from different
    levels sit side by side (a short paragraph next to the sentences of a long one); the merge
    step does not care where a piece came from. That is simpler than LangChain's merge-per-level
    and lets the overlap run across every boundary instead of only inside one level.
    """
    span = text[lo:hi]
    tokens = budget.count(span)
    if tokens <= budget.size:
        return [_Piece(lo, hi, tokens)]
    for depth, level in enumerate(levels):
        cuts = [lo + cut for cut in level.cuts(span)]
        if cuts:
            pieces: list[_Piece] = []
            for start, end in pairwise([lo, *cuts, hi]):
                pieces.extend(_atomize(text, start, end, levels[depth + 1 :], budget))
            return pieces
    # No boundary at all (a base64 blob, a long unbroken CJK passage): fixed runs of characters.
    # ``size`` characters never exceed ``size`` tokens, so every run fits.
    runs = [(start, min(start + budget.size, hi)) for start in range(lo, hi, budget.size)]
    return [_Piece(start, end, budget.count(text[start:end])) for start, end in runs]


# --- pass 3: windows -------------------------------------------------------------------------


def _windows(pieces: Sequence[_Piece], budget: _Budget) -> list[tuple[int, int]]:
    """Merge pieces left to right into chunks; return each chunk's ``(start, end)``.

    A chunk takes pieces while their token counts sum to at most ``size`` (a safe upper bound for
    the merged text, see ``estimate_tokens``). The next chunk starts a few pieces back, as many as
    fit in ``overlap`` tokens, but never so many that the next piece no longer fits beside them.
    Without that last condition the next chunk would add nothing new and repeat its predecessor.
    Every chunk therefore ends at a later piece than the one before: the loop always finishes.
    """
    spans: list[tuple[int, int]] = []
    count = len(pieces)
    first = 0
    while first < count:
        end = first
        total = 0
        while end < count and total + pieces[end].tokens <= budget.size:
            total += pieces[end].tokens
            end += 1
        if end == first:  # a piece over the budget: only a counter that breaks its contract
            end = first + 1
        spans.append((pieces[first].start, pieces[end - 1].end))
        if end >= count:
            break
        # How many tokens the next chunk may reopen with: the overlap budget, less whatever the
        # next piece needs to still fit beside them.
        limit = min(budget.overlap, budget.size - pieces[end].tokens)
        back = end
        carried = 0
        while back > first + 1 and carried + pieces[back - 1].tokens <= limit:
            back -= 1
            carried += pieces[back].tokens
        first = back
    return spans


_LEADING_BLANK_LINES: Final = re.compile(r"(?:[ \t\r]*\n)+")


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Drop blank lines before and whitespace after a chunk; ``None`` if nothing is left.

    The indentation of the first real line stays: for code it is meaning, and the chunk must
    still be an exact slice of the text.
    """
    span = text[start:end]
    blank = _LEADING_BLANK_LINES.match(span)
    lead = blank.end() if blank else 0
    body = span[lead:].rstrip()
    if not body:
        return None
    return start + lead, start + lead + len(body)


def _spans(
    text: str, sections: Sequence[_Section], levels: Sequence[_Level], budget: _Budget
) -> Iterator[tuple[int, int, tuple[str, ...]]]:
    """The chunk spans of one segment as ``(start, end, section_path)``, offsets into ``text``.

    Each section is chunked on its own, so no chunk reaches across a heading.
    """
    for section in sections:
        pieces = _atomize(text, section.start, section.end, levels, budget)
        for start, end in _windows(pieces, budget):
            trimmed = _trim(text, start, end)
            if trimmed is not None:
                yield trimmed[0], trimmed[1], section.path
