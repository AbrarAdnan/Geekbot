"""Server-sent events: writing and reading the wire format (step S4.2).

An SSE response is an ordinary HTTP response that does not end: the server keeps the connection open
and writes *frames* as things happen, and the client reads them as they arrive. A frame is a few
``field: value`` lines and a blank line::

    event: text
    data: {"text":"Ninety days"}

Geekbot uses two fields. ``event`` names the kind of frame and ``data`` carries one JSON value. The
point of putting JSON on a single line is that the framing and the payload never interfere: a model
can write any text, newlines included, and it travels inside a JSON string.

This module is the one place that knows the format, for both ends: the server writes frames with
``format_event`` (``server/threads.py``), the tests and later the Streamlit client (S4.4) read them with
``parse_events`` / ``parse_text``. It imports nothing from Geekbot, so a client can use it without
pulling in the storage modules (the UI never imports those, PLAN.md section 6.2).

**Why the writer is more careful than ``json.dumps``.** A frame must stay on one line, and
``json.dumps`` escapes every control character below U+0020, so a ``\\n`` in the text cannot start a
new line. Three characters get through that it leaves alone, because JSON allows them raw inside a
string: U+0085, U+2028 and U+2029. ``str.splitlines`` (which httpx's ``iter_lines`` and plenty of
other Python code use) treats each as a line break, so a document that happens to contain one would
cut a ``data:`` line in two on the client. They are escaped here. The second hazard is a text with a
lone surrogate (a model can emit one): it cannot be encoded as UTF-8, and the server would die in the
middle of a stream. Such a payload is written in pure ASCII instead, where every non-ASCII character
is a ``\\uXXXX`` escape that the client decodes back to the same text.
"""

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["SseEvent", "format_event", "parse_events", "parse_text"]

# Lower-case words, so a name can never carry a colon, a space or a line break into the frame.
_EVENT_NAME: Final = re.compile(r"[a-z][a-z0-9_-]{0,31}")

# The three characters ``json.dumps(ensure_ascii=False)`` leaves raw and ``str.splitlines`` splits on.
_LINE_BREAKS_JSON_ALLOWS: Final = {
    0x0085: "\\u0085",
    0x2028: "\\u2028",
    0x2029: "\\u2029",
}


@dataclass(frozen=True)
class SseEvent:
    """One frame as a reader sees it: the event name and its decoded JSON value."""

    event: str
    data: Any


def format_event(event: str, data: object) -> str:
    """One SSE frame: ``event: <name>``, ``data: <json>`` and the blank line that ends it.

    ✍️ Written by Claude (prototyping mode, D-047, step S4.2); on the ✍️ backlog in STATUS.md.

    Rules, each with a test:

    1. ``event`` is a lower-case word (letters, digits, ``_`` and ``-``, at most 32 characters, starting
       with a letter); anything else is a ``ValueError``, because a name with a line break in it would
       let its author add fields to the frame.
    2. ``data`` is any JSON value, written compactly on **one** line. NaN and infinity are refused
       (``ValueError``): they are not JSON, and a browser's ``JSON.parse`` would reject them.
    3. Non-ASCII text is written as it is (UTF-8), not as ``\\uXXXX`` escapes, so Bangla or Chinese
       text costs three bytes a character instead of six.
    4. U+0085, U+2028 and U+2029 are escaped, so that no line splitter used by a reader can cut the
       ``data:`` line (see the module docstring).
    5. A text that cannot be encoded as UTF-8 (a lone surrogate) is written in pure ASCII instead,
       so this function never produces a frame the server cannot send.
    6. Whatever the data, splitting the frame on line breaks of any kind gives exactly the event
       line, the data line and one empty line, and ``parse_text`` returns the same event and value.
    """
    if not _EVENT_NAME.fullmatch(event):
        raise ValueError(
            f"{event!r} is not a usable event name: use a lower-case word of letters, digits, "
            "'_' and '-', starting with a letter, at most 32 characters"
        )
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    try:
        payload.encode("utf-8")
    except UnicodeEncodeError:
        payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return f"event: {event}\ndata: {payload.translate(_LINE_BREAKS_JSON_ALLOWS)}\n\n"


def parse_events(lines: Iterable[str]) -> Iterator[SseEvent]:
    """The frames in a stream of lines, one ``SseEvent`` each, as they complete.

    ``lines`` are lines **without** their line endings, split on ``\\n``, ``\\r`` or ``\\r\\n`` only
    (what an SSE stream defines; not ``str.splitlines``, which splits on more). The rules are the
    standard's, as far as Geekbot's frames need them:

    - a blank line ends a frame; a frame with no ``data:`` line produces nothing;
    - a line that starts with ``:`` is a comment and is skipped;
    - ``field: value`` has one optional space after the colon; several ``data:`` lines are joined
      with a newline; ``event:`` sets the name (``"message"`` when absent);
    - fields other than ``event`` and ``data`` are ignored;
    - a frame still open when the lines end is not delivered (the connection broke mid-frame).

    A ``data`` value that is not JSON raises ``ValueError``: it is the sender that broke the format.
    """
    name = "message"
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                yield SseEvent(name, json.loads("\n".join(data)))
            name, data = "message", []
        elif line.startswith(":"):
            continue
        else:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)


def parse_text(text: str) -> list[SseEvent]:
    """Every complete frame in ``text`` (a whole response body), split the way SSE defines lines."""
    return list(parse_events(re.split(r"\r\n|\r|\n", text)))
