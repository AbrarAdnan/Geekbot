"""From the graph's stream to SSE frames, and from an exception to a sentence (step S4.2).

``agent.runner.stream_turn`` hands over the events of one turn exactly as LangGraph produces them:
``("custom", StreamEvent)`` for each piece of the answer or of the reasoning, and ``("values", state)``
after every step. ``TurnTranslator`` turns them into the frames of the ``/messages`` stream
(``sse.format_event``). It does no I/O and holds no model, so it is tested on its own.

What goes out as a frame is only what the graph already cleaned: the pieces are the output of
``StreamSanitizer`` (G3, S3.5), sent by ``generate`` after the cleaning. This module adds nothing to
them and never looks inside the text, except to drop leading whitespace from the very first piece of
the answer, as the terminal does (``agent.render.StreamCollector``), so an answer never starts with a
blank line.
"""

from collections.abc import Mapping
from typing import Any, Final

from geekbot.agent.mode import ModeError
from geekbot.csv_mode.errors import CsvLoadError
from geekbot.csv_mode.sessions import SessionTooLargeError
from geekbot.index.dense import DenseIndexError
from geekbot.index.embedder import EmbeddingError
from geekbot.llm.chat import ChatError
from geekbot.llm.models import OllamaError
from geekbot.retrieval.rerank import RerankError
from geekbot.sse import format_event

__all__ = ["TurnTranslator", "describe_error"]

# Failures the program expects and words itself: the message is meant for the user. Anything else is
# a bug, and its text (a library's, a path, a stack) stays in the server log.
EXPECTED_ERRORS: Final = (
    ChatError,
    CsvLoadError,
    DenseIndexError,
    EmbeddingError,
    ModeError,
    OllamaError,
    RerankError,
    SessionTooLargeError,
)


class TurnTranslator:
    """Turns one turn's stream events into SSE frames, and keeps the last state it saw.

    Feed it every ``(kind, payload)`` pair; ``feed`` returns the frames to send (possibly none). The
    state of the finished turn is ``final`` once the stream is over; it is ``None`` if the graph never
    produced one.

    - ``text`` and ``thinking`` events become ``text`` and ``thinking`` frames carrying ``{"text":
      piece}``. Empty pieces send nothing. The first piece of the *answer* has its leading whitespace
      removed, and so does each following piece until one has something left.
    - ``values`` payloads are the whole state after a step; only the latest is kept.
    - Anything else (a stream mode this does not know, an event of another kind) is ignored.
    """

    def __init__(self) -> None:
        self.final: dict[str, Any] | None = None
        self._text_started = False

    def feed(self, kind: str, payload: Any) -> list[str]:
        """The frames for one event."""
        if kind == "values":
            self.final = dict(payload)
            return []
        if kind != "custom" or not isinstance(payload, Mapping):
            return []
        piece = str(payload.get("text", ""))
        event = payload.get("event")
        if event == "thinking":
            return [format_event("thinking", {"text": piece})] if piece else []
        if event == "text":
            if not self._text_started:
                piece = piece.lstrip()
            if not piece:
                return []
            self._text_started = True
            return [format_event("text", {"text": piece})]
        return []


def describe_error(error: Exception, *, think: bool = False) -> tuple[str, bool]:
    """The sentence the ``error`` event carries, and whether the failure was an expected kind.

    An expected failure (the chat model or Ollama is down, a model is missing, the thread is in
    another mode) is reported in its own words, which were written for a person. With ``think`` on, a
    model that has no thinking mode gets the same hint the command line gives. Anything else is
    reported as "unexpected" with its type only: the details are in the server log, not on the wire.
    """
    if isinstance(error, EXPECTED_ERRORS):
        message = " ".join(str(error).split())
        if think and "support thinking" in message:
            message += ". Ask without thinking, or choose a chat model with a thinking mode."
        return message, True
    return f"unexpected {type(error).__name__}; the server log has the details", False
