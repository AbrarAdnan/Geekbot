"""Reading a chat model's stream: the answer, the thinking and the time to the first token (S3.3).

``ChatOllama.stream`` yields small message chunks while the model generates. Each chunk carries a
piece of the answer (``content``), sometimes a piece of the model's reasoning (``additional_kwargs``),
and the last real one carries Ollama's own numbers (``response_metadata``: how long the model took to
load, how many tokens it read and wrote, how long each took). ``read_stream`` consumes the whole
stream, hands every piece on the moment it arrives (so a terminal can print it), and returns the
finished text together with a ``RunStats``.

Why this is its own function. "How fast is it?" has two answers that feel very different: the time
until the *first* token (how long the user waits before anything moves) and the time for the whole
answer. Streaming exists to make the first number small, and the plan's targets are written on it
(PLAN.md §6.5: TTFT p50 2.5 s). A call timed only from the outside, around ``invoke``, cannot tell
the two apart. The clock is a parameter, so a test can feed chunks and tick a fake clock.
"""

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from langchain_core.messages import BaseMessage

from geekbot.domain import RunStats

__all__ = ["REASONING_KEY", "StreamResult", "content_text", "read_stream", "stats_from_metadata"]

# Where langchain-ollama puts a model's reasoning when thinking is on (``reasoning=True``).
REASONING_KEY: Final = "reasoning_content"

_NANOSECONDS: Final = 1e9  # Ollama reports its durations in nanoseconds


@dataclass(frozen=True)
class StreamResult:
    """A finished stream: the answer, the reasoning (empty when thinking is off), the readings."""

    text: str
    thinking: str
    stats: RunStats


def content_text(content: str | Sequence[str | dict[str, Any]]) -> str:
    """The text of a message's ``content``, whether a string or a list of content blocks.

    Not stripped: a streamed piece such as ``" days"`` has a meaningful space in front of it.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _number(metadata: Mapping[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _seconds(metadata: Mapping[str, Any], key: str) -> float | None:
    nanoseconds = _number(metadata, key)
    return None if nanoseconds is None else nanoseconds / _NANOSECONDS


def stats_from_metadata(
    metadata: Mapping[str, Any],
    *,
    seconds: float,
    first_token_s: float | None = None,
    thinking_chars: int = 0,
) -> RunStats:
    """``RunStats`` from the numbers Ollama puts in its last message plus Geekbot's own clock.

    ``seconds`` and ``first_token_s`` are Geekbot's wall-clock readings; everything else is
    Ollama's (``load_duration``, ``prompt_eval_count``, ``prompt_eval_duration``, ``eval_count``,
    ``eval_duration``). A key that is missing, or not a plain number, gives ``None`` rather than a
    guess; tokens per second needs both the count and a non-zero generation time.
    """
    prompt_tokens = _number(metadata, "prompt_eval_count")
    output_tokens = _number(metadata, "eval_count")
    generating = _seconds(metadata, "eval_duration")
    return RunStats(
        seconds=seconds,
        first_token_s=first_token_s,
        load_s=_seconds(metadata, "load_duration"),
        prompt_tokens=None if prompt_tokens is None else round(prompt_tokens),
        prompt_eval_s=_seconds(metadata, "prompt_eval_duration"),
        output_tokens=None if output_tokens is None else round(output_tokens),
        tokens_per_second=(
            output_tokens / generating if output_tokens is not None and generating else None
        ),
        thinking_chars=thinking_chars,
    )


def read_stream(
    chunks: Iterable[BaseMessage],
    *,
    started: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    on_text: Callable[[str], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
) -> StreamResult:
    """Consume a chat stream; return its text, its reasoning and what the call cost. The ✍️ piece.

    ✍️ Written by Claude (prototyping mode, D-047); on the ✍️ backlog in STATUS.md. The rules:

    1. **The clock starts at ``started``**, or when this function is entered if none is given.
       ``ChatOllama.stream`` is lazy (the request goes out on the first ``next``), so entering
       the loop is the moment the call begins. A caller whose stream is not lazy passes the time it
       took *before* creating the stream.
    2. **Time to first token is the first piece of answer text that is not only whitespace**
       (``RunStats.first_token_s``): a lone newline shows the user nothing. Reasoning text never
       counts. ``None`` when no such piece ever arrives.
    3. **Pieces are passed on as they come.** ``on_text`` gets every answer piece, whitespace and
       all (a space between two words is a piece); ``on_thinking`` gets every reasoning piece from
       ``additional_kwargs["reasoning_content"]``. Empty pieces are not passed on.
    4. **The text is the pieces joined, then stripped**, for the answer and for the reasoning.
    5. **Ollama's numbers are merged over all chunks** (later chunks win a clash): the final chunk
       carries them, and langchain appends an empty marker chunk after it that must not erase them.
       ``stats_from_metadata`` turns them into the readings.
    6. **Total seconds is the clock at the end of the stream** minus the start. An exception from
       the stream propagates as it is; the caller decides how to report it.
    """
    begin = clock() if started is None else started
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    metadata: dict[str, Any] = {}
    first_token_s: float | None = None
    for chunk in chunks:
        metadata.update(chunk.response_metadata)
        reasoning = chunk.additional_kwargs.get(REASONING_KEY)
        if isinstance(reasoning, str) and reasoning:
            thinking_parts.append(reasoning)
            if on_thinking is not None:
                on_thinking(reasoning)
        piece = content_text(chunk.content)
        if piece:
            text_parts.append(piece)
            if first_token_s is None and piece.strip():
                first_token_s = clock() - begin
            if on_text is not None:
                on_text(piece)
    seconds = clock() - begin
    thinking = "".join(thinking_parts).strip()
    stats = stats_from_metadata(
        metadata, seconds=seconds, first_token_s=first_token_s, thinking_chars=len(thinking)
    )
    return StreamResult(text="".join(text_parts).strip(), thinking=thinking, stats=stats)
