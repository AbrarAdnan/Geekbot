"""Plain records that cross layers (PLAN.md §6.6).

A record here is data and nothing else: no behaviour that touches a store, a model or the disk, so
the ingest, index, retrieval and agent packages can all import it without importing one another.
S1.4 adds the ones the store hands out; S3.2 adds ``Label``; S3.3 adds ``RunStats``. The plan's
other record (``Citation``) arrives with the step that first needs it.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Self

# The outbox states of a document (PLAN.md §6.4): its chunk rows are written while it is
# ``pending``, its vectors while it is ``embedding``, and retrieval may only see it once it is
# ``ready``. ``failed`` keeps the reason in ``Document.error``. Which moves between states are
# allowed is the pipeline's rule (S1.6); the store only checks that the state is one of these.
DocState = Literal["pending", "embedding", "ready", "failed"]
DOC_STATES: Final[tuple[DocState, ...]] = ("pending", "embedding", "ready", "failed")


class Label(StrEnum):
    """Who wrote a piece of text, which decides how far an answer built on it can be trusted.

    PLAN.md §8.2: every piece of text that goes into a prompt has a label, and a model call's
    output carries the union of the labels of everything in its prompt. ``USER`` is the person at the
    keyboard; ``KB`` text came out of an ingested document; ``WEB`` from a web page; ``TOOL`` from a
    tool's output (CSV mode); ``UNKNOWN`` is anything whose origin cannot be shown (for example an
    assistant message a client sends back to the stateless API).
    """

    USER = "USER"
    KB = "KB"
    WEB = "WEB"
    TOOL = "TOOL"
    UNKNOWN = "UNKNOWN"


# Text that someone other than the user wrote, or whose author is unknown. A prompt that holds any of
# it can steer the model, so what the model produces from it is *tainted* (§8.2).
TAINTED_LABELS: Final = frozenset({Label.KB, Label.WEB, Label.TOOL, Label.UNKNOWN})


def is_tainted(labels: Iterable[Label]) -> bool:
    """Whether any of the labels is one that an outsider's text carries (``USER`` alone is not)."""
    return not TAINTED_LABELS.isdisjoint(labels)


@dataclass(frozen=True)
class RunStats:
    """What one chat-model call cost, in plain numbers (PLAN.md §6.5, step S3.3).

    - ``seconds``: wall-clock time of the call, from asking to the last token.
    - ``first_token_s``: time to the first piece of *answer* text, the TTFT that the latency targets
      are written on. ``None`` when no answer text ever arrived. Reasoning text does not count: a
      model that thinks for ten seconds has not yet shown the user anything.
    - ``load_s``: how long Ollama spent loading the model for this call. After a warm-up it must be
      close to 0 (rule R-2); seconds here mean the model was reloaded during a question.
    - ``prompt_tokens`` and ``prompt_eval_s``: how many prompt tokens Ollama processed for this call
      and how long that took. Ollama can reuse the work it did for an unchanged start of the prompt
      (the fixed system message), so on a warm server these should be smaller than the prompt.
    - ``output_tokens``: tokens the model wrote, reasoning included. ``tokens_per_second`` is those
      over the time Ollama itself spent generating them (not including loading or reading the prompt).
    - ``thinking_chars``: characters of reasoning text that streamed (0 when thinking is off).

    A reading Ollama did not report is ``None`` and is left out of ``as_dict``: an absent number is
    not a zero. ``as_dict`` / ``from_mapping`` are the plain-data form the graph's state keeps (a
    checkpointer can only save plain data, and the numbers are all floats there).
    """

    seconds: float
    first_token_s: float | None = None
    load_s: float | None = None
    prompt_tokens: int | None = None
    prompt_eval_s: float | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    thinking_chars: int = 0

    def as_dict(self) -> dict[str, float]:
        """The readings that exist, under the keys ``agent.render`` prints (see ``from_mapping``)."""
        named: dict[str, float | None] = {
            "seconds": self.seconds,
            "first_token_seconds": self.first_token_s,
            "load_seconds": self.load_s,
            "prompt_tokens": self.prompt_tokens,
            "prompt_eval_seconds": self.prompt_eval_s,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "thinking_chars": self.thinking_chars or None,
        }
        return {name: value for name, value in named.items() if value is not None}

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> Self:
        """The inverse of ``as_dict``. ``ValueError`` without ``seconds``: nothing was timed."""
        if "seconds" not in values:
            raise ValueError("the readings have no 'seconds': no model call was timed")

        def whole(name: str) -> int | None:
            return round(values[name]) if name in values else None

        return cls(
            seconds=values["seconds"],
            first_token_s=values.get("first_token_seconds"),
            load_s=values.get("load_seconds"),
            prompt_tokens=whole("prompt_tokens"),
            prompt_eval_s=values.get("prompt_eval_seconds"),
            output_tokens=whole("output_tokens"),
            tokens_per_second=values.get("tokens_per_second"),
            thinking_chars=whole("thinking_chars") or 0,
        )


def embedding_text(header: str, text: str, *, with_header: bool) -> str:
    """The exact string the embedder sees for a chunk: its text, with the header in front if asked.

    One definition for the chunker's draft and the stored chunk, so a chunk embedded at ingest
    and the same chunk re-embedded later can never differ by the header's spelling. The header is
    kept out of the chunk's identity and out of BM25 (D-055); this is the only place it joins
    the text, and only for the dense side.
    """
    if with_header and header:
        return f"{header}\n\n{text}"
    return text


@dataclass(frozen=True)
class NewDocument:
    """What the pipeline knows about a file when it hands it to the store.

    ``doc_id`` comes from ``ids.doc_id_of``, ``file_hash`` from ``ids.file_hash_of``. The path is
    kept for display and citations, so it keeps its spelling (the ID uses a normalized one).
    ``pipeline_version`` names the loader and chunker that produced the chunks: an unchanged file
    still has to be processed again when that changes, which ``file_hash`` alone cannot say.
    """

    doc_id: str
    path: str
    title: str
    kind: str
    file_hash: str
    pipeline_version: str


@dataclass(frozen=True)
class Document:
    """A stored document: a ``NewDocument`` plus where it is in the outbox."""

    doc_id: str
    path: str
    title: str
    kind: str
    file_hash: str
    pipeline_version: str
    state: DocState
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Chunk:
    """A stored chunk, with the path and title of its document so a citation needs no second query.

    The fields are those of ``IdentifiedChunk`` (S1.3) plus ``path`` and ``title``. ``char_start``
    and ``char_end`` are offsets into the document's loaded text and ``page`` is 1-based, as in
    S1.1 and S1.2; ``section_path`` is the trail of headings above the chunk.
    """

    chunk_id: str
    doc_id: str
    path: str
    title: str
    content_hash: str
    occurrence: int
    ordinal: int
    char_start: int
    char_end: int
    page: int | None
    section_path: tuple[str, ...]
    header: str
    text: str

    def embedding_text(self, *, with_header: bool) -> str:
        """The string to embed for this chunk (see the module-level ``embedding_text``)."""
        return embedding_text(self.header, self.text, with_header=with_header)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk and how well a retriever scored it. A higher ``score`` is better.

    What the score means depends on the retriever (a BM25 value, a cosine similarity), so two
    scores from different retrievers are never compared; fusion (S2.2) works on ranks.
    """

    chunk: Chunk
    score: float
