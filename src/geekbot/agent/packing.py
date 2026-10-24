"""The only place a prompt is assembled (steps S3.1 and S3.2, PLAN.md §6.5, §8.2).

Everything the model reads is built here and nowhere else: the standing instructions, the retrieved
sources, the question. That is an architectural rule, not a convenience. §8.2 labels every piece of
text in a prompt (who wrote it, how far it can be trusted) and derives the labels of the answer from
the labels of everything in the prompt; that only works if there is exactly one function that puts text
into a prompt, so this is also the one place that decides the labels (``PackedPrompt.labels``).

**Layout.** A fixed system message, then one user message that holds, in this order, the earlier
turns of the conversation (if the thread has any, S3.4), the sources and, last, the question. The
system message is read from ``prompts/system.md`` and is identical for every question, which lets
Ollama reuse the work it did on that prefix (the "prefix KV cache", PLAN.md §6.5) instead of reading it
again each time. Nothing that changes per question may go into it. The earlier turns come before the
sources because they only grow at their end, so a thread's next turn starts with the same text as its
last one for as long as the history fits.

**History (S3.4).** Earlier messages are quoted data too: an assistant message was written by a model
that had read untrusted documents, and a model can repeat what a document told it to say, so it is
marked line by line like a source, squeezed onto one line per message (a message cannot fake a new
speaker with a line break), and stripped of its ``[n]`` marks (they numbered the sources of *that*
turn and would mean the wrong chunk now). Whole turns are taken newest first until
``history_budget_tokens`` is used. The labels of the messages that went in are part of the prompt's
labels, which is how a thread stays tainted after a tainted answer (no laundering through history).

**Data-marking.** The sources are untrusted text: a document can say "ignore your instructions". Each
source is a header line (``[n] title | section | page``) followed by its text with ``> `` in front of
*every* line. The system prompt tells the model that a line starting with ``> `` is quoted data. The
point of marking every line, instead of fencing the block, is that a document has no way to fake the
end of the block, a new header, or a line outside the quote: whatever it writes, each of its lines
still starts with the mark. Unicode line breaks (``\\u2028``, ``\\x85`` and the like) are turned into
plain newlines first, so they cannot hide a line from the marking. Marking costs about one token per
line, keeps the words intact for a small model, and keeps the system prompt fixed (a random fence
per request would not). It lowers the risk and guarantees nothing: the real defences are the output
filter (S3.5) and the injection evaluation that measures what gets through (S5.4).

**The budget.** The sources may take ``budget_tokens`` of the prompt (3,000 by default, out of an
8,192-token window, PLAN.md §6.5). They are added in rank order and the first one that does not fit
stops the packing, so what the model sees is always sources ``1..used`` with no gap: a number the
model cites always means the same chunk. The size is an estimate (``estimate_tokens``), which is not
guaranteed to be an upper bound (D-064), so the budget leaves room. The question gets its own cap
(``MAX_QUESTION_TOKENS``): a prompt larger than ``num_ctx`` would be cut by Ollama, and the part it
cuts must never be the rules.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Final, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from geekbot.agent.citations import check_citations
from geekbot.agent.state import HistoryMessage, SourceRef
from geekbot.domain import Label, label_of
from geekbot.ingest.chunking import estimate_tokens

__all__ = [
    "DATAMARK",
    "HISTORY_HEADING",
    "MAX_QUESTION_TOKENS",
    "SYSTEM_PROMPT",
    "PackedHistory",
    "PackedPrompt",
    "QuestionError",
    "check_question",
    "datamark",
    "history_line",
    "load_prompt",
    "message_labels",
    "pack_history",
    "pack_prompt",
    "source_block",
    "source_header",
    "source_label",
    "source_origin",
]

DATAMARK: Final = "> "
HISTORY_HEADING: Final = "Conversation so far (earlier in this chat, quoted):"

# The question's share of the prompt: about 4,000 characters, a long paragraph or two. With the
# system prompt (about 250 tokens), the sources (3,000) and the answer (512) it keeps the whole
# request far below the 8,192-token window even if the token estimate is off by a factor of two.
MAX_QUESTION_TOKENS: Final = 1000


class QuestionError(ValueError):
    """The question cannot be sent: it is empty, or too long to leave room for the sources."""


@cache
def load_prompt(name: str) -> str:
    """The text of ``agent/prompts/<name>.md`` without its trailing newline, read once.

    The prompts are data files so that changing one is a visible diff in a file of its own, not an
    edit inside Python. ``importlib.resources`` finds them next to the code, whether the package is
    installed in editable mode or as a wheel. A missing file raises ``FileNotFoundError`` when this
    module is imported, which is where a broken install should fail.
    """
    resource = files("geekbot.agent") / "prompts" / f"{name}.md"
    return resource.read_text(encoding="utf-8").rstrip("\n")


SYSTEM_PROMPT: Final = load_prompt("system")


@dataclass(frozen=True)
class PackedPrompt:
    """The messages to send, how many sources they contain (``1..used``), and their labels.

    ``labels`` are the §8.2 labels of everything in the messages: always ``USER`` (the question),
    ``KB`` when at least one source is in, and whatever labels the earlier messages that went in
    carry (``history_used`` says how many). The system prompt is the application's own text and
    carries no label.
    """

    messages: list[BaseMessage]
    used: int
    labels: frozenset[Label]
    history_used: int = 0


def check_question(question: str) -> None:
    """Raise ``QuestionError`` for an empty question or one over ``MAX_QUESTION_TOKENS``.

    Called first by ``pack_prompt`` and also by the command line before it loads a model, so a
    mistake costs nothing. The question is never cut: a silently shortened question would be
    answered as if it were the whole thing.
    """
    if not question.strip():
        raise QuestionError("the question is empty")
    tokens = estimate_tokens(question)
    if tokens > MAX_QUESTION_TOKENS:
        raise QuestionError(
            f"the question is too long (about {tokens} tokens; the limit is {MAX_QUESTION_TOKENS}); "
            "shorten it"
        )


def _one_line(text: str) -> str:
    """Collapse every run of whitespace, line breaks included, to one space."""
    return " ".join(text.split())


def datamark(text: str) -> str:
    """``text`` with ``DATAMARK`` in front of every line (an empty line becomes a bare ``>``).

    Lines are found with ``str.splitlines``, which also splits on ``\\r``, ``\\x0b``, ``\\x0c``,
    ``\\x1c`` to ``\\x1e``, ``\\x85``, ``\\u2028`` and ``\\u2029``, and they are rejoined with
    ``\\n``: a text that uses one of those as a line break is marked per line like any other, and the
    break itself does not survive. So no character of ``text`` can sit at the start of a line in the
    result without the mark in front of it.
    """
    lines = text.splitlines() or [""]
    return "\n".join(f"{DATAMARK}{line}" if line else DATAMARK.rstrip() for line in lines)


def source_label(source: SourceRef) -> str:
    """Where a source comes from, on one line: ``title | section | page 3`` (parts that exist).

    The title and the heading trail come from the document, so they are untrusted too: each is
    squeezed onto one line, so a title that contains a line break cannot start a line of its own.
    """
    parts = [_one_line(source["title"]), _one_line(source["section"])]
    if source["page"] is not None:
        parts.append(f"page {source['page']}")
    if source_origin(source) is Label.WEB:
        parts.insert(0, "web")  # the model and the reader can tell a search result from a document
    return " | ".join(part for part in parts if part)


def source_origin(source: SourceRef) -> Label:
    """Who wrote a source's text: ``KB`` unless it says otherwise (S5.3). An origin that is not a
    label is ``UNKNOWN`` (``domain.label_of``), never ``KB``."""
    return label_of(source.get("origin", Label.KB.value))


def message_labels(message: HistoryMessage) -> frozenset[Label]:
    """The labels of an earlier message. A message without any is not "clean": an assistant message
    is ``UNKNOWN`` (nothing shows it was written by this program), a user message ``USER``."""
    # ``labels`` is a required key of ``HistoryMessage``, but a thread saved by something other than
    # ``remember`` (or damaged) may lack it, and a missing label must never read as "no taint".
    names: object = cast("Mapping[str, object]", message).get("labels")
    if not isinstance(names, list):
        return frozenset({Label.USER if message.get("role") == "user" else Label.UNKNOWN})
    return frozenset(label_of(str(name)) for name in names)


def source_header(source: SourceRef) -> str:
    """The header line of a source in the prompt: ``[n] title | section | page``."""
    return f"[{source['n']}] {source_label(source)}".rstrip()


def source_block(source: SourceRef) -> str:
    """One source as the model sees it: its header line, then its text marked line by line."""
    return f"{source_header(source)}\n{datamark(source['text'])}"


def history_line(message: HistoryMessage) -> str:
    """One earlier message as one marked line: ``> User: ...`` or ``> Assistant: ...``.

    The text is squeezed onto one line (so it cannot start a line of its own that looks like another
    speaker) and an assistant message loses its ``[n]`` marks: they numbered the sources of the turn
    it came from, and in a new prompt ``[1]`` would name a different chunk.
    """
    if message["role"] == "assistant":
        speaker, text = "Assistant", check_citations(message["content"], 0).text
    else:
        speaker, text = "User", message["content"]
    return f"{DATAMARK}{speaker}: {_one_line(text)}".rstrip()


@dataclass(frozen=True)
class PackedHistory:
    """The earlier messages that fit: their marked lines, how many, and the labels they carry."""

    lines: list[str]
    labels: frozenset[Label]

    @property
    def count(self) -> int:
        return len(self.lines)


def pack_history(history: Sequence[HistoryMessage], *, budget_tokens: int) -> PackedHistory:
    """The newest whole turns of ``history`` that fit in ``budget_tokens``, oldest first.

    Messages are taken from the end until one does not fit, and the packing stops there (a gap in
    the middle of a conversation would make it read as something that was never said). A turn is a
    user message and the answer to it, so a first message that is an answer is dropped: the model
    would otherwise see a reply to a question it cannot see. With a budget below 1 or no history the
    result is empty.
    """
    chosen: list[HistoryMessage] = []
    spent = 0
    if budget_tokens >= 1:
        for message in reversed(history):
            cost = estimate_tokens(history_line(message))
            if spent + cost > budget_tokens:
                break
            chosen.append(message)
            spent += cost
    chosen.reverse()
    while chosen and chosen[0]["role"] != "user":
        chosen.pop(0)
    labels = frozenset(label for message in chosen for label in message_labels(message))
    return PackedHistory(lines=[history_line(message) for message in chosen], labels=labels)


def pack_prompt(
    question: str,
    sources: Sequence[SourceRef],
    *,
    budget_tokens: int,
    history: Sequence[HistoryMessage] = (),
    history_budget_tokens: int = 0,
) -> PackedPrompt:
    """Build the prompt for ``question`` from as many of ``sources`` as fit in ``budget_tokens``.

    ``sources`` must already be numbered and in rank order. With none that fit (or none given),
    ``used`` is 0 and the user message says so; the graph does not send such a prompt (it answers
    "nothing found" without calling the model), but this function stays total. ``history`` is the
    thread's earlier messages; the newest whole turns that fit in ``history_budget_tokens`` go in
    front of the sources (none by default). ``ValueError`` if ``budget_tokens`` is below 1;
    ``QuestionError`` if the question is empty or too long (``check_question``).
    """
    if budget_tokens < 1:
        raise ValueError(f"budget_tokens must be at least 1, got {budget_tokens}")
    check_question(question)
    blocks: list[str] = []
    origins: set[Label] = set()
    spent = 0
    for source in sources:
        block = source_block(source)
        cost = estimate_tokens(block)
        if spent + cost > budget_tokens:
            break  # stop, do not skip: skipping would leave a gap in the numbers the model sees
        blocks.append(block)
        origins.add(source_origin(source))
        spent += cost
    earlier = pack_history(history, budget_tokens=history_budget_tokens)
    conversation = (
        f"{HISTORY_HEADING}\n" + "\n".join(earlier.lines) + "\n\n" if earlier.lines else ""
    )
    listing = "\n\n".join(blocks) if blocks else "(no sources)"
    user = f"{conversation}Sources:\n\n{listing}\n\nQuestion: {question}"
    labels = frozenset({Label.USER, *origins})  # each source's own label: KB, WEB, ...
    return PackedPrompt(
        messages=[SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)],
        used=len(blocks),
        labels=labels | earlier.labels,
        history_used=earlier.count,
    )
