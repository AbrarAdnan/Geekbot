"""The only place a prompt is assembled (steps S3.1 and S3.2, PLAN.md §6.5, §8.2).

Everything the model reads is built here and nowhere else: the standing instructions, the retrieved
sources, the question. That is an architectural rule, not a convenience. §8.2 labels every piece of
text in a prompt (who wrote it, how far it can be trusted) and derives the labels of the answer from
the labels of everything in the prompt; that only works if there is exactly one function that puts text
into a prompt, so this is also the one place that decides the labels (``PackedPrompt.labels``).

**Layout.** A fixed system message, then one user message that holds the sources and, last, the
question. The system message is read from ``prompts/system.md`` and is identical for every question,
which lets Ollama reuse the work it did on that prefix (the "prefix KV cache", PLAN.md §6.5) instead
of reading it again each time. Nothing that changes per question may go into it.

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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Final

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from geekbot.agent.state import SourceRef
from geekbot.domain import Label
from geekbot.ingest.chunking import estimate_tokens

__all__ = [
    "DATAMARK",
    "MAX_QUESTION_TOKENS",
    "SYSTEM_PROMPT",
    "PackedPrompt",
    "QuestionError",
    "check_question",
    "datamark",
    "load_prompt",
    "pack_prompt",
    "source_block",
    "source_header",
    "source_label",
]

DATAMARK: Final = "> "

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
    and ``KB`` when at least one source is in. The system prompt is the application's own text and
    carries no label.
    """

    messages: list[BaseMessage]
    used: int
    labels: frozenset[Label]


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
    return " | ".join(part for part in parts if part)


def source_header(source: SourceRef) -> str:
    """The header line of a source in the prompt: ``[n] title | section | page``."""
    return f"[{source['n']}] {source_label(source)}".rstrip()


def source_block(source: SourceRef) -> str:
    """One source as the model sees it: its header line, then its text marked line by line."""
    return f"{source_header(source)}\n{datamark(source['text'])}"


def pack_prompt(question: str, sources: Sequence[SourceRef], *, budget_tokens: int) -> PackedPrompt:
    """Build the prompt for ``question`` from as many of ``sources`` as fit in ``budget_tokens``.

    ``sources`` must already be numbered and in rank order. With none that fit (or none given),
    ``used`` is 0 and the user message says so; the graph does not send such a prompt (it answers
    "nothing found" without calling the model), but this function stays total.
    ``ValueError`` if ``budget_tokens`` is below 1; ``QuestionError`` if the question is empty or
    too long (``check_question``).
    """
    if budget_tokens < 1:
        raise ValueError(f"budget_tokens must be at least 1, got {budget_tokens}")
    check_question(question)
    blocks: list[str] = []
    spent = 0
    for source in sources:
        block = source_block(source)
        cost = estimate_tokens(block)
        if spent + cost > budget_tokens:
            break  # stop, do not skip: skipping would leave a gap in the numbers the model sees
        blocks.append(block)
        spent += cost
    listing = "\n\n".join(blocks) if blocks else "(no sources)"
    user = f"Sources:\n\n{listing}\n\nQuestion: {question}"
    labels = frozenset({Label.USER, Label.KB}) if blocks else frozenset({Label.USER})
    return PackedPrompt(
        messages=[SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)],
        used=len(blocks),
        labels=labels,
    )
