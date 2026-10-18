"""Turning a follow-up into a question that stands on its own (step S3.4, PLAN.md §6.5, §7).

"And what about audit logs?" means nothing to a search engine: the words that say what it is about
were in the previous turn. Retrieval needs a standalone question, so on a follow-up (and only then:
the first turn of a thread makes no model call before retrieval) one model call rewrites the last
message using the conversation so far. The answer is still written from the user's own words and the
quoted conversation; the rewrite is only what retrieval searches for.

This module is the part a person can read and argue with: the shape the model must answer in
(``CondenseOutput``), the instructions (``prompts/condense.md``), how the conversation is put in front
of the model (``build_condense_messages``) and what is done with its reply (``clean_standalone``).
The call itself is made by a ``StructuredChat`` (``llm.chat``), which asks Ollama to produce JSON in
exactly this shape.

Why a structured reply and not free text: a small model asked to "rewrite this question" tends to add
a greeting, answer the question, or explain itself. Constraining the output to one field removes the
room for all three. It does not make the rewrite *correct*: that is measured later (C6 for CSV mode,
and the multi-turn items of the golden set once they are scored).

The earlier chat is quoted data, as everywhere else (``packing.history_line``): an assistant message
was written by a model that had read untrusted documents. It goes through the same marking, and the
rewrite inherits the labels of the thread (``thread_taint``), which is how §8.2 wants a tainted web
query caught later.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from geekbot.agent.packing import (
    HISTORY_HEADING,
    MAX_QUESTION_TOKENS,
    load_prompt,
    pack_history,
)
from geekbot.agent.state import HistoryMessage
from geekbot.ingest.chunking import estimate_tokens

__all__ = [
    "CONDENSE_PROMPT",
    "CondenseOutput",
    "build_condense_messages",
    "clean_standalone",
]

CONDENSE_PROMPT = load_prompt("condense")


class CondenseOutput(BaseModel):
    """What the rewrite step must return: one question that makes sense without the chat.

    Ollama is given this class's JSON schema and constrains its output to it, so the reply is always
    this one field. ``clean_standalone`` still checks the text, because a reply in the right shape
    can be empty, absurdly long or several lines.
    """

    standalone_question: str = Field(
        description="The last message rewritten as one question that needs no earlier context."
    )


def build_condense_messages(
    question: str, history: Sequence[HistoryMessage], *, budget_tokens: int
) -> list[BaseMessage]:
    """The messages for the rewrite call: the fixed instructions, then the chat and the last message.

    The conversation is the newest whole turns that fit in ``budget_tokens``, packed exactly like the
    history of the answer prompt. With none that fit, the model sees only the last message, and the
    instructions tell it to return that unchanged.
    """
    earlier = pack_history(history, budget_tokens=budget_tokens)
    conversation = (
        f"{HISTORY_HEADING}\n" + "\n".join(earlier.lines) + "\n\n" if earlier.lines else ""
    )
    return [
        SystemMessage(content=CONDENSE_PROMPT),
        HumanMessage(content=f"{conversation}Last message: {question}"),
    ]


def clean_standalone(raw: str, *, fallback: str) -> str:
    """The rewrite as one clean line, or ``fallback`` (the user's own words) if it is unusable.

    Whitespace and line breaks collapse to single spaces, and one pair of quotation marks around the
    whole text is removed (models like to quote). The text is unusable if nothing is left, or if it
    is longer than a question may be (``MAX_QUESTION_TOKENS``: a rewrite that long is the model
    answering or rambling, not asking). Falling back costs retrieval quality on a follow-up and never
    stops the turn.
    """
    text = " ".join(raw.split())
    if len(text) > 1 and text[0] in "\"'" and text[-1] == text[0]:
        text = text[1:-1].strip()
    if not text or estimate_tokens(text) > MAX_QUESTION_TOKENS:
        return fallback
    return text
