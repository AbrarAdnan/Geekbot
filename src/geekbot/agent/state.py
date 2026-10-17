"""What flows through the answer graph (step S3.1, PLAN.md §7, §8.2).

The state is **plain data**: strings, numbers, lists and dictionaries of those. No retriever, no
model, no database handle, not even a ``Chunk`` object. Two reasons. S3.4 saves the state in a
checkpointer after every step so a conversation can be resumed, and a checkpointer can only save what
it can serialize and read back safely; and the taint labels of §8.2 live in the state too, next to
the text they describe. What the graph is *given* to work with (the retriever, the chat model) is
passed separately, as a runtime context (``agent.runtime``).

A node returns only the keys it changes, and LangGraph merges them into the state. A key that no node
has written yet is simply absent, which is why the state class is ``total=False``.
"""

from typing import Literal, TypedDict

__all__ = ["AgentState", "SourceRef", "StreamEvent"]


class SourceRef(TypedDict):
    """One retrieved chunk, as the graph sees it: what a prompt and a citation need.

    ``n`` is the number the model is told to cite, ``[n]``. It is the chunk's rank in the
    retrieval result, starting at 1, and it stays the same everywhere: in the prompt, in the
    model's answer, and in the list printed under it. The packer may drop sources from the *end*
    to fit the budget, so the numbers the model sees are always ``1..used`` with no gaps.
    """

    n: int
    chunk_id: str
    title: str
    path: str
    page: int | None
    section: str  # the heading trail, "A > B"; empty when the chunk has none
    text: str
    score: float


class AgentState(TypedDict, total=False):
    """The answer graph's state. Only ``question`` is given at the start."""

    question: str
    sources: list[SourceRef]  # every chunk retrieval returned, best first
    retrieval_ms: dict[str, float]  # per stage, from the retriever's own timings
    used: int  # how many of the sources made it into the prompt
    answer: str  # what the user sees; after S3.2's check, without citations that point nowhere
    raw_answer: str  # the model's own text, before the citation check (absent when it was not run)
    cited: list[int]  # the valid [n] marks left in the answer, first appearance first
    dropped_citations: list[int]  # the marks the check removed because no such source was shown
    answer_labels: list[str]  # PLAN.md §8.2: the labels of everything in the prompt, sorted
    thinking: str  # the model's reasoning, only when thinking was on and the model produced some
    generation: dict[str, float]  # what the model call took: ``RunStats.as_dict()`` (S3.3)


class StreamEvent(TypedDict):
    """One piece of the answer while it is being written (S3.3), sent by ``generate``.

    This is not state: nothing keeps it. ``generate`` hands it to the graph's stream writer for
    whoever is listening (``graph.stream(..., stream_mode="custom")``); with plain ``invoke`` the
    writer does nothing. ``text`` is a piece of the answer, ``thinking`` a piece of the reasoning.
    Both come straight from the model, so they are untrusted: the terminal prints them as text.
    """

    event: Literal["text", "thinking"]
    text: str
