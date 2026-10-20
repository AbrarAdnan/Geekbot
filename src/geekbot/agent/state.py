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

from typing import Literal, NotRequired, TypedDict

__all__ = [
    "AgentState",
    "CitationScore",
    "HistoryMessage",
    "HistorySource",
    "SourceRef",
    "StreamEvent",
]


class CitationScore(TypedDict):
    """How well one cited source fits the sentence that cites it (S3.5 ``cite_check``).

    ``score`` is the cross-encoder's number for (sentence, source text): higher is a better fit, and it
    is only meaningful next to other scores of the same model. ``weak`` is true when it is below the
    configured limit (``Settings.cite_weak_below``, an unmeasured starting point). A report, not a
    verdict: nothing in the answer changes.
    """

    n: int
    score: float
    weak: bool
    sentence: str


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


class HistorySource(TypedDict):
    """A source an earlier answer was built from, kept with that answer (step S4.2).

    Only the reference, not the chunk's text: a thread keeps up to 40 messages and the checkpoint is
    rewritten after every step, so the text of five chunks per answer would be copied again and again.
    The reference is enough to show *where* an answer came from after a reload ("citations persist",
    Gate A); ``chunk_id`` is there so a later route can fetch the text if the UI wants it.
    """

    n: int  # the number the answer's [n] mark means
    chunk_id: str
    title: str
    path: str
    page: int | None
    section: str


class HistoryMessage(TypedDict):
    """One message of the conversation so far, as plain data (step S3.4, PLAN.md §8.2).

    ``labels`` say who wrote the text, in the vocabulary of ``domain.Label``: a user's message is
    ``["USER"]``; an assistant message carries the labels of everything that was in the prompt that
    produced it (``["KB", "USER"]`` for an answer built from documents; none for the application's
    own "nothing found" sentence). The labels live next to the text they describe, never on a
    string, so they survive being saved and loaded again.

    An assistant message that was built from sources also carries ``sources`` (the ones the prompt
    held, numbered as the model saw them) and ``cited`` (the numbers its text still cites after the
    citation check), so a thread read back later can show what each answer rested on (S4.2). The
    two keys are absent on user messages, on "nothing found" answers, on CSV answers, and on every
    message a thread saved before S4.2.
    """

    id: str  # "t<turn>-user" / "t<turn>-assistant"
    role: Literal["user", "assistant"]
    content: str
    labels: list[str]
    cited: NotRequired[list[int]]
    sources: NotRequired[list[HistorySource]]


class AgentState(TypedDict, total=False):
    """The answer graph's state: what one turn works on, and what a thread remembers.

    Only ``question`` is given at the start of a turn. Without a checkpointer every run starts from
    nothing. With one (``geekbot ask --thread``) the keys below are saved after every step and the
    next turn starts from them, so ``begin_turn`` first resets the keys that belong to one turn only
    (everything from ``standalone_question`` to ``generation``, and ``requested_mode`` and
    ``csv_status``): without that, a turn that finds nothing would still show the previous turn's
    sources. ``history``, ``thread_taint``, ``turn``, ``mode`` and ``csv_file`` are the thread's own.
    """

    question: str
    requested_mode: (
        str  # input only: the mode the caller asks for ("chat" or "csv"), "" if none (C4)
    )
    mode: str  # the thread's mode, fixed at its first turn (C4): "chat" or "csv"
    # CSV mode: the file the thread was started on. The command line keeps the plain name (display
    # only); the server keeps the name of the stored upload, "<sha16>-name.csv", so that it can open
    # the same file again after a restart (C5, ``server.csvmode.stored_file``).
    csv_file: str
    csv_status: str  # CSV mode: the package's verdict for this turn: answered, abstained, ...
    standalone_question: str  # what retrieval searches for: the question, or its rewrite (S3.4)
    condense_seconds: float  # how long the rewrite took (only on a follow-up)
    condense_error: str  # why the rewrite fell back to the user's own words (only if it failed)
    sources: list[SourceRef]  # every chunk retrieval returned, best first
    retrieval_ms: dict[str, float]  # per stage, from the retriever's own timings
    used: int  # how many of the sources made it into the prompt
    history_used: int  # how many earlier messages made it into the prompt (S3.4)
    answer: str  # what the user sees; after S3.2's check, without citations that point nowhere
    raw_answer: str  # the model's own text, before the citation check ("" when it was not run)
    cited: list[int]  # the valid [n] marks left in the answer, first appearance first
    dropped_citations: list[int]  # the marks the check removed because no such source was shown
    abstained: (
        bool  # the answer is a refusal and cites nothing (S3.5, D-082): "not in the documents"
    )
    citation_checks: list[CitationScore]  # S3.5: how well each cited source fits its sentence
    citation_check_error: str  # why the relevance check did not run ("" if it ran or was not asked)
    removed_markup: list[str]  # S3.5, G3: what the output cleaning took out, "image: <address>"
    answer_labels: list[str]  # PLAN.md §8.2: the labels of everything in the prompt, sorted
    thinking: str  # the model's reasoning, "" unless thinking was on and the model produced some
    generation: dict[str, float]  # what the model call took: ``RunStats.as_dict()`` (S3.3)
    # The thread's memory (S3.4). Written only by ``remember``, at the end of a turn that answered.
    history: list[HistoryMessage]  # earlier messages, oldest first; the newest are kept
    thread_taint: list[str]  # PLAN.md §8.2: every label seen in this thread; only ever grows
    turn: int  # how many turns the thread has completed


class StreamEvent(TypedDict):
    """One piece of the answer while it is being written (S3.3), sent by ``generate``.

    This is not state: nothing keeps it. ``generate`` hands it to the graph's stream writer for
    whoever is listening (``graph.stream(..., stream_mode="custom")``); with plain ``invoke`` the
    writer does nothing. ``text`` is a piece of the answer, ``thinking`` a piece of the reasoning.
    Both come straight from the model, so they are untrusted: the terminal prints them as text.
    """

    event: Literal["text", "thinking"]
    text: str
