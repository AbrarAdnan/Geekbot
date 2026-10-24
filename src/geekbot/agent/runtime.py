"""What the answer graph is given to work with (steps S3.1 to S3.4 and C4, PLAN.md §7).

LangGraph separates two kinds of input. The **state** is the data that changes as the graph runs and
is saved between steps. The **context** is fixed for one run and is not data: here, the retriever, the
chat models and (in CSV mode) the loaded table. Putting them in the state would make the state
impossible to serialize (S3.4's checkpointer saves it), and putting them in module globals would make
the graph impossible to test without patching. The graph declares ``context_schema=AgentContext`` and a
node receives it as ``runtime.context``.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage

from geekbot.llm.chat import ChatModel, StructuredChat
from geekbot.retrieval import RetrievalResult, Retriever
from geekbot.retrieval.rerank import PassageScorer
from geekbot.tools.egress import GatedTool

__all__ = [
    "AgentContext",
    "CsvReply",
    "CsvTool",
    "UnavailableChat",
    "UnavailableRetriever",
]


class CsvReply(Protocol):
    """What the graph needs from one answered CSV question (``csv_mode.adapter.CsvAnswer`` has it)."""

    @property
    def markdown(self) -> str:
        """The reply as the package renders it: the answer, its confidence, the SQL, a result table."""
        ...

    @property
    def status(self) -> str:
        """``answered``, ``abstained``, ``needs_clarification`` or ``failed``."""
        ...

    @property
    def model(self) -> str:
        """The model tag that produced it."""
        ...

    @property
    def seconds(self) -> float: ...

    @property
    def model_load_ms(self) -> int: ...


class CsvTool(Protocol):
    """A loaded CSV that answers questions about itself (``csv_mode.adapter.CsvAnalyst``).

    ``ask`` blocks for seconds (it makes several model calls and runs SQL), so the graph calls it from
    a node, which LangGraph runs in a worker thread (PLAN.md §6.3). The graph is typed against this
    Protocol and not against the adapter so that ``agent`` never imports the CSV package (§6.7: only
    ``geekbot.csv_mode`` may) and a test passes a fake.
    """

    def ask(self, question: str, /) -> CsvReply: ...


class UnavailableRetriever:
    """The retriever of a run that has no knowledge base (a CSV conversation): it refuses to search."""

    def search(self, query: str, k: int) -> RetrievalResult:
        raise RuntimeError("this run has no knowledge base: it is a CSV conversation")


class UnavailableChat:
    """The chat model of a run that has none loaded (a CSV conversation): it refuses to answer."""

    def stream(self, messages: list[BaseMessage], /) -> Iterator[BaseMessage]:
        raise RuntimeError("this run has no chat model loaded: it is a CSV conversation")


@dataclass(frozen=True)
class AgentContext:
    """The tools and limits of one graph run, all given explicitly.

    ``model`` is the chat model's name, for messages and reports. ``top_k`` is how many chunks
    retrieval is asked for, and ``context_budget_tokens`` how much of the prompt they may take
    (``Settings.context_budget_tokens`` has the default: one place says 3,000). ``retriever`` is
    any ``Retriever``; if it is a ``retrieval.staged.StagedSearch`` the graph runs its stages as
    steps, the two searches side by side (S3.6), and otherwise it calls it whole.

    The next three are conversation memory (S3.4). ``condenser`` is the model call that rewrites a
    follow-up into a standalone question; without one, a follow-up is searched for as the user typed
    it. ``history_budget_tokens`` is how much of a prompt the earlier turns may take and
    ``history_keep_messages`` how many messages a thread remembers.

    ``csv`` (C4) is the loaded CSV of a CSV conversation; ``None`` in chat mode. A run is one mode or
    the other: a CSV run passes ``UnavailableRetriever`` and ``UnavailableChat``, a chat run no CSV.

    ``citation_scorer`` (S3.5) is the cross-encoder that ``cite_check`` uses to score each cited
    source against the sentence that cites it; ``None`` (no reranker in the retriever, or a test)
    skips the check. ``citation_weak_below`` is the score under which a citation is called weak.

    ``web`` (S5.3) is the web search tool behind the egress gate (``tools.egress.GatedTool``); ``None``
    when the web is off, which is the default and means a turn that asks for a search is told there is
    none. ``approvals`` says that a person can be asked a question in the middle of a turn: it is true
    when the turn is saved in a thread (the runner sets it), because a question needs a checkpoint to
    wait in. A gate that needs to ask and cannot treats that as a denial.
    """

    retriever: Retriever
    llm: ChatModel
    model: str
    top_k: int
    context_budget_tokens: int
    condenser: StructuredChat | None = None
    history_budget_tokens: int = 1000
    history_keep_messages: int = 40
    csv: CsvTool | None = None
    citation_scorer: PassageScorer | None = None
    citation_weak_below: float = -3.0
    web: GatedTool | None = None
    approvals: bool = False
