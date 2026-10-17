"""What the answer graph is given to work with (step S3.1, PLAN.md §7).

LangGraph separates two kinds of input. The **state** is the data that changes as the graph runs and
is saved between steps. The **context** is fixed for one run and is not data: here, the retriever and
the chat model. Putting them in the state would make the state impossible to serialize (S3.4), and
putting them in module globals would make the graph impossible to test without patching. The graph
declares ``context_schema=AgentContext`` and a node receives it as ``runtime.context``.
"""

from dataclasses import dataclass

from geekbot.llm.chat import ChatModel
from geekbot.retrieval import Retriever

__all__ = ["AgentContext"]


@dataclass(frozen=True)
class AgentContext:
    """The tools and limits of one graph run, all given explicitly.

    ``model`` is the chat model's name, for messages and reports. ``top_k`` is how many chunks
    retrieval is asked for, and ``context_budget_tokens`` how much of the prompt they may take
    (``Settings.context_budget_tokens`` has the default: one place says 3,000).
    """

    retriever: Retriever
    llm: ChatModel
    model: str
    top_k: int
    context_budget_tokens: int
