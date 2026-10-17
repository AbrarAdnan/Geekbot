"""The answer graph, wired (steps S3.1 and S3.2, PLAN.md §7).

::

    START → retrieve ─┬ sources found ─► generate ─► validate_citations ─► END
                      └ nothing found ─► no_context ─────────────────────► END

Every edge is written down. There is no agent loop and no model deciding what to do next: the only
decision (``has_sources``) is a plain function of the state. That is the point of using LangGraph
as a *graph* and not as an agent runtime: what the system does is something you can read, draw and
test, and what it costs is known in advance (one retrieval, at most one model call).

``validate_citations`` (S3.2) is plain code too: it checks the ``[n]`` marks in the model's answer
against the sources the prompt held. It never calls the model, and an answer with bad or missing
citations is reported, not regenerated (a retry loop would make the cost unbounded).

``build_graph`` returns a compiled graph. Compiling checks the wiring (every edge points at a node
that exists) and turns the description into something that can be run with ``invoke``; later
steps pass a checkpointer to ``compile`` (S3.4) and add nodes and branches (S3.3 to S3.6).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from geekbot.agent.nodes import generate, has_sources, no_context, retrieve, validate_citations
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = ["build_graph"]


def build_graph() -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """The compiled answer graph. Run it with ``graph.invoke(state, context=AgentContext(...))``."""
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("no_context", no_context)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve", has_sources, {"generate": "generate", "no_context": "no_context"}
    )
    builder.add_edge("generate", "validate_citations")
    builder.add_edge("validate_citations", END)
    builder.add_edge("no_context", END)
    return builder.compile()
