"""The answer graph, wired (step S3.1, PLAN.md §7).

::

    START → retrieve ─┬ sources found ─► generate ─► END
                      └ nothing found ─► no_context ─► END

Every edge is written down. There is no agent loop and no model deciding what to do next: the only
decision (``has_sources``) is a plain function of the state. That is the point of using LangGraph
as a *graph* and not as an agent runtime: what the system does is something you can read, draw and
test, and what it costs is known in advance (one retrieval, at most one model call).

``build_graph`` returns a compiled graph. Compiling checks the wiring (every edge points at a node
that exists) and turns the description into something that can be run with ``invoke``; later
steps pass a checkpointer to ``compile`` (S3.4) and add nodes and branches (S3.2 to S3.6).
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from geekbot.agent.nodes import generate, has_sources, no_context, retrieve
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = ["build_graph"]


def build_graph() -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """The compiled answer graph. Run it with ``graph.invoke(state, context=AgentContext(...))``."""
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("no_context", no_context)
    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges(
        "retrieve", has_sources, {"generate": "generate", "no_context": "no_context"}
    )
    builder.add_edge("generate", END)
    builder.add_edge("no_context", END)
    return builder.compile()
