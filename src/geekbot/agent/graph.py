"""The answer graph, wired (steps S3.1 to S3.4 and C4, PLAN.md §7).

::

    START → begin_turn ─┬ first turn ─────────────┬─► retrieve   (a chat thread)
                        └ follow-up ─► condense ──┴─► csv_ask    (a CSV thread)

    retrieve ─┬ sources found ─► generate ─► validate_citations ─┐
              └ nothing found ─► no_context ─────────────────────┤
    csv_ask ─────────────────────────────────────────────────────┴─► remember ─► END

The chat branch is ``retrieve`` and what follows it; the CSV branch is the single node ``csv_ask``.
Both branches end in ``remember``.

Every edge is written down. There is no agent loop and no model deciding what to do next: the three
decisions (``route_after_begin``, ``route_after_condense``, ``has_sources``) are plain functions of the
state. That is the point of using LangGraph as a *graph* and not as an agent runtime: what the system
does is something you can read, draw and test, and what it costs is known in advance (one retrieval or
one CSV question, at most one rewrite call and one answer call).

``begin_turn`` clears what the previous turn of a saved thread left in the state and settles the
thread's mode (chat or CSV, fixed for the thread); ``remember`` writes the finished turn into the
thread's memory (S3.4); without a checkpointer both are invisible. ``condense`` (S3.4) runs only when
the thread already has history, so the first question of a conversation makes no model call before its
branch starts; in a CSV thread it is the same step with the CSV model (C4), because the package keeps
no memory of its own. ``validate_citations`` (S3.2) is plain code too: it checks the ``[n]`` marks in
the model's answer against the sources the prompt held. It never calls the model, and an answer with
bad or missing citations is reported, not regenerated (a retry loop would make the cost unbounded).

PLAN.md §7 also draws a ``sanitize`` step at the end of both branches (the output filter, G3). It is
S3.5's, and it will be added in front of ``remember`` on both branches there; until then the reply of
the CSV branch reaches the terminal as the package rendered it.

``build_graph`` returns a compiled graph. Compiling checks the wiring (every edge points at a node that
exists) and turns the description into something that can be run with ``invoke`` or ``astream``. Given
a checkpointer (``AsyncSqliteSaver``, see ``agent.runner``), the state is saved after every step under
the ``thread_id`` of the run's config, which is what makes a conversation resumable.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from geekbot.agent.mode import route_after_begin, route_after_condense
from geekbot.agent.nodes import (
    begin_turn,
    condense,
    csv_ask,
    generate,
    has_sources,
    no_context,
    remember,
    retrieve,
    validate_citations,
)
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = ["build_graph"]


def build_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """The compiled answer graph. Run it with ``graph.invoke(state, context=AgentContext(...))``.

    ``checkpointer`` makes it a conversation: pass the same ``{"configurable": {"thread_id": ...}}``
    config to every run of a thread and the thread's memory carries over. Without one every run
    starts from nothing.
    """
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("begin_turn", begin_turn)
    builder.add_node("condense", condense)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("no_context", no_context)
    builder.add_node("csv_ask", csv_ask)
    builder.add_node("remember", remember)
    builder.add_edge(START, "begin_turn")
    builder.add_conditional_edges(
        "begin_turn",
        route_after_begin,
        {"condense": "condense", "retrieve": "retrieve", "csv_ask": "csv_ask"},
    )
    builder.add_conditional_edges(
        "condense", route_after_condense, {"retrieve": "retrieve", "csv_ask": "csv_ask"}
    )
    builder.add_conditional_edges(
        "retrieve", has_sources, {"generate": "generate", "no_context": "no_context"}
    )
    builder.add_edge("generate", "validate_citations")
    builder.add_edge("validate_citations", "remember")
    builder.add_edge("no_context", "remember")
    builder.add_edge("csv_ask", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer)
