"""The answer graph, wired (steps S3.1 to S3.4 and C4, PLAN.md §7).

::

    START → begin_turn ─┬ first turn ─────────────┬─► retrieve   (a chat thread)
                        └ follow-up ─► condense ──┴─► csv_ask    (a CSV thread)

    retrieve ─┬ sources found ─► generate ─► validate_citations ─► cite_check ─► sanitize ─┐
              ├ nothing found ─► no_context ──────────────────────────────────────────────┤
              └ staged search ─► search_dense ┐                                           │
                                 search_lexical┴─► fuse ─► rerank ─► (sources? as above)  │
    csv_ask ─────────────────────────────────────────────────────────────► sanitize ─────┴─► remember ─► END

The chat branch is ``retrieve`` and what follows it; the CSV branch is the single node ``csv_ask``.
Both branches end in ``remember``, and both reply texts pass ``sanitize`` first (S3.5, G3), except the
application's own "nothing found" sentence, which no model wrote.

**Staged search (S3.6).** When the retriever has stages (``retrieval.staged.StagedSearch``) ``retrieve``
only announces the search and the conditional edge after it names *two* steps, ``search_dense`` and
``search_lexical``. LangGraph runs the steps named by one edge in the same round, in parallel, and a
list-form edge (``["search_dense", "search_lexical"] → "fuse"``) makes ``fuse`` wait for both. After
``rerank`` the turn takes the same decision as after a whole-retriever search (``has_sources``). Any
other retriever never enters the four staged steps.

**Retries (S3.6).** The steps that call Ollama and change nothing (``retrieve``, ``search_dense`` and
``generate``) carry a ``RetryPolicy`` that retries transient failures only (``agent.resilience``).
``csv_ask`` has none: a CSV question can run for a minute and the package has its own retry logic.
``condense`` has none because it never raises (it falls back to the user's own words).

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

**S3.5** added the end of both branches. ``cite_check`` scores each citation against the sentence that
makes it (a report, never a verdict) and ``sanitize`` is the output filter of PLAN.md §8.1 G3: no
images, no links to places the sources do not name, on a chat reply and on a CSV reply alike. There
is no step that grades the sources and abstains below a threshold (the plan had one before rev 20,
D-082): the model's own refusal, recognised in ``validate_citations``, is the abstention.

``build_graph`` returns a compiled graph. Compiling checks the wiring (every edge points at a node that
exists) and turns the description into something that can be run with ``invoke`` or ``astream``. Given
a checkpointer (``AsyncSqliteSaver``, see ``agent.runner``), the state is saved after every step under
the ``thread_id`` of the run's config, which is what makes a conversation resumable.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from geekbot.agent.mode import route_after_begin, route_after_condense
from geekbot.agent.nodes import (
    begin_turn,
    cite_check,
    condense,
    csv_ask,
    fuse,
    generate,
    has_sources,
    no_context,
    remember,
    rerank,
    retrieve,
    route_after_retrieve,
    sanitize,
    search_dense,
    search_lexical,
    validate_citations,
)
from geekbot.agent.resilience import transient_retry
from geekbot.agent.runtime import AgentContext
from geekbot.agent.state import AgentState

__all__ = ["build_graph"]


def build_graph(
    checkpointer: BaseCheckpointSaver[str] | None = None,
    *,
    retry: RetryPolicy | None = None,
) -> CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]:
    """The compiled answer graph. Run it with ``graph.invoke(state, context=AgentContext(...))``.

    ``checkpointer`` makes it a conversation: pass the same ``{"configurable": {"thread_id": ...}}``
    config to every run of a thread and the thread's memory carries over. Without one every run
    starts from nothing. ``retry`` is the policy of the steps that call Ollama and change nothing
    (S3.6); the default retries transient failures a few times (``agent.resilience.transient_retry``),
    and a test passes a policy that does not wait.
    """
    policy = retry if retry is not None else transient_retry()
    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node("begin_turn", begin_turn)
    builder.add_node("condense", condense)
    builder.add_node("retrieve", retrieve, retry_policy=policy)
    builder.add_node("search_dense", search_dense, retry_policy=policy)
    builder.add_node("search_lexical", search_lexical)
    builder.add_node("fuse", fuse)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate, retry_policy=policy)
    builder.add_node("validate_citations", validate_citations)
    builder.add_node("cite_check", cite_check)
    builder.add_node("sanitize", sanitize)
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
        "retrieve",
        route_after_retrieve,
        {
            "generate": "generate",
            "no_context": "no_context",
            "search_dense": "search_dense",
            "search_lexical": "search_lexical",
        },
    )
    builder.add_edge(["search_dense", "search_lexical"], "fuse")
    builder.add_edge("fuse", "rerank")
    builder.add_conditional_edges(
        "rerank", has_sources, {"generate": "generate", "no_context": "no_context"}
    )
    builder.add_edge("generate", "validate_citations")
    builder.add_edge("validate_citations", "cite_check")
    builder.add_edge("cite_check", "sanitize")
    builder.add_edge("csv_ask", "sanitize")
    builder.add_edge("sanitize", "remember")
    builder.add_edge("no_context", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer)
