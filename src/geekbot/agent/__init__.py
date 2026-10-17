"""The answer graph (Phase 3, PLAN.md §7).

A question goes in, an answer with its sources comes out, through a fixed LangGraph state machine:
no agent loop, no tool choice, every edge written down. S3.1 builds the smallest useful one
(``retrieve`` then ``generate``); S3.2 to S3.6 grow it with citations, streaming, memory, abstention
and parallel branches.

This package imports LangGraph and LangChain, which are slow to import, so nothing here is imported
by ``geekbot --version`` or ``geekbot doctor``: the commands that need it import it inside their
function.

Where things live: ``state`` (what flows through the graph, plain data), ``runtime`` (what the
graph is *given*: the retriever and the chat model), ``packing`` (the only place a prompt is
assembled), ``nodes`` (one function per step), ``graph`` (the wiring), ``render`` (the answer as
text for a terminal).
"""
