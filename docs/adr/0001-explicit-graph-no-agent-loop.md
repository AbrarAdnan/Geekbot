# ADR-0001: An explicit LangGraph state machine, no agent loop

- **Status:** accepted (PLAN.md §7; D-002, D-005, D-006 in `project/DECISIONS.md`)
- **Date:** 2026-10-07 (built across steps S3.1 to S3.6, C4, S5.3)

## Context
Geekbot answers a question from a knowledge base with a small local model. Frameworks offer an *agent
loop*: the model picks a tool, reads the result, picks another, until it decides to stop. That is
flexible, and on a 12-billion-parameter model on one consumer GPU it is also unbounded in cost
(every iteration is a model call), hard to test (what happens next is decided by sampled text) and the
worst place to put security (the model that reads untrusted documents would also choose what to call).

## Decision
The answer path is a fixed graph (`src/geekbot/agent/graph.py`, LangGraph `StateGraph`). Every edge is
written down; the few decisions (`route_after_begin`, `route_after_condense`, `after_search`,
`route_after_gate`, `has_sources`) are plain functions of the state, never a model's choice. One turn
costs at most one rewrite call and one answer call. There is no `create_agent`, no ReAct loop, and the
model is given no tools (guarantee G1).

LangGraph is used for what a graph runtime is good at here: the two searches run side by side, the
state is saved after every step so a conversation resumes, `interrupt()` stops a turn at the question
the egress gate asks the user, nodes have retry policies, and the stream carries step events.

## Consequences
- What the system does can be read, drawn and tested; what it costs is known in advance.
- The graph cannot "decide to do more": a capability is a new node and a new edge, in a diff.
- The web search is a step the *user* asks for (`web` on the turn), not something the model calls; the
  gate in front of it asks about the exact text that would leave the machine (ADR-0004).
- A flexible multi-hop agent is out of scope on purpose (parked in PLAN §10).

## Evidence
`tests/unit/test_agent_graph.py` (the sixteen steps), `test_web_graph.py` (the gate inside the graph),
`project/steps/S3.6-parallelism-resilience.md`, `S5.3-web-taint.md`.
