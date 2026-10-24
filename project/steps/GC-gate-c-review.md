# GC — Gate C review (the final review)

- **Phase / Gate:** the review at the end of Phase 6 (PLAN §11 "Gate C review", §14 item 9), and the end of autopilot run 5 (D-097, 2026-10-24). Done by Claude; **the verdict is yours.**
- **Commit:** branch `step/GC-gate-c-review` (stacked on `step/S6.1-mcp`)

## Gate C's definition: the MCP server (PLAN §3, §11, §14 item 9)

| Item | Verdict | Evidence and what to keep with it |
|---|---|---|
| The MCP server is mounted at `/mcp` with read-only tools | **pass** | `server/mcp.py`; two tools, both `readOnlyHint`; no tool writes, ingests, deletes or reaches the network (`test_mcp.py` asserts the exact list). |
| §14 item 9: `claude mcp add --transport http geekbot http://127.0.0.1:8000/mcp` then `search_knowledge_base` | **pass in substance, not to the letter** | The official MCP SDK 2.3 client did exactly that exchange against the running server with `gemma4:12b` loaded (both negotiation modes), and in a test over a real socket; raw JSON-RPC and the stdio relay were checked live. **`claude mcp add` itself was not run**: it changes your Claude configuration. The command with the key header is in the README; the step is yours. |
| The stdio proxy | **pass** | `geekbot mcp-stdio`; three messages in, two answers out, live. |
| Same perimeter, no new door that can change anything | **pass** | key, Host allowlist, body cap, no CORS, plus an `Origin` check (401, 400, 403 and 405 checked live and in tests). |

## The final look-back (the whole project against the plan)

- **Delivered, to Gate C:** Gate A (17 of 17 items, run by Claude), S5.1, S5.2 (the judge unvalidated), C6, S5.3, S5.4, S5.5, Gate B, S6.1. Everything is built, measured on the target models, committed and pushed on **ten stacked branches**; **no pull request exists yet** (the GitHub sign-in was lost in the built-in browser).
- **What the project can now say about itself** (README, `docs/`): a measured retrieval ladder with its unflattering parts, speed targets met on a 10 GB GPU with one named miss, an answer evaluation with an honestly unvalidated judge, a CSV mode whose follow-up rewrite caused none of three failures in 15 items, and a security story in two parts: guarantees with their tests, and 0 of 129 bypasses plus 15 of 129 hijacks, with an effective N of 43 attacks and three filter holes found and fixed on the way.
- **What the run did that you did not ask for:** found and fixed nine holes in the output filter across two steps (D-100, D-103); fixed a tie-order bug in rank fusion (a float sum's last bit); added a dev dependency (`mcp`) and regenerated the lock; created ten branches.

## A pass over the notes: where the plan and the code disagree (nothing edited without your yes)

PLAN.md was **not** edited in this run (it changes only with your approval, CLAUDE.md). The disagreements, with the decision that records each, are proposals for a **PLAN rev 23**:

1. **§7 (the graph) and §6.7 draw the web edge as `weak, web on ─► egress_gate ─► web`.** In the code a web search is **asked for on the turn** (the UI toggle, `ask --web`) and the gate decides whatever the knowledge base found; there is no "weak" grading step since rev 20. (D-102 item 8.) Proposed: redraw as `web asked for ─► egress_gate ─► web`.
2. **§6.6 (layout) and §11 (S5.3 row) name `taint.py` and `EgressGate` under `agent/` and `tools/web.py`**; the code has `agent/taint.py`, `tools/web.py` and `tools/egress.py`, and the server's routes are flat files (`routes.py`, `threads.py`, `csvroutes.py`, `mcp.py`), not `routes/{openai,...}.py`. The label file is `evals/labels/labels.jsonl`, not `evals/datasets/human_labels.jsonl`. Proposed: refresh the layout block.
3. **§8.1 G3** could say that raw HTML anchors and `[x][y](url)` are covered (D-103), and that an address a source names is kept by design (the "exposure" the injection report counts).
4. **§11 S5.5/S6.1 ✍️ cells**: the README results section and the MCP tool were written by Claude under D-097 and are on the ✍️ backlog for you to redo.
5. **STATUS.md's header and first paragraph were stale since 2026-10-17**; the new first paragraph under "Next up" is the authoritative one. (Edited here: notes, not the plan.)

## What is yours (the end of the run, D-097)

1. **Sign in to GitHub in the built-in browser**, then open the pull requests **in this order, each with base `v2`** (never `main`), wait for the three checks and squash-merge, deleting the branch after each (D-056): `step/GA-gate-a-review`, `step/S5.1-bench-sweep`, `step/S5.2-answer-eval`, `step/C6-csv-followups`, `step/S5.3-web-taint`, `step/S5.4-injection-eval`, `step/S5.5-writeup`, `step/GB-gate-b-review`, `step/S6.1-mcp`, `step/GC-gate-c-review`. (Each branch was cut from the one before; after a squash merge of one, the next one's pull request shows only its own commit once GitHub re-bases the comparison, or merge them in order and resolve the repeated history if GitHub does not.)
2. **The judge:** label about 60 answers on the Labels page and read κ (or pull `qwen3:8b` first). Until then it supports no claim.
3. **Your own golden questions and CSV items**; reword C6's F-006 and F-010.
4. **The decisions listed in the Gate B log** (the reranker, the 400 ms target, q8_0, top-k, the embedder, a real web backend).
5. **Connect a client** (`claude mcp add ...`, README) and say whether the two tools are what you want.
6. **PLAN rev 23** (the five items above), if you agree.
7. **Older results hold CPU-gate process names** (S2.6, committed before S5.1 found the problem): you may want them scrubbed.

## State at the end of the run
- The server and the UI are stopped; the scratch data folders (`scratch/s53-data`, `scratch/gate-a-*`) are ignored by git; **no model is loaded**; the no-run policy is back in force for the next chat (the memory file says so).
- Models used: `gemma4:12b`, `qwen2.5-coder:7b`, `nomic-embed-text`, the MiniLM reranker, `gemma3:4b`/`qwen2.5-coder:3b` only in the early proto-profile runs. Nothing over 12B; nothing pulled; one PyPI package added (`mcp`, dev only) plus its four dependencies.

## Slips (told straight)
- Listed in the run report: a read-only `ls` at the start (no-run policy then in force), a `Set-Content` BOM, the `@(@())` flattening, a few no-op edits.

## Handoff
The run is over. The next chat reads STATUS.md, this file and the run report, and starts with whichever decision you take first.
