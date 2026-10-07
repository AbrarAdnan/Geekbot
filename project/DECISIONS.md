# Geekbot v2 — Decision log

Every decision that shapes the project, with its reason and where it came from. New entries are appended; old entries are never edited. If a decision is reversed, a new entry supersedes it and names the old ID.

**Sources:**
- **user:** decided by the user.
- **review:** self-review of the code (PLAN.md §5).
- **CGR-1 / CGR-2:** ChatGPT review #1 / #2 (`project/reviews/`).
- **claude:** proposed by Claude and accepted with the approved plan.

| ID | Date | Decision | Why | Source |
|---|---|---|---|---|
| D-001 | 2026-10-07 | Rebuild v1 as v2 instead of patching it | v1 has verified correctness bugs (Chroma `include=["ids"]`, re-embedding on every rerun, hard-coded web search, stateless chat) and uses removed LangChain APIs | review |
| D-002 | 2026-10-07 | **Stack: LangChain 1.x + LangGraph**. Direct `chromadb` / `sqlite3` clients where control matters | User choice over a lean ollama + chromadb core. LangChain is used where it adds value (models, splitters, messages, structured output, graph) | user |
| D-003 | 2026-10-07 | **Thesis = three pillars:** retrieval quality, inference cost, untrusted-content security | Gives each technology a job; anything else is parked | CGR-1 #20 |
| D-004 | 2026-10-07 | **Gates A/B/C** with definitions of done; an explicit Parked list (Docker, OpenTelemetry/Phoenix, multimodal, folder watch, time-travel, LLM contextual retrieval, `Send` multi-hop, nightly GPU CI) | Prevents scope explosion | CGR-1 #13, #19 |
| D-005 | 2026-10-07 | Explicit `StateGraph`; **no ReAct / `create_agent` loop** | More reliable with local models; enables tool isolation (G1). LangGraph is justified by state and conditional transitions, not "agentic" for its own sake | CGR-1 #10, claude |
| D-006 | 2026-10-07 | **Exception-driven routing:** zero LLM calls before retrieval on turn 1; one structured `condense` call on follow-ups; route returned by the same call when optional routes exist | Speed; a planner call is not needed for the default path | CGR-1 #11 |
| D-007 | 2026-10-07 | **Default chat model `gemma4:12b`; no model over 12B anywhere** | User requirement: speed on a 10 GB RTX 3080 | user |
| D-008 | 2026-10-07 | **Speed is a hard requirement.** Latency SLOs are targets until measured: retrieval + rerank p95 ≤ 400 ms; TTFT p50 ≤ 2.5 s / p95 ≤ 4 s; follow-up TTFT p95 ≤ 5 s; end-to-end p95 ≤ 15 s; 0 model loads per query | v1 answers can take about a minute | user, claude |
| D-009 | 2026-10-07 | **Embedder and reranker on CPU** (`OllamaEmbeddings(num_gpu=0)`, fastembed ONNX). gemma4 must be 100% GPU-resident. Thinking off by default, with a per-question toggle | The 7.6 GB model + KV + buffers barely fit next to the desktop's VRAM use; swaps or CPU offload destroy latency | claude |
| D-010 | 2026-10-07 | **Conda environment `geekbot`** (Python 3.12) + pyproject; Windows lock via `pip-compile`; Linux CI resolves fresh from ranges. No uv | User preference; 3.12 has the broadest wheel support for chromadb / onnxruntime / fastembed on Windows | user, claude |
| D-011 | 2026-10-07 | **No-run policy:** Claude never executes anything on this PC; the user runs every command | The PC also runs the user's CSV-query project | user |
| D-012 | 2026-10-07 | **Teacher/learner, step-by-step delivery** (~30 steps): lesson → design approval → build → user verifies → walkthrough + questions (+ optional "Your turn") → green light | The user wants to learn every component and review each feature | user |
| D-013 | 2026-10-07 | **One chat per task or phase.** Central notes are committed in-repo under `project/`; a root `CLAUDE.md` bootstraps every chat; `PLAN.md` changes only with user approval | Transparency and continuity across chats; reviewers can read the build log on GitHub | user |
| D-014 | 2026-10-07 | **Web search is opt-in (off by default), built in Gate B** together with the security work | The security pillar needs a real egress channel to defend; it resolves a contradiction in CGR-1 (which placed web in Gate C) | user |
| D-015 | 2026-10-07 | **Security eval and VRAM/latency bench are mandatory Gate B evidence** | Resolves the CGR-1 contradiction ("pick only 2" vs listing both as mandatory) | user |
| D-016 | 2026-10-07 | **CSV interim handling:** always index a schema card; embed row text only for files ≤2,000 rows while no tabular backend is registered | Keeps v1's ability to answer about CSVs until the CSV project is connected | user |
| D-017 | 2026-10-07 | **Gate C order:** MCP server first, then the CSV-project adapter (`TabularBackend` port + conformance suite) | MCP is small once the API exists; the CSV adapter is a proof of integration, not a feature build | user |
| D-018 | 2026-10-07 | **Judge:** a fixed non-Gemma model ≤12B (Qwen3 ~8B class, tag verified at S5.2), chosen before any generator run. Config rejects a judge from the generator's family. If κ < 0.6 against ~60 blind human labels, judge scores are excluded from headline claims | Avoids self-preference bias; respects the ≤12B cap | CGR-2 #3, user |
| D-019 | 2026-10-07 | **Storage invariant:** exactly one OS process owns SQLite/Chroma (writer lock + `workers=1` + import-linter) | Multi-process writers corrupt state | CGR-2 #1 |
| D-020 | 2026-10-07 | **Blocking-I/O rule:** sync SQLite/Chroma/parsing/rerank never run on the event loop (writer thread, thread pool, process pool), verified by a heartbeat test | An "async" server that blocks is worse than a sync one | CGR-2 #2 |
| D-021 | 2026-10-07 | **Chunk identity vs fingerprint:** `chunk_id = sha256(doc_id ‖ content_hash ‖ occurrence)`; position is metadata only | Fixes the rev-2 collision for duplicate chunks; position-based IDs would break incremental re-embedding | CGR-1 #5, claude |
| D-022 | 2026-10-07 | **Outbox consistency:** SQLite is the source of truth; Chroma is a derived index; `doctor --repair` converges them | Crash recovery story across two stores | claude |
| D-023 | 2026-10-07 | **Taint model:** labels in graph state (never on strings); `packing.py` is the only labeling point; union propagation; monotonic; client-supplied history = `UNKNOWN`; gated egress needs user approval (HITL `interrupt()`) | Prevents tool- and render-based exfiltration by construction | CGR-2 #5, claude |
| D-024 | 2026-10-07 | **Output sanitization (G3) in Gate A:** strip markdown/HTML images; allow only links to URLs among the sources | Markdown-image exfiltration needs no tool call | claude |
| D-025 | 2026-10-07 | **Claims discipline:** guarantees and empirical results reported separately; BEIR = standardized retrieval evidence, golden corpus = system-specific behavioral evaluation; "significant" only with Δ + 95% CI | Interview-proof credibility | CGR-2 #4, #6 |
| D-026 | 2026-10-07 | **Ablation:** one variable per rung (R0–R7) in a pre-registered order + leave-one-out; primary metrics pre-registered (nDCG@10 SciFact / NFCorpus, Recall@5 golden) with Holm–Bonferroni; latency recorded per rung | Attributable gains; no confounds | CGR-2 #7, claude |
| D-027 | 2026-10-07 | **Phase 2 "first evidence" checkpoint:** review the ablation table + v1 latency baseline before building the graph/UI | Avoids building infrastructure before producing evidence | CGR-2 (final remark) |
| D-028 | 2026-10-07 | **Benchmarks only on an idle GPU:** `bench` detects other GPU processes and refuses to record official numbers | The PC is shared with the CSV project | claude |
| D-029 | 2026-10-07 | v1 code stays untouched and runnable until S4.4; tag `v1.0` preserves it; the v1 → v2 story is documented in `docs/evolution.md` | Honest evolution narrative + a measurable baseline | CGR-1 #23 |
