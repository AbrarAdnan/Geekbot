# Lesson notes — your Geekbot v2 course

Every step starts with a lesson. Each note covers:
- the concepts (what, why, trade-offs, pitfalls)
- how they map to Geekbot's code
- 3–5 interview-style questions with short model answers

Together they form a study guide for the whole project. Notes are written as each step runs; this index lists what's planned.

**Deferred (D-047, fast-prototyping mode):** every row below marked "planned" is deferred. Lessons 01–06 stay as written. A lesson is written when the user asks for it. Lessons are numbered in the order the steps happen (renumbered at plan rev 11, when the CSV track C1–C6 was added).

| # | Topic | Step | Status |
|---|---|---|---|
| 01 | [Python packaging: src layout, pyproject, entry points, conda vs pip, lockfiles, linters/type checkers](01-packaging.md) | S0.1 | written |
| 02 | [12-factor configuration, pydantic-settings, logging, typed CLIs](02-config-cli.md) | S0.2 | written |
| 03 | [How Ollama loads models: keep_alive, residency, VRAM math (weights + KV cache + buffers), gemma4's sliding-window KV](03-ollama-gpu.md) | S0.3 | written |
| 04 | [CI pipelines: matrix builds, lock vs fresh resolution; branches and pull requests](04-ci.md) | S0.5 | written |
| 05 | [Working in a codebase you didn't just write: vendoring vs packages vs git subtree, provenance, gradual typing; a guided tour of `csv_analyst`](05-vendoring.md) (characterization tests: not used in C1) | C1 | written |
| 06 | [Ports and adapters: dependency injection through a Protocol, connecting two settings systems, Ollama JSON mode and structured output](06-ports-adapters.md) | C2 | written |
| 07 | Two models, one GPU: explicit swaps, measuring swap cost, reproduction runs and run-to-run noise | C3 | planned |
| 08 | Document parsing: PDFs, office files, page metadata, tabular schema cards | S1.1 | planned |
| 09 | Chunking: tokens vs characters, structure-aware splitting, contextual headers | S1.2 | planned |
| 10 | Content addressing: identity vs fingerprint, idempotency, property-based testing | S1.3 | planned |
| 11 | SQLite internals for RAG: WAL, FTS5, BM25, migrations, single-writer | S1.4 | planned |
| 12 | Embeddings: vector spaces, cosine vs L2, HNSW, task prefixes, embedding caches, fingerprints | S1.5 | planned |
| 13 | The outbox pattern: consistency across two stores, crash recovery | S1.6 | planned |
| 14 | Dense retrieval and asymmetric search | S2.1 | planned |
| 15 | Lexical vs semantic retrieval; Reciprocal Rank Fusion | S2.2 | planned |
| 16 | Bi-encoders vs cross-encoders; rerank depth vs latency | S2.3 | planned |
| 17 | Building eval sets: BEIR, qrels, golden sets, unanswerable questions | S2.4 | planned |
| 18 | IR metrics (nDCG, Recall, MRR) and statistics (bootstrap CI, permutation test, Holm) | S2.5 | planned |
| 19 | Ablation design: confounds, leave-one-out, threshold calibration | S2.6 | planned |
| 20 | LangGraph fundamentals: state, nodes, edges, compile; ChatOllama options | S3.1 | planned |
| 21 | Prompting for RAG: token budgets, prefix-KV reuse, datamarking, citations | S3.2 | planned |
| 22 | Streaming, TTFT, measuring latency properly (warm vs cold, percentiles), thinking models | S3.3 | planned |
| 23 | Conversational RAG: checkpointers, query condensation, structured output | S3.4 | planned |
| 24 | Mode routing: deterministic vs LLM routing, per-mode state, rewriting follow-ups for a tool with no memory | C4 | planned |
| 25 | Abstention, citation relevance vs entailment, markdown-image exfiltration | S3.5 | planned |
| 26 | Graph parallelism, retries, stream events, SLO smoke tests | S3.6 | planned |
| 27 | ASGI and async: lifespan, event-loop blocking, single-owner processes | S4.1 | planned |
| 28 | SSE and API compatibility: stateless vs stateful chat APIs | S4.2 | planned |
| 29 | Threat modeling basics: auth, CORS, path traversal, upload hygiene | S4.3 | planned |
| 30 | Streamlit's execution model (the root of v1's re-index bug) | S4.4 | planned |
| 31 | Stateful sessions in a single-owner server: uploads, in-memory sessions, eviction, memory limits | C5 | planned |
| 32 | Performance engineering: KV-cache quantization, flash attention, batch size vs VRAM | S5.1 | planned |
| 33 | LLM-as-judge, self-preference bias, Cohen's κ | S5.2 | planned |
| 34 | Evaluating a stateful tool mode; prompt injection through data | C6 | planned |
| 35 | Taint tracking, HITL interrupts, egress control | S5.3 | planned |
| 36 | Prompt-injection attack classes and how to measure defenses | S5.4 | planned |
| 37 | Technical writing: ADRs, presenting results honestly | S5.5 | planned |
| 38 | The MCP protocol: tools, resources, transports | S6.1 | planned |

**Dropped:**
- The S0.4 lesson (measuring latency) moved into lesson 22 (S3.3), because S0.4 was dropped (D-031).
- The old S6.2 lesson (ports and adapters) moved into lesson 06 (C2), because S6.2 was dropped (D-038).
