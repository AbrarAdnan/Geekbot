# Lesson notes — your Geekbot v2 course

Every step starts with a lesson. Each note covers:
- the concepts (what, why, trade-offs, pitfalls)
- how they map to Geekbot's code
- 3–5 interview-style questions with short model answers

Together they form a study guide for the whole project. Notes are written as each step runs; this index lists what's planned.

| # | Topic | Step | Status |
|---|---|---|---|
| 01 | Python packaging: src layout, pyproject, entry points, conda vs pip, lockfiles, linters/type checkers | S0.1 | planned |
| 02 | 12-factor configuration, pydantic-settings, logging, typed CLIs | S0.2 | planned |
| 03 | How Ollama loads models: keep_alive, residency, VRAM math (weights + KV cache + buffers), gemma4's sliding-window KV | S0.3 | planned |
| 04 | ~~Measuring latency properly~~ (S0.4 was dropped; the topic moves into lesson 20) | ~~S0.4~~ | dropped (D-031) |
| 05 | CI pipelines: matrix builds, lock vs fresh resolution | S0.5 | planned |
| 06 | Document parsing: PDFs, office files, page metadata, tabular schema cards | S1.1 | planned |
| 07 | Chunking: tokens vs characters, structure-aware splitting, contextual headers | S1.2 | planned |
| 08 | Content addressing: identity vs fingerprint, idempotency, property-based testing | S1.3 | planned |
| 09 | SQLite internals for RAG: WAL, FTS5, BM25, migrations, single-writer | S1.4 | planned |
| 10 | Embeddings: vector spaces, cosine vs L2, HNSW, task prefixes, embedding caches, fingerprints | S1.5 | planned |
| 11 | The outbox pattern: consistency across two stores, crash recovery | S1.6 | planned |
| 12 | Dense retrieval and asymmetric search | S2.1 | planned |
| 13 | Lexical vs semantic retrieval; Reciprocal Rank Fusion | S2.2 | planned |
| 14 | Bi-encoders vs cross-encoders; rerank depth vs latency | S2.3 | planned |
| 15 | Building eval sets: BEIR, qrels, golden sets, unanswerable questions | S2.4 | planned |
| 16 | IR metrics (nDCG, Recall, MRR) and statistics (bootstrap CI, permutation test, Holm) | S2.5 | planned |
| 17 | Ablation design: confounds, leave-one-out, threshold calibration | S2.6 | planned |
| 18 | LangGraph fundamentals: state, nodes, edges, compile; ChatOllama options | S3.1 | planned |
| 19 | Prompting for RAG: token budgets, prefix-KV reuse, datamarking, citations | S3.2 | planned |
| 20 | Streaming, TTFT, measuring latency properly (warm vs cold, percentiles), thinking models | S3.3 | planned |
| 21 | Conversational RAG: checkpointers, query condensation, structured output | S3.4 | planned |
| 22 | Abstention, citation relevance vs entailment, markdown-image exfiltration | S3.5 | planned |
| 23 | Graph parallelism, retries, stream events, SLO smoke tests | S3.6 | planned |
| 24 | ASGI and async: lifespan, event-loop blocking, single-owner processes | S4.1 | planned |
| 25 | SSE and API compatibility: stateless vs stateful chat APIs | S4.2 | planned |
| 26 | Threat modeling basics: auth, CORS, path traversal, upload hygiene | S4.3 | planned |
| 27 | Streamlit's execution model (the root of v1's re-index bug) | S4.4 | planned |
| 28 | Performance engineering: KV-cache quantization, flash attention, batch size vs VRAM | S5.1 | planned |
| 29 | LLM-as-judge, self-preference bias, Cohen's κ | S5.2 | planned |
| 30 | Taint tracking, HITL interrupts, egress control | S5.3 | planned |
| 31 | Prompt-injection attack classes and how to measure defenses | S5.4 | planned |
| 32 | Technical writing: ADRs, presenting results honestly | S5.5 | planned |
| 33 | The MCP protocol: tools, resources, transports | S6.1 | planned |
| 34 | Ports and adapters; contract testing; plugin discovery | S6.2 | planned |
