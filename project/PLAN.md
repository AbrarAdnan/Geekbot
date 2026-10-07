# Geekbot v2 — implementation plan (rev 9)

> **The measured results are the product; the architecture exists to make them credible.**
> (This line also opens the v2 README.)

## 1. Context

Geekbot (Sept 2025) is a 4-file Streamlit + LangChain 0.3 + Chroma + Ollama RAG chatbot. It has verified correctness bugs (§5), uses removed or deprecated LangChain APIs, makes README claims the code doesn't meet, and its answers can take around a minute (user-reported, never measured). v2 rebuilds it into a fast, measurable project the user can put on a resume and defend in a deep technical interview. **They will also learn every component as we build it together.**

**User decisions**
- **Stack:** LangChain 1.x + LangGraph.
- **Model and speed:** `gemma4:12b` is the default chat model; no model over 12B; speed is a hard requirement.
- **Environment:** a **conda** environment (no uv).
- **No running:** **I don't execute anything on this PC** (it runs the user's CSV-query project). The user runs every command.
- **Pace:** **step-by-step delivery**. Each feature is reviewed and green-lit before the next, in a **teacher/learner** format.
- **Multi-chat:** **each task or phase runs in its own chat.** This plan lives in the repo at `project/PLAN.md`, and all notes live in a **committed central directory `project/`**, bootstrapped for every chat by a root `CLAUDE.md` (§2.1).
- **Web search:** opt-in, built in Gate B with the security work.
- **CSV handling before the CSV project plugs in:** schema card, plus row text for small files.
- **Gate C order:** MCP, then the CSV adapter.

**This machine (measured once, before the no-run rule)**
- **Hardware:** RTX 3080 with 10,240 MiB VRAM (1,311 MiB in use at check time); ~48 GB RAM.
- **Python:** system 3.13.9, untouched.
- **Ollama:** 0.34.2.
- **`gemma4:12b`:**
  - **Size:** 11.9B, Q4_K_M, 7.6 GB file.
  - **Capabilities:** tools, thinking, vision, audio.
  - **Attention layout:** 48 layers. 40 sliding-window layers (window 1,024; 8 KV heads × 256) and 8 global layers (1 KV head × 512).
  - **KV cache estimate:** ≈448 MiB (≈0.44 GiB) at 8k context (f16): 40 layers × 1,024 tokens × 8 KiB + 8 layers × 8,192 tokens × 2 KiB.
- **Note:** my earlier `ollama list` check auto-started the Ollama desktop app. It was idle with no models loaded.

## 2. How we work together (applies to every step in §11)

**No-run policy.** I only read and write files in this repo.
- I run no shell commands, installs, tests, Ollama calls, benchmarks or git commands. The exception is a step where you explicitly allow a specific command.
- For each step I give you the exact commands. You run them in the `geekbot` conda env when your PC is free and paste back the output.

**Per-step loop (one feature at a time):**
1. **Lesson:** the concepts behind the feature (what, why, trade-offs, pitfalls), mapped to our design. It's saved as a study note at `project/learn/NN-<topic>.md`, so you end up with a full course plus interview prep.
2. **Design:** the files, interfaces and tests for this step. You approve or adjust it *before* I write code.
3. **Build:** I write a small, reviewable diff (code + tests + docstrings). Nothing is run.
4. **You verify:** you run the listed commands and share the output. I fix anything that fails.
5. **Walkthrough and check:** I walk through the diff, then ask 3–5 understanding / interview questions. An optional **"Your turn"** exercise has you implement a small piece yourself against tests I provide (e.g. RRF, nDCG), so the resume claims are genuinely yours.
6. **Green light:** you approve, and I give you the commit command and message. Then we move to the next step.

**GPU etiquette:** steps that load models (doctor warm-up, ask, bench, eval) say so explicitly. Benchmarks are only valid on an idle GPU, so `bench` detects other GPU processes (e.g. your CSV project) and refuses to record official numbers while they run.

### 2.1 Central notes directory and multi-chat protocol
Everything below is committed with the code, so any chat (or a reviewer such as ChatGPT or Codex reading GitHub) sees the same state.

```
CLAUDE.md                   auto-loaded by every Claude Code chat in this repo: the rules + the session protocol
                            (short; points into project/, never duplicates the plan)
project/
  PLAN.md                   this plan: the single source of truth; changes only with your approval
  STATUS.md                 step tracker (step · status · chat date · commit · log link) + a "Next up" handoff
  DECISIONS.md              dated decision log: every decision so far and every later one, with the reason
  steps/S0.1-skeleton.md    one log per step: goal, approved design, files changed, commands you ran + results,
                            review Q&A, open issues, handoff to the next chat
  learn/01-packaging.md     lesson notes: your study guide + interview prep (index in learn/README.md)
  reviews/                  external reviews (the two ChatGPT reviews, verbatim) + how each point was resolved
```
- **Status values:** `todo → design → build → verify → green-lit`, or `blocked`.
- **Session start:**
  1. Read `CLAUDE.md`, `PLAN.md`, `STATUS.md`, `DECISIONS.md` and the previous step's log.
  2. Confirm with you which step(s) this chat covers.
  3. Restate the no-run policy.
- **During:** follow the §2 loop. **`PLAN.md` is never changed silently.** A change is proposed, you approve it, and it's recorded in `DECISIONS.md` and in the plan's revision history.
- **Session end:**
  1. Update the step log, `STATUS.md` (including the "Next up" handoff) and `DECISIONS.md`.
  2. Give you the commit command; notes are committed together with that step's code.
- **Scope of a chat:** one step or a whole phase, with the same protocol either way.
- **Conflicts:** if `STATUS.md`, the notes and the code disagree, the chat stops and asks you before doing anything.
- **Memory:** I also save the standing rules (no-run, conda, teaching format, notes location) to Claude's project memory as a backstop. The repo files stay authoritative.

## 3. Thesis — three pillars

> **A local RAG engine that measures retrieval quality, controls inference cost, and treats retrieved content as untrusted data.**

| Pillar | Contents | README evidence |
|---|---|---|
| **Retrieval** | dense + BM25 → RRF → cross-encoder rerank → calibrated abstention | Δ nDCG@10 / Recall@5 with 95% CI and p, from a one-variable ablation; abstention accuracy |
| **Systems** | 12B model 100% GPU-resident on a 10 GB consumer GPU, latency SLOs, no model swaps, incremental ingestion, single storage owner, crash recovery, non-blocking async | latency vs the SLOs and vs the one-time v1 reference run (same model), tok/s, VRAM budget table, re-ingest cost, repair test |
| **Security** | untrusted-source model, tool isolation, formal taint-gated egress, output sanitization, path sandbox | **Architectural guarantees** and, separately, the **empirical result** (*k/N planted attacks bypassed a control*) plus the hijack rate |

## 4. Revision history (condensed)

- **rev 1–2:** feature checklist → thesis, BEIR, single writer, outbox, security. Too much scope, and an ID collision bug.
- **rev 3:** ChatGPT review #1 (gates, ID fix, statistics wording, image/history attack defenses).
- **rev 4:** ChatGPT review #2 (storage invariant, blocking-I/O rule, fixed judge, formal taint, one-variable ablation, risk register).
- **rev 5:** `gemma4:12b` fixed, measured 10 GB VRAM budget + latency SLOs, CPU embedder/reranker, thinking off by default.
- **rev 6:**
  - **Conda instead of uv**
  - **No-run policy** (§2)
  - **The milestones become ~30 small, individually reviewed teaching steps** (§11)
  - Benchmark validity on a shared PC added as a risk
- **rev 7:**
  - The plan moves into the repo as `project/PLAN.md`
  - One chat per task or phase
  - A committed central notes directory + root `CLAUDE.md` session protocol (§2.1)
  - New step **P0** bootstraps it
- **rev 8** (P0 verification, before the Codex review):
  - v1 is no longer run or benchmarked: S0.4 and all v1 → v2 comparisons are dropped (D-031)
  - Blank templates for the test documents and questions under `evals/` (D-032)
  - Eval-design questions deferred to S2.4–S2.6 (D-033)
  - Factual corrections to §5, the KV estimate and command formatting (D-030)
- **rev 9** (P0, user decision): a **one-time v1 reference run** at checkpoint CP2: the real v1 from tag `v1.0`, in its own `geekbot-v1` conda env, with `gemma4:12b`, measuring speed and search quality on the golden questions. Rung R0 stays as the ablation's starting point (D-034, partly reverses D-031)

## 5. v1 findings (verified; all fixed by v2)

| # | Issue | Where |
|---|---|---|
| 1 | **Every Streamlit rerun re-embeds every uploaded file**: the `file_uploader` block runs on each interaction, including chat submits and `st.rerun()` | app.py:221-228 |
| 2 | `get(include=["ids"])` raises `ValueError` in chromadb (verified in 1.0.20 source) and is swallowed. As a result:<br>• the existence check always fails, so everything is re-embedded (LangChain's upsert hides the duplicates)<br>• **delete-by-source silently does nothing**<br>• "Show indexed sources" is always empty | storage_utils.py:82,116,150 |
| 3 | `use_web=True` is hard-coded: 3 *sequential* DuckDuckGo calls per question, contradicting the privacy claims and adding latency | app.py:264, rag_pipeline.py:125-130 |
| 4 | docx/xlsx/md/pptx/html loaders need `unstructured` (not in requirements); `JSONLoader` needs `jq_schema` + `jq`. They crash, and one bad file aborts a whole folder | app.py:77-112 |
| 5 | Chat history is stored but never sent to the model, so follow-ups fail | rag_pipeline.py:141 |
| 6 | Sources are wiped by `st.rerun()` (not persisted) and print as raw dicts | app.py:267-273 |
| 7 | "Rerank" re-embeds the retrieved chunks and re-scores them with the same bi-encoder: no new relevance signal; it only merges the multi-query results. "Multi-query" is 5 string templates. Together: up to 21 embedding calls per question (5 searches + 1 query + up to 15 chunks) | rag_pipeline.py:43-62,78-97 |
| 8 | `nomic-embed-text` is used without its required `search_query:` / `search_document:` prefixes | get_embedding_function.py |
| 9 | No `num_ctx`, no token budget. With web results the prompt grows to an estimated ~3k tokens (never measured); anything over the context window is truncated from the **front**, where the local document context sits | rag_pipeline.py:141-147 |
| 10 | Deprecated `langchain_community` `Ollama` LLM with no reasoning control: thinking-capable models can think at length and emit `<think>…</think>` into answers. No streaming | rag_pipeline.py:147 |
| 11 | PDF pages are 0-indexed; `line` is never set (dead branch); a small model formats citations itself | rag_pipeline.py:110-117 |
| 12 | Untrusted web/doc text goes into the prompt with only plain separators (`---(Local)---`, `---(Web)---`); nothing marks it as untrusted data. Web results aren't de-duplicated | rag_pipeline.py:117-119,132-139 |
| 13 | An empty "Index folder" input resolves to the cwd, so one click tries to index the whole repo, including the virtualenv (the README creates `venv/` inside the repo). With #4, the walk likely crashes before anything is embedded. `.gitignore` misses `venv/` | app.py:230-233, README |
| 14 | Same-name uploads overwrite each other; directory prefix match (`/a/b` matches `/a/bc`); default L2 space (probably harmless: if Ollama returns unit-length vectors, L2 and cosine rank identically; checked at S1.5) | app.py:223, storage_utils.py:128 |
| 15 | Dead branch `loader(...) if not callable else loader(...)`; unused `CHUNK_SIZE` and `CHUNK_OVERLAP`; the `RESET_MARKER` hack works around Windows file locks held by Chroma clients that are opened on every call and never closed | app.py:27-46,98,192-194 |
| 16 | Raw `pip freeze` requirements missing actual deps; no tests, CI, config, license, logging or packaging; hard-coded models | repo |
| 17 | **Latency compounds** (3 sequential web searches, up to 21 embedding calls, default 5-minute `keep_alive` cold reloads, no thinking control, no streaming; with a chat model much larger than the default `qwen3:0.6b`, possibly embedder ↔ chat-model swapping or CPU offload on a 10 GB card). These are *hypotheses*; the one-time v1 reference run at CP2 measures v1's speed with `gemma4:12b` (D-034) | rag_pipeline.py |

## 6. Architecture

```
Streamlit UI ──┐                                    ┌─ SQLite (WAL) = source of truth: documents, chunks,
CLI (client) ──┼─► geekbot.client (httpx) ─► FastAPI │   chunks_fts (FTS5), jobs, threads, labels, feedback
MCP stdio proxy┘   (Gate C)                  server  ├─ Chroma = derived dense index (cosine; vectors + doc_id)
OpenAI SDK / Open WebUI ──────────────────►  (ONE   ├─ Ollama GPU: gemma4:12b (100% resident, warm, keep_alive)
Claude Code / Cursor ── /mcp (Gate C) ────►  storage├─ Ollama CPU: embedder (num_gpu=0) ── 0 VRAM
                                              owner) └─ fastembed ONNX cross-encoder (CPU) ─ 0 VRAM
```

### 6.1 Environment and packaging (conda)
- **`environment.yml`** (conda-forge): `python=3.12` (the broadest wheel support for chromadb, onnxruntime and fastembed on Windows), `pip`, `pip-tools`.
  - **Bootstrap (S0.1):** the pip section installs `-e .[dev,eval]` from the pyproject ranges.
  - You then run `pip-compile --extra dev --extra eval -o requirements/lock-win.txt pyproject.toml`.
  - From then on, the pip section installs `-r requirements/lock-win.txt` followed by `-e .`. Re-creating the env reproduces the exact versions.
- **`pyproject.toml`:** the src layout, version-ranged direct dependencies, extras `[dev]` and `[eval]`, the `geekbot` console script, and the ruff / mypy / pytest / import-linter config.
- **Lock:** `requirements/lock-win.txt` is generated by `pip-compile` (pip-tools) on your Windows machine. That makes it platform-specific, which is fine because it's for your dev box.
- **CI** (GitHub Actions + `conda-incubator/setup-miniconda`) runs on GitHub's machines, not yours:
  - **windows / py3.12:** installs from the lock.
  - **ubuntu / py3.12 and 3.13:** fresh resolution from the pyproject ranges, which catches upstream breakage early.
- Every eval/bench result records the exact package versions (`importlib.metadata`), Ollama version, model digests and git SHA.

### 6.2 Storage invariant (enforced)
> **Exactly one OS process owns SQLite/Chroma.** Every other component is an HTTP client.

- `geekbot serve` and `geekbot up` call `uvicorn.run(app, workers=1)`, and no `--workers` option is exposed.
- An exclusive writer lock (`data/.storage.lock`, via `filelock`) is taken before any store is opened. A second storage-owning process fails fast.
- `--embedded` mode (tests and offline eval only) takes the same lock.
- The UI and CLI never import the storage modules (import-linter contract in CI).

### 6.3 Concurrency and blocking-I/O rule
> **No synchronous I/O or CPU-bound work runs on the event loop.**

| Work | Where it runs |
|---|---|
| SQLite writes | one dedicated writer thread with a queue |
| SQLite reads / FTS5 | read-connection pool via `anyio.to_thread.run_sync` (WAL) |
| chromadb calls | `anyio.to_thread.run_sync` |
| file parsing | `ProcessPoolExecutor` (spawn-safe; never touches storage) |
| fastembed rerank | worker thread |
| Ollama | native async |
| checkpoints | `AsyncSqliteSaver` |

Verified by a heartbeat test: `/healthz` p99 < 200 ms during a concurrent ingest + streaming answer, with asyncio debug `slow_callback_duration=0.1`.

### 6.4 Consistency, identity, fingerprints
- **Outbox:** each document goes `pending → embedding → ready | failed`.
  1. One SQLite transaction writes the chunk rows and FTS entries.
  2. Vectors are upserted to Chroma.
  3. The state is flipped to `ready`.
  
  Retrieval only returns `ready` chunks. `geekbot doctor --repair` converges the two stores.
- **Identity vs fingerprint:**
  - `doc_id = sha256(normalized_abs_path)[:16]`: which source.
  - `file_hash`: skip unchanged files.
  - `content_hash = sha256(normalized_chunk_text)`: did the content change? Also the embedding-cache key, together with the embedder fingerprint.
  - `occurrence`: the nth repeat of the same `content_hash` within a doc.
  - **`chunk_id = sha256(doc_id ‖ content_hash ‖ occurrence)[:24]`**: stable identity.
  - `ordinal` / `char_start` / `char_end` / `page` / `section_path` are position metadata only, never identity.
  - Re-ingest = set diff on `chunk_id`.
- **Embedder fingerprint** (model + digest + prefixes + dimension) is stored in the collection metadata. A mismatch blocks queries until a re-index.
- **Where LangChain is used:** where it adds value: `ChatOllama`, `OllamaEmbeddings`, splitters, messages, structured output, LangGraph.
- **Where direct clients are used:** where control matters: `chromadb` (precomputed, cached vectors) and `sqlite3` FTS5.

### 6.5 Latency and VRAM design

**VRAM budget on the RTX 3080** (estimates; verified with `ollama ps` and `nvidia-smi`):

| Item | Budget |
|---|---|
| Total | 10,240 MiB |
| Desktop / other apps (live value from `doctor`) | ~1,300 MiB |
| `gemma4:12b` weights (Q4_K_M, incl. vision/audio towers) | ~7,250 MiB |
| KV cache @ 8k, f16 (estimate, §1) | ~448 MiB |
| Compute buffers (`num_batch` lever) | measured |
| Embedder | 0 (CPU, `OllamaEmbeddings(num_gpu=0)`) |
| Reranker | 0 (CPU ONNX) |
| **Headroom** | **tight: measured, never assumed** |

**Residency rules:**
- **R-1:** the chat model is **100% GPU** (`ollama ps`). `doctor` and `bench` fail otherwise.
- **R-2:** **no model loads during a query:** warm-up at server start, a long `keep_alive`, `OLLAMA_MAX_LOADED_MODELS ≥ 2`, and load events logged in `RunStats`.
- **R-3:** `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_FLASH_ATTENTION=1`; KV cache type chosen by the sweep.

**Speed design:**
- Thinking is off by default, with a per-question "think harder" toggle.
- One model serves every LLM call.
- `condense` runs only on follow-ups (`num_predict` ≤ 96, structured output).
- Packed context ≈3k tokens; `num_ctx` 8,192.
- Prompt layout for Ollama prefix-KV reuse: a static system prompt → history → sources → question.
- `num_predict` cap (512); streaming from the first token; `cite_check` after the stream; web only as an opt-in fallback.

**Latency SLOs** (targets until measured; warm server, local KB, no web):

| Metric | Target |
|---|---|
| Retrieval + rerank p95 | ≤ 400 ms |
| First-turn TTFT p50 / p95 | ≤ 2.5 s / ≤ 4 s |
| Follow-up TTFT p95 | ≤ 5 s |
| End-to-end p95 (≤ 400 tokens) | ≤ 15 s |
| Model loads during queries | 0 |

**If an SLO is missed,** apply these levers in order:
1. Restore residency: `num_batch` 256, q8_0 KV, 4k context, close GPU-heavy apps.
2. Shrink top-k.
3. Lower `num_predict`.
4. Reduce rerank depth.
5. Only then, ask the user about a smaller model.

### 6.6 Repo layout
```
src/geekbot/
  config.py · domain.py (Chunk, ScoredChunk, Citation, RunStats, Label)
  ingest/    loaders/{pdf,office,text,notebook,tabular}.py · chunking.py · ids.py · pipeline.py · jobs.py
  index/     sqlite.py (schema+migrations, writer thread) · dense.py · consistency.py · lock.py
  retrieval/ lexical.py · dense.py · fusion.py · rerank.py · retriever.py (rungs R0–R7) · calibrate.py
  llm/       models.py (factories, warm-up, keep_alive) · vram.py (nvidia-smi / ollama ps parsing)
  agent/     state.py · runtime.py · prompts/*.md · packing.py · citations.py · sanitize.py · taint.py · nodes/* · graph.py
  tools/     web.py (Gate B) · tabular.py (Gate C)
  server/    app.py · routes/{openai,kb,threads,jobs,health}.py · security.py · mcp.py (Gate C)
  client/sdk.py · ui/ (chat · knowledge · settings · evals+labeling) · cli.py
  evals/     datasets.py · metrics.py · stats.py · judge.py · injection.py · bench.py · report.py
  testing/tabular_contract.py (Gate C)
evals/ README.md · corpus/ · datasets/{golden,injection,human_labels}.jsonl + golden-questions.md · configs/*.toml · results/
docs/ architecture.md · evolution.md · security.md · performance.md · adr/ · plugins.md      (product docs)
project/ PLAN.md · STATUS.md · DECISIONS.md · steps/ · learn/ · reviews/                   (build process, §2.1)
tests/ unit/ · integration/ · live/
CLAUDE.md · environment.yml · pyproject.toml · requirements/lock-win.txt · LICENSE (MIT) · .github/workflows/ci.yml
```

## 7. LangGraph design (explicit state and conditional transitions; no ReAct loop)

```
START → condense? → [retrieve_dense ‖ retrieve_lexical] → fuse(RRF) → rerank → grade ─┬ ok ─────────► generate → sanitize → END ⇢ cite_check (after stream)
        (follow-ups only)                                                             ├ weak, web on ─► egress_gate ─► web ─► generate …
                                                                                      └ weak ─────────► abstain → END
```
- **Exception-driven routing:**
  - Turn 1 means zero LLM calls before retrieval.
  - Follow-ups make one structured `condense` call.
  - When optional routes exist, that same call also returns `route`.
- **Why LangGraph:** parallel branches, conditional transitions, the `AsyncSqliteSaver` checkpointer, `interrupt()` for egress approval, `get_stream_writer()` events, node `RetryPolicy`, and `context_schema`.
- **`generate`:** `ChatOllama(model="gemma4:12b", num_ctx, num_predict, keep_alive, temperature=0.2, reasoning=False)`. When thinking is toggled on, reasoning goes into `additional_kwargs["reasoning_content"]`, shown collapsed.
  - `packing.py` enforces the token budget and returns labels.
  - Sources are datamarked.
  - The model emits only `[n]`; the app validates and renders the citations.
- **`cite_check`:** a citation *relevance* check (cross-encoder), not entailment. Groundedness is measured offline by the judge.
- **Abstain:** the threshold is calibrated F1-optimally on the golden set.
- **API semantics:** `/v1/chat/completions` is stateless; the UI uses `/v1/threads/*`.

## 8. Security model (`docs/security.md`, ADR-0004)

### 8.1 Architectural guarantees (each with a dedicated test)
- **G1, tool isolation:** nodes that read untrusted text have no tools bound.
- **G2, taint-gated egress:** no egress with tainted arguments without explicit user approval of the exact arguments.
  - The single `EgressGate` wraps every `network=True` tool (registry test).
  - G2 guarantees no *unapproved* tainted egress.
- **G3, output sanitization** (Gate A): a streaming-safe filter strips images and drops links to URLs that aren't among the sources (UI, REST, SSE).
- **Perimeter:**
  - `127.0.0.1` bind; a non-local bind requires `GEEKBOT_API_KEY`; CORS allowlist.
  - `GEEKBOT_ALLOWED_ROOTS` path sandbox (no symlink escape, empty input rejected).
  - Size and page limits; `<sha>-<safe-name>` uploads.
  - MCP is read-only by default.
- **Not guaranteed:** answer manipulation by poisoned documents. That is measured (hijack rate), never claimed solved.

### 8.2 Taint model (bookkeeping from Phase 3, enforced in Gate B)
- **Labels:** `{USER, KB, WEB, TOOL, UNKNOWN}`. Tainted means the set intersects `{KB, WEB, TOOL, UNKNOWN}`.
- **Where taint lives:** in **graph state**, never on strings.
  - `state.labels: dict[message_id, frozenset[Label]]`, checkpointed.
  - `state.thread_taint`, monotonic.
- **Propagation rule:** a model call's output labels = the union of the labels of everything in its context window. `packing.py` is the **only** prompt assembler, so it is the only labeling point.
- **Per-step behaviour:**
  - `condense` outputs carry the history labels, so a tainted web query triggers `interrupt()` (approve / edit / deny).
  - Citations are labeled `KB`.
  - Streaming: labels are fixed before the call.
  - Serialization keeps labels as fields.
  - Client-supplied assistant messages on the stateless OpenAI endpoint are labeled `UNKNOWN`.
  - MCP outputs are labeled `KB`.
- **Monotonic, no declassification.** Covered by property tests, an `UNKNOWN`-labeling test and a registry test.

### 8.3 Injection evaluation (empirical; reported separately)
~30 planted docs/pages covering four vectors:
1. A web call carrying secret text.
2. Markdown-image exfiltration.
3. History laundering.
4. Citation forgery / hijack.

A simulated user denies every approval. The report gives bypasses k/N (target 0), gate triggers, and the hijack rate.

## 9. Evaluation methodology

- **Standardized retrieval evidence:** BEIR SciFact (5,183 docs / 300 test queries) and NFCorpus (3,633 / 323).
  - Downloaded at runtime with checksums; HF mirror as fallback.
  - Chunk hits are aggregated to doc level (max).
  - Wording: "standard BEIR datasets with standard metrics".
- **System-specific behavioral evaluation:** the golden corpus (~15 docs). ~100 items: ~50 single-hop, ~15 multi-hop, ~15 multi-turn, ~20 unanswerable. No general-superiority claims.
- **Measurement hierarchy:** retrieval metrics → deterministic answer checks (citation hit, abstention accuracy, must-include recall, invalid-citation rate) → human labels → calibrated judge (supporting only).
- **Primary metrics (pre-registered):** nDCG@10 on SciFact, nDCG@10 on NFCorpus, Recall@5 on golden, with Holm–Bonferroni across them. Everything else is descriptive.
- **Ablation, one variable per rung:** cumulative, in a fixed order, with the embedder held at `nomic-embed-text`, run on BEIR and golden, latency recorded per rung.

  | Rung | Single change |
  |---|---|
  | R0 | legacy v1 replica |
  | R1 | L2 → cosine space (expected Δ≈0 if vectors are unit-length; checked at S1.5) |
  | R2 | remove template multi-query (its re-sort was the merge step, so it goes too) |
  | R3 | + prefixes |
  | R4 | new chunking strategy |
  | R5 | + contextual header (BEIR: title) |
  | R6 | + BM25 via RRF (k=60) |
  | R7 | + rerank (depth tuned under the 400 ms SLO) |

  Plus leave-one-out from R7 (−BM25, −rerank, −header).
- **Statistics:** Δ first, then the paired-bootstrap 95% CI (10,000 resamples), then a paired sign-flip permutation p (10,000), Holm-adjusted. "Significant" is never used without Δ and the CI.
- **Judge:**
  - The generator is `gemma4:12b`.
  - The judge is fixed in advance: a **non-Gemma model ≤12B** (Qwen3 ~8B class, tag verified at step S5.2). Config rejects a judge from the generator's family.
  - Runs are phased by model (generate all, then judge all). Verdicts are cached.
- **Human labels:** ~60 answers, shuffled and labeled blind. If κ < 0.6, the judge is excluded from headline claims.
- **Bake-offs:**
  - **3 CPU embedders:** `nomic-embed-text` with prefixes, `qwen3-embedding:0.6b`, `embeddinggemma`. Quality + query ms + ingest chunks/s are reported.
  - **2 CPU rerankers:** MiniLM-L-6 (Apache-2.0) and bge-reranker-base (MIT). jina-v2 is excluded (CC-BY-NC).
- **gemma4 tuning sweep:** `num_ctx` {4k, 8k} × KV {f16, q8_0} × `num_batch` {256, 512} × top-k {3, 5}, plus thinking on/off as information. The default is the best quality that meets all SLOs at 100% residency.
- **One-time v1 reference run** (at checkpoint CP2, D-034):
  - The real v1 from tag `v1.0` (a `git worktree`, so S4.4's removal of the v1 files doesn't matter), in its own conda env `geekbot-v1` (Python 3.10 + v1's `requirements.txt`), because v1 needs LangChain 0.3.
  - `gemma4:12b` is set at runtime (v1's files stay untouched) and web search is off, matching v2's default, so speed differences come from the pipeline, not the model.
  - On the golden corpus and questions, it records speed per question and search quality (doc-level Recall@5). The search scores also check that rung R0 really replicates v1.
  - How v1 handles the formats it can't load (§5 #4) is settled when the run is designed.
- **Bench:**
  - `geekbot bench` records TTFT (first turn and follow-up), tok/s, end-to-end, retrieval/rerank ms, ingest chunks/s, peak VRAM, residency, load events, and **other GPU processes**. Official numbers are only recorded when the GPU is idle.
  - Every README number is regenerable with one command from `evals/configs/*.toml`.

## 10. Phases at a glance

| Phase | Gate | Outcome |
|---|---|---|
| P0 Notes bootstrap | — | `project/` central notes + `CLAUDE.md`; the plan lives in the repo (this chat) |
| 0 Setup | A | conda env, tooling, CI, `doctor` |
| 1 Ingestion & storage | A | incremental, crash-safe indexing into SQLite + Chroma |
| 2 Retrieval & first evidence | A | hybrid retrieval + rerank; ⛳ **first ablation table** + the one-time v1 reference run |
| 3 Answering with LangGraph | A | fast streaming answers with memory, citations, abstention |
| 4 Server & UI | A | single-owner API, OpenAI-compatible endpoint, Streamlit client; ⛳ **Gate A review** |
| 5 Evidence | B | bench/VRAM/latency, answer eval + judge, web + taint, injection eval, README; ⛳ **Gate B review** |
| 6 Integration | C | MCP server, CSV-project adapter; ⛳ **Gate C review** |
| Parked | — | Docker, OpenTelemetry/Phoenix, multimodal ingestion, folder watch, time-travel, LLM contextual retrieval, `Send` multi-hop, nightly GPU CI, any model over 12B |

## 11. Step-by-step roadmap (each step follows the §2 loop and needs your green light)

**Legend:** 🧠 what you learn · 🛠 what gets built · ▶ what you run (in `conda activate geekbot`) · 🎮 = loads a model on the GPU (run when your CSV project isn't using it) · ✋ optional "Your turn" exercise

### Step P0 — Project notes bootstrap (this planning chat, right after approval; writes files only, runs nothing)
- **`project/PLAN.md`:** this plan, verbatim.
- **`CLAUDE.md`:** project one-liner, standing rules (no-run, conda env `geekbot`, LangChain 1.x + LangGraph, `gemma4:12b` default and nothing over 12B, SLO-first speed, teaching loop), the §2.1 session start/end protocol, and code conventions.
- **`project/STATUS.md`:** every step from §11 listed as `todo`, P0 as `green-lit` once you approve, and "Next up: S0.1" with a handoff note.
- **`project/DECISIONS.md`:** seeded with every decision so far, dated 2026-10-07:
  - framework
  - default model and the ≤12B cap
  - speed SLOs
  - conda
  - no-run
  - teaching format
  - multi-chat + notes location
  - web in Gate B
  - CSV interim handling
  - Gate C order
  - judge fixed non-Gemma ≤12B
  - storage invariant
  - taint model
- **`project/reviews/`:** both ChatGPT reviews verbatim (`chatgpt-review-1.md`, `chatgpt-review-2.md`), each followed by a resolution table (adopted / changed / disagreed, with the reason).
- **Indexes:** `project/steps/README.md` and `project/learn/README.md`.
- **You run:**
  ```
  git tag v1.0
  git checkout -b v2
  git add CLAUDE.md project
  git commit -m "chore: add v2 plan and project notes"
  ```
  The tag goes on current `main`. The exact message is given at the time.
- Then **S0.1 starts in a new chat**, which reads `CLAUDE.md` → `project/STATUS.md`.

### Phase 0 — Setup
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S0.1 Skeleton + conda env** | src layout, `pyproject`, entry points, conda vs pip, lockfiles, ruff/mypy/pytest | `environment.yml`, `pyproject.toml`, `src/geekbot/__init__.py`, smoke test, ruff/mypy/pytest config, pre-commit, MIT LICENSE, `.gitignore` (`venv/`, `.venv/`, `data/`, `.env`); v1 files untouched | `conda env create -f environment.yml` (bootstrap) → `pip-compile …` to create the lock → `conda env update -f environment.yml --prune` (from the lock) → `pytest` · `ruff check .` · `mypy src` |
| **S0.2 Config, logging, CLI** | 12-factor config, pydantic-settings, typed CLIs | `config.py` (gemma4:12b defaults, SLOs, paths), logging, Typer `geekbot` with stub commands, `.env.example` | `geekbot --help` · `pytest` |
| **S0.3 `doctor` + GPU introspection** 🎮 | how Ollama loads models, `keep_alive`, VRAM math (weights + KV + buffers), the gemma4 sliding-window KV calculation | `llm/vram.py` (parsers, unit-tested on recorded outputs), `llm/models.py` (warm-up), `geekbot doctor` | `pytest` · `geekbot doctor` (warm-up loads gemma4) |
| ~~S0.4 v1 latency baseline~~ | — | **Dropped** (D-031). v1 runs once instead, at checkpoint CP2 (D-034). The latency-measurement lesson moves to S3.3 | — |
| **S0.5 CI** | CI pipelines, matrix builds, lock vs fresh resolution | `.github/workflows/ci.yml` (setup-miniconda) | `git push` to a branch, then check the Actions tab |

### Phase 1 — Ingestion and storage
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S1.1 Loaders** | document parsing, why MarkItDown over unstructured, page metadata, tabular schema cards | `ingest/loaders/*` (pypdf 1-indexed, MarkItDown docx/pptx/html, text/code/ipynb, csv/xlsx cards + ≤2,000-row text), per-file error report | `pytest tests/unit/test_loaders.py` |
| **S1.2 Chunking** | chunking trade-offs, tokens vs characters, structure-aware splitting, contextual headers | `ingest/chunking.py` (header → token-recursive 512/64, code splitter, header text) | `pytest` |
| **S1.3 IDs and hashing** ✋ | content addressing, identity vs fingerprint, idempotency, property-based testing | `ingest/ids.py` + hypothesis tests (duplicates get distinct IDs; an insert keeps unchanged IDs). ✋ write `occurrence` numbering yourself | `pytest` |
| **S1.4 SQLite store + FTS5** | WAL, FTS5 and BM25 internals, migrations, single-writer thread | `index/sqlite.py` (schema, migrations, writer queue, read pool), `index/lock.py` | `pytest` |
| **S1.5 Embeddings + Chroma** | embeddings, cosine vs L2, HNSW, CPU embedding, embedding cache, fingerprints | `index/dense.py`, `PrefixedEmbeddings` (CPU `num_gpu=0`), cache, fingerprint guard | `pytest` · a tiny live embed test (CPU only; also checks whether vectors are unit-length, which sets the R1 expectation) |
| **S1.6 Ingest pipeline** | the outbox pattern, consistency across two stores, crash recovery, process pools on Windows | `ingest/pipeline.py`, `jobs.py`, `consistency.py`, `geekbot ingest`, `doctor --repair` | `geekbot ingest evals/corpus` twice (2nd = 0 embeds; needs the corpus, D-032) · crash/repair test |

### Phase 2 — Retrieval and first evidence
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S2.1 Dense retrieval** | asymmetric retrieval, task prefixes, the "ready-only" filter | `retrieval/dense.py`, `geekbot search` | `geekbot search "…"` |
| **S2.2 BM25 + RRF** ✋ | lexical vs semantic retrieval, why fuse by rank not score | `retrieval/lexical.py`, `fusion.py`. ✋ implement RRF against my tests | `pytest` · `geekbot search --mode hybrid` |
| **S2.3 Reranking** | bi-encoder vs cross-encoder, rerank depth vs latency | `retrieval/rerank.py` (fastembed CPU), latency timing | `geekbot search --mode hybrid_rerank` |
| **S2.4 Eval datasets** | qrels, BEIR, building a golden set, unanswerable questions | `evals/datasets.py`, `eval generate`, golden-set format (from `evals/datasets/golden-questions.md`); settle the deferred eval questions (D-033). **You curate ~100 items** (~1 h) | `geekbot eval generate` 🎮 → curate |
| **S2.5 Metrics and statistics** ✋ | nDCG, Recall, MRR; bootstrap CI, permutation test, Holm | `evals/metrics.py`, `stats.py`, the Holm test family (D-033). ✋ implement nDCG@k against hand-computed tests | `pytest` |
| **S2.6 Ablation + bake-offs** | ablation design, confounds, leave-one-out, threshold calibration | rungs R0–R7, LOO, embedder/reranker bake-offs, `calibrate.py`, report writer | `geekbot eval retrieval --config evals/configs/ablation.toml` (CPU-heavy, long run) |
| ⛳ **Checkpoint: first evidence** | — | one-time v1 reference run (D-034) + draft README results section | create the `geekbot-v1` env, then the v1 run 🎮 (commands given then); we review the ablation table and the v1 reference together **before Phase 3** |

### Phase 3 — Answering with LangGraph
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S3.1 Graph basics** 🎮 | LangGraph state, nodes, edges, compile; `ChatOllama` options | minimal `retrieve → generate` graph, `geekbot ask` (non-streaming) | `geekbot ask "…"` |
| **S3.2 Prompting, packing, citations** | context windows, token budgets, prefix-KV reuse, datamarking, citation validation | `prompts/*.md`, `packing.py` (+ label bookkeeping), `citations.py` | `pytest` · `geekbot ask` |
| **S3.3 Streaming + speed** 🎮 | streaming, TTFT, measuring latency properly (warm vs cold, percentiles), thinking models, warm-up, `keep_alive` | streaming, think toggle, `RunStats` (TTFT, tok/s, loads) | `geekbot ask --stream` (watch TTFT) |
| **S3.4 Memory and follow-ups** 🎮 | conversational RAG, checkpointers, structured output | `AsyncSqliteSaver` threads, `condense` node | multi-turn `geekbot ask --thread t1` |
| **S3.5 Abstain, cite_check, sanitize** | abstention, relevance vs entailment, markdown-image exfiltration | grade/abstain, post-stream `cite_check`, `sanitize.py` (G3) | `pytest` · unanswerable question demo |
| **S3.6 Parallelism + resilience** 🎮 | parallel branches, `RetryPolicy`, stream events, SLO smoke testing | dense ‖ lexical branches, retries, step events, live SLO smoke | `pytest -m live` |

### Phase 4 — Server and UI
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S4.1 FastAPI core** | ASGI, lifespan, event-loop blocking, single-owner process | `server/app.py`, kb/jobs/health routes, writer lock at startup, heartbeat test | `geekbot serve` · `pytest` |
| **S4.2 Threads + OpenAI-compatible API** | SSE, API compatibility, stateless vs stateful | thread routes, `/v1/chat/completions`, client SDK, OpenAI-SDK contract test | `pytest` · the OpenAI SDK snippet (given) |
| **S4.3 Perimeter security** | threat modeling, auth, CORS, path traversal, upload hygiene | `server/security.py`, allowlist, limits | `pytest` |
| **S4.4 Streamlit UI** | Streamlit's rerun model (the root of v1 bug #1), API-client UIs | chat / knowledge / settings / evals pages; v1 files removed | `geekbot up` → try it in the browser |
| ⛳ **Gate A review** | — | DoD checklist below | You run the checklist and we review together |

**Gate A definition of done:**
- CI is green.
- Re-ingesting unchanged data costs 0 embeddings.
- `--repair` heals a crash.
- A second storage process is refused.
- The heartbeat test passes.
- gemma4 is 100% GPU-resident with 0 loads during queries.
- The SLO smoke run passes.
- A follow-up resolves.
- An unanswerable question abstains.
- Citations persist.
- The OpenAI SDK works.
- A planted image is stripped.

### Phase 5 — Evidence
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S5.1 Bench + tuning sweep** 🎮 | performance engineering, KV-cache quantization, flash attention, batch size vs VRAM | `evals/bench.py` (idle-GPU guard), sweep config, VRAM table | `geekbot bench --config evals/configs/bench.toml` (idle GPU) |
| **S5.2 Answer eval + judge** 🎮 | LLM-as-judge, self-preference bias, Cohen's κ | deterministic answer metrics, `judge.py`, labeling view. **You label ~60 answers** (~1 h) | `geekbot eval answers` → label → κ report |
| **S5.3 Web tool + taint enforcement** | taint tracking, HITL interrupts, egress control | `tools/web.py`, `taint.py`, `EgressGate`, `UNKNOWN` labeling | `pytest` · approval-prompt demo |
| **S5.4 Injection eval** | prompt-injection attack classes, measuring security | `evals/injection.py` + dataset | `geekbot eval injection` |
| **S5.5 Write-up** | technical writing, ADRs, presenting results | README v2, `evolution.md`, `performance.md`, 6 ADRs, resume bullets | Read and review |
| ⛳ **Gate B review** | — | — | Every README number regenerable; bypasses 0/N; κ rule applied; SLO table measured |

### Phase 6 — Integration
| Step | 🧠 Learn | 🛠 Build | ▶ You run |
|---|---|---|---|
| **S6.1 MCP server** | the MCP protocol, tools vs resources, transports | `server/mcp.py` (`MCPServer`, mounted `/mcp`), `mcp-stdio` proxy, read-only tools | `claude mcp add --transport http geekbot http://127.0.0.1:8000/mcp` → call a tool |
| **S6.2 CSV-project adapter** | ports and adapters, contract testing, plugin discovery | `TabularBackend`, contract suite, fixture backend, tabular route, `docs/plugins.md`; then **we connect your CSV project** | contract tests against your project |
| ⛳ **Gate C review** | — | — | Final review |

## 12. Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | gemma4:12b not 100% GPU-resident on 10 GB | High | CPU embedder + reranker; `doctor` residency check; ordered levers (§6.5) |
| 2 | Model swaps / cold loads mid-query | High | Warm-up, long `keep_alive`, `MAX_LOADED_MODELS ≥ 2`, load events in DoD |
| 3 | **Shared PC: your CSV project competes for GPU/CPU, invalidating numbers** | High | Idle-GPU guard in `bench`; official numbers only on an idle machine; 🎮-marked steps |
| 4 | Infrastructure before evidence | High | Phase 2 checkpoint with the first ablation table |
| 5 | Taint stripped on some path | High | Graph-state labels; single labeling point; `EgressGate`; property and registry tests; `UNKNOWN` default |
| 6 | Multiple storage writers | High | Writer lock + `workers=1` + import-linter |
| 7 | Event loop blocked by sync I/O | Medium | §6.3 boundary + heartbeat test |
| 8 | CPU embedder slow on big ingests | Medium | Batching, embedding cache, background jobs; throughput reported |
| 9 | Judge unreliable or biased | Medium | Fixed non-Gemma judge; blind human labels; κ rule; deterministic metrics stay primary |
| 10 | Ablation confounds | Medium | One variable per rung, LOO, Holm |
| 11 | Over-claiming | Medium | Guarantees vs empirical results kept separate; BEIR vs golden terminology; SLOs are targets until measured |
| 12 | Windows / conda specifics (wheels, spawn, file locks) | Medium | Python 3.12 env; Windows CI from the lock; spawn-safe pools; one Chroma client |
| 13 | Lockfile is platform-specific | Low | Windows lock for the dev box; Linux CI resolves fresh from ranges |
| 14 | Ecosystem drift | Medium | Versions and digests recorded; tags verified at S0.3/S5.2; re-bench after an Ollama update |
| 15 | Learning pace vs momentum | Medium | Small steps; lesson notes; optional ✋ exercises never block progress |

## 13. Testing (built incrementally, step by step)
- **Unit:**
  - RRF invariants; chunker no-loss; ID properties
  - Packing budget + labels
  - Metrics/stats against hand-computed cases
  - Citation validation; sanitizer (incl. split stream chunks)
  - Taint rules; path allowlist; judge-family guard
  - `ollama ps` / `nvidia-smi` parsers on recorded fixtures
- **Integration** (fake embeddings + a fake chat model with structured output and streaming; temp dirs):
  - ingest → re-ingest → edit → delete
  - Crash → `--repair`
  - A second writer is refused
  - Heartbeat
  - Graph paths (zero-LLM first turn, condense, abstain, taint interrupt → resume)
  - OpenAI-SDK contract; injection regression
  - MCP and tabular contract (Gate C)
- **Live** (`pytest -m live`, needs Ollama + GPU 🎮): smoke tests, residency assertion, SLO smoke.

## 14. Verification commands (you run them, in `conda activate geekbot`)
1. Static checks and tests (one command per line; `&&` fails in Windows PowerShell 5.1):
   ```
   ruff check .
   mypy src
   lint-imports
   pytest
   ```
2. `geekbot doctor` 🎮: gemma4 at 100% GPU after warm-up, embedder on CPU, free VRAM shown, lock free.
3. Ingest:
   - `geekbot ingest evals/corpus` twice: the second run does 0 embeddings.
   - Edit a section, then delete a file: incremental.
   - Kill mid-ingest, then `geekbot doctor --repair`: consistent.
   - Start a second `geekbot serve`: refused.
4. `geekbot eval retrieval --config evals/configs/ablation.toml`: the Phase 2 checkpoint table, plus the one-time v1 reference run 🎮 in the `geekbot-v1` env (D-034).
5. `geekbot up` 🎮, then in the browser:
   - First token in ~2–3 s, warm.
   - No re-indexing or model loads.
   - Citations and the think-harder expander survive a reload.
   - Follow-up resolves; unanswerable abstains.
   - Web off: no egress.
   - A tainted web query shows the approval prompt.
6. The API: the OpenAI SDK streams; a non-local bind without a key is refused; an out-of-allowlist ingest returns 403; a planted image is stripped.
7. Evidence:
   - `geekbot bench --config evals/configs/bench.toml` 🎮 on an idle GPU: SLO + VRAM tables.
   - `geekbot eval injection`.
   - `geekbot eval answers` 🎮: κ rule.
8. Gate C: `claude mcp add --transport http geekbot http://127.0.0.1:8000/mcp` → `search_knowledge_base`; the tabular contract passes.

## 15. Prerequisites and your effort
- **Miniconda/Anaconda** installed; you run `conda env create -f environment.yml` in S0.1, and create the separate `geekbot-v1` env once at CP2 (D-034).
- **Ollama:** 0.34.2 is fine. If you update Ollama, everything is re-benched; versions are recorded.
- **Model pulls (you run them):**
  - `qwen3-embedding:0.6b` and `embeddinggemma` (S2.6, ~1 GB, CPU use)
  - the judge, Qwen3 ~8B class (~5 GB, S5.2; tag verified first)
- **No model over 12B** is pulled or used.
- **Your time:**
  - a review and green light per step
  - ~1 h to curate the golden set (S2.4)
  - ~1 h to blind-label ~60 answers (S5.2)
  - optional ✋ exercises (S1.3, S2.2, S2.5)
- **For fast answers and valid benchmarks:** keep the GPU free of other heavy work (your CSV project, games, GPU-accelerated browsers). `doctor` and `bench` show what's using it.
