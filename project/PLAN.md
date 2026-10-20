# Geekbot v2 — implementation plan (rev 21)

> **The measured results are the product; the architecture exists to make them credible.**
> (This line also opens the v2 README.)

## 1. Context

Geekbot (Sept 2025) is a 4-file Streamlit + LangChain 0.3 + Chroma + Ollama RAG chatbot. It has verified correctness bugs (§5), uses removed or deprecated LangChain APIs, makes README claims the code doesn't meet, and its answers can take around a minute (user-reported, never measured). v2 rebuilds it into a fast, measurable project the user can put on a resume and defend in a deep technical interview. **They will also learn every component as we build it together, and write one piece of each step themselves (§2).**

**Rev 11 adds a second mode.** Each conversation starts in one of two modes:
- **Chat mode:** the RAG engine answers from the knowledge base (everything below).
- **CSV mode:** the user uploads a CSV and asks questions about it. The answers come from `csv_analyst`, the package from the user's Gigalogy project (`C:\Work\Gigalogy`, guide copied to `project/sources/csv-analyst-INTEGRATION.md`). The model writes one read-only SQL query, the query runs on an in-memory copy of the CSV, and the answer is written from the real result, with its numbers checked against it. The model never states a number on its own. Design: §6.7. Steps: track C (§11).

**User decisions**
- **Stack:** LangChain 1.x + LangGraph.
- **Model and speed:** `gemma4:12b` is the default chat model; no model over 12B; speed is a hard requirement.
- **Environment:** a **conda** environment (no uv).
- **No running:** **I don't execute anything on this PC** (it runs the user's CSV-query project). The user runs every command.
- **Pace:** **step-by-step delivery**. Each feature is reviewed and green-lit before the next, in a **teacher/learner** format.
- **Multi-chat:** **each task or phase runs in its own chat.** This plan lives in the repo at `project/PLAN.md`, and all notes live in a **committed central directory `project/`**, bootstrapped for every chat by a root `CLAUDE.md` (§2.1).
- **Web search:** opt-in, built in Gate B with the security work.
- **CSV analyst mode (rev 11):**
  - The `csv_analyst` package is **copied into the repo** with a note of where it came from (D-036).
  - CSV mode uses **`qwen2.5-coder:7b`**, the model Gigalogy measured. Chat mode keeps `gemma4:12b`. The two never share the GPU; Geekbot swaps them (D-037).
  - The work is a **track of 6 steps, C1–C6**, each placed where the parts it needs exist (D-038).
- **CSV files in chat mode (rev 16, D-051):** a schema card only (columns, types, a few example values); row text is never indexed. Numbers come from CSV mode (§6.7). This replaces D-016's "row text for files up to 2,000 rows".
- **Gate C:** the MCP server only. The old CSV-adapter step (S6.2) is replaced by track C (D-038, replaces D-017).
- **Pair programming (rev 11):** in every step the user writes one piece of code themselves, against tests Claude provides (D-039).

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

> **Prototyping mode (rev 14, D-047).** The table below is the full teaching loop. While prototyping, stages 1–2 shrink to a short outline from Claude (with a little interesting context), stages 3–4 become one build by Claude (including the ✍️ piece, which goes on the ✍️ backlog in `STATUS.md` so the user can redo it later), and stage 5 starts with Claude's summary. Stages 6–7 stay: the user runs the commands, gives the green light and pushes. New lesson notes are deferred.

**Per-step loop (one feature at a time; pair programming D-039; branches, pull requests and look-backs D-040).** Each step is a small, complete software lifecycle you can watch from start to finish:

| # | Loop stage | Lifecycle stage | What happens | Where it's kept |
|---|---|---|---|---|
| 1 | **Lesson** | research | I explain the concepts (what, why, trade-offs, pitfalls) and map them to our design | `project/learn/NN-<topic>.md`: a full course plus interview prep |
| 2 | **Design approval** | design | the files, interfaces and tests, plus the **spec of your ✍️ piece** (signature, behaviour, its tests). You approve or adjust it *before* I write code | step log, "Approved design" (+ an ADR if it's an architecture choice) |
| 3 | **Build** | coding | I write a small, reviewable diff (code + tests + docstrings). Your ✍️ piece is left as a stub, with tests that fail until it works. Nothing is run | the step's branch |
| 4 | **✍️ You write** | coding | you write your piece (usually 20–60 lines) | the step's branch |
| 5 | **You verify** | testing | you run the listed commands and paste the output. I fix anything of mine that fails; you fix your piece, with hints if you want them | step log |
| 6 | **Review** | code review | I review your code, you review my diff, then I ask 3–5 understanding / interview questions | step log |
| 7 | **Green light** | release | you approve, and I give you the exact git commands. From S0.5 on (when CI exists) each step runs on its own branch `step/<id>-<slug>` and is merged into `v2` through a pull request once CI passes | `STATUS.md` (commit SHA) |
| — | **Look-back** (end of each phase) | retrospective | planned vs actual, what went well, what to change | "Phase look-back" section of the phase's last step log |

**Your ✍️ piece:**
- Every step in §11 names one, so the resume claims are genuinely yours.
- **If you get stuck,** help comes in three levels:
  1. a hint (the concept or the doc to read)
  2. a rough outline (pseudocode)
  3. I write it and explain it line by line
- If level 3 is used, the step log says "written by Claude".
- I never fill in your piece unless you ask for level 3. A stuck piece never blocks the project.
- How branches are merged (merge commit or squash) and whether PRs use the `gh` CLI or the GitHub web page is settled at S0.5.

**GPU etiquette:** steps that load models (doctor warm-up, ask, bench, eval) say so explicitly. Benchmarks are only valid on an idle GPU, so `bench` detects other GPU processes (e.g. a Gigalogy benchmark or training run) and refuses to record official numbers while they run.

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
  sources/                  outside documents the plan builds on, copied verbatim (the csv_analyst INTEGRATION.md)
```
- **Status values:** `todo → design → build → verify → green-lit`, or `blocked`, or `dropped` (removed by a plan change).
- **Session start:**
  1. Read `CLAUDE.md`, `PLAN.md`, `STATUS.md`, `DECISIONS.md` and the previous step's log.
  2. Confirm with you which step(s) this chat covers.
  3. Restate the no-run policy.
- **During:** follow the §2 loop. **`PLAN.md` is never changed silently.** A change is proposed, you approve it, and it's recorded in `DECISIONS.md` and in the plan's revision history.
- **Session end:**
  1. Update the step log, `STATUS.md` (including the "Next up" handoff) and `DECISIONS.md`.
  2. Give you the commit command (and, from S0.5 on, the push + pull request commands); notes are committed together with that step's code.
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

**CSV analyst mode (rev 11)** runs under the same three pillars:

| Pillar | In CSV mode | Evidence |
|---|---|---|
| **Retrieval → grounding** | the answer comes from one executed read-only SQL query; its numbers are checked against the result; unanswerable questions are refused | C3 reproduction run of the Gigalogy benchmark; C6 follow-up results |
| **Systems** | one model on the GPU per mode (`qwen2.5-coder:7b` in CSV mode), explicit measured swap, profiling and queries off the event loop | swap time and seconds per question (C3, S5.1) |
| **Security** | CSV cells are untrusted data; replies pass the same output filter (G3); the CSV never leaves the machine | the read-only SQL test; the CSV-cell attacks in the injection eval (C6, S5.4) |

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
- **rev 10** (S0.1 design, user-approved), §6.1 and the S0.1 row in §11 (D-035):
  - Each dependency, extra and tool config arrives in the step that first uses it
  - conda installs only Python + pip; pip-tools comes from pip
  - `environment.yml` always installs from the lock; a one-time bootstrap creates the first lock
- **rev 11** (P1, user decisions, 2026-10-11): **CSV analyst mode + pair programming** (D-036 – D-041):
  - Gigalogy's `csv_analyst` is copied into `src/csv_analyst/` and becomes a second conversation mode (new §6.7)
  - CSV mode uses `qwen2.5-coder:7b`; it's swapped with `gemma4:12b`, never loaded together; residency rules restated per mode (§6.5)
  - New track C1–C6 (§11); S6.2 is dropped; Gate C is MCP only
  - Open question for S1.1: does chat mode still need row text for small CSVs (D-016)?
  - The §2 loop becomes pair programming: one ✍️ piece per step, a hint ladder, a lifecycle map, branch + PR per step from S0.5, and a look-back per phase
  - Lesson list renumbered in execution order
- **rev 12** (2026-10-14, user decision): **prototyping profile with lighter models** (D-042):
  - While building, Geekbot can run on lighter models: chat `gemma3:4b`, CSV mode `qwen2.5-coder:3b` (Gigalogy's own "fast" tier)
  - The plan's models stay the target: `gemma4:12b` and `qwen2.5-coder:7b`. Every reported number comes from them and names its model
  - New paragraph in §6.5, a note in S0.2's row (§11) and a new risk, 22 (§12)
- **rev 13** (C1 design, 2026-10-16, user decisions; D-045): **what C1 copies, and from where**:
  - The copy comes from the Gigalogy working tree of 2026-10-16 (the user's newer, then-uncommitted edits), not from commit `f482455`. `VENDORED.md` says so. Gigalogy's numbers (35/40, about 3.0 s) still describe the commit they were measured on, not this exact copy; Geekbot reports its own results from C3 on
  - The pandas engine stays in the copy, unused (the default engine is SQL). Dropping it needs more than small edits (`pipeline.py` and three other modules depend on it)
  - Cuts: `interface/`, the two hosted providers, the 27B tier, and the 8 tests that need Gigalogy's `evaluation/` folder or its API
  - C1 has no ✍️ piece (the user's choice, to keep the step small); §11's C1 row is updated
- **rev 14** (C2, 2026-10-16, user decision; D-047): **fast-prototyping mode**:
  - Per step: a short outline with some interesting context, one build by Claude, one summary; the user runs the commands, gives the green light and pushes
  - No separate lesson or design-approval gate; new lesson notes are deferred
  - Claude writes the ✍️ pieces for now (marked in the step log, listed in the ✍️ backlog in `STATUS.md`) so the user can redo them later
  - The §11 ✍️ column stays as the list of pieces; the no-run policy, the model cap and the claims rules are unchanged
- **rev 15** (C3, 2026-10-16, user approval; D-048, D-049): **first CSV-mode measurements and speed targets** (§6.5):
  - The explicit swap, the reproduction run and its design are recorded in D-048 (this settles §9's "imported read-only or ported": the grader is ported, and the benchmark is read from the Gigalogy folder and never copied)
  - CSV-mode targets are set from the first measurements on the target models: question p50 ≤ 3 s and p95 ≤ 10 s, swap ≤ 6 s in either direction, 0 model loads during questions (R-2). They stay targets until S5.1 re-measures them
  - The CSV VRAM line is checked against a measurement (§6.5); the question cap (60 s) and the per-call timeout (60 s) are unchanged because neither came close to firing
- **rev 16** (S1.1, 2026-10-16, user decision; D-051, D-052): **chat mode indexes a schema card for tables, never their rows** (§1, §6.7, §11 S1.1):
  - The open question from rev 11 is settled. D-016's "row text for files up to 2,000 rows" is superseded: a CSV, TSV or Excel sheet becomes a card (columns, types, empty share, three example values per column) that points number questions to CSV mode
  - Reason: numbers read from embedded rows are exactly the guessing CSV mode avoids (the model there writes one SQL query and the answer's numbers are checked against the result)
  - The loader design (what each format yields, what is dropped on purpose) is recorded in D-052
- **rev 17** (S1.2, 2026-10-16, user decision; D-053): **the chunker is hand-written, not LangChain's splitters** (§6.4, §11 S1.2):
  - LangChain's splitters return text without positions, and the plan needs `char_start` / `char_end` for citations and an exact "no text is lost" property; a hand-written splitter makes every chunk an exact slice of the document text
  - §6.4's "where LangChain is used" no longer lists splitters; `ingest/chunking.py` joins the places where our own code is used
  - Token counts for chunk sizes come from a dependency-free estimate; the embedder's real tokenizer is compared with it at S1.5. `hypothesis` enters the `[dev]` extra at S1.2 (its first use), not S1.3
- **rev 18** (S2.4, 2026-10-17, user decision; D-067): **Claude writes the first golden corpus and golden questions; the user's come later** (§9, §11 S2.4, §12 "Your time"):
  - The ~15-document golden corpus and ~100 questions are drafted by Claude as **"golden v0", provisional** (a fictional studio's AI-rollout documents, all original, in the formats the loaders support). The user reviews them and adds their own questions after the prototype works, in a **separate `golden-user.jsonl`**, scored on its own as the independent set (adding them to v0 would move v0's development/test split)
  - Reason: the user wants a working prototype first and is overseeing the build; the questions are not on the critical path until S2.6 needs them
  - Cost, stated: the author of the documents, the questions and the system is one, so v0 is easy mode and its numbers are never a headline. Every v0 item carries a "provisional" flag, and results from it are reported as "Claude-authored golden v0, provisional" (CLAUDE.md rule 8). D-065 stays provisional as well
  - §9's description of the golden corpus is unchanged otherwise; §11's S2.4 row no longer asks the user for ~1 h of curation now (that time moves to the review of v0 and the user's own set)
- **rev 19** (2026-10-17, user decision after S2.4; D-068): **the answering path comes before the measured ladder: S3.1 to S3.4 now, then S2.6 and CP2, then S3.5** (§10, §11):
  - The user wants a working prototype before the ablation. S2.6 (rungs R0–R7, leave-one-out, the embedder and reranker bake-offs, `calibrate.py`; a long CPU run) and the CP2 checkpoint (the first evidence table and the one-time v1 reference run) move from the end of Phase 2 to **just before S3.5**, because S3.5 (abstain) is the first step that needs an output of S2.6: the calibrated abstain threshold
  - The order of work becomes: Phase 1 → S2.1 to S2.5 → S3.1 to S3.4 (C4 after S3.4) → S2.6 → CP2 → S3.5 → S3.6 → Phase 4 → Gate A. CP2's "review before Phase 3" becomes "review before S3.5"
  - Until S2.6, the answering graph uses the retrieval defaults that already exist (hybrid with reranking, 50 candidates a side, rerank depth 30, top-k 5). They are starting points, not measurements, and the code and notes say so
  - Cost, stated: the first four steps of Phase 3 are built on retrieval settings that the ablation has not checked yet. If S2.6 changes a default (the rerank depth, say) that is a configuration change, but prompts and packing written meanwhile may need a second look. Phase 2 is not finished until S2.6 runs. D-065 must still be revisited before S2.6 runs anything, and everything computed on golden v0 stays provisional (rev 18)
- **rev 20** (2026-10-18, user decisions after CP2 and before the end-to-end prototype; D-082, D-083): **abstention without a score threshold, and S3.6 after the prototype** (§7, §10, §11):
  - S2.6 and CP2 are done (D-079 to D-082). The planned abstain signal, a threshold on the reranker's best score, measured weak on golden v0 (dev F1 0.41, test F1 0.39). The user chose **option B for S3.5: the chat model's own refusal (the system prompt already tells it to say it could not find the answer) plus the post-stream citation check, no score gate**; abstention is *measured* on the 20 unanswerable golden questions, and an LLM grading step or other signals are added only if too many of them get an answer. §7's "the threshold is calibrated F1-optimally" and §11's S3.5 row are changed accordingly; the calibrated threshold stays as data in `evals/results/s26-run1/abstain.json`
  - **S3.6 (parallel search branches, retries, step events, live SLO smoke) moves from before Phase 4 to after the end-to-end prototype**, so a chatbot the user can run in a browser comes first: the order of work becomes S3.5 → S4.1 → S4.2 → S4.3 → S4.4 → C5 → S3.6 → Gate A. Nothing in Phase 4 depends on S3.6 (it changes how the existing retrieval is scheduled, not what it returns)
  - The reranker, the embedder, the 400 ms retrieval target and the golden set are left as they are until the whole project has been used (D-082); the user then writes evaluation questions from their own source files and the deferred decisions are gone over together
  - **No external API feature in the prototype (user, 2026-10-18; D-084):** S4.2's OpenAI-compatible `/v1/chat/completions`, its OpenAI-SDK contract test and the client SDK snippet are left out for now; S4.2 builds only the thread routes and the streamed answer route (server-sent events) that the Streamlit UI uses, under Geekbot's own `/v1/threads` paths. S4.3 keeps the protections of the local server and its upload and drops what only outside clients need. The endpoint comes back when the user wants other programs to talk to Geekbot
  - Cost, stated: the answering path reaches the user before its parallelism and retry behaviour exist, so the first prototype searches dense and BM25 one after the other (about 25 to 50 ms without the reranker, measured) and a failing model call is not retried; S3.6 measures and fixes that afterwards
- **rev 21** (2026-10-20, user decisions at S4.3; D-089, D-090): **the perimeter as built** (§8.1 "Perimeter", the S4.3 row of §11, the API line of §14):
  - **The API key is required on every route except `/healthz`, always**, not only for a non-local bind. The server cannot bind off the machine at all (`HOST` is a constant and `serve` has no host option), so the old condition never applied, and loopback alone keeps out other machines but not a web page open in the user's browser, another Windows account on the PC, or a local program. The key is a **local password shared by the server and the UI** (`GEEKBOT_API_KEY`, else `data/api-key` made on first start), not a cloud or provider key; it is never sent anywhere
  - **"CORS allowlist" becomes "no CORS headers"**: the UI is a server-side Python client with no browser origin (D-084), so there is no origin to allow and a browser is never let read a response. A `Host` header allowlist (`127.0.0.1`, `localhost`) is added against DNS rebinding
  - The path sandbox also refuses network and device paths before touching the disk, never reads the data folder, and checks every file of a folder walk again (a Windows junction can lead outside an allowed folder); with no `GEEKBOT_ALLOWED_ROOTS` set, only uploaded files can be ingested
  - **A knowledge-base upload route, `POST /v1/kb/upload`, is added to S4.3** (raw body, stored as `<sha16>-<safe-name>` in `data/uploads`, type and size checked, then ingested). The user said to make it as needed; C5's CSV upload reuses its code
  - Cost, stated: nothing in this revision is verified until the user's runs of S4.3; the limits (1 MB JSON, 100 MB a file, 2,000 PDF pages) are unmeasured starting points

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
OpenAI SDK / Open WebUI ──────────────────►  (ONE   ├─ Ollama GPU, one chat model at a time (100% resident, warm, keep_alive):
Claude Code / Cursor ── /mcp (Gate C) ────►  storage│   gemma4:12b in chat mode ⇄ qwen2.5-coder:7b in CSV mode (§6.7)
                                              owner) ├─ Ollama CPU: embedder (num_gpu=0) ── 0 VRAM
                                                     ├─ fastembed ONNX cross-encoder (CPU) ─ 0 VRAM
                                                     └─ csv_analyst (CSV mode): uploads in data/uploads/, queried
                                                         read-only in an in-memory SQLite copy, one session per thread
```

### 6.1 Environment and packaging (conda)
- **`environment.yml`** (conda-forge only, `nodefaults`): conda installs just `python=3.12` (the broadest wheel support for chromadb, onnxruntime and fastembed on Windows) and `pip`. Every Python package, pip-tools included, comes from pip, so no package has two installers (D-035).
  - The pip section always installs `-r requirements/lock-win.txt` followed by `-e .`. Re-creating the env reproduces the exact versions.
  - **Bootstrap (S0.1, once):**
    1. Create a bare env with `conda create` (Python 3.12 + pip).
    2. Run `pip install -e ".[dev]"` from the pyproject ranges.
    3. Run `pip-compile --extra dev --strip-extras --output-file requirements/lock-win.txt pyproject.toml`.
    4. Delete the env and re-create it from `environment.yml`, which proves it reproduces.
- **`pyproject.toml`:** the src layout, version-ranged direct dependencies, extras, the `geekbot` console script, and tool config.
  - **Each dependency, extra and tool config is added in the step that first uses it** (D-035):
    - S0.1 brings the `[dev]` extra and the ruff / mypy / pytest config.
    - `[eval]` arrives with the first eval dependency (around S2.4).
    - The import-linter config arrives when the client / UI modules exist (Phase 4).
  - Each such step regenerates the lock (`pip-compile` with every extra defined at that point).
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
| csv_analyst: loading + profiling a CSV (`from_bytes`), answering (`ask`) | worker thread (`anyio.to_thread.run_sync`) |
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
- **Where LangChain is used:** where it adds value: `ChatOllama`, `OllamaEmbeddings`, messages, structured output, LangGraph.
- **Where direct clients or our own code are used:** where control matters: `chromadb` (precomputed, cached vectors), `sqlite3` FTS5, and the chunker (`ingest/chunking.py`, rev 17, D-053): LangChain's splitters return text without positions, and every chunk here must be an exact slice of the document so that its offsets are facts.

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
| **Headroom** | **tight: measured, never assumed.** First reading, at C3 (`gemma4:12b` warm at 8k through the swap, desktop included): 8,981 MiB in use, 1,259 MiB free. S5.1 re-measures it with the full tuning settings |

**CSV mode** replaces gemma4 on the GPU (the two never fit together):

| Item | Budget |
|---|---|
| `qwen2.5-coder:7b` file | 4.7 GB (INTEGRATION.md) |
| KV cache @ 16k (`num_ctx` 16,384, the package's setting), f16 | ≈896 MiB, an **estimate**: 28 layers × 4 KV heads × 128 dims × 2 (K and V) × 2 bytes × 16,384 tokens. At C3 the GPU went from 869 MiB (idle) to 6,457 MiB with the model warm, +5,588 MiB for weights, KV cache and compute buffers together: consistent with 4.7 GB + ≈ 0.9 GB, but the KV cache itself was not measured separately |
| Headroom | measured at C3: 3,783 MiB free (6,457 of 10,240 MiB in use, desktop included) |

**Residency rules** (restated per mode in rev 11, D-037):
- **R-1:** the **active mode's model** is **100% GPU** (`ollama ps`). `doctor` and `bench` fail otherwise.
- **R-2:** **no model loads during a query:** warm-up at server start, a long `keep_alive`, `OLLAMA_MAX_LOADED_MODELS ≥ 2`, and load events logged in `RunStats`. **One exception:** the first CSV question after a swap. The swap starts as soon as a CSV conversation begins, so it overlaps the upload and profiling; it is logged as a load event, and the UI shows "loading the CSV model…".
- **R-3:** `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_FLASH_ATTENTION=1`; KV cache type chosen by the sweep.
- **R-4:** **swaps are explicit:** unload the other chat model (`keep_alive: 0`), then warm up the new one. They're never left to Ollama's own eviction, which could leave part of a model on the CPU. `doctor` checks residency after a swap (C3).

**Speed design:**
- Thinking is off by default, with a per-question "think harder" toggle.
- One model serves every LLM call **within a mode**: gemma4 in chat mode; qwen in CSV mode, including the rewriting of CSV follow-ups (calling gemma4 there would force a swap on every question).
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
| Model loads during queries | 0 (except the first CSV question after a swap, R-2) |

**CSV-mode targets** (rev 15, set at C3; targets until S5.1 re-measures them; target models, warm server, CSV already loaded):

| Metric | Target | Measured at C3 |
|---|---|---|
| Seconds per question, p50 | ≤ 3 s | 2.3 s |
| Seconds per question, p95 | ≤ 10 s | 7.7 s |
| Swap between the two models, either direction | ≤ 6 s | chat→csv median 2.7 s (max 4.3 s); csv→chat median 4.5 s (max 4.5 s) |
| Model loads during questions | 0 (except the first one after a swap, R-2) | 0 of 120 question-runs |

Each target is the measured value plus about 30% headroom, rounded. The measurement is one reproduction run (40 questions × 3 seed sets on the hospital and sales datasets, an idle GPU, `gemma4:12b` ↔ `qwen2.5-coder:7b`; the C3 log has the details): short questions on two datasets, so wider CSVs with longer prompts may need the targets revisited at S5.1. The per-question cap (60 s) and the per-call timeout (60 s) stay as they are: the slowest question took 11.9 s and the slowest single step 2.7 s. For reference only, Gigalogy measured about 3.0 s per question on average with `qwen2.5-coder:7b` on this GPU (its README); Geekbot's own run gave a mean of 2.9 s. The first is a source-project number, the second a Geekbot result.

**Prototyping profile** (rev 12, D-042). For fast iteration on a PC shared with your Gigalogy work, the config can point both modes at lighter models:
- **Prototyping models:** chat `gemma3:4b`, CSV `qwen2.5-coder:3b` (Gigalogy's own "fast" tier, which its notes keep for cheap iteration).
- **Target models:** `gemma4:12b` and `qwen2.5-coder:7b`. The latency targets above, the VRAM budget, the benchmarks, the evaluations and every README number come only from the target models. Each result records the model tag that produced it (§6.1 already records model digests).
- **Rules R-1 to R-4** above apply unchanged to whichever profile is active.
- **What doesn't carry over:**
  - Gemma 3 has no thinking mode (its Ollama page shows no "thinking" label), so the "think harder" toggle is tested on the target profile only.
  - Gigalogy's notes call the 3b "more brittle" than the 7b, and its headline CSV numbers are for the 7b. CSV answer quality is judged on the 7b only (D-041).
- **Mechanism:** how the profile is picked, and which one applies when nothing is set, is settled in S0.2's design.

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
  llm/       models.py (factories, warm-up, keep_alive, explicit model swap) · vram.py (nvidia-smi / ollama ps parsing)
  csv_mode/  provider.py (csv_analyst's LLMProvider on Geekbot's Ollama settings) · sessions.py (one session per thread)
  agent/     state.py · runtime.py · prompts/*.md · packing.py · citations.py · sanitize.py · taint.py · nodes/* · graph.py
  tools/     web.py (Gate B)
  server/    app.py · routes/{openai,kb,threads,jobs,health,csv}.py · security.py · mcp.py (Gate C)
  client/sdk.py · ui/ (chat · csv · knowledge · settings · evals+labeling) · cli.py
  evals/     datasets.py · metrics.py · stats.py · judge.py · injection.py · bench.py · report.py
src/csv_analyst/   copied from Gigalogy (D-036); VENDORED.md = source commit, cuts, local edits, how to re-sync
evals/ README.md · corpus/ · datasets/{golden,injection,human_labels}.jsonl + golden-questions.md · configs/*.toml · csv/ (C3, C6) · results/
docs/ architecture.md · evolution.md · security.md · performance.md · adr/                (product docs)
project/ PLAN.md · STATUS.md · DECISIONS.md · steps/ · learn/ · reviews/ · sources/       (build process, §2.1)
tests/ unit/ · integration/ · live/ · csv_analyst/ (the copied package's own tests, no model needed)
CLAUDE.md · environment.yml · pyproject.toml · requirements/lock-win.txt · LICENSE (MIT) · .github/workflows/ci.yml
```

### 6.7 CSV analyst mode (rev 11; D-036, D-037, D-038, D-041)
The host-app guide is `project/sources/csv-analyst-INTEGRATION.md` (a verbatim copy of Gigalogy's `INTEGRATION.md`).

**What the package does.** `CsvAnalystSession.from_bytes(data, filename)` loads and profiles one CSV. `session.ask(question)` returns a `RunRecord`:
- The model writes one read-only SQLite `SELECT`.
- The query runs on an in-memory copy of the CSV.
- Two samples must agree; a third runs only if they don't.
- A second model call explains the result, and the numbers in that answer are checked against the result.

`record.status` is one of:
- `ANSWERED`
- `ABSTAINED`: the data can't answer it
- `NEEDS_CLARIFICATION`
- `FAILED`: no reliable answer, the time limit, or the backend is down

`to_markdown(record)` renders the answer, the confidence, the SQL and a result table. A bad file raises `IngestionError`.

**How it comes in (C1, D-036).**
- **The copy:** `csv_analyst/` from the Gigalogy working tree of 2026-10-16 (based on commit `f482455` plus the user's then-uncommitted edits; rev 13, D-045) is copied to `src/csv_analyst/`. It keeps its package name and public API.
- **Provenance:** `src/csv_analyst/VENDORED.md` records the source commit, what was cut, every local edit, and how to re-sync from Gigalogy.
- **Gigalogy's role:** it stays the home of the benchmark, datasets and experiments (INTEGRATION.md: "keep them in this repository for re-measuring").
- **Cuts, settled in C1's design:**
  - **Must go:** the heavy tier `qwen3.8:27b` (over the 12B cap).
  - **Recommended:**
    - drop `interface/` (Gigalogy's own API, UI and CLI)
    - drop the hosted providers (Anthropic, OpenAI-compatible): they would send CSV data off the machine on a path the egress gate (G2) doesn't control
    - drop the deprecated pandas engine and its subprocess sandbox, if `pipeline.py` allows it with small edits. **Settled in C1: it does not allow it, so the engine stays, unused** (rev 13)
- **Dependencies:** pandas, numpy, pydantic, pydantic-settings, python-dotenv, requests, charset-normalizer, pyarrow and sqlglot enter `pyproject.toml` in C1, and the lock is regenerated (D-035).
- **Quality gates:** the copied code isn't written for mypy strict, so it gets a per-package override, recorded in `VENDORED.md` as a typing debt.
- **Tests:** its no-model tests move to `tests/csv_analyst/`.

**How Geekbot talks to it (C2, ports and adapters).**
- **Provider:** `geekbot.csv_mode.provider` implements the package's `LLMProvider` protocol (`generate_json(prompt, *, system, temperature, seed, schema)`, in `csv_analyst/llm/base.py`). It is passed in through `CsvAnalystSession(..., llm=...)`. It uses Geekbot's own config: Ollama host, `keep_alive`, `num_ctx` 16,384 and timeouts. It keeps the package's "prompt too long" guard and records load events in `RunStats`.
- **Settings:** Geekbot's config (S0.2, `GEEKBOT_` prefix) owns the CSV model name and the per-question time limit, and passes them in.
  - The package's own `CSV_ANALYST_*` prompt and pipeline settings stay at the defaults Gigalogy chose by measurement. Changing them means re-running the benchmark.
  - Its default time limit is 10 minutes per question, too long for Geekbot's speed goal. C2 sets Geekbot's cap (60 s); at C3 no benchmark question took longer than 11.9 s, so the cap cut nothing.
- **Import boundary:** only `geekbot.csv_mode` imports `csv_analyst` (an import-linter contract, Phase 4).
- **CLI:** `geekbot csv describe <file>` and `geekbot csv ask <file> "<question>"`.

**Conversations (C4, C5).**
- **Mode per thread:** the user picks `chat` or `csv` when a conversation starts, and it can't change mid-conversation. That keeps memory, taint and model residency simple.
- **One session per thread:** `geekbot.csv_mode.sessions` keeps one `CsvAnalystSession` per thread, with a size cap and least-recently-used eviction. A new upload replaces the thread's session.
- **Uploads:** the server saves uploads to `data/uploads/<sha>-<safe-name>`, so only the server touches files (§6.2). After a restart, a session is rebuilt from that file the next time it's used.
- **Follow-ups:** the package has no conversation memory. The graph's `condense` node (S3.4) turns a follow-up such as "and by month?" into a full question first, using the CSV model (§6.5). Its quality is measured in C6.
- **Blocking work:** loading and asking are blocking, so they run in worker threads (§6.3).

**Models (C3, D-037).**
- **The model:** CSV mode uses `qwen2.5-coder:7b`, the configuration Gigalogy measured.
- **No co-residency:** gemma4 and qwen never share the GPU. Swaps follow R-2 and R-4 (§6.5).
- **C3 measures:**
  - the swap time in both directions
  - qwen's residency at 16k context
  - a reproduction run of the Gigalogy benchmark through Geekbot's adapter, to check the copy behaves like the original

**Claims (D-041).**
- **Gigalogy's numbers:** quoted only as "measured in the source project": 35/40 solved; 0 invented answers in 30 attempts on unanswerable questions there, but 6 in 27 on its Chinook dev set; about 3.0 s per question, with `qwen2.5-coder:7b` on this GPU.
- **Geekbot's own numbers:** reported separately, from C3, C6 and S5.1.
- **SQL safety:** the package's read-only SQL checks are described the way INTEGRATION.md describes them: prototype-grade, not a security audit (§8).

**Chat mode and CSV files (settled at S1.1, rev 16, D-051).**
- **The old rule:** D-016 indexed a schema card plus row text for files ≤2,000 rows, "while no tabular backend is registered". From C2 on, CSV mode is that backend.
- **The decision:** chat mode indexes **a schema card only** for every CSV, TSV and Excel sheet: the columns, their types, the empty share and three example values per column, plus a line that sends number questions to CSV mode. Row text is never indexed.
- **Why:** numbers read from embedded rows are exactly the guessing CSV mode avoids. The card is still enough to find the right file for a question.

## 7. LangGraph design (explicit state and conditional transitions; no ReAct loop)

```
START → condense? → [retrieve_dense ‖ retrieve_lexical] → fuse(RRF) → rerank → grade ─┬ ok ─────────► generate → sanitize → END ⇢ cite_check (after stream)
        (follow-ups only)                                                             ├ weak, web on ─► egress_gate ─► web ─► generate …
                                                                                      └ weak ─────────► abstain → END
```
**Mode router (rev 11, C4).** The graph starts with a plain-code router on the thread's mode:
```
START → mode? ─ chat ─► (the RAG graph above)
              └ csv ──► condense? (CSV model, follow-ups only) → csv_ask (worker thread) → sanitize → END
```
- `mode?` is a plain function of the thread state: no LLM call, no agent loop.
- `csv_ask` runs the package's fixed pipeline (§6.7).
- The CSV branch reuses `condense` and `sanitize`. Its output is labeled `TOOL` (§8.2).

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
- **Abstain (rev 20, D-082):** the model's own refusal plus the post-stream citation check; no score threshold as a gate. Measured on the unanswerable golden questions; a grading call or another signal is added only if too many of them get an answer. (Before rev 20: a threshold calibrated F1-optimally on the golden set; it measured weak, see `evals/results/s26-run1/abstain.json`.)
- **API semantics:** `/v1/chat/completions` is stateless; the UI uses `/v1/threads/*`.

## 8. Security model (`docs/security.md`, ADR-0004)

### 8.1 Architectural guarantees (each with a dedicated test)
- **G1, tool isolation:** nodes that read untrusted text have no tools bound.
- **G2, taint-gated egress:** no egress with tainted arguments without explicit user approval of the exact arguments.
  - The single `EgressGate` wraps every `network=True` tool (registry test).
  - G2 guarantees no *unapproved* tainted egress.
- **G3, output sanitization** (Gate A): a streaming-safe filter strips images and drops links to URLs that aren't among the sources (UI, REST, SSE). It also runs on **CSV-mode replies**: CSV cells are untrusted, and `to_markdown` escapes `|` and newlines but not `![…](…)`.
- **CSV mode** (rev 11, D-041):
  - **Never off the machine:** the CSV never leaves the machine. The hosted providers are removed (§6.7), and the mode is fixed per thread, so CSV data never reaches the web tool.
  - **Read-only SQL:** a dedicated Geekbot test proves that the package refuses writes, `ATTACH` and `PRAGMA`. Only that tested property is claimed. Beyond it, the SQL checks are prototype-grade (INTEGRATION.md).
- **Perimeter:**
  - `127.0.0.1` bind only (no host option); every route except `/healthz` requires the API key (`GEEKBOT_API_KEY`, else generated into `data/api-key`; a local password, not a cloud key); `Host` header allowlist against DNS rebinding; no CORS headers (rev 21).
  - `GEEKBOT_ALLOWED_ROOTS` path sandbox (no symlink escape, empty input rejected; network paths refused before the disk is touched; the data folder never readable; every file of a folder walk checked again).
  - Size and page limits; `<sha>-<safe-name>` uploads.
  - MCP is read-only by default.
- **Not guaranteed:** answer manipulation by poisoned documents, or by instructions hidden in CSV cells. That is measured (hijack rate), never claimed solved.

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
  - CSV-mode outputs (answer, SQL, result table) are labeled `TOOL`.
- **Monotonic, no declassification.** Covered by property tests, an `UNKNOWN`-labeling test and a registry test.

### 8.3 Injection evaluation (empirical; reported separately)
~30 planted docs/pages covering four vectors:
1. A web call carrying secret text.
2. Markdown-image exfiltration.
3. History laundering.
4. Citation forgery / hijack.
5. CSV cells (rev 11, C6): a markdown image or link in a cell, and instructions in a cell aimed at the SQL planner.

A simulated user denies every approval. The report gives bypasses k/N (target 0), gate triggers, and the hijack rate.

## 9. Evaluation methodology

- **Standardized retrieval evidence:** BEIR SciFact (5,183 docs / 300 test queries) and NFCorpus (3,633 / 323).
  - Downloaded at runtime with checksums; HF mirror as fallback.
  - Chunk hits are aggregated to doc level (max).
  - Wording: "standard BEIR datasets with standard metrics".
- **System-specific behavioral evaluation:** the golden corpus (~15 docs). ~100 items: ~50 single-hop, ~15 multi-hop, ~15 multi-turn, ~20 unanswerable. No general-superiority claims. **Golden v0 is written by Claude and provisional (rev 18, D-067); the user's own questions come later as a separate `golden-user.jsonl`, scored separately.**
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
- **CSV-mode evaluation** (rev 11). This is system-specific behavioral evaluation, like the golden set:
  - **C3 reproduction run:**
    - *What:* Gigalogy's 40-question benchmark (hospital + sales datasets; 30 answerable, 10 unanswerable), 3 runs each, through Geekbot's adapter.
    - *Rule:* a question counts as solved when it's right in at least 2 of the 3 runs (Gigalogy's rule, because local models vary run to run).
    - *Purpose:* it checks that the copy behaves like the original. It's a check, not a new claim.
    - *Harness:* whether the Gigalogy grader is imported read-only or ported into `evals/csv/` is settled in C3's design.
  - **C6 follow-up set:**
    - *What:* ~15 multi-turn items on Gigalogy's datasets (e.g. "total revenue by region" → "and by month?").
    - *Scoring:* scored the same 2-of-3 way.
    - *Purpose:* it measures the `condense` rewriting, which INTEGRATION.md lists as unmeasured.
  - **C6 CSV-cell attacks:** fed into the S5.4 injection eval (§8.3, vector 5).
  - **S5.1 bench:** adds CSV-mode rows: seconds per question p50/p95, swap time, residency.
- **Bench:**
  - `geekbot bench` records TTFT (first turn and follow-up), tok/s, end-to-end, retrieval/rerank ms, ingest chunks/s, peak VRAM, residency, load events, and **other GPU processes**. Official numbers are only recorded when the GPU is idle.
  - Every README number is regenerable with one command from `evals/configs/*.toml`.

## 10. Phases at a glance

| Phase | Gate | Outcome |
|---|---|---|
| P0 Notes bootstrap | — | `project/` central notes + `CLAUDE.md`; the plan lives in the repo |
| P1 CSV integration plan | — | rev 11: CSV analyst mode + pair programming (notes only) |
| 0 Setup | A | conda env, tooling, CI, `doctor` |
| C1–C3 CSV analyst in | A | the copied package, its adapter, `geekbot csv ask`; measured model swap + reproduction run |
| 1 Ingestion & storage | A | incremental, crash-safe indexing into SQLite + Chroma |
| 2 Retrieval & first evidence | A | hybrid retrieval + rerank, the metrics, golden v0; ⛳ **first ablation table** (S2.6) + the one-time v1 reference run (CP2), which **run after S3.4 and before S3.5** (rev 19) |
| 3 Answering with LangGraph | A | fast streaming answers with memory, citations, abstention; **C4** mode router + CSV branch; S3.1 to S3.4 come first, S2.6 and CP2 before S3.5 (rev 19) |
| 4 Server & UI | A | single-owner API, OpenAI-compatible endpoint, Streamlit client; **C5** CSV upload + mode picker; ⛳ **Gate A review** |
| 5 Evidence | B | bench/VRAM/latency, answer eval + judge, **C6** CSV-mode eval, web + taint, injection eval, README; ⛳ **Gate B review** |
| 6 Integration | C | MCP server; ⛳ **Gate C review** |
| Parked | — | Docker, OpenTelemetry/Phoenix, multimodal ingestion, folder watch, time-travel, LLM contextual retrieval, `Send` multi-hop, nightly GPU CI, any model over 12B; **CSV mode:** XLSX files, joins across several CSVs, hosted model providers, Gigalogy's fine-tuned model (only after a measured decision) |

## 11. Step-by-step roadmap (each step follows the §2 loop and needs your green light)

**Legend:**
- 🧠 what you learn · 🛠 what I build · ✍️ **what you write yourself** (§2) · ▶ what you run (in `conda activate geekbot`)
- 🎮 = loads a model on the GPU (run when your Gigalogy work isn't using it)
- **C1–C6** = the CSV track (rev 11). "C" stands for CSV, not Gate C.

**Order of work (rev 20):** P0 → P1 → Phase 0 → C1–C3 → Phase 1 → S2.1–S2.5 → S3.1–S3.4 (C4 after S3.4) → S2.6 → CP2 → S3.5 → Phase 4 (S4.1–S4.4, C5 after S4.4) → **S3.6** → Gate A → Phase 5 (C6 after S5.2) → Gate B → Phase 6 → Gate C. (Before rev 20: …S3.5 → S3.6 → Phase 4…; before rev 19: …Phase 2 → CP2 → Phase 3….)

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

### Step P1 — CSV integration plan (2026-10-11; writes notes only, runs nothing)
- **Plan:** PLAN rev 11 (this file).
- **Decisions:** D-036 – D-041.
- **Notes:** STATUS rows for C1–C6, the `CLAUDE.md` rules, the renumbered lesson list, the step-log template's ✍️ and look-back sections, and the P1 log.
- **Guide copy:** a verbatim copy of Gigalogy's `INTEGRATION.md` in `project/sources/`.

### Phase 0 — Setup
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S0.1 Skeleton + conda env** | src layout, `pyproject`, entry points, conda vs pip, lockfiles, ruff/mypy/pytest | `environment.yml`, `pyproject.toml`, `src/geekbot/__init__.py`, placeholder `geekbot --version` (`cli.py`, replaced in S0.2), smoke test, ruff/mypy/pytest config, pre-commit, MIT LICENSE, `.gitignore` (`venv/`, `.venv/`, `data/`, `.env`); v1 files untouched | — (built before rev 11) | one-time bootstrap (bare env → `pip install -e ".[dev]"` → `pip-compile …` creates the lock) → delete the env and re-create it with `conda env create -f environment.yml` (from the lock) → `pytest` · `ruff check .` · `mypy src` · `geekbot --version` (D-035) |
| **S0.2 Config, logging, CLI** | 12-factor config, pydantic-settings, typed CLIs | `config.py` (the target models `gemma4:12b` / `qwen2.5-coder:7b`, the prototyping models `gemma3:4b` / `qwen2.5-coder:3b` and how a profile is picked (D-042), SLOs, paths), logging, Typer `geekbot` with stub commands, `.env.example` | `geekbot config show`, which prints the settings with secrets masked | `geekbot --help` · `pytest` |
| **S0.3 `doctor` + GPU introspection** 🎮 | how Ollama loads models, `keep_alive`, VRAM math (weights + KV + buffers), the gemma4 sliding-window KV calculation | `llm/vram.py` (parsers, unit-tested on recorded outputs), `llm/models.py` (warm-up), `geekbot doctor` | the `ollama ps` output parser, tested on recorded output | `pytest` · `geekbot doctor` (warm-up loads gemma4) |
| ~~S0.4 v1 latency baseline~~ | — | **Dropped** (D-031). v1 runs once instead, at checkpoint CP2 (D-034). The latency-measurement lesson moves to S3.3 | — | — |
| **S0.5 CI** | CI pipelines, matrix builds, lock vs fresh resolution, branches + pull requests | `.github/workflows/ci.yml` (setup-miniconda); from here on, each step runs on its own branch and lands through a PR (§2) | the Ubuntu fresh-install job in `ci.yml` | `git push` to a branch, open a PR, then check the Actions tab |

### Track C, part 1 — CSV analyst in (right after Phase 0)
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **C1 Copy in `csv_analyst`** | working in a codebase you didn't just write; vendoring vs packages vs git subtree; provenance; characterization tests; gradual typing. A guided code tour: load → profile → plan → check → run SQL → agree → explain → number check | `src/csv_analyst/` from the Gigalogy working tree of 2026-10-16 (rev 13) with the cuts (§6.7), `VENDORED.md`, runtime dependencies + regenerated lock, mypy/ruff override for the copy, its no-model tests in `tests/csv_analyst/` | none in C1 (the user dropped it, rev 13); the characterization-test idea (pinning down what existing code does today, e.g. for `sql_engine/rewrite.py`) moves to C2's or a later step's ✍️ if wanted | `pytest` · `ruff check .` · `mypy src` |
| **C2 Connect it to Geekbot** | ports and adapters, dependency injection through a Protocol, connecting two settings systems, Ollama JSON mode and structured output | `csv_mode/provider.py`, Geekbot's time-limit cap, `geekbot csv describe` / `geekbot csv ask`, the read-only SQL test (§8.1) | the provider's `generate_json`, tested against a fake HTTP transport | `pytest` · one live `geekbot csv ask` 🎮 |
| **C3 CSV model: swap + reproduction run** 🎮 | sharing one GPU between two models, measuring swap cost, reproduction runs and run-to-run noise (the 2-of-3 rule) | explicit swap in `llm/models.py` (R-4), `doctor` checks the CSV model's residency, the Gigalogy benchmark through Geekbot's adapter (§9); CSV-mode speed targets set from the results; Phase 0 + C1–C3 look-back | the swap function (unload, warm up, timings), tested with a fake client, then run live | `ollama list` (is `qwen2.5-coder:7b` there?) · `geekbot doctor` · the benchmark run (idle GPU, no Gigalogy jobs running; commands given then) |

### Phase 1 — Ingestion and storage
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S1.1 Loaders** | document parsing, why MarkItDown over unstructured, page metadata, tabular schema cards | `ingest/loaders/*` (pypdf 1-indexed, MarkItDown docx/pptx/html, text/code/ipynb, csv/tsv/xlsx schema cards only, never row text (rev 16, D-051)), per-file error report | the text/code loader | `pytest tests/unit/test_loaders.py` |
| **S1.2 Chunking** | chunking trade-offs, tokens vs characters, structure-aware splitting, contextual headers | `ingest/chunking.py`, hand-written (rev 17): header → token-recursive 512/64 with exact offsets, code splitter, header text | the chunker's no-loss property test (hypothesis, added to `[dev]` here) | `pytest` |
| **S1.3 IDs and hashing** | content addressing, identity vs fingerprint, idempotency, property-based testing | `ingest/ids.py` + hypothesis tests (duplicates get distinct IDs; an insert keeps unchanged IDs) | `occurrence` numbering | `pytest` |
| **S1.4 SQLite store + FTS5** | WAL, FTS5 and BM25 internals, migrations, single-writer thread | `index/sqlite.py` (schema, migrations, writer queue, read pool), `index/lock.py` | the BM25 search query (FTS5 `MATCH` + `bm25()`, ready-only) | `pytest` |
| **S1.5 Embeddings + Chroma** | embeddings, cosine vs L2, HNSW, CPU embedding, embedding cache, fingerprints | `index/dense.py`, `PrefixedEmbeddings` (CPU `num_gpu=0`), cache, fingerprint guard | the embedding-cache lookup (key = content hash + embedder fingerprint) | `pytest` · a tiny live embed test (CPU only; also checks whether vectors are unit-length, which sets the R1 expectation) |
| **S1.6 Ingest pipeline** | the outbox pattern, consistency across two stores, crash recovery, process pools on Windows | `ingest/pipeline.py`, `jobs.py`, `consistency.py`, `geekbot ingest`, `doctor --repair` | the outbox state machine (`pending → embedding → ready / failed`; wrong moves raise) | `geekbot ingest evals/corpus` twice (2nd = 0 embeds; needs the corpus, D-032) · crash/repair test |

### Phase 2 — Retrieval and first evidence
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S2.1 Dense retrieval** | asymmetric retrieval, task prefixes, the "ready-only" filter | `retrieval/dense.py`, `geekbot search` | turning Chroma hits back into SQLite chunk rows, dropping chunks that aren't ready | `geekbot search "…"` |
| **S2.2 BM25 + RRF** | lexical vs semantic retrieval, why fuse by rank not score | `retrieval/lexical.py`, `fusion.py` | RRF (Reciprocal Rank Fusion), against my tests | `pytest` · `geekbot search --mode hybrid` |
| **S2.3 Reranking** | bi-encoder vs cross-encoder, rerank depth vs latency | `retrieval/rerank.py` (fastembed CPU), latency timing | the rerank-depth timing harness (p50/p95) | `geekbot search --mode hybrid_rerank` |
| **S2.4 Eval datasets** | qrels, BEIR, building a golden set, unanswerable questions | `evals/datasets.py`, `eval generate`, golden-set format (from `evals/datasets/golden-questions.md`); settle the deferred eval questions (D-033). **Claude writes golden v0 (the corpus and ~100 provisional items, rev 18, D-067); you review it and add your own set later** | the golden-set JSONL validator | `geekbot eval generate` 🎮 → curate |
| **S2.5 Metrics and statistics** | nDCG, Recall, MRR; bootstrap CI, permutation test, Holm | `evals/metrics.py`, `stats.py`, the Holm test family (D-033) | nDCG@k against hand-computed tests, and the paired bootstrap CI | `pytest` |
| **S2.6 Ablation + bake-offs** (runs after S3.4, before S3.5: rev 19) | ablation design, confounds, leave-one-out, threshold calibration | rungs R0–R7, LOO, embedder/reranker bake-offs, `calibrate.py`, report writer | the abstain-threshold calibration (F1-optimal sweep) | `geekbot eval retrieval --config evals/configs/ablation.toml` (CPU-heavy, long run) |
| ⛳ **Checkpoint: first evidence** (after S2.6; rev 19) | — | one-time v1 reference run (D-034) + draft README results section; Phase 1–2 look-back | — | create the `geekbot-v1` env, then the v1 run 🎮 (commands given then); we review the ablation table and the v1 reference together **before S3.5** (was: before Phase 3) |

### Phase 3 — Answering with LangGraph
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S3.1 Graph basics** 🎮 | LangGraph state, nodes, edges, compile; `ChatOllama` options | minimal `retrieve → generate` graph, `geekbot ask` (non-streaming) | the `generate` node | `geekbot ask "…"` |
| **S3.2 Prompting, packing, citations** | context windows, token budgets, prefix-KV reuse, datamarking, citation validation | `prompts/*.md`, `packing.py` (+ label bookkeeping), `citations.py` | the citation validator (`[n]` → source; invalid ones dropped) | `pytest` · `geekbot ask` |
| **S3.3 Streaming + speed** 🎮 | streaming, TTFT, measuring latency properly (warm vs cold, percentiles), thinking models, warm-up, `keep_alive` | streaming, think toggle, `RunStats` (TTFT, tok/s, loads) | TTFT and tok/s measurement in `RunStats` | `geekbot ask --stream` (watch TTFT) |
| **S3.4 Memory and follow-ups** 🎮 | conversational RAG, checkpointers, structured output | `AsyncSqliteSaver` threads, `condense` node | the `condense` output schema + prompt | multi-turn `geekbot ask --thread t1` |
| **C4 Mode switch in the graph** 🎮 | deterministic routing vs LLM routing, per-mode state, rewriting follow-ups for a tool with no memory | thread `mode`, the mode router (§7), the CSV branch (`condense` with the CSV model → `csv_ask` in a worker thread → `sanitize`), `TOOL` labels, the swap at mode entry | the mode router (a pure function) + its tests | `pytest` · a multi-turn `geekbot csv ask --thread` demo |
| **S3.5 Abstain, cite_check, sanitize** | abstention, relevance vs entailment, markdown-image exfiltration | abstention by the model's own refusal and the post-stream `cite_check`, no score gate (rev 20, D-082), measured on the 20 unanswerable golden questions; `sanitize.py` (G3, also on CSV replies) | the streaming-safe image stripper | `pytest` · unanswerable question demo |
| **S3.6 Parallelism + resilience** 🎮 (runs after C5, rev 20) | parallel branches, `RetryPolicy`, stream events, SLO smoke testing | dense ‖ lexical branches, retries, step events, live SLO smoke; Phase 3 look-back | the `RetryPolicy` and parallel-branch wiring | `pytest -m live` |

### Phase 4 — Server and UI
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S4.1 FastAPI core** | ASGI, lifespan, event-loop blocking, single-owner process | `server/app.py`, kb/jobs/health routes, writer lock at startup, heartbeat test | the `/healthz` route + the heartbeat test | `geekbot serve` · `pytest` |
| **S4.2 Threads + streamed answers** (OpenAI-compatible API left out for now, rev 20, D-084) | SSE, stateless vs stateful | thread routes, the streamed question-and-answer route (SSE: answer pieces, sources, readings) under `/v1/threads`; **not** `/v1/chat/completions`, the client SDK or the OpenAI-SDK contract test (they come back when other programs need to talk to Geekbot) | the SSE event formatter | `pytest` |
| **S4.3 Perimeter security** (outside-client items left out for now, rev 20, D-084) | threat modeling, auth, path traversal, upload hygiene | `server/security.py`, allowlist, limits, the UI's own API key, 127.0.0.1 only; the upload route `POST /v1/kb/upload` and `server/uploads.py` (rev 21) | the path allowlist check (no symlink escape) | `pytest` |
| **S4.4 Streamlit UI** | Streamlit's rerun model (the root of v1 bug #1), API-client UIs | chat / knowledge / settings / evals pages; v1 files removed | the knowledge page | `geekbot up` → try it in the browser |
| **C5 CSV mode in the API + UI** | stateful sessions in a single-owner server, upload hygiene, memory limits | `POST /v1/threads` with `mode`, the CSV upload route (S4.3 limits, `data/uploads/`), `csv_mode/sessions.py`, the UI mode picker + upload + evidence view (SQL + result table), import-linter rule (§6.7); heartbeat test with a CSV question running | the session registry (one per thread, LRU eviction, size cap) | `pytest` · `geekbot up` → pick CSV mode, upload, ask |
| ⛳ **Gate A review** | — | DoD checklist below; Phase 4 look-back | — | You run the checklist and we review together |

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
- **CSV mode (rev 11):**
  - `geekbot csv ask` answers with the SQL and a result table, and an unanswerable CSV question is refused with a reason.
  - After a CSV conversation starts, `qwen2.5-coder:7b` is 100% on the GPU. Back in chat mode, gemma4 is 100% on the GPU again (`doctor`).
  - A CSV follow-up ("and by month?") resolves.
  - An image hidden in a CSV cell is stripped from the reply.
  - The heartbeat test passes while a CSV question runs.

### Phase 5 — Evidence
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S5.1 Bench + tuning sweep** 🎮 | performance engineering, KV-cache quantization, flash attention, batch size vs VRAM | `evals/bench.py` (idle-GPU guard), sweep config, VRAM table, CSV-mode rows (§9) | the idle-GPU guard | `geekbot bench --config evals/configs/bench.toml` (idle GPU) |
| **S5.2 Answer eval + judge** 🎮 | LLM-as-judge, self-preference bias, Cohen's κ | deterministic answer metrics, `judge.py`, labeling view. **You label ~60 answers** (~1 h) | Cohen's κ + the judge-family guard | `geekbot eval answers` → label → κ report |
| **C6 CSV-mode evaluation** 🎮 | evaluating a stateful tool mode, injection through data | the ~15-item CSV follow-up set + runner (§9), the CSV-cell attack documents for S5.4 (§8.3, vector 5) | the follow-up scorer (2-of-3 rule) | `geekbot eval csv` (idle GPU; commands given then) |
| **S5.3 Web tool + taint enforcement** | taint tracking, HITL interrupts, egress control | `tools/web.py`, `taint.py`, `EgressGate`, `UNKNOWN` labeling | the `EgressGate` decision function | `pytest` · approval-prompt demo |
| **S5.4 Injection eval** | prompt-injection attack classes, measuring security | `evals/injection.py` + dataset (incl. the CSV-cell vectors) | the bypass scorer (k/N) | `geekbot eval injection` |
| **S5.5 Write-up** | technical writing, ADRs, presenting results | README v2 (with a CSV-mode section), `evolution.md`, `performance.md`, 6 ADRs, resume bullets | the first draft of the README results section | Read and review |
| ⛳ **Gate B review** | — | Phase 5 look-back | — | Every README number regenerable; bypasses 0/N; κ rule applied; SLO table measured (incl. CSV mode) |

### Phase 6 — Integration
| Step | 🧠 Learn | 🛠 Build | ✍️ You write | ▶ You run |
|---|---|---|---|---|
| **S6.1 MCP server** | the MCP protocol, tools vs resources, transports | `server/mcp.py` (`MCPServer`, mounted `/mcp`), `mcp-stdio` proxy, read-only tools | one read-only MCP tool | `claude mcp add --transport http geekbot http://127.0.0.1:8000/mcp` → call a tool |
| ~~S6.2 CSV-project adapter~~ | — | **Dropped** (D-038): the CSV project now comes in through track C1–C6 | — | — |
| ⛳ **Gate C review** | — | final look-back | — | Final review |

## 12. Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| 1 | gemma4:12b not 100% GPU-resident on 10 GB | High | CPU embedder + reranker; `doctor` residency check; ordered levers (§6.5) |
| 2 | Model swaps / cold loads mid-query | High | Warm-up, long `keep_alive`, `MAX_LOADED_MODELS ≥ 2`, load events in DoD |
| 3 | **Shared PC: your Gigalogy runs compete for GPU/CPU, invalidating numbers** | High | Idle-GPU guard in `bench`; official numbers only on an idle machine; 🎮-marked steps |
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
| 15 | Learning pace vs momentum | Medium | Small steps; lesson notes; ✍️ pieces are 20–60 lines, with the hint ladder (§2), so a stuck piece never blocks progress |
| 16 | Model swap delay, or swapping back and forth between chat and CSV conversations | High | Mode fixed per thread; swap starts at mode entry (overlaps upload + profiling); swap time measured at C3 and shown in the UI |
| 17 | Part of a model left on the CPU after a swap | High | Explicit unload with `keep_alive: 0` before warm-up (R-4); `doctor` residency check after a swap |
| 18 | The copied `csv_analyst` drifts from Gigalogy | Medium | `VENDORED.md` (source commit, cuts, local edits); re-sync only by decision, followed by a re-run of the reproduction check |
| 19 | Text inside CSV cells used as an attack | Medium | Cells treated as untrusted; G3 on CSV replies; read-only SQL test; CSV-cell attacks measured (C6, S5.4) |
| 20 | Memory use of CSV sessions (a session holds the whole table) | Medium | Upload size limit; session registry with a cap and LRU eviction (C5) |
| 21 | CSV-mode quality differs inside Geekbot (adapter, settings) | Medium | C3 reproduction run against Gigalogy's numbers; the package's measured settings stay at their defaults |
| 22 | A prototyping-profile result is reported as a real result (lighter models give different speed and quality) | Medium | Reported numbers come only from the target models; every result records its model tag (D-042) |

## 13. Testing (built incrementally, step by step)
- **Unit:**
  - RRF invariants; chunker no-loss; ID properties
  - Packing budget + labels
  - Metrics/stats against hand-computed cases
  - Citation validation; sanitizer (incl. split stream chunks)
  - Taint rules; path allowlist; judge-family guard
  - `ollama ps` / `nvidia-smi` parsers on recorded fixtures
  - CSV mode: the copied package's own tests (no model; a fake LLM); the Geekbot provider against a fake HTTP transport; the read-only SQL test; the mode router; the session registry; the sanitizer on CSV cells
- **Integration** (fake embeddings + a fake chat model with structured output and streaming; temp dirs):
  - ingest → re-ingest → edit → delete
  - Crash → `--repair`
  - A second writer is refused
  - Heartbeat
  - Graph paths (zero-LLM first turn, condense, abstain, taint interrupt → resume)
  - OpenAI-SDK contract; injection regression
  - CSV branch of the graph (fake LLM): upload → ask → follow-up → refusal
  - MCP (Gate C)
- **Live** (`pytest -m live`, needs Ollama + GPU 🎮): smoke tests, residency assertion (both modes, after a swap), SLO smoke.

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
   - CSV mode: pick it, upload a CSV, ask; the answer shows the SQL and the result table; a follow-up resolves; an unanswerable question is refused with a reason.
6. The API: the OpenAI SDK streams; a request without the API key is refused (401) and one with a foreign `Host` header too (400); an out-of-allowlist ingest returns 403; a planted image is stripped.
7. Evidence:
   - `geekbot bench --config evals/configs/bench.toml` 🎮 on an idle GPU: SLO + VRAM tables.
   - `geekbot eval injection`.
   - `geekbot eval answers` 🎮: κ rule.
   - `geekbot eval csv` 🎮: the CSV follow-up results.
8. CSV mode on its own (from C2 / C3):
   - `geekbot csv ask <file.csv> "<question>"` 🎮: the answer, the SQL and the result table.
   - `geekbot doctor` after a swap: the active mode's model is 100% on the GPU.
9. Gate C: `claude mcp add --transport http geekbot http://127.0.0.1:8000/mcp` → `search_knowledge_base`.

## 15. Prerequisites and your effort
- **Miniconda/Anaconda** installed; you run `conda env create -f environment.yml` in S0.1, and create the separate `geekbot-v1` env once at CP2 (D-034).
- **Ollama:** 0.34.2 is fine. If you update Ollama, everything is re-benched; versions are recorded.
- **Model pulls (you run them):**
  - `qwen3-embedding:0.6b` and `embeddinggemma` (S2.6, ~1 GB, CPU use)
  - the judge, Qwen3 ~8B class (~5 GB, S5.2; tag verified first)
  - `qwen2.5-coder:7b` (4.7 GB) for CSV mode: Gigalogy already uses it, so it's probably pulled; `ollama list` checks at C3
- **No model over 12B** is pulled or used (the copied package's 27B "heavy" tier is removed at C1).
- **Your time:**
  - a review and green light per step
  - **one ✍️ piece per step** (usually 20–60 lines)
  - the C1 code tour of `csv_analyst`
  - ~1 h to review golden v0 and add your own questions (S2.4, rev 18; after the prototype works)
  - ~1 h to blind-label ~60 answers (S5.2)
- **For fast answers and valid benchmarks:** keep the GPU free of other heavy work (Gigalogy runs, games, GPU-accelerated browsers). `doctor` and `bench` show what's using it.
