# Geekbot v2 — Status

_Last updated: 2026-10-17, in the S1.4 chat. Fast-prototyping mode is on (D-047). Phase 0, C1–C3 and S1.1–S1.3 are green-lit and merged. S1.4 is built and its tests pass; a mypy re-run and the green light are pending. Next: S1.5._

## Next up
**S1.4 — SQLite store + FTS5** (Phase 1, Gate A) is **built and run by the user** (status `verify`): pytest 789 passed, ruff and `pip-compile` reported good, and one mypy error in `Store.read` was fixed afterwards, so `mypy src` must be re-run before the green light. Claude ran nothing. The [S1.4 log](steps/S1.4-sqlite.md) has the design, the commands, the results and what to watch. The branch is `step/S1.4-sqlite` (from an up-to-date `v2` at `88e7e4f`). This file, the S1.3 log (the recorded merge SHA) and S1.4's new files are all uncommitted and ride along in one commit, so read `git status` before `git add -A`. After the green light and the merge, **S1.5 — Embeddings + Chroma** is next, in a new chat on `step/S1.5-embeddings`. **Decided at S1.4 (D-055, no plan change):** `index/lock.py` is a fail-fast `filelock` that `Store` takes before it opens the database; `index/sqlite.py` has one writer thread plus a pool of four read-only connections in WAL mode, numbered migrations on `PRAGMA user_version`, and `documents` / `chunks` / `chunks_fts` (FTS5 over the chunk **text only**, never the header, so ablation rung R5 stays one variable; tokenizer `porter unicode61 remove_diacritics 2`); `sync_chunks` is the set difference on `chunk_id` and returns the IDs added, removed, updated and unchanged; search is ready-only inside the same statement as `MATCH` and `LIMIT`, and `fts_match_query` quotes every typed word so FTS5 syntax in user text is inert. **For S1.5/S1.6:** embed exactly `ChunkSync.added` and drop the vectors of `removed`; `Chunk` (the stored record) has `header` and `text` but not yet the header-aware `embedding_text`; an edited document is invisible to search until it is `ready` again; the outbox state machine and the surrogate-text gap (a lone surrogate fails at insert) are S1.6's. **S1.3 — IDs and hashing** was green-lit and merged as `88e7e4f` (the squash commit of its pull request; the three CI checks passed; the user ran pytest and reported ruff and mypy clean; the [S1.3 log](steps/S1.3-ids.md) has the design, the results and what was assumed). **Decided at S1.3 (D-054, no plan change):** `ingest/ids.py` gives `doc_id` (16 hex) and `chunk_id` (24 hex) from SHA-256; the path is normalized lexically (no disk access, symlinks not followed), a chunk's text by NFC, line endings and a trimmed end; `occurrence` numbers repeats from 0, so a text that appears `k` times owns exactly the IDs `0 .. k-1`. **For S1.4–S1.6:** the store keys rows on `chunk_id` and updates position columns in place; S1.5's cache key is `content_hash` + embedder fingerprint + header mode; S1.6 refreshes the positions of kept IDs, passes absolute paths and keys "skip unchanged file" on `file_hash` plus a pipeline version. **S1.2 — Chunking** is green-lit and merged as `2dfcd71` (the squash commit of its pull request; the user ran ruff, mypy and pytest and reported all good; the [S1.2 log](steps/S1.2-chunking.md) has what was pasted). **Decided at S1.2:** the chunker is hand-written, not LangChain's splitters (D-053, PLAN **rev 17**); a chunk is an exact slice of the document with its page, heading path and a separate contextual header; token counts are an estimate until S1.5 compares them with the embedder's, and S1.5's embedding-cache key must cover the header mode. **S1.2 consumed S1.1's output:** it iterates a `LoadedDocument`'s `segments` to keep `page`, uses `language` to pick the code splitter and `title` for the contextual header. Its ✍️ piece (the no-loss property test) uses hypothesis, which therefore entered `[dev]` at S1.2 (D-035's own rule). **Decided at S1.1:** chat mode indexes a schema card only for CSV, TSV and Excel files, never their rows (D-051, which supersedes D-016; PLAN **rev 16**); the loader design is D-052 (see the [S1.1 log](steps/S1.1-loaders.md) for what each format yields and what is dropped on purpose). Track C part 1 (C1–C3) is done and merged: `geekbot csv ask|describe|swap|repro`, `geekbot doctor --mode csv`, an explicit measured model swap, and a reproduction of the Gigalogy benchmark inside the band fixed beforehand (33/40 solved, 32.3 correct per run, 0 of 30 invented on the target models). The [C3 log](steps/C3-csv-swap-repro.md) has the open items from that track (none blocks S1.2).
- **S1.3** was merged as `88e7e4f` (the squash commit of its pull request). **S1.2** as `2dfcd71` (the squash commit of its pull request; branch tip `6a9e0cd`). **S1.1** as `5991e92` (the squash commit of its pull request). **C3** as `86d2cc2` (PR #4). **C2** as `9d5dab5` (PR #3). The merged step branches were deleted locally and on the remote after S1.1.
- **Reply style (D-050):** chat replies use the caveman skill (level full) by default; code, commits, PR text and repo notes stay normal prose.
- **Mode (D-047):** fast prototyping. Per step: Claude's outline (with some interesting context), one build, one summary; the user runs the tests, gives the green light and pushes. New lessons are deferred; ✍️ pieces Claude writes go on the backlog below.
- **C1 handoff:** see the C1 step log (the vendored package, its known gaps, what to watch in C2/C3).
- **Workflow since S0.5 (D-044):** every step starts with `git switch -c step/<id>-<slug>` from an up-to-date `v2`, ends with a push and a pull request into `v2` (base `v2`, not `main`), and is squash-merged after the three CI checks are green. First mistake to avoid (it happened twice, at Phase 0 and at C3): committing before creating the branch. Run `git switch -c step/<id>-<slug>` as the very first command of the step, and read `git status` before `git add -A`.
- Merged on `v2`: C1 as `3bd19db` (merge commit of PR #2); S0.5 as `e800a2b` (merge commit of PR #1); before them S0.3 `6a1dba2`, S0.2 `2febc8a`, P1 + S0.1 `0d09179`. D-046: the rule is squash; the repo setting to allow only squash is the user's to change in GitHub.
- `C:\Work\Gigalogy` is no longer at that path (the user's rename or move); the only copy of the package source is `scratch/gigalogy-latest/` (git-ignored, read-only, not backed up for lack of disk space).
- Older handoffs below are history; where they disagree with the table, the table wins.

**Handoff from the 2026-10-14 chat (notes only, nothing run):**
- **New file `project/README.md`:** the user-facing guide to running a step chat (one step = one chat). It is uncommitted; commit it together with P1. PLAN §2.1's folder tree doesn't list it (rev 12 didn't change the tree). **S0.5** must add the exact branch and pull-request commands to its section 4.
- **Decided: D-042, plan rev 12 (prototyping profile).** While building, the models are `gemma3:4b` (chat) and `qwen2.5-coder:3b` (CSV mode, Gigalogy's own "fast" tier). The target models stay `gemma4:12b` and `qwen2.5-coder:7b`, and every reported number comes from them and names its model. Recorded in PLAN §4, §6.5, the S0.2 row and risk 22, in CLAUDE.md rule 4, and in `project/DECISIONS.md`. **S0.2's design** must settle how the profile is picked and which one applies when nothing is set; **S0.3's `doctor`** must check the models of the active profile. The user needs `gemma3:4b` pulled (new) and `qwen2.5-coder:3b` (probably already pulled for Gigalogy) by S0.3 at the latest.
- **P1 is still waiting** for the user's review (PLAN §2, §6.7, §11, plus the rev 12 additions in §4 and §6.5) and green light. Close it first. Nothing from P1 or S0.1 is committed yet; the P1 commit also carries rev 12 and `project/README.md`.

**Handoff from the P1 chat (2026-10-11):**
- **The plan is now rev 11** (PLAN.md §4; D-036 – D-041):
  - Gigalogy's `csv_analyst` becomes a second conversation mode, "CSV analyst". It is copied into `src/csv_analyst/` at C1 (design: PLAN §6.7)
  - CSV mode uses `qwen2.5-coder:7b`, swapped with `gemma4:12b`; they're never on the GPU together
  - New track **C1–C6**; S6.2 is dropped
  - **Pair programming:** every step names one ✍️ piece that you write; the ✍️ column is in PLAN §11
  - From S0.5, a branch + pull request per step; a look-back at the end of each phase
- **P1 wrote notes only:** PLAN, DECISIONS, this file, `CLAUDE.md`, the lesson list, the step-log template, the P1 log, and `project/sources/csv-analyst-INTEGRATION.md`. No code; nothing was run, except one read-only listing command in the Gigalogy folder, a no-run slip recorded in the P1 log.
- **Open question for S1.1's design:** does chat mode still need row text for small CSVs (D-016), now that CSV mode exists? Claude recommends a schema card only (PLAN §6.7).
- **Commit:** ask the user for the P1 commit SHA and record it in the P1 row below.

**Handoff from the P0 chats:**
- **P0 is green-lit** (2026-10-07). It wrote only notes and blank templates. **No v2 code exists yet**; the repo still contains the v1 code.
- **Commits:** `f7c2100` (the P0 notes) plus the P0 verification commit. Ask the user for the verification commit's SHA and record it in the P0 row below.
- **The plan is now rev 9** (PLAN.md §4):
  - S0.4 (the v1 latency baseline) is dropped (D-031). Instead, v1 runs **once**, at checkpoint CP2, in its own `geekbot-v1` env with `gemma4:12b`, measuring speed and search quality (D-034)
  - blank templates for the test documents and questions are in `evals/` (D-032)
  - some eval-design questions wait for S2.4–S2.6 (D-033)
  - factual fixes from the P0 self-audit (D-030)
- **Codex review pending.** When the user pastes it, save it verbatim as `project/reviews/codex-review-1.md`, followed by a resolution table (same structure as the ChatGPT reviews), and list it in `project/reviews/README.md`. Any plan change it causes follows the usual approval rule.
- **Nothing has been installed.** The `geekbot` conda env doesn't exist yet. Whether Miniconda/Anaconda is installed **hasn't been verified**, so S0.1 starts by asking the user to run `conda --version`.
- **Measured facts** (once, before the no-run rule):
  - RTX 3080 with 10,240 MiB VRAM (1,311 MiB in use by the desktop at check time)
  - Ollama 0.34.2, with `gemma4:12b` (11.9B, Q4_K_M, 7.6 GB) and `nomic-embed-text` pulled
  - system Python 3.13.9
  - See PLAN.md §1.

## Steps
Status values: `todo → design → build → verify → green-lit`, or `blocked`, or `dropped` (removed by a plan change).

| Step | Title | Gate | Status | Chat date | Commit | Log |
|---|---|---|---|---|---|---|
| P0 | Project notes bootstrap | — | green-lit | 2026-10-07 | `f7c2100` + `b57696f` | [P0](steps/P0-notes-bootstrap.md) |
| P1 | CSV analyst integration plan (rev 11 + rev 12, notes only) | — | green-lit (2026-10-15; user skimmed and approved) | 2026-10-11 | `0d09179` | [P1](steps/P1-csv-integration-plan.md) |
| S0.1 | Skeleton + conda env | A | green-lit (2026-10-15; all 4 checks green on the rebuilt env) | 2026-10-07 – 15 | `0d09179` | [S0.1](steps/S0.1-skeleton.md) |
| S0.2 | Config, logging, CLI | A | green-lit (2026-10-15; pytest 31 passed, ruff + mypy clean) | 2026-10-15 | `2febc8a` | [S0.2](steps/S0.2-config-cli.md) |
| S0.3 | `doctor` + GPU introspection 🎮 | A | green-lit (2026-10-15; 68 tests, live doctor run clean on the proto profile) | 2026-10-15 | `6a1dba2` | [S0.3](steps/S0.3-doctor-gpu.md) |
| S0.4 | ~~v1 latency baseline~~ | A | dropped (D-031) | | | |
| S0.5 | CI (+ branch and pull request per step from here) | A | green-lit (2026-10-16; all three CI checks green on the first run) | 2026-10-16 | `e800a2b` (merge commit of PR #1; branch commits `1e4a869` + `8c81f1b`) | [S0.5](steps/S0.5-ci.md) |
| C1 | Copy in `csv_analyst` | A | green-lit (2026-10-16; 382 passed, 10 skipped, 1 xfailed) | 2026-10-16 | `3bd19db` (PR #2; branch commit `adea115`) | [C1](steps/C1-csv-analyst.md) |
| C2 | Connect it to Geekbot (`geekbot csv ask`) | A | green-lit (2026-10-16; 445 passed, 10 skipped, 1 xfailed; ruff, format, mypy clean; live `csv ask` and live tests passed on the proto profile) | 2026-10-16 | `9d5dab5` (PR #3, squash) | [C2](steps/C2-csv-adapter.md) |
| C3 | CSV model: swap + reproduction run 🎮 | A | green-lit (2026-10-16; 557 passed, 10 skipped, 1 xfailed; ruff, format, mypy clean; target-model reproduction inside the band fixed beforehand: 33/40 solved, 32.3 per run, 0 of 30 invented) | 2026-10-16 | `86d2cc2` (PR #4, squash; branch commits `b186795`, `ef1aa6f`, `32277ab`) | [C3](steps/C3-csv-swap-repro.md) |
| S1.1 | Loaders (chat mode indexes schema cards, never CSV rows: D-051) | A | green-lit (2026-10-16; 607 passed, 10 skipped, 1 xfailed; ruff, format, mypy clean) | 2026-10-16 | `5991e92` (squash commit of the pull request; branch tip `6f2726d`) | [S1.1](steps/S1.1-loaders.md) |
| S1.2 | Chunking (hand-written splitter, D-053) | A | green-lit (2026-10-16; user ran ruff, mypy and pytest and reported all good; counts not pasted) | 2026-10-16 | `2dfcd71` (squash commit of the pull request; branch tip `6a9e0cd`) | [S1.2](steps/S1.2-chunking.md) |
| S1.3 | IDs and hashing (D-054) | A | green-lit (2026-10-16; 695 passed, 11 skipped, 4 deselected, 1 xfailed; ruff and mypy reported clean by the user) | 2026-10-16 | `88e7e4f` (squash commit of the pull request; branch tip `072abba`) | [S1.3](steps/S1.3-ids.md) |
| S1.4 | SQLite store + FTS5 (D-055) | A | verify (2026-10-17; pytest 789 passed, 11 skipped, 4 deselected, 1 xfailed; one mypy error fixed, `mypy src` re-run pending) | 2026-10-16 – 17 | | [S1.4](steps/S1.4-sqlite.md) |
| S1.5 | Embeddings + Chroma | A | todo | | | |
| S1.6 | Ingest pipeline (needs the corpus, D-032) | A | todo | | | |
| S2.1 | Dense retrieval | A | todo | | | |
| S2.2 | BM25 + RRF | A | todo | | | |
| S2.3 | Reranking | A | todo | | | |
| S2.4 | Eval datasets (user curates golden set) | A | todo | | | |
| S2.5 | Metrics and statistics | A | todo | | | |
| S2.6 | Ablation + bake-offs | A | todo | | | |
| CP2 | ⛳ Checkpoint: first evidence (+ one-time v1 reference run 🎮, D-034) | A | todo | | | |
| S3.1 | Graph basics 🎮 | A | todo | | | |
| S3.2 | Prompting, packing, citations | A | todo | | | |
| S3.3 | Streaming + speed 🎮 | A | todo | | | |
| S3.4 | Memory and follow-ups 🎮 | A | todo | | | |
| C4 | Mode switch in the graph 🎮 | A | todo | | | |
| S3.5 | Abstain, cite_check, sanitize | A | todo | | | |
| S3.6 | Parallelism + resilience 🎮 | A | todo | | | |
| S4.1 | FastAPI core | A | todo | | | |
| S4.2 | Threads + OpenAI-compatible API | A | todo | | | |
| S4.3 | Perimeter security | A | todo | | | |
| S4.4 | Streamlit UI (removes v1 files) | A | todo | | | |
| C5 | CSV mode in the API + UI | A | todo | | | |
| GA | ⛳ Gate A review | A | todo | | | |
| S5.1 | Bench + tuning sweep 🎮 | B | todo | | | |
| S5.2 | Answer eval + judge 🎮 (user labels ~60) | B | todo | | | |
| C6 | CSV-mode evaluation 🎮 | B | todo | | | |
| S5.3 | Web tool + taint enforcement | B | todo | | | |
| S5.4 | Injection eval | B | todo | | | |
| S5.5 | Write-up (README, ADRs, resume bullets) | B | todo | | | |
| GB | ⛳ Gate B review | B | todo | | | |
| S6.1 | MCP server | C | todo | | | |
| S6.2 | ~~CSV-project adapter~~ | C | dropped (D-038) | | | |
| GC | ⛳ Gate C review | C | todo | | | |

Legend:
- Rows are in the order the work happens.
- 🎮 loads a model on the GPU (run when Gigalogy isn't using it).
- **C1–C6** = the CSV track ("C" for CSV, not Gate C).
- Every step from S0.2 on has a ✍️ piece (PLAN §11 names it). The user wrote S0.2's, S0.3's and S0.5's. From C2 on (fast-prototyping mode, D-047) Claude writes them and they go on the ✍️ backlog below.

## ✍️ backlog (written by Claude in prototyping mode, D-047; to redo as exercises later)
| Step | Piece | Where | Spec and tests |
|---|---|---|---|
| C2 | `OllamaCsvProvider.generate_json`: build the Ollama request, map failures to `LLMProviderError`, return the raw reply | `src/geekbot/csv_mode/provider.py` | the docstring of the method; `tests/unit/test_csv_provider.py` |
| C3 | `swap_to`: unload the other mode's model, wait until Ollama stops listing it, warm the new one up, return the timings | `src/geekbot/llm/models.py` | the docstring of the function; `tests/unit/test_swap.py` (fake Ollama and clock), `tests/live/test_swap_live.py` |
| S1.1 | `load_text`: read a text, Markdown or code file as one segment (kind and language from the extension, UTF-8 then a guessed encoding, binary / oversized / empty files are errors) | `src/geekbot/ingest/loaders/text.py` | the docstring of the function; the text, Markdown and code tests in `tests/unit/test_loaders.py` |
| S1.2 | The chunker's no-loss property test: arbitrary text in, every chunk an exact slice and every non-whitespace character in some chunk (plus three more properties: budgets and order, one page per chunk, the token estimate's subadditivity) | `tests/unit/test_chunking.py` | the four `@given` tests at the end of the file; they test `src/geekbot/ingest/chunking.py` |
| S1.3 | `assign_occurrences`: number the repeats of a chunk hash within one document from 0, in document order | `src/geekbot/ingest/ids.py` | the docstring of the function; the two numbering examples and the `@given` tests at the end of `tests/unit/test_ids.py` (numbering, distinct IDs, insert adds one ID, removal removes one, order does not change the set) |
| S1.4 | The BM25 search query: `fts_match_query` (typed text to a safe FTS5 expression: each word quoted, words OR-joined) and `Store.search_bm25` (`MATCH` + `bm25()`, ready documents only, best first, ties by `chunk_id`) | `src/geekbot/index/sqlite.py` | the docstrings of both functions; the query-builder and search tests in `tests/unit/test_sqlite_store.py`, including the two `@given` properties at the end |

To redo one: restore the stub (`raise NotImplementedError`), keep the tests, and write it yourself; the hint ladder still applies.

## Measured results so far
**C3, 2026-10-16, target profile** (`gemma4:12b` chat, `qwen2.5-coder:7b` CSV, Ollama 0.34.2, RTX 3080, idle GPU; details in the [C3 log](steps/C3-csv-swap-repro.md); result file `evals/csv/results/repro-target-qwen2.5-coder-7b-20261016-1657.json`):
- **Swap** (3 round trips): chat→csv median 2.7 s (max 4.3 s); csv→chat median 4.5 s (max 4.5 s); unload ≈ 0.1 s; the new model was 100% on the GPU and the old one gone after every swap. The setup swap that first loaded `gemma4:12b` took 13.5 s (it depends on what was loaded before, so it is not a swap time).
- **VRAM in use** (desktop included, ≈ 870 MiB idle): CSV mode 6,457 MiB (3,783 MiB free); chat mode 8,981 MiB (1,259 MiB free).
- **Reproduction run** (Gigalogy's 40 questions × 3 seed sets, official): 33/40 solved (source project: 35/40), 32.3 correct per run (34.0), 0 of 30 invented answers (0), 2.9 s per question on average (3.0 s), p95 7.7 s, slowest 11.9 s; 0 model loads during questions. Inside the acceptance band fixed before the run; at its low edge.
- Earlier numbers on the light profile (`gemma3:4b`, `qwen2.5-coder:3b`) were plumbing checks and are not results.
The first retrieval and chat-latency numbers arrive at S0.3's successors (S2.x, S3.3) and at the CP2 ablation table.
