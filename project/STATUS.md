# Geekbot v2 — Status

_Last updated: 2026-10-16, in a multi-step chat (user's choice: S0.1 → S0.2 → S0.3 → S0.5 → C1–C3 → Phase 1 in one chat). P1, S0.1, S0.2, S0.3 and S0.5 are green-lit; Phase 0 is complete after S0.5 merges._

## Next up
**C1 — Copy in `csv_analyst`** (Track C, Gate A), on branch `step/C1-csv-analyst`. The multi-step chat plan: C1 → C2 → C3 → Phase 1, then a handoff written here.
- **Before C1's design:** the user checks that `C:\Work\Gigalogy` is still at commit `f482455` (`git -C C:\Work\Gigalogy log --oneline -3`).
- **Record at the start of C1:** the squash-commit SHA of the S0.5 pull request on `v2` (`git switch v2`, `git pull`, `git log -1 --format=%h`), in the S0.5 row.
- **Workflow since S0.5 (D-044):** every step starts with `git switch -c step/<id>-<slug>` from an up-to-date `v2`, ends with a push and a pull request into `v2` (base `v2`, not `main`), and is squash-merged after the three CI checks are green. First mistake to avoid (it happened once): committing before creating the branch.
- `v2` on GitHub now holds the work through S0.3 (`6a1dba2`); S0.5 lands through its own pull request.
- The P1 + S0.1 commit is `0d09179`.
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
| S0.5 | CI (+ branch and pull request per step from here) | A | green-lit (2026-10-16; all three CI checks green on the first run) | 2026-10-16 | `1e4a869` on the branch; squash SHA on `v2` to be recorded at the start of C1 | [S0.5](steps/S0.5-ci.md) |
| C1 | Copy in `csv_analyst` | A | todo | | | |
| C2 | Connect it to Geekbot (`geekbot csv ask`) | A | todo | | | |
| C3 | CSV model: swap + reproduction run 🎮 | A | todo | | | |
| S1.1 | Loaders (open question: CSV row text, PLAN §6.7) | A | todo | | | |
| S1.2 | Chunking | A | todo | | | |
| S1.3 | IDs and hashing | A | todo | | | |
| S1.4 | SQLite store + FTS5 | A | todo | | | |
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
- Every step from S0.2 on has a ✍️ piece you write yourself; PLAN §11 names it.

## Measured results so far
_None yet. The first numbers arrive at S0.3 (`doctor`: VRAM use and GPU residency), at C3 (CSV-model swap time + reproduction run) and at the CP2 ablation table._
