# Geekbot v2 — Status

_Last updated: 2026-10-07, in the P0 verification chat._

## Next up
**S0.1 — Skeleton + conda env** (Phase 0, Gate A): set up the project skeleton and the `geekbot` conda environment. Start it in a **new chat**; that chat reads `CLAUDE.md` → this file.

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
| P0 | Project notes bootstrap | — | green-lit | 2026-10-07 | `f7c2100` + verification commit (SHA pending) | [P0](steps/P0-notes-bootstrap.md) |
| S0.1 | Skeleton + conda env | A | todo | | | |
| S0.2 | Config, logging, CLI | A | todo | | | |
| S0.3 | `doctor` + GPU introspection 🎮 | A | todo | | | |
| S0.4 | ~~v1 latency baseline~~ | A | dropped (D-031) | | | |
| S0.5 | CI | A | todo | | | |
| S1.1 | Loaders | A | todo | | | |
| S1.2 | Chunking | A | todo | | | |
| S1.3 | IDs and hashing ✋ | A | todo | | | |
| S1.4 | SQLite store + FTS5 | A | todo | | | |
| S1.5 | Embeddings + Chroma | A | todo | | | |
| S1.6 | Ingest pipeline (needs the corpus, D-032) | A | todo | | | |
| S2.1 | Dense retrieval | A | todo | | | |
| S2.2 | BM25 + RRF ✋ | A | todo | | | |
| S2.3 | Reranking | A | todo | | | |
| S2.4 | Eval datasets (user curates golden set) | A | todo | | | |
| S2.5 | Metrics and statistics ✋ | A | todo | | | |
| S2.6 | Ablation + bake-offs | A | todo | | | |
| CP2 | ⛳ Checkpoint: first evidence (+ one-time v1 reference run 🎮, D-034) | A | todo | | | |
| S3.1 | Graph basics 🎮 | A | todo | | | |
| S3.2 | Prompting, packing, citations | A | todo | | | |
| S3.3 | Streaming + speed 🎮 | A | todo | | | |
| S3.4 | Memory and follow-ups 🎮 | A | todo | | | |
| S3.5 | Abstain, cite_check, sanitize | A | todo | | | |
| S3.6 | Parallelism + resilience 🎮 | A | todo | | | |
| S4.1 | FastAPI core | A | todo | | | |
| S4.2 | Threads + OpenAI-compatible API | A | todo | | | |
| S4.3 | Perimeter security | A | todo | | | |
| S4.4 | Streamlit UI (removes v1 files) | A | todo | | | |
| GA | ⛳ Gate A review | A | todo | | | |
| S5.1 | Bench + tuning sweep 🎮 | B | todo | | | |
| S5.2 | Answer eval + judge 🎮 (user labels ~60) | B | todo | | | |
| S5.3 | Web tool + taint enforcement | B | todo | | | |
| S5.4 | Injection eval | B | todo | | | |
| S5.5 | Write-up (README, ADRs, resume bullets) | B | todo | | | |
| GB | ⛳ Gate B review | B | todo | | | |
| S6.1 | MCP server | C | todo | | | |
| S6.2 | CSV-project adapter | C | todo | | | |
| GC | ⛳ Gate C review | C | todo | | | |

Legend: 🎮 loads a model on the GPU (run when the CSV project isn't using it) · ✋ optional "Your turn" exercise.

## Measured results so far
_None yet. The first numbers arrive at S0.3 (`doctor`: VRAM use and GPU residency) and at the CP2 ablation table._
