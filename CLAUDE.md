# Geekbot v2 — instructions for every Claude chat in this repo

Geekbot is being rebuilt (v1 → v2) into a local RAG engine that **measures retrieval quality, controls inference cost, and treats retrieved content as untrusted data**. The full plan is **`project/PLAN.md`**, the single source of truth. This file holds only the standing rules and the session protocol; it never duplicates the plan.

## Standing rules
These rules change only when the user says so. Any change is recorded in `project/DECISIONS.md`.

1. **No-run policy.** Never execute anything on this PC: no shell commands, installs, tests, Ollama/model calls, benchmarks or git. Only read and write files in this repo.
   - Give the user the exact commands to run in the `geekbot` conda env, and work from the output they paste back.
   - The only exception is a specific command the user explicitly allows in the current chat.
   - *Why:* this PC also runs the user's CSV-query project, and its GPU/CPU must not be disturbed.
2. **Environment:** conda env `geekbot` (`environment.yml`, Python 3.12), with the project pip-installed from `pyproject.toml` and `requirements/lock-win.txt`. No uv, no venv.
   - The only other env is `geekbot-v1`, for the one-time v1 reference run at checkpoint CP2 (`project/DECISIONS.md` D-034).
3. **Stack:**
   - LangChain 1.x + LangGraph: an explicit `StateGraph`, **no ReAct/`create_agent` loop**.
   - Ollama.
   - SQLite + FTS5 (source of truth) and Chroma (derived dense index).
   - FastAPI as the **only** storage-owning process.
   - Streamlit as an API client only.
4. **Models:** the default chat model is **`gemma4:12b`**. **Never use or suggest a model over 12B.**
   - Embedder and reranker run on **CPU**, so gemma4 stays 100% GPU-resident on the RTX 3080 (10 GB).
   - Thinking is off by default.
5. **Speed is a hard requirement.** Respect the latency SLOs in PLAN.md §6.5. Never state a number that hasn't been measured.
6. **Teacher mode.** The user is learning every component. Follow the per-step loop in PLAN.md §2:
   1. lesson
   2. **design approval**
   3. small build
   4. user verifies
   5. walkthrough + questions (+ optional "Your turn" exercise)
   6. **green light**
   7. commit command

   Never skip design approval or the green light.
7. **One step at a time.** Don't start the next step without the user's green light. Don't build anything from the Parked list in PLAN.md §10.
8. **Claims discipline:**
   - BEIR is *standardized retrieval evidence*; the golden corpus is *system-specific behavioral evaluation*.
   - Security guarantees and empirical results are reported separately.
   - SLOs are targets until measured.

## Session protocol
**Start of every chat**
1. Read, in this order:
   1. `project/STATUS.md`
   2. `project/PLAN.md` (at least §2, §6 and the current step's row in §11)
   3. `project/DECISIONS.md`
   4. the most recent log in `project/steps/`
2. Tell the user where things stand (the "Next up" in STATUS.md) and confirm which step(s) this chat covers.
3. Restate the no-run policy in one line.
4. If STATUS.md, the notes and the code disagree, **stop and ask** before doing anything.

**During a step**
- Work only on the confirmed step(s), with small, reviewable diffs.
- Write the lesson note to `project/learn/NN-<topic>.md` and add it to `project/learn/README.md`.
- **Never edit `project/PLAN.md` silently.** Propose the change and get approval. Then add a line to PLAN.md §4 (revision history) and an entry to `project/DECISIONS.md`.

**End of every step**
1. Write the step log `project/steps/<step-id>-<slug>.md` from `project/steps/_TEMPLATE.md`.
2. Update `project/STATUS.md`: status, date, the "Next up" handoff.
3. Add any new decisions to `project/DECISIONS.md`.
4. Give the user the exact git commands. Code and notes go in one commit.
5. The user reports the commit SHA, which is recorded in STATUS.md (at the latest at the start of the next chat).

## Code conventions
- src layout (`src/geekbot/`), Python 3.12, full type hints (mypy strict on core), ruff for lint + format.
- pytest (+ hypothesis for property tests). Every feature ships with tests. Tests that need Ollama/GPU are marked `@pytest.mark.live`.
- Architectural invariants:
  - Blocking I/O never runs on the event loop (PLAN.md §6.3).
  - Only the server process owns storage (§6.2).
  - `packing.py` is the only prompt assembler and taint-labeling point (§8.2).
- Docstrings explain *why*, not just *what*. The code is also teaching material.
- Commit messages follow Conventional Commits and reference the step, e.g. `feat(ingest): S1.1 loaders`. Messages drafted by Claude end with a `Co-Authored-By:` trailer for the Claude model used.

## Map
- `project/PLAN.md`: the plan
- `project/STATUS.md`: progress + "Next up"
- `project/DECISIONS.md`: decision log
- `project/steps/`: per-step logs
- `project/learn/`: lesson notes
- `project/reviews/`: external reviews + resolutions
- v1 code (`app.py`, `rag_pipeline.py`, `storage_utils.py`, `get_embedding_function.py`, `requirements.txt`) stays untouched until step S4.4 removes it. The git tag `v1.0` preserves it.
