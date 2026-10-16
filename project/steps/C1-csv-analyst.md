# C1 — Copy in `csv_analyst`

- **Phase / Gate:** Track C, part 1 (right after Phase 0), Gate A
- **Chat date(s):** 2026-10-16
- **Status:** green-lit (2026-10-16; 382 passed, 10 skipped, 1 xfailed; ruff, format check and mypy clean)
- **Commit:** `3bd19db` (merge commit of PR #2; branch commit `adea115`)
- **Lesson note:** [learn/05-vendoring.md](../learn/05-vendoring.md)

## Goal
Bring the Gigalogy CSV analyst into the repo as a vendored package, with provenance, the agreed cuts and passing no-model tests (PLAN.md §6.7, §11 C1).

## Approved design
Approved by the user on 2026-10-16, with these changes to the plan (rev 13, D-045):
- **Source:** the Gigalogy working tree of 2026-10-16 (newer than `f482455`, uncommitted edits), not the commit. The user chose "use the files I just copied".
- **Pandas engine kept**, unused (the cut would need more than small edits).
- **No ✍️ piece** in C1 (the user's choice: "keep minimal edits").
- Cuts: `interface/`, the two hosted providers, the 27B tier, 8 tests needing `evaluation/` or the API.
- Quality gates: ruff excludes the copy; mypy ignores its errors (typing debt); `pyproject.toml` gets the runtime dependencies (pandas, numpy, pyarrow, pydantic, requests, charset-normalizer, sqlglot) and builds both packages.
- Copy via `robocopy` run by the user (byte-exact), local edits by Claude marked `GEEKBOT EDIT (C1)`.

## ✍️ Your piece
None in C1 (user's choice; recorded in PLAN rev 13). The characterization-test idea for `sql_engine/rewrite.py` can be offered again in C2 or later.

## Files changed
- `src/csv_analyst/` (46 copied files), `src/csv_analyst/VENDORED.md` (new)
- Local edits: `config.py` (heavy tier), `session.py` (hosted providers)
- `tests/csv_analyst/vendored/` (19 copied files; 2 edited)
- `pyproject.toml`, `.pre-commit-config.yaml`, `.gitignore` (`/scratch/`)
- `project/PLAN.md` (rev 13), `project/DECISIONS.md` (D-045), `project/learn/05-vendoring.md` + index, `project/STATUS.md`, `project/steps/S0.5-ci.md` (S0.5 SHA)

## Commands the user ran + results
- Verified Gigalogy at `f482455` (master); then copied the whole working tree (including `.venv` and `.git`, ~42,700 files) into the git-ignored `scratch/gigalogy-latest/`.
- `robocopy` of the package (46 files) and of the tests (19 files); `git status` showed no `scratch/`.
- After the build: lock regenerated and env updated (pandas 3.0.6, numpy 2.5.4, pyarrow 26.0.0, sqlglot 30.23.0; the user's `gigalogy` conda env has pandas 3.0.5, numpy 2.5.3, pyarrow 25.0.1, sqlglot 30.21.0, Python 3.13.15; Geekbot's env is Python 3.12).
- First `pytest`: 380 passed, 13 failed. Causes: 12 tests assume Gigalogy's repo layout (relative paths to `data/hospital`, `data/sales`, `csv_analyst/llm/prompts`); 1 `test_facts` failure.
- Fix: `tests/csv_analyst/conftest.py` (runs the vendored tests from `src/`, skips the 10 dataset tests). Second run: 382 passed, 10 skipped, 1 failed.
- The `test_facts` failure was diagnosed with a throwaway script (outside both repos): `visit_date` is `datetime64` after profiling, so the sentence says "holds dates" (the test expects "holds ISO text"). The profiler converts ISO text dates on purpose. Marked expected-fail, cause unconfirmed (see VENDORED.md).
- A planned comparison run of that test inside Gigalogy's env could not be done: `C:\Work\Gigalogy` was no longer reachable (the user thinks it was renamed or moved; Claude never ran anything against it). `scratch/gigalogy-latest` is now the only copy and is treated as read-only.

- Final run: `pytest` 382 passed, 10 skipped, 1 xfailed (40.9 s) · `ruff check .` clean · `ruff format --check .` 19 files · `mypy src` clean, 46 source files.
- The user could not back up `scratch/gigalogy-latest` (no disk space); it is the only copy of the source and stays read-only.

## Review + walkthrough Q&A
- Questions asked (the user asked for the answers): why exclude the copy from ruff instead of fixing its style (keeps the copy identical to its source so re-sync is a copy plus four marked edits; the cost is recorded as typing debt); why an expected failure instead of deleting or fixing the test (deleting hides the change, fixing means guessing without the original, xfail keeps it visible and flags when it starts passing).
- Claude's review of its own diff: four small marked edits (`config.py`, `session.py`, two vendored tests), plus `tests/csv_analyst/conftest.py`. No ✍️ piece in C1.

## Decisions
D-045 (plan rev 13).

## Open issues / follow-ups
- Nothing here was run by Claude. Possible surprises: a vendored test that depends on something not copied; `sqlglot>=30.0` resolution; Python 3.13 wheels for the new dependencies in the Ubuntu CI job.
- The user's copy of Gigalogy's working tree includes a `CLAUDE.md` with instructions for a different project ("talk like caveman"); Claude ignored it. It stays in `scratch/` (git-ignored), nothing from it was copied.
- Provenance is weak until Gigalogy's edits are committed (see `VENDORED.md`).
- The pandas engine (and its subprocess sandbox) is unused code in the repo; revisit after C3.
- `requests` is used only by the package's Ollama provider; C2 replaces the provider.
- **CI flake, seen once after the merge (found in C2's chat, 2026-10-16):** on the Ubuntu job (Python 3.13.16), `tests/csv_analyst/vendored/test_runner.py::test_execute_unsupported_result_type_fails` hit the sandbox's 15 s timeout (15.02 s) and passed when the job was re-run. The failed run was CI run #5, the push to `v2` after the merge of PR #2 (`3bd19db`). The code path is as cheap as the other 23 tests in that file, which passed in the same run, so the cause is unconfirmed (most likely one slow or stalled sandbox process start on the shared runner). The timeout path in `runner.py` discards the child's output, so a recurrence can't be diagnosed from the log. If it recurs: a small fix on its own branch, recorded in `VENDORED.md`.

## Handoff to the next chat
- **State:** `src/csv_analyst/` is in place with its tests; Geekbot code does not import it yet. Python 3.12 (Geekbot) vs 3.13 (the user's Gigalogy env) is the one known environment difference.
- **Next:** C2 (connect it to Geekbot): `geekbot.csv_mode.provider` implementing `LLMProvider.generate_json` with Geekbot's Ollama settings (a Protocol, tested on a fake HTTP transport), Geekbot's per-question time cap, `geekbot csv describe` / `csv ask`, and the read-only SQL test (PLAN §8.1). Branch `step/C2-csv-adapter`, created before any change. The C2 ✍️ in the plan is the provider's `generate_json`; C1 had none.
- **Watch in C2/C3:** the `test_facts` expected-failure (dates label); `requests` in the package's own Ollama provider vs Geekbot's httpx; the package's 10-minute time limit (too slow for Geekbot's speed goal).
- **User-side open items:** back up or free space for `scratch/gigalogy-latest`; commit Gigalogy's edits so `VENDORED.md` can name a real SHA; the Gigalogy folder is no longer at `C:\Work\Gigalogy` (renamed or moved by the user, not by Claude).
