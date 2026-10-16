# C2 — Connect it to Geekbot (`geekbot csv ask`)

- **Phase / Gate:** Track C, part 1 (right after Phase 0), Gate A
- **Chat date(s):** 2026-10-16
- **Status:** green-lit (2026-10-16; 445 passed, 10 skipped, 1 xfailed; ruff, format and mypy clean)
- **Commit:** <SHA reported by the user>
- **Lesson note:** [learn/06-ports-adapters.md](../learn/06-ports-adapters.md)

## Goal
Connect the vendored `csv_analyst` package to Geekbot's own Ollama settings through an adapter, and make `geekbot csv describe` / `geekbot csv ask` real (PLAN.md §6.7, §11 C2). C2 also sets Geekbot's per-question time cap and adds the dedicated read-only-SQL test (§8.1, D-041). It does no model swap (C3), no session cache (C5) and no benchmark (C3).

## Chat housekeeping before the design (2026-10-16)
- C1's commit recorded in STATUS and the C1 log: `3bd19db` (merge commit of PR #2; branch commit `adea115`).
- Git history showed merge commits for PRs #1 and #2, not squashes. The S0.5 row was corrected, and **D-046** records that D-044 (squash) stays the rule and the repository is to be set to squash only (a GitHub setting the user changes).
- A CI flake in C1's vendored test (`test_runner.py`, Ubuntu / Python 3.13, CI run #5, passed on re-run) is noted in C1's open issues.

## Approved design
Approved by the user on 2026-10-16 ("OK GO"), as proposed, including all six decisions and the extras.

**Files**
- New, under `src/geekbot/csv_mode/`:
  - `__init__.py`: the only package in Geekbot that imports `csv_analyst`.
  - `provider.py`: `OllamaCsvProvider`, which fits the package's `LLMProvider` protocol. The ✍️ piece lives here.
  - `adapter.py`: builds the provider and the session from Geekbot's settings, runs one question, collects timing.
- New: `tests/fixtures/sales_small.csv` (12 rows); tests `test_csv_provider.py`, `test_csv_adapter.py`, `test_csv_cli.py`, `test_csv_sql_readonly.py`, one `live`-marked smoke test, and an interim import-boundary test.
- Changed: `config.py` (two settings; `mask_settings` untouched, it is the user's S0.2 piece), `.env.example`, `cli.py` (the two stubs become real commands), `tests/unit/test_cli.py` (the two `csv` stub rows removed), `src/csv_analyst/session.py` (a 2-line edit marked `GEEKBOT EDIT (C2)`), `src/csv_analyst/VENDORED.md`.

**Decisions approved**
1. **Time cap:** the marked edit in `session.py` so `ask(question, time_limit_s=...)` passes the limit to `run_pipeline`. The cap is soft: the package checks it before each attempt or sample batch, so the worst case is the cap plus one model call.
2. **Defaults:** `GEEKBOT_CSV_LLM_TIMEOUT_S` (per Ollama call) and `GEEKBOT_CSV_QUESTION_TIME_LIMIT_S` (per question), both validated as greater than 0, both 60 s. These are unmeasured starting values; C3 checks how often the cap would cut an answer.
3. **Prompt-too-long guard:** import the package's `estimate_tokens` and reserve constants (one source of truth).
4. **Request parity with the package's provider,** plus `keep_alive` from Geekbot's settings (R-2). The `thinking` fallback when `response` is empty is kept.
5. **Load events:** an optional `on_response` hook; `csv ask` sums Ollama's `load_duration` and prints a one-line footer on stderr naming the model, profile, status, time and model-load time (D-043). `RunStats` does not exist yet.
6. **Extras:** the interim test that only `geekbot.csv_mode` imports `csv_analyst` (the import-linter contract is Phase 4), and a pre-check that Ollama is reachable and the model is pulled.

**Command behaviour**
- `csv describe` prints the package's "Loaded **file**…" text and never calls a model.
- `csv ask` prints the package's markdown reply.
- Exit codes: 0 for ANSWERED, ABSTAINED and NEEDS_CLARIFICATION; 1 for FAILED, a bad CSV, Ollama unreachable or a model not pulled.
- No unload or swap in C2. The live run needs a free GPU, as for any 🎮 step.

**Read-only SQL test (PLAN §8.1).** Calls the package's `execute_sql` directly (which skips the static check), so the SQLite authorizer is what refuses each statement: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, ATTACH, PRAGMA, a read of `sqlite_master`, two statements in one string, and `WITH … INSERT`. Only that tested property is claimed (D-041).

**Changes made during the build:**
- **Workflow change mid-step (D-047, PLAN rev 14):** the user switched to fast-prototyping mode before writing the ✍️ piece, so Claude wrote `generate_json` (see below). From then on: outline, one build, one summary.
- `CsvLoadError` (Geekbot's own exception in `adapter.py`) wraps the package's `IngestionError`, so only `csv_mode` ever sees the package's error type.
- `cli.py` imports `geekbot.csv_mode.adapter` inside the two commands, not at the top, so `--version` and `doctor` do not wait for pandas and pyarrow. A test pins this.
- Tests beyond the design: three settings tests in `test_config.py`, an import-boundary check that the vendored package never imports Geekbot, and the lazy-import check above.

## ✍️ Your piece
- **What you write:** `OllamaCsvProvider.generate_json` in `src/geekbot/csv_mode/provider.py` (about 35–45 lines). Claude provides the class, `__init__` (takes an optional `httpx` transport), `close`, and a helper `_check_fits(prompt, system)`. The method, in order:
  1. Call `self._check_fits(prompt, system)`; if it raises `PromptTooLongError`, nothing is sent.
  2. Build the JSON body: `model`, `prompt`, `system`, `format` (the `schema` if given, else the string `"json"`), `stream: False`, `keep_alive`, and `options` with `temperature`, `num_ctx`, plus `seed` only when it is not `None`.
  3. `POST /api/generate` through `self._client`.
  4. On a timeout, connection error or non-2xx status, or a body that is not a JSON object, raise `LLMProviderError` naming the model, with the original error chained.
  5. If `on_response` is set, call it with the parsed response.
  6. Return `data["response"]`; if empty, `data.get("thinking")`, or `""` if that is missing too; if `response` is missing or not text, raise `LLMProviderError`. Never parse or repair the model's JSON.
- **Tests it must pass** (written by Claude, failing until it works): request shape with and without a schema; the seed present only when given; the exact string returned; a 500; a refused connection; a timeout; a non-JSON body; an over-long prompt (the fake server is never called); the `on_response` call.
- **Result:** written by Claude in fast-prototyping mode (D-047) after the user changed the workflow. The user ran the tests once against the untouched stub (21 failed with `NotImplementedError`, 1 passed, as expected for a stub). It is on the ✍️ backlog in `STATUS.md`. Whether it passes is the user's next run.
- **Help used:** written by Claude (prototyping mode).

## Files changed
- New: `src/geekbot/csv_mode/__init__.py`, `provider.py` (the ✍️ `generate_json`, written by Claude), `adapter.py`
- New tests: `tests/unit/test_csv_provider.py`, `test_csv_adapter.py`, `test_csv_cli.py`, `test_csv_sql_readonly.py`, `test_csv_import_boundary.py`; `tests/live/test_csv_provider_live.py`; `tests/fixtures/sales_small.csv`
- Changed: `src/geekbot/config.py` (two CSV settings), `src/geekbot/cli.py` (real `csv describe` / `csv ask`), `.env.example`, `src/csv_analyst/session.py` (`GEEKBOT EDIT (C2)`) and `src/csv_analyst/VENDORED.md`, `tests/unit/test_config.py`, `tests/unit/test_cli.py` (the two `csv` stub rows removed)
- Notes: `project/learn/06-ports-adapters.md` + index, this log, `STATUS.md`, `DECISIONS.md` (D-046, D-047), `PLAN.md` (rev 14), `CLAUDE.md` (rule 6), the C1 log (commit SHA, CI flake)

## Commands the user ran + results
```
pytest tests/unit/test_csv_provider.py -q
21 failed, 1 passed in 1.21s   (all 21 failures: NotImplementedError from the untouched stub; the
                                 passing test is the signature check). Run before Claude wrote
                                 generate_json; the next run is the user's, after the summary.

(after Claude wrote generate_json)
pytest -q
2 failed, 443 passed, 10 skipped, 3 deselected, 1 xfailed in 43.8 s
  - test_csv_sql_readonly[read the schema]: SQLite says "access to sqlite_master.type is
    prohibited", not "not authorized" (Claude's wrong guess). The test now accepts both authorizer
    wordings.
  - test_log.py::test_setup_is_idempotent_and_changes_the_level: "assert 3 == 1". Cause not
    observed; probable cause: the new CLI tests leave the global `geekbot` logger configured.
    Claude added tests/unit/conftest.py, which restores the logger after every test. To be
    confirmed by the user's re-run.
ruff check .
2 x RUF100 (unused `noqa: PLC0415` in cli.py); `ruff check . --fix` removed them; now clean
ruff format --check .      28 files already formatted
mypy src                   Success: no issues found in 49 source files
geekbot csv describe tests/fixtures/sales_small.csv
Loaded **sales_small.csv**: 12 rows, 4 columns (`region`, `month`, `units`, `revenue`).
geekbot csv ask tests/fixtures/sales_small.csv "Which region has the highest total revenue?"
answered "North" (total 520.0), confidence high
[csv] model=qwen2.5-coder:3b profile=proto status=answered time=7.0s model_load=5115ms
(one run on the proto model: a plumbing check, not a result)
pytest -m live tests/live/test_csv_provider_live.py     2 passed in 1.04 s

(final run, after the two fixes)
pytest -q                 445 passed, 10 skipped, 3 deselected, 1 xfailed in 39.75 s
ruff check .              All checks passed!
ruff format --check .     29 files already formatted
(mypy src was clean on the earlier run; the fixes touched only tests)
```

## Review + walkthrough Q&A
To be filled in.

## Decisions
D-046 (merge method, from this chat's housekeeping) and D-047 (fast-prototyping mode). The time-cap edit and its 60 s defaults are recorded here and in `VENDORED.md`, with no separate decision.

## Open issues / follow-ups
- Nothing was run by Claude. The test expectations were reasoned, not observed. The user's first full run confirmed them except one: SQLite words the refusal of a `sqlite_master` read as "access to … is prohibited", not "not authorized" (test fixed).
- The package's own too-long message tells the user to "Raise LLM_NUM_CTX or load fewer columns" (`pipeline.py`). In Geekbot that setting is `GEEKBOT_CSV_NUM_CTX` and the package's variable is not read, so the hint misleads. Options: a marked edit of the message (recorded in `VENDORED.md`) or rewriting the text in `CsvAnalyst.ask`. Not done in C2.
- Ollama truncates an over-long prompt rather than refusing it; whether it logs a warning varies between the (secondary) sources found, and Ollama 0.34.2 was not checked. Gigalogy's own note: an 8,192 window cut an 8,309-token prompt to its last ~4k tokens. The provider's guard refuses first. A possible extra tripwire (a warning when `prompt_eval_count` is close to `num_ctx`) is untested: prefix caching may lower that count.
- The read-only SQL claim covers statement-form PRAGMA only. The table-valued `pragma_*()` functions are untested (written in the test's docstring).
- The 60 s defaults (one Ollama call, one question) are unmeasured starting values, and the question cap is soft (worst case: the cap plus one call). C3 measures both.
- `OllamaCsvProvider` and the package's own `OllamaProvider` send the same request apart from `keep_alive`; the package's provider stays in the copy (still imported by `session.py`), so `requests` is still a dependency.
- The characterization-test idea for `sql_engine/rewrite.py` stays parked unless the user asks for it.
- The user still has to set the repository to squash-only merges in GitHub (D-046).
- The C1 CI flake (run #5, Ubuntu / Python 3.13) is in the C1 log; if it comes back, fix it on its own branch.

## Handoff to the next chat
- **State:** C2 is green-lit; the user pushes it and merges the pull request into `v2`. Record the merge commit SHA in `STATUS.md` and in this log's `Commit:` line at the start of the next chat.
- **Next:** C3, "CSV model: swap + reproduction run" 🎮. It adds an explicit model swap (unload one, warm up the other, R-4) in `llm/models.py`, a `doctor` check for the CSV model's residency, and the Gigalogy benchmark through Geekbot's adapter. It needs `qwen2.5-coder:7b` pulled, an idle GPU, and `GEEKBOT_PROFILE=target` for any number that is reported.
- **Use from C2:** the footer of `geekbot csv ask` (model, profile, status, time, `model_load`) is the raw material for C3's swap and load measurements; `CsvAnalyst` is the object C5 will keep per conversation.
- **Mode:** fast prototyping (D-047): outline, one build, one summary.
