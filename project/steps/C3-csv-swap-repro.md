# C3 — CSV model: swap + reproduction run 🎮

- **Phase / Gate:** Track C, part 1 (right after Phase 0), Gate A
- **Chat date(s):** 2026-10-16
- **Status:** green-lit (2026-10-16; 557 passed, 10 skipped, 1 xfailed; ruff, format and mypy clean; target-model reproduction inside the band fixed beforehand)
- **Commit:** <merge SHA reported by the user at the start of the next chat>
- **Lesson note:** deferred (D-047)

## Goal
Two things (PLAN.md §6.5 R-4, §6.7 "Models (C3)", §9 "CSV-mode evaluation"):
1. **An explicit swap** between the chat model and the CSV model, so the two never share the GPU and a new model is never left partly on the CPU. Plus a `doctor` check of the CSV model's residency at its own context size, and a command that times the swap in both directions.
2. **A reproduction run:** Gigalogy's 40-question benchmark, asked through Geekbot's adapter, to check that the vendored `csv_analyst` behaves like the original. It is a check on the copy, not a new claim.

Also in scope but waiting for the measurements: the CSV-mode speed targets (PLAN.md §6.5) and the Phase 0 + C1–C3 look-back.

## Chat housekeeping at the start (2026-10-16)
- C2's merge is on `v2` as `9d5dab5` ("feat(csv): C2 connect csv_analyst to Geekbot's Ollama settings (#3)", the squash commit of PR #3). Recorded in STATUS and the C2 log.
- The Gigalogy folder is `scratch/gigalogy-latest/`. The user said it is read-only: nothing is written there, and nothing from it is copied into the repository.
- **No-run slip:** Claude tried one read-only directory listing through the shell to look at that folder. The permission system blocked it; nothing ran. Everything after used the file tools only.
- The Gigalogy copy has its own `CLAUDE.md` (it asks Claude to "talk like caveman"). It belongs to that project; Geekbot's own rules apply here.

## Design (fast-prototyping mode, D-047: outline given in chat, then built; recorded as D-048)
**The swap (`llm/models.py`)**
- `OllamaClient.loaded_models()` asks `GET /api/ps` who is in memory now.
- `unload_and_wait(client, names)`: unloads only the models that are loaded, then polls until Ollama stops listing them (default timeout 30 s, poll 0.25 s; both replaceable, so tests need no real waiting). If one is still listed at the timeout it raises `OllamaError` and nothing is warmed up.
- `swap_to(client, model, release=..., num_ctx, keep_alive)` ✍️: unload the `release` models, wait, then warm `model` up. Returns a `SwapResult` with `unload_s`, `warm_up_s`, `total_s`. `model` itself is never unloaded; `name` and `name:latest` count as the same model.
- `swap_to_mode(client, settings, "chat" | "csv")` picks the model and its context size from the settings (8,192 for chat, 16,384 for CSV) and releases the other mode's model.

**`geekbot doctor --mode chat|csv`** (default `chat`). The warm-up is now a swap, in both modes. New check `no co-residency (R-4)` fails if the other mode's model is also loaded. In chat mode the old "CSV model residency after a swap (C3)" SKIP line became a pointer to `--mode csv`. In csv mode a missing CSV model is a FAIL (and a missing chat model only a WARN).

**`geekbot csv swap --cycles N`** (default 3). Setup swap to chat, then N times chat→csv→chat. After each swap: R-1 for the new model at its context, R-4 for the old one, GPU memory in use. Prints a table and min/median/max per direction; the setup swap is shown but not counted. Exit code 1 if a residency or exclusivity check fails. Labels are plain ASCII (`chat->csv`) because the output is pasted from a Windows terminal.

**`geekbot csv repro`** (module `geekbot/evals/csv_repro.py`)
- Reads questions, gold answers and CSVs from `--source` (default `scratch/gigalogy-latest`), read-only. The datasets come from third parties and this repository is public, so none of it is copied here. Result files hold ids, statuses, correctness, times and model-load times, never question or answer text.
- Preflight: unload both of our models and wait, read GPU memory (that reading is the "idle GPU" check, D-028), warm the CSV model (the cold-load time), check R-1 at 16,384.
- Loop: seed set → suite → question (as Gigalogy's harness does), 3 seed sets by default. `CsvAnalyst.ask` gained `seed_offset` (below); seed set `k` uses offset `k × 10,000`, Gigalogy's stride.
- **Solved** = right in more than half of the runs (at least 2 of 3), Gigalogy's rule.
- The run uses the package's own **600 s** per-question cap, not Geekbot's 60 s, so the cap cannot disguise a difference between the copy and the original. The report counts the question-runs that took longer than 60 s: an upper bound on what Geekbot's cap would cut (the cap is only checked between model calls).
- The report prints our numbers next to the source project's README numbers (35/40 solved, 10/8/7 easy/medium/hard, 10/10 refused, 34.0 correct per run, 0 invented in 30, 3.0 s per question), labelled as the source project's. Those were measured on Gigalogy's commit and harness, not on this working-tree copy (VENDORED.md).
- Also reported: correct per run, unstable questions, false abstentions, failures, latency mean/p50/p95/max, and model loads during questions (reply `load_duration` over 500 ms; the 500 ms is a heuristic I chose, and the median and maximum are printed so it can be judged).
- **Official** only if: the CSV model is the target model, the GPU was idle (≤ 2,048 MiB used with our models unloaded), residency was PASS, and the run was full (both suites, 3+ runs, no `--limit`). Otherwise the file says `"official": false` with the reasons, and it is a plumbing check (D-043).
- The JSON result is rewritten after every seed set, so a crash or Ctrl-C keeps the finished runs.

**The grader** is ported into `csv_mode/grading.py` instead of imported from the reference copy (git-ignored, may move; a silently changed grader would make "reproduces" meaningless). It reproduces Gigalogy's comparator and grade rules: row order and column names ignored, tolerance for floats, number-vs-numeric-text and date-vs-ISO-text coercions only, a one-row table counts as a scalar, a many-row table never does. One deliberate difference: rows of different lengths are never equal. Gigalogy's own grader tests were carried over to `tests/unit/test_csv_grading.py`.

**Vendored edit (6th in `VENDORED.md`):** `CsvAnalystSession.ask` takes `seed_offset` and passes it to `run_pipeline`, which already had it. Marked `GEEKBOT EDIT (C3)`.

**New dependency:** `pyyaml>=6.0` in `pyproject.toml` (the question files are YAML). It is already pinned in the lock (`pyyaml==6.0.3`, pulled in by pre-commit), so only the lock's "via" note changes when it is regenerated. It has no type stubs here, so its import carries one `type: ignore[import-untyped]`.

## ✍️ Piece
- **What:** `swap_to` in `src/geekbot/llm/models.py` (the swap function with its timings; PLAN.md §11 C3).
- **Result:** written by Claude (prototyping mode, D-047); on the ✍️ backlog in `STATUS.md`. Its tests are in `tests/unit/test_swap.py` (fake Ollama, fake clock); the live check is `tests/live/test_swap_live.py`.
- **Help used:** written by Claude (prototyping mode).

## Files changed
- New, source: `src/geekbot/swapcheck.py`, `src/geekbot/csv_mode/grading.py`, `src/geekbot/evals/__init__.py`, `src/geekbot/evals/csv_repro.py`
- Changed, source: `src/geekbot/llm/models.py` (swap, `loaded_models`, `same_model`), `src/geekbot/llm/vram.py` (`try_read`), `src/geekbot/doctor.py` (`--mode`, swap warm-up, `check_exclusive`), `src/geekbot/config.py` (`Mode`), `src/geekbot/cli.py` (`doctor --mode`, `csv swap`, `csv repro`), `src/geekbot/csv_mode/adapter.py` (`ask(seed_offset=)`, `CsvAnswer.reason`, `.retries`), `src/csv_analyst/session.py` (`GEEKBOT EDIT (C3)`) and `src/csv_analyst/VENDORED.md`, `pyproject.toml` (pyyaml)
- New, tests: `tests/unit/test_swap.py`, `test_swapcheck.py`, `test_csv_grading.py`, `test_csv_repro.py`, `test_csv_swap_repro_cli.py`; `tests/live/test_swap_live.py`
- Changed, tests: `tests/unit/test_doctor.py` (fake client gained `loaded_models`/`unload`; the "C3 not built yet" assertion replaced by new mode tests), `test_models_client.py`, `test_vram.py`, `test_csv_adapter.py`, `test_csv_import_boundary.py`, `conftest.py` (a `clean_geekbot_env` fixture)
- Notes: `evals/csv/README.md`, `evals/README.md` (one row), this log, `STATUS.md`, `DECISIONS.md` (D-048), the C2 log (commit SHA)

## Acceptance band for the reproduction (settled by the user on 2026-10-16, before any run: "keep as written")
Gigalogy's README says two identical runs differ on about 5 % of outputs and that "2–3 answers out of 120" is noise, and the copy is a later working tree run through a different HTTP client and harness. So, on the **target model**, a full run **reproduces** the original if:
- questions solved is **33 to 37** of 40 (source: 35);
- correct answers per run (mean) is **32.0 to 36.0** (source: 34.0);
- invented answers are **at most 2 of 30** (source: 0);
- no model load during a question beyond the first (R-2), and every swap leaves the new model 100% on the GPU.
Seconds per question is **not** judged against 3.0 s (different harness and prompt-cache state); it is reported, and C3's CSV-mode speed targets are set from it. A result outside the band is investigated, not hidden. The band comes from the README's own noise statement; it was fixed before the first run and is not to be changed after seeing the numbers.

## Commands the user ran + results
Order planned: static checks and tests; `ollama list`; `geekbot doctor --mode csv`; `pytest -m live tests/live/test_swap_live.py`; `geekbot csv swap --cycles 3`; a three-question smoke run of `geekbot csv repro`; then the full target-profile run. All numbers must name the model and profile that produced them.

**First live session, 2026-10-16, proto profile (chat `gemma3:4b`, CSV `qwen2.5-coder:3b`). Plumbing checks, not results (D-043).** The static-check and pytest outputs were not pasted yet.
```
geekbot doctor --mode csv
all PASS: qwen2.5-coder:3b loaded in 2.2 s, 100% GPU at context 16384, gemma3:4b not loaded;
GPU 1020 MiB before -> 3841 MiB after (+2821 MiB)

geekbot csv swap            (3 cycles; setup swap 4.6 s, not counted)
chat->csv   total 1.9 s / 1.9 s / 2.2 s (min / median / max), unload 0.1 s each, warm-up 1.8-2.1 s
csv->chat   total 3.0 s / 3.0 s / 3.0 s, unload 0.1 s each, warm-up 2.9 s
every swap: checks PASS; GPU used 3.8 GB with the CSV model, 5.1 GB with the chat model

geekbot csv repro           (full, proto profile, qwen2.5-coder:3b, official: NO)
correct per run [27, 28, 29]; solved 29/40 (easy/medium/hard 10/6/4; unanswerable 9/10 refused);
invented 4 of 30; false abstentions 6; failed 0; unstable questions 3
seconds per question: mean 2.7, p50 1.4, p95 3.6, max 61.2; 2 of 120 question-runs over 60 s
model loads during questions: 0 (load_duration median 15 ms, max 64 ms)
GPU: 1030 MiB idle -> 3852 MiB with the 3b loaded at 16k; cold load 2.2 s; R-1 PASS
(3b model: not comparable to the 7b acceptance band, and Gigalogy's README calls the 3b "more brittle")
```
What the 3b run showed:
- **The plumbing works end to end:** the Gigalogy folder is read, the questions are asked, the grader runs, and the two seed sets differ (e.g. hospital e01 wrong in set 0, right in set 1), so `seed_offset` does its job.
- **The failures I looked at are model mistakes, not grader bugs:** counting rows instead of de-duplicated patients (hospital e01, m02, m03, m05, h01), an extra column or an id instead of a name (hospital h03, sales m02, h02: the strict output-shape rule Gigalogy's README describes), `needs_clarification` on hospital h04 and sales h01, and invented answers on sales u02 and u03 (set 0) and u02 (set 1). Not every failure was inspected.
- **Wrong answers repeat across seeds** (the same city-count table, the same extra columns): agreement across runs is not correctness, as Gigalogy's README says.
- **No model load during any question** (max `load_duration` 64 ms against 5,115 ms for a cold load seen at C2), so the 500 ms threshold separates the two cases by a wide margin, on this model. R-2 held for this run.
- **Two slow questions, both exactly 61.2 s:** hospital m03 in set 0 and in set 2 (2.5 s in set 1). The run used the 600 s question cap, so this is not the cap. 61.2 s is suspiciously close to one 60 s model-call timeout (`csv_llm_timeout_s`) plus ~1.2 s of other work, i.e. probably a runaway generation. That is a hypothesis: the run did not record per-step times. Fixed for the next run: each question-run now records its failed steps and slowest single step, and the report prints them. If the 7b shows the same, the per-call timeout, not the question cap, is the number to set at C3.
- **The final report block printed correctly** on real data (the layout was untested until now).
- **Found and fixed in this chat (2 changes, both built after the run and not yet run):** (1) a mismatch note holds the expected and the actual table in full (hundreds of rows), which buried the progress lines and would have put rows of the third-party dataset into the result file. Notes are now cut to 160 characters (`MAX_NOTE_CHARS`, with tests). The partial proto result file written by this run (`evals/csv/results/repro-proto-qwen2.5-coder-3b-20261016-1544.json`) has the long notes: delete it, do not commit it. (2) Per-step data (`CsvAnswer.failed_steps` / `slowest_step_ms`, two more fields in each result row and two summary lines) so a slow question can be explained.
- After both changes: `pytest tests/unit -q` → **243 passed**; `geekbot csv repro --runs 1 --limit 2` (proto) ran clean and printed the new lines (0 of 4 with a failed step, slowest step 1.2 s). The full `pytest`, `ruff` and `mypy` outputs were not pasted (the files on disk were reformatted since, probably by `ruff format .`); CI runs all of them on the pull request.

**Second live session, 2026-10-16, TARGET profile (chat `gemma4:12b`, CSV `qwen2.5-coder:7b`, Ollama 0.34.2), idle GPU.** These are Geekbot's first reportable CSV-mode numbers.
```
geekbot csv swap --cycles 3          (keep_alive 30m; chat num_ctx 8192, csv num_ctx 16384)
setup (cold chat load)    13.5 s (not counted)       GPU used 9,026 MiB
chat->csv   total 4.3 / 2.7 / 2.7 s   (median 2.7, max 4.3)   unload 0.1 s   GPU used 6,414 / 6,360 / 6,359 MiB
csv->chat   total 4.5 / 4.2 / 4.5 s   (median 4.5, max 4.5)   unload 0.1 s   GPU used 8,981 MiB each time
every swap: residency PASS (new model 100% GPU at its own context) and old model gone (R-4)

geekbot csv repro                    (official: yes; 40 questions x 3 seed sets = 120 question-runs; 600 s cap used)
CSV model: cold load 2.7 s; GPU 869 MiB idle -> 6,457 MiB with the 7b loaded at 16k; R-1 PASS
correct per run [31, 33, 33] (mean 32.3); solved 33/40 (easy/medium/hard 9/7/7); unanswerable refused 10/10
invented answers 0 of 30; false abstentions 0; failed 0; unstable questions 4
seconds per question: mean 2.9, p50 2.3, p95 7.7, max 11.9; question-runs over 60 s: 0 of 120
question-runs with a failed step: 7 of 120; slowest single step 2.7 s
model loads during questions: 0 (load_duration median 17 ms, max 56 ms)
source project (Gigalogy README, its harness and commit): 35/40 solved, 10/8/7, 10/10, 34.0 per run, 0 of 30 invented, 3.0 s
```
**Against the band fixed before the run:** solved 33 (band 33–37) ✓, mean correct per run 32.3 (32.0–36.0) ✓, invented 0 of 30 (≤ 2) ✓, no model load during a question ✓, every swap 100% GPU ✓. **The copy is accepted as reproducing the original, within the pre-registered band.** Said plainly: both figures sit at the low edge of the band, 5 answers of 120 below the source (34.0 to 32.3 per run), and Gigalogy's own noise statement is "2–3 answers of 120". So the result is consistent with the original under the band we chose, not a demonstration that the two are identical.

**Which questions differ from the source project's list.** Gigalogy's README lists five questions the 7b gets wrong in all three runs. Four of them are unsolved here too (hospital m03 allergies by popularity, hospital m04 doctors' admissions, sales h02 East vs West by month, sales h03 churned customers), and the fifth (sales h05, days from order to ship per ship mode) is solved 3 of 3 here. Three questions are unsolved here that the README does not list: hospital h03 (provinces with more male patients), sales e04 (the missing-values check: "expected 0, got None" in two runs, right in the third) and sales m02 (top 10 customers). The notes were cut to 160 characters, so this paste does not show why; `geekbot csv ask` on those three would. Possible causes (none checked): the vendored working tree has edits that were never committed (VENDORED.md), the grader strictness on output shape, run-to-run variation.

**Speed and cost, measured (target models).** Mean 2.9 s per question against the source project's 3.0 s; p95 7.7 s; slowest question 11.9 s; none near Geekbot's 60 s cap or the 60 s per-call timeout (slowest single step 2.7 s), so the earlier 61.2 s cases were specific to the 3b model. Swap cost: chat→csv median 2.7 s, csv→chat median 4.5 s, unload about 0.1 s. VRAM: CSV mode 6,457 MiB used (3,783 MiB free); chat mode with `gemma4:12b` 8,981 MiB used, which leaves 1,259 MiB free. The PLAN's CSV-mode estimate (weights 4.7 GB + KV ≈ 896 MiB) is consistent with the +5,588 MiB seen between idle and loaded, which also includes compute buffers.

**Static checks, 2026-10-16** (run by the user after the target runs):
```
pytest                  557 passed, 10 skipped, 4 deselected (live), 1 xfailed in 44.3 s
                        (the skips and the xfail are the vendored ones from C1, unchanged)
mypy src                Success: no issues found in 53 source files
ruff format .           8 files reformatted
ruff check .            15 errors: 14 x ISC004 (implicit string concatenation in a list, in
                        format_report and format_swap_report) and 1 x PYI034 (`__enter__` in a
                        test fake). Fixed by Claude afterwards (parentheses around each
                        multi-line string; `Self`); the report text is unchanged.
re-run after the fix   pytest: 557 passed, 10 skipped, 4 deselected, 1 xfailed in 45.3 s; the
                        user reported ruff as clean ("rest all good")
```
The `xfail` in `test_facts` ("holds dates" vs "holds ISO text"; VENDORED.md says "check again in C3") was not looked at in this step and is still open.

## Review + walkthrough Q&A
To be filled in.

## Decisions
D-048 (the design) and D-049 (the CSV-mode speed targets and the verdict; PLAN rev 15, approved by the user on 2026-10-16).

## Open issues / follow-ups
- Nothing was run by Claude. All tests were reasoned, not observed; the first full `pytest`, `ruff`, `mypy` run will show what needs fixing. `ruff format` was not run either, so `ruff format .` first.
- `/api/ps` is used for "is it still loaded?". Its `name` field is what the code reads; checked against Ollama's documented response, not against Ollama 0.34.2. If a swap reports "still loaded" although nothing is, look at that response first.
- The load-event threshold (500 ms) and the 30 s unload timeout are guesses to be judged from the first measurements.
- CSV-mode speed targets (seconds per question, swap time) are still to be set: they need the target-model numbers and a PLAN edit (rev 15) that the user approves.
- The chat model `gemma4:12b` must be pulled for the swap measurements on the target profile (`ollama list`).
- The KV-cache estimate for `qwen2.5-coder:7b` at 16k (≈896 MiB, PLAN.md §6.5) is checked by the preflight's GPU memory readings (idle vs loaded); the difference includes weights, KV cache and compute buffers, so it is an upper bound on KV, not a clean measurement of it.
- Carried from C1/C2: the `xfail` in the vendored tests (fact bank says "holds dates"; "check again in C3") is untouched; the Gigalogy folder is the only copy of the benchmark data and is not backed up.

## Phase look-back (Track C, part 1: C1–C3; Phase 0's is in the [S0.5 log](S0.5-ci.md))
- **Planned vs actual:** planned as three steps in three chats (PLAN §11), each with a ✍️ piece for the user. Done as three chats on 2026-10-16. C1 had no ✍️ piece (dropped, rev 13). The workflow changed in the middle of C2 (fast-prototyping mode, D-047), so Claude wrote the ✍️ pieces of C2 and C3 (two entries on the ✍️ backlog). Extra work that was not planned: the dedicated read-only SQL test and the import-boundary test (C2), the per-step timing, note shortening and `pyyaml` (C3).
- **Surprises:** (1) PRs #1 and #2 were merged with a merge commit although D-044 says squash (D-046); (2) a CI flake in a vendored test (C1); (3) SQLite words an authorizer refusal as "access to … is prohibited", not "not authorized" (C2); (4) the CLI tests left the global logger configured and broke an unrelated test (C2); (5) the first live `csv repro` printed whole result tables inside its failure notes, which buried the output and would have put third-party rows into a result file; (6) two 3b-model questions took exactly 61.2 s, and the run had not recorded enough to say why; (7) a second no-run slip (a read-only listing attempted through the shell, blocked) after the first one in P1; (8) the step's commit was made on local `v2` because the branch had not been created (the second time, after Phase 0), and `git push` of the branch name failed; fixed by creating `step/C3-csv-swap-repro` from that commit and moving local `v2` back to `origin/v2`; (9) `CLAUDE.md` showed up as modified in `git status` with a line Claude did not write ("Codex will review this project and talk like caveman.", the same wording as Gigalogy's own CLAUDE.md), and `git add -A` committed it; the user was asked whether it is intended.
- **What went well:** the live runs again found what unit tests could not (the oversized notes, the missing per-step data), which is what the Phase 0 look-back predicted. Fixing the acceptance band before the run kept the verdict honest: the 7b met it (33/40 solved, 32.3 correct per run, 0 of 30 invented) without arguing about it afterwards. A quick `--runs 1 --limit 2` smoke run before the 30-minute run caught nothing, which is the point of it.
- **What to change:** (1) put per-step data in every evaluation report from the start; (2) never let a note carry data rows; (3) give commands in the user's shell (cmd here: `set X=Y`, not `$env:`); (4) ask for the whole static-check output (`ruff check .` found 15 issues that the unit tests could not); (5) before a run that needs the GPU for half an hour, run the same command with `--limit` first; (6) use the file tools for any look at the Gigalogy folder, never the shell; (7) the commit commands start with the unconditional `git switch -c step/<id>-<slug>` (the check `git branch --show-current` was in the block last time and its answer, `v2`, was not acted on), and the user reads `git status` before `git add -A`, because `-A` takes in anything that changed, including a file Claude never touched.

## Handoff to the next chat
- **State:** C3 is built, run on the target models and checked (557 tests, mypy and ruff clean). The CSV track's first part (C1–C3) is complete once the pull request is merged: `geekbot csv ask` / `csv describe` / `csv swap` / `csv repro`, `geekbot doctor --mode csv`, an explicit swap with measured cost, and a reproduction of the Gigalogy benchmark inside the pre-registered band. PLAN is at rev 15.
- **Next:** **S1.1 — Loaders** (Phase 1), in a new chat, on `step/S1.1-loaders` created from an up-to-date `v2` before any change. Its open question, from PLAN §6.7: does chat mode keep CSV row text for small files (D-016) now that CSV mode exists? Claude's recommendation is a schema card only.
- **Use from C3:** `swap_to_mode` (the server and C4's mode router will call it); `unload_and_wait`; the `Patience` object (a fake clock for tests); `geekbot csv repro` and its result files (C6 adds its own sets next to them); `CsvAnswer.failed_steps` / `slowest_step_ms`.
- **Open items (none blocks S1.1):**
  - Why hospital h03, sales e04 and sales m02 fail here but are not on the source project's unsolved list. Not diagnosed: `geekbot csv ask` on those three would show the SQL and the result.
  - The vendored `test_facts` expected failure ("holds dates" vs "holds ISO text"), flagged "check again in C3", was not looked at.
  - The 30 s unload timeout and the 500 ms load-event threshold held up in the runs (unload ≈ 0.1 s; max `load_duration` 56 ms), but they are guesses that only these runs have tested.
  - `/api/ps` was read for `name`; it worked on Ollama 0.34.2 in the live runs.
  - Carried over: `scratch/gigalogy-latest/` is the only copy of the benchmark data and is not backed up; Gigalogy's edits are not committed, so `VENDORED.md` still names no real SHA; branch protection and the squash-only repository setting (D-046) are the user's GitHub settings; `config show` does not print the S0.3 settings.
- **Mode:** fast prototyping (D-047): outline, one build, one summary.
