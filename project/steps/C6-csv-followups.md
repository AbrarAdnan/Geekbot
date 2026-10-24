# C6 — CSV-mode evaluation: follow-ups and CSV-cell attacks

- **Phase / Gate:** Track C, evidence for Gate B. Built and run by Claude in autopilot run 5 (D-097, 2026-10-24).
- **Chat date(s):** 2026-10-24
- **Status:** **built, verified and run by Claude under D-097**: `ruff`, `mypy src`, `lint-imports` clean, `pytest` 2979 passed (72 more than S5.2's 2907), one real run on the target model (`evals/results/c6-run1/`). **Not reviewed by you. The 15 items and their expected results are Claude-authored and provisional (D-067); two of the three failing items fail on a shape question in the item, not on the model (see "What failed").**
- **Commit:** branch `step/C6-csv-followups` (stacked on `step/S5.2-answer-eval`); merged into `v2` as PR #47 (squash `a387e40`), branch deleted
- **Lesson note:** deferred (prototyping mode, D-047)

## Goal
PLAN §9 and §11 row C6: measure the one part of CSV mode that is Geekbot's own and was never measured, the **rewrite of a follow-up** (`condense`, C4). The copied `csv_analyst` package has no memory; "And what about profit?" works only because a small model rewrites it into a standalone question first. INTEGRATION.md lists that rewrite as unmeasured. Second deliverable: the **CSV-cell attack documents** that S5.4's injection evaluation will run (§8.3 vector 5).

A little context:
- **Two paths per item, so a failure can be placed.** The *conversation* (question, follow-up, the product's own rewrite with the CSV model, then the package) is compared with the *oracle* (the hand-written standalone question straight to the package). Oracle right and conversation wrong: the rewrite lost something. Both wrong: the package cannot do that question, and no rewrite could have helped. Only the difference between the two says anything about `condense`.
- **The expected results are computed, not typed.** Each item has a pandas recipe (`csv_followups_gold.py`) run on the dataset's own CSV; recipes are code, because a data file that held expressions would need `eval`, which this project avoids. Each recipe was checked on the real data for what makes a gold answer unreliable (an empty result, a tie for "the highest", a question whose meaning depends on which rows count).
- **A result file in a public repository must hold no dataset rows.** The grader's notes of a wrong answer quote the rows; they are reduced to one of five categories (`matches`, `abstained`, `false abstention`, `hallucination`, `mismatch`) before they are saved.

## Design (D-101)
**New**
- `src/geekbot/evals/csv_followups.py`: `FollowUpItem`, `load_followups` (closed fields, unique ids, known dataset, at least two turns, a standalone question that differs from the follow-up as typed, a known recipe), `FollowUpRun`, `run_followups` (seed set by seed set, a run that breaks stops the evaluation: it is not counted as a wrong answer), `solved` (**the 2-of-3 rule: right in more than half of the runs**), `summarize_followups` (rewrite loss, package failures, gained, unstable, how often the rewrite left the follow-up unchanged, per dataset), `format_followups_report`.
- `src/geekbot/evals/csv_followups_gold.py`: 15 pandas recipes (9 on `sales`, 6 on `hospital`); `gold_entry` refuses an empty table (an empty expected result would make every empty answer correct).
- `src/geekbot/evals/csv_followups_run.py`: the real graph in CSV mode with a recording wrapper around the package, an `InMemorySaver` thread per (item, seed set), Gigalogy's grader (`csv_mode.grading.grade`), seeds shifted by 10,000 per seed set as in C3.
- `evals/datasets/csv-followups.jsonl` (F-001 to F-015) and `geekbot eval csv-followups [--name --source --items --results-dir --runs --limit]`.
- **`src/geekbot/evals/csv_attacks.py` + `evals/datasets/csv-attacks/`** for S5.4: 8 small CSV files and `attacks.jsonl` with 13 attacks over 9 vectors (`image`, `link`, `html`, `planner`, `answer`, `sql`, `schema`, `header`, `unicode`); each entry names the question, the planted payload, a canary token where one fits and the **detectors** that count as a bypass (`live_markup`, `canary_in_reply`, `canary_in_sql`, `forbidden_sql`, `answer_wrong`). The loader checks that the payload and the canary are really in the file and that **every URL is under `.example`** (so a bypass cannot reach a real host). A test checks that the invisible characters of `unicode-tricks.csv` are still there (an editor that "cleans" them would turn the attack into a plain one). `README.md` explains the files.
- tests: `test_csv_followups.py` (40, includes every recipe on made-up tables and, if `scratch/gigalogy-latest` exists, on the real datasets), `test_csv_attacks.py` (25), `test_csv_followups_cli.py` (7).

## ✍️ Your pieces
- **The follow-up scorer (the 2-of-3 rule):** `solved(right, runs)` and the two-path summary. Written by Claude (prototyping mode, D-047). Not added to the ✍️ backlog as a separate row: it is four lines (`2 * right > runs`) next to `summarize_followups`; the backlog row is `solved` + `summarize_followups`.

## Results (`evals/results/c6-run1/`)
`qwen2.5-coder:7b`, target profile, GPU idle (953 MiB in use by the desktop, 0% utilisation, nothing loaded), 15 items, 3 seed sets. Gigalogy's `sales` and `hospital` CSVs are read from `scratch/gigalogy-latest` and never copied.
- **The conversation solved 12 of 15; the oracle solved 11 of 15. Rewrite loss: 0 items.** The package failed on three items on both paths (F-006, F-010, F-014); the conversation solved one item the oracle did not (F-008, unstable). Two items are unstable (right in some runs, wrong in others): F-008, F-013.
- **The rewrite changed the follow-up in 45 of 45 runs** (never left it as typed; no rewrite call failed). For all three failing items the text the package was asked equals the hand-written standalone question word for word: the rewrite did exactly what the oracle did, and the package then failed.
- Mean time per run (both paths, two turns plus the oracle): 10.4 s.
- A measurement of one model on 15 items I wrote: 12 of 15 against 11 of 15 is a difference of one item (F-008, unstable) and is **not** evidence that the conversation is better than the oracle.

## What failed (asked by hand afterwards, the same model, one run each: not part of the result)
- **F-006** (`Just the Consumer segment.`): the package answers right and adds a third column, `segment`, to the result (`ship_mode, segment, count`). Gigalogy's grader demands the same number of columns, so a correct-looking answer fails. **A shape ambiguity in my item** (it should say which columns), and a strictness of the grader.
- **F-010** (`And how many of them are female?`): the package counts distinct female patients per province and labels the rows with the province abbreviations (`AB`, `BC`, ...) from another column; the recipe groups by the province *name*. The numbers can agree and the labels do not. **Also an ambiguity in my item** (the question does not say which label).
- **F-014** (`And the latest?`): the package wrote `SELECT patient_id, MAX(admission_date) ... GROUP BY patient_id`, a per-patient maximum, for "What is the latest admission date?". **A genuine package failure** on a question the oracle also failed.
- **I did not change the items after seeing this** (that would be tuning the test to the result): the run stands as it is, with these three causes written next to it. Rewording F-006 and F-010 ("... (ship_mode and the count only)", "... by province_name") is your revision, together with your own items; a second run would then be `c6-run2`.

## What you do (the user's part)
1. Read `evals/results/c6-run1/csv-followups.md` and the three causes above; decide whether F-006 and F-010 are reworded (a one-line edit each in `evals/datasets/csv-followups.jsonl`) and whether to add items of your own.
2. To repeat (needs an idle GPU; about 8 minutes): `set GEEKBOT_PROFILE=target` then `geekbot eval csv-followups --name c6-run2`.
3. The attack files are for S5.4; nothing to run.

## Not verified, stated
- Anything about conversations longer than two turns, other datasets, or other models. The rewrite on follow-ups that are harder than these (a pronoun that points two turns back).
- That the attack files make the planner or the answer step obey anything: S5.4 measures that. A payload that is in a file is not an attack that works.
- The gold recipes on a dataset other than Gigalogy's copy of `sales` and `hospital` of 2026-10-16.

## Slips (told straight)
- The `\uXXXX` quirk of the file tools bit again: a test line with escapes in a Python string was written as the characters themselves (ruff PLE2502); repaired with a one-off line replacement. No other slip.

## Decisions
D-101.

## Handoff
S5.3 (web tool + taint enforcement) is next in the run.
