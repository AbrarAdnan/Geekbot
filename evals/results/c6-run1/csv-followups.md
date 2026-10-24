# CSV follow-up evaluation: c6-run1

CSV model `qwen2.5-coder:7b` (profile `target`); 15 two-turn items on Gigalogy's `sales` and `hospital` datasets, 3 runs each, 2026-10-24T14:19:15+06:00. The items and the expected results (computed with pandas) are Claude-authored; the grader is Gigalogy's, ported in C3. A measurement of this model on a small set, not a general claim.

## Result (the 2-of-3 rule: solved = right in more than half of the runs)
| path | solved |
|---|---|
| **the conversation** (question, follow-up, rewrite, package) | **12 of 15** |
| **the oracle** (the hand-written standalone question straight to the package) | **11 of 15** |

**Rewrite loss** (the oracle solves it, the conversation does not): 0 items . **Package failures** (neither path solves it; no rewrite could have helped): 3 items (F-006, F-010, F-014). Solved only through the conversation: 1 (F-008). Unstable (right in some runs, wrong in others): 2 (F-008, F-013).

| dataset | items | conversation solved | oracle solved |
|---|---|---|---|
| sales | 9 | 8 | 7 |
| hospital | 6 | 4 | 4 |

The rewrite equalled the follow-up as typed in 0 of 45 runs (0 of them because the rewrite call failed). Mean time per run (both paths): 10.4 s.

## Failing items: what was asked
| item | follow-up as typed | what the package was asked (first failing run) | standalone, by hand |
|---|---|---|---|
| F-006 | Just the Consumer segment. | How many rows are there for each ship mode in the Consumer segment? | How many rows are there for each ship mode in the Consumer segment? |
| F-010 | And how many of them are female? | How many distinct female patients are there in each province? | How many distinct female patients are there in each province? |
| F-014 | And the latest? | What is the latest admission date? | What is the latest admission date? |
