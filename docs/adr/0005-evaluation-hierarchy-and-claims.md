# ADR-0005: Evidence in a fixed order, and claims that say what they are

- **Status:** accepted (PLAN.md §9; D-015, D-018, D-025, D-026, D-067, D-082, D-099)
- **Date:** 2026-10-07 (built across S2.1 to S2.6, S5.1, S5.2, C6, S5.4)

## Context
Evaluations of RAG systems are easy to make flattering: a judge model that prefers its own family's
writing, a test set written by the system's author, a number measured on a busy machine, a table where
five things changed at once. A reader cannot tell which numbers to trust.

## Decision
- **Two kinds of retrieval evidence, never mixed.** Standard BEIR datasets (SciFact, NFCorpus) with
  standard metrics are *standardized retrieval evidence*; the golden corpus (15 documents, 100 questions)
  is *system-specific behavioural evaluation*. Golden v0 is Claude-authored and provisional; the user's own
  questions are a separate file, scored separately.
- **One variable per rung.** The ablation adds one change at a time in a fixed order and prints the gain
  with its confidence interval; only three pre-registered comparisons carry a p-value (Holm-adjusted).
  The word "significant" is not used.
- **A measurement hierarchy for answers.** Deterministic checks first (abstention, must-include recall,
  citation hit, invalid citations), then human labels, then a calibrated judge as *supporting* evidence:
  the judge is another model family, 12B or fewer, chosen before any generator run, its verdicts are
  cached, and **if its agreement with about 60 blind human labels is below Cohen's kappa 0.6 it is
  excluded from headline claims.** Claude writes no label.
- **A target is a target until measured.** Latency SLOs are asserted by no test (a shared PC would make
  the assertion measure noise); a bench reports misses next to the targets. Official numbers need an idle
  GPU, and a run that cannot prove it is marked not official.
- **Every number names its model and profile, and says so when the PC was busy.** A CSV number from the
  source project is quoted only as "measured in the source project".
- **Security is reported in two parts**: bypasses of guarantees (target 0) and hijacks (rates)
  (ADR-0004).

## Consequences
- Several results are deliberately unflattering and are in the README as they are (the reranker did not
  help on BEIR; most of the gain over the v1 replica is removing something v1 did wrong; the retrieval
  target is missed on long documents).
- A judge, a labelled set and a golden set that a person has not yet checked are *named* as unvalidated
  wherever they appear.

## Evidence
`docs/performance.md`, `README.md` (results), `project/steps/CP2-first-evidence.md`, `S5.2-answer-eval.md`.
