# ADR-0006: CSV mode is a copied package behind one adapter, and a thread has one mode

- **Status:** accepted (PLAN.md §6.7; D-036 to D-041, D-045)
- **Date:** 2026-10-11 (built in C1 to C6)

## Context
Numbers in a table cannot be read reliably from retrieved text: a model that is shown row fragments
guesses. The author's separate project (Gigalogy) already answers questions about one CSV by writing one
read-only SQL query, running it, and explaining the result, with its own benchmark. Re-implementing it
would lose that work; depending on it as a library would couple two repositories that change at different
speeds.

## Decision
- **Copy the package in** (`src/csv_analyst/`), not as a dependency. `VENDORED.md` records the source
  commit, what was cut (the 27B heavy tier, the hosted providers, the interface layer), every local edit
  and how to re-sync. The package's own no-model tests move to `tests/csv_analyst/`.
- **One adapter.** Only `geekbot.csv_mode` imports `csv_analyst` (an import-linter contract). The adapter
  gives the package Geekbot's Ollama settings, a per-question time limit and the model name in every
  result.
- **A thread has one mode, fixed at its first turn** (`chat` or `csv`), so memory, taint and model
  residency stay simple. The package has no memory, so a follow-up is rewritten into a standalone
  question by the graph's `condense` step with the CSV model (C4).
- **CSV cells are untrusted.** Replies pass the same output filter (G3); the SQL engine is read-only
  (tested); the CSV never reaches the web tool or leaves the machine.
- **Results are reported separately.** Gigalogy's numbers are quoted only as measured in the source
  project; Geekbot reports its own reproduction (C3), its follow-up measurement (C6), its speed (S5.1)
  and its CSV-cell attack results (S5.4).

## Consequences
- Following the source project means a deliberate re-sync and a re-run of the reproduction check.
- The rewrite step is the part of CSV mode that is Geekbot's own and the part C6 measures: it separates
  failures the rewrite caused from failures the package would have had anyway.
- A table is held in memory per session; a registry bounds the count and the bytes and evicts the least
  recently used (C5).

## Evidence
`src/csv_analyst/VENDORED.md`, `project/steps/C3-swap-and-reproduction.md`, `C6-csv-followups.md`,
`evals/results/c6-run1/`, `docs/performance.md`.
