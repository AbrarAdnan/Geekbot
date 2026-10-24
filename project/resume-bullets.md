# Resume bullets (a draft for you to edit)

_Written by Claude in autopilot run 5 (D-097) as the S5.5 deliverable; the first draft of your own wording is
yours. **Every number below is from a result file named after it, measured on the models named, and says what
it is.** Do not strengthen a bullet beyond its caveat; the caveat is part of the claim (CLAUDE.md rule 8). Drop
a bullet rather than drop its caveat._

## Short list (pick three to five)

- **Built a local RAG engine** (LangGraph explicit state machine, SQLite FTS5 + Chroma, FastAPI, Streamlit,
  Ollama) whose design goal was to *measure* quality, speed and security rather than assert them: every number in
  its README regenerates from one command.
- **Measured retrieval honestly:** an ablation, one change per rung, on two standard BEIR sets (SciFact,
  NFCorpus): the rebuilt pipeline beats a replica of the original by +0.068 and +0.115 nDCG@10 (95% CIs excluding
  zero), **and found that most of the gain was removing a bug and that the cross-encoder reranker did not help**
  (`evals/results/s26-run1`).
- **Met the latency targets that depend on the GPU, and said which one it missed:** with a 12B model kept 100% on
  a 10 GB GPU, first token p50 0.68 to 0.76 s (target 2.5 s) and zero model loads during questions across a
  16-setting sweep on an idle machine; retrieval p95 missed its 400 ms target by up to 35 ms at 11 of 16 points
  (`evals/results/s51-run2`, `docs/performance.md`).
- **Security as a guarantee plus a measurement, kept apart:** a taint-label model with an egress gate (a web query
  built from untrusted text is never sent without the user approving the exact text), an output filter against
  markdown-image exfiltration, and a 43-attack injection evaluation: **0 of 129 attempts bypassed a control and
  the model obeyed in 5 of 43 attacks**; the evaluation also found three holes in the filter, which were fixed
  (`docs/security.md`, D-100, D-103).
- **Built evaluation that can fail:** a blind human-labelling page, Cohen's kappa with an exclusion rule for a
  judge model, an idle-GPU guard for official benchmarks, detectors written independently of the controls they
  test, and controls broken on purpose in tests to check the evaluation reports them.
- **Integrated a second system without coupling the repositories:** copied the author's CSV analyst package
  behind one import-linter-enforced adapter, reproduced its benchmark within a band fixed beforehand (33 of 40
  against the source's 35), and measured the one part that is new (the follow-up rewrite: no failure was caused by
  it on 15 items, which are the author's own and few).

## Longer versions and what each rests on

| claim | evidence | caveat to keep with it |
|---|---|---|
| +0.068 / +0.115 nDCG@10 over the v1 replica | `evals/results/s26-run1`, README "Retrieval quality" | one embedder, one reranker depth; most of the gain is rung R2 (removing something v1 did wrong) |
| reranker no help on BEIR | same | it helped on the provisional golden set; the decision is open |
| first token p50 0.68 to 0.76 s, 0 loads, 100% GPU | `evals/results/s51-run2` | one PC, 65-chunk corpus, an idle-GPU guard passed; golden questions are easy |
| retrieval p95 389 to 435 ms vs 400 ms | same | the CPU and reranker depth, not the GPU |
| 19/20 unanswerable refused, 6/80 answerable refused | `evals/results/s52-run1` | golden v0 is written by the system's author; one model; run-to-run variation (Gate A saw 5/65) |
| judge model | `evals/results/s52-run1` | **unvalidated: no human labels yet; do not cite it** |
| CSV 33/40 (source 35/40), 0/30 invented | `project/steps/C3-csv-swap-repro.md` | a check on the copy, not a new claim; the source project's own numbers are only quoted as its |
| follow-up rewrite 12/15 vs oracle 11/15, rewrite loss 0 | `evals/results/c6-run1` | 15 items I wrote; two of the three failures are ambiguities in the items; one unstable item is the whole difference |
| 0/129 bypasses, 15/129 hijacks (5/43 attacks) | `evals/results/s54-run2` | the 3 runs of each attack agreed (effective N = 43); my attacks; poison always in the prompt; the filter had three holes in the first sitting and "none left" is not shown |

## What to say about the process (if asked)

- The plan went through 22 revisions and 103 logged decisions; the build was done in small reviewed steps, with
  a step log, a decision log and a status file per step, so every claim above has a trail.
- Findings that cost something to report: v1's 9 of 15 files unloadable in its own README environment; a
  reranker that did not earn its latency; a security filter that had holes; a judge that is unvalidated.
