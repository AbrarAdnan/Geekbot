# CP2 — Checkpoint: first evidence

- **Phase / Gate:** Phase 2 close, Gate A. In the order of PLAN rev 19 the checkpoint comes after S2.6 and before S3.5: **the user reviews this before S3.5 starts.**
- **Chat date(s):** 2026-10-18 (autopilot run 3, D-078)
- **Status:** prepared and verified by Claude under D-078; **reviewed by the user on 2026-10-18; closed by D-082** (their answers to the six decisions below are in D-082: 1 to 3 wait until the project has been used, 4 is option B, 5 keep golden v0 for now, 6 delegated to Claude, who green-lit S3.3, S3.4 and C4)
- **Commit:** `8be0862` (squash commit of PR #36 into `v2`, 3 CI checks passed; branch tip `23e1808`, identical tree; the branch was deleted afterwards, D-056). Merged by Claude under D-078, awaiting the user's review.
- **Lesson note:** deferred (prototyping mode, D-047)

## Goal
PLAN §11: the first ablation table, the one-time v1 reference run (D-034), a draft README results section and the Phase 1–2 look-back, so that the user and Claude can look at the evidence together before the answering path grows further (S3.5 onward). Nothing here is a verdict: the run's job was to produce numbers with their intervals and to keep every choice visible (D-079, D-080, D-081).

## What exists
- The numbers: `evals/results/s26-run1/report.md` (generated; ladder, leave-one-out, sweeps, bake-offs, abstain) and the JSON files beside it; `evals/results/v1-reference.json` (the v1 run).
- The draft README results section: `project/cp2/README-results-draft.md` (not in `README.md`: the README is rewritten in Phase 5, §11).
- The v1 reference-run scripts: `evals/v1_reference/` (`run_v1_reference.py`, and the two-step splitter check `dump_splitter_cases.py` + `compare_splitter.py`).
- The Phase 1–2 look-back: Phase 1 in the [S1.6 log](S1.6-ingest.md), Phase 2 in the [S2.6 log](S2.6-ablation-tooling.md) (it covers S2.1 to S2.6).

## The first-evidence table
`nomic-embed-text` (CPU, held), chunks: legacy 800 characters (R0 to R3) or v2 512 tokens (R4 on), reranker `Xenova/ms-marco-MiniLM-L-6-v2` at depth 10, 50 fusion candidates a side, top 10 chunks, documents ranked at their best chunk. **BEIR rows: standard datasets, standard metrics. Golden v0: Claude-authored and provisional (D-067), not a headline.** Latency is retrieval including the question's embedding and the rerank, on an idle PC (CPU gate), p50 in ms.

| rung | change | SciFact nDCG@10 | NFCorpus nDCG@10 | golden test Recall@5 | p50 ms (SciFact / NFCorpus / golden) |
|---|---|---|---|---|---|
| R0 | legacy v1 replica | 0.6440 | 0.2353 | 0.8681 | 463 / 533 / 429 |
| R1 | L2 to cosine | 0.6440 | 0.2356 | 0.8681 | 473 / 528 / 416 |
| R2 | remove template multi-query | 0.6737 | 0.2961 | 0.9375 | 25 / 19 / 21 |
| R3 | + prefixes | 0.6820 | 0.3148 | 0.8958 | 26 / 20 / 22 |
| R4 | new chunking | 0.6839 | 0.3366 | 0.8438 | 26 / 20 / 23 |
| R5 | + contextual header | 0.6992 | 0.3383 | 0.8854 | 26 / 19 / 22 |
| R6 | + BM25 via RRF | 0.7375 | 0.3492 | 0.9514 | 48 / 25 / 24 |
| R7 | + rerank | 0.7124 | 0.3502 | 0.9410 | 390 / 348 / 351 |

The pre-registered family (R7 against R0, Holm over three, D-065): SciFact nDCG@10 Δ +0.0684 [+0.0345, +0.1039] (adjusted p 0.0003, n=300); NFCorpus Δ +0.1148 [+0.0972, +0.1330] (adjusted p 0.0003, n=323); golden test Recall@5 Δ +0.0729 [−0.0069, +0.1562] (adjusted p 0.1117, n=48, provisional). Rung-by-rung Δ with intervals, leave-one-out, the sweeps and the bake-offs are in the report; **three results to look at first:** (1) most of the distance between v1 and v2 is rung R2, removing v1's five-query search and its re-sort (SciFact +0.0297 [+0.0189, +0.0417], NFCorpus +0.0606 [+0.0501, +0.0714]); (2) the reranker did not help on the two BEIR sets (SciFact −0.0250 [−0.0455, −0.0042]); (3) the retrieval-plus-rerank p95 at depth 10 is 528 to 626 ms on BEIR against the 400 ms target (375 to 400 ms on golden).

## The v1 reference run (D-034, D-081)
The real v1 (tag `v1.0`) in `geekbot-v1` (Python 3.10, v1's `requirements.txt`), `gemma4:12b`, web off. What it found, as measured:
- **9 of the 15 golden files cannot be loaded** in the environment v1 documents (`unstructured` is not in its requirements): the Word, PowerPoint, Excel, Markdown and HTML files. 6 files, 27 chunks, were indexed.
- **Quality:** document-level Recall@5 over the 65 single- and multi-hop questions: 0.3974 (1.0 on the 22 whose sources v1 can load, 0.0891 on the 43 it cannot).
- **Speed:** retrieval alone, median 0.31 s over 100 questions (max 2.2 s). A full question with `gemma4:12b`: 35 to 755 s, median 400 s, over 20 questions (every 5th; asking all 100 would have taken about 10 hours). Cause seen: v1 sets no context size, Ollama loaded the model with 262,144 tokens, 44% on the CPU; v1 does not stream. The slowness is measured; its causes are only partly observed (the context size and the CPU split were seen in `ollama ps`; whether the model "thinks" was not checked), so PLAN §5 #17 stays a hypothesis about the cause.
- **R0 replicates v1?** v1's real splitter (0.3.11) and the 1.x splitter of R0 cut 624 texts (1,525 chunks) identically. On the 22 questions v1 can answer from, Recall@5 is 1.0 for v1 and 0.9394 for R0, identical on 20 of 22 (R0 sees all 15 documents, v1 six). Supported, not proven; the stricter test (R0 on the same six files) was not run.
- **Not compared:** v2's speed on `gemma4:12b` is not measured (S5.1), so there is no v1-against-v2 speed claim anywhere.

## Decisions for you (nothing was changed by any result)
1. **The reranker:** keep MiniLM at depth 10, change it, or drop it? It helped on golden test, did not on BEIR, and costs about 330 ms of the 400 ms budget. `bge-reranker-base` is not distinguishable from it and is 4 to 5 times slower. (The depth sweep gave no quality reason to leave 10 on BEIR.)
2. **The embedder:** `embeddinggemma` was clearly better dense-only on both BEIR sets (SciFact +0.0638 [+0.0311, +0.0962], NFCorpus +0.0382 [+0.0212, +0.0556]) and the gap mostly closes after fusion and rerank; it ingests at about 75 to 90% of `nomic`'s speed. Switching means re-indexing. Its prefixes are as recalled from its model card (unchecked).
3. **The 400 ms target:** it holds on the small golden corpus and not on BEIR's long abstracts; change the target, the depth, or the way it is measured?
4. **S3.5's abstain signal:** the planned threshold on the reranker's best score is weak on golden v0 (dev F1 0.41, test F1 0.39, 19 of 32 answerable dev questions withheld to catch 7 of 8 unanswerable). What should S3.5 try first? (D-080.)
5. **Your own questions** (`golden-user.jsonl`): golden v0 is easy and small (32 development and 48 test retrieval questions) and cannot separate close systems; this is the most useful next input.
6. **Review S3.3, S3.4 and C4** (merged under D-071), which are still awaiting your review.

## Phase 1–2 look-back
In the [S1.6 log](S1.6-ingest.md) (Phase 1) and the [S2.6 log](S2.6-ablation-tooling.md) (Phase 2). In one paragraph: Phase 1 and 2 were built as planned in count (six steps each), with the golden set (S2.4) and the ablation (S2.6) moved by your decisions, and the real measurement came last, as it had to. The surprises were all about what was *not* needed or *not* true: most of v1's gap was something v1 did wrong, the reranker the plan treated as the finishing rung did not pay on standard data, and the cheap abstain signal was weak. The practical lesson is the one the Phase 0 look-back already had: run the real thing early. Next time, measure retrieval on the real corpus before building the layers that depend on it.

## Files changed
- `evals/results/v1-reference.json`, `evals/v1_reference/*`: the v1 run and the splitter check
- `project/cp2/README-results-draft.md`, this log, `project/STATUS.md`, `project/DECISIONS.md` (D-081), the run report

## Commands run (by Claude, D-078)
```
git worktree add scratch/v1-worktree v1.0
conda create -n geekbot-v1 python=3.10 -y ; pip install -r scratch/v1-worktree/requirements.txt   # in geekbot-v1, PyPI only
python evals/v1_reference/run_v1_reference.py ... --full-every 5     # in geekbot-v1, from the worktree; GPU; about 2 hours
python evals/v1_reference/dump_splitter_cases.py scratch/splitter-cases.json                     # geekbot env
python evals/v1_reference/compare_splitter.py scratch/splitter-cases.json                        # geekbot-v1 env: identical for 624 of 624 texts
```
GPU gate before the v1 run: only desktop applications on the GPU (1.1 GB of 10 GB), no model loaded; CPU load 5 to 10%. The models the run loaded were unloaded with `ollama stop`.

## Decisions
D-081 (the v1 run's design and findings). D-079 and D-080 were written for S2.6.

## Open issues / follow-ups
- Run R0 on the six files v1 can load, to turn "supports" into a stricter check (small: a bundle with six documents).
- `geekbot-v1` and `scratch/v1-worktree` stay in place (the worktree is in `scratch/`, git-ignored; `git worktree remove scratch/v1-worktree` when you no longer need it).

## Handoff to the next chat
After your review: S3.5 (abstain, cite_check, sanitize), where the abstain signal is the open design question (D-080), and `sanitize` for both branches (D-075). The retrieval defaults the graph uses (hybrid + MiniLM at depth 10, top-k 5) stay until you decide the items above.
