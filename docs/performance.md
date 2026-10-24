# Performance: what was measured, on what, and what it says

Every number here is from a result file in `evals/results/` or a step log in `project/steps/`, names the
model that produced it, and says when the PC was not idle. **A target is a target until a row below
measures it.** Nothing here is a claim about any other machine.

## The setup
One Windows 11 PC with an RTX 3080 (10 GB), Ollama 0.34.2, which also runs the author's other projects.
Chat model **`gemma4:12b`**, CSV model **`qwen2.5-coder:7b`** (the target profile); `nomic-embed-text` for
embeddings and the `ms-marco-MiniLM-L-6-v2` cross-encoder (rerank depth 10) both on the **CPU**, so the
active model keeps the whole GPU. Context 8,192 (chat) and 16,384 (CSV). The knowledge base is the golden
corpus (15 files, 65 chunks), the same one the answer evaluation uses.

**Official numbers need an idle machine.** The bench's guard decides on GPU memory in use (limit 2,048 MiB
with our models unloaded), GPU utilisation (15%) and CPU load (15%); a run that fails it still runs and is
marked *not official* on the first lines of its report. The Windows driver lists every program with a window
as a "compute process", so the process list is only evidence, never the decision (D-098). `s51-run2` is the
official run (GPU 0% busy with 969 MiB in use, CPU 4%); `s51-run1` is not (the desktop kept the GPU 21% busy
before it started) and agrees closely.

## Against the targets of PLAN §6.5 (`evals/results/s51-run2`, 16 settings, 24 runs each)

| measure | target | measured (range over the 16 settings) | verdict |
|---|---|---|---|
| question to first token, p50 | 2.5 s | 0.68 to 0.76 s | met under every setting |
| question to first token, p95 | 4 s | 0.88 to 1.67 s | met |
| whole turn, p95 | 15 s | 1.12 to 1.91 s | met |
| a follow-up (rewrite, retrieval, first token), p50 / p95 | 5 s | 2.05 s / 2.33 s (shipped setting) | met |
| model loads during questions | 0 | 0 in all 16 settings | met |
| `gemma4:12b` on the GPU | 100% | 100% in all 16 settings | met |
| **retrieval plus rerank, p95** | **400 ms** | **389 to 435 ms** | **met at 5 of 16 points, missed at 11, by at most 35 ms** |
| CSV question p50 / p95 | 3 s / 10 s | 2.8 s / 6.4 s (max 9.8 s) | met |
| swap chat to CSV | 6 s | median 2.7 s | met |
| swap CSV to chat | 6 s | median 5.0 s (Gate A: median 5.2 s, once 6.1 s) | met on the median, missed once |

The retrieval target is missed by a little almost everywhere, and **it is the CPU and the reranker depth, not
the GPU**: retrieval does not depend on any swept setting, almost all of its time is the cross-encoder
(681 to 756 ms of the 716 to 788 ms retrieval totals of the four ungated questions of Gate A part C, when
the CPU load was not checked; 330 ms median on an idle PC at depth 10 in D-077), and on BEIR's long abstracts the target fails at the 95th percentile while on the small golden corpus
it holds. The reranker did not improve nDCG@10 on BEIR (docs/evolution.md); whether to keep it, reduce its
depth or relax the target is the author's decision (D-082).

Other readings:
- **Thinking on** ("think harder"): question to first token 5.9 s p50 (6.6 s in run 1), whole turn about 5.9 s.
  For information; no target.
- **A cold start** (unload, load, first question): 1.5 s from question to first token (run 1, two rounds).
- **Ingest:** 65 chunks in 6.2 s, **10.5 chunks per second** (embedder on the CPU, a fresh folder).
- **CSV follow-up evaluation** (`evals/results/c6-run1`, `qwen2.5-coder:7b`): a two-turn item plus the oracle
  question took a mean of 10.4 s per run.

## VRAM (the product's own server and settings, `s51-run2`)
| state | in use | free |
|---|---|---|
| nothing of ours loaded (the desktop) | 969 MiB | |
| `gemma4:12b` warm, context 8,192 | 9,306 MiB | 934 MiB |
| `qwen2.5-coder:7b` warm, context 16,384 | 6,556 MiB | 3,684 MiB |

Peak during the sweep: 9,104 to 9,320 MiB with an f16 KV cache, **8,892 to 9,071 MiB with q8_0** (about 200 MiB
less). Free memory under `gemma4:12b` is under 1 GiB: the plan's residency risk is real and checked, not assumed.

## What the sweep changed (nothing) and what it found
The sweep varied context size (4,096 or 8,192), KV-cache type (f16 or q8_0), batch size (256 or 512) and
top-k (3 or 5) on a **private Ollama server on port 11435** (the KV type is a server setting; the author's
server on 11434 was never restarted).
- **The KV-cache type is the one setting that moved speed**, in both runs: the model's median time to first
  token was **0.33 to 0.39 s with q8_0 against 0.57 to 0.92 s with f16**; the text then arrives a little slower
  (64 to 67 tokens/s against 69 to 71). Context size and batch size changed nothing the runs could tell apart;
  top-k 3 was a little faster than 5 (a shorter prompt).
- **Quality did not separate the settings**: must-include recall 0.67 to 0.83, and the order of the settings
  changed between the two runs. A difference of one missed phrase moves the number by 0.04. Whether q8_0 hurts
  long or hard prompts is **not measured**.
- Nothing was changed: the product ships context 8,192, f16, batch 512, top-k 5. Whether to ship q8_0
  (`OLLAMA_KV_CACHE_TYPE=q8_0` on the Ollama server) is the author's decision.

## Retrieval speed on its own (`evals/results/s26-run1`, idle PC)
Retrieval including the question's embedding and the rerank takes a median of 350 to 390 ms at depth 10
(95th percentile 528 to 626 ms on the two BEIR sets, 375 to 400 ms on the golden corpus); without the
reranker 19 to 48 ms. Depths 20 and 30 cost two and three times as much and did not improve quality on the
tuning questions. The parallel schedule of the two searches (S3.6) is **not faster** than the sequential one:
the lexical search takes 1.8 ms, so the overlap can save at most that (362.8 ms sequential against 369.6 ms
parallel, whole turn p50, 40 runs each, with an instant fake model); it stays on for the stage-level retry and
the "Ranking the passages" step.

## What these numbers do not show
- Anything about another machine, another corpus size (65 chunks), or long conversations.
- Quality differences between settings, or q8_0 on long prompts (too few, easy questions).
- A third sitting or a quieter hour could move the retrieval p95 either way.
- The out-of-memory corner: every one of the 16 settings fitted, so that failure path is tested with fakes only.
- A speed comparison with v1: v1 was run once at CP2 (median 400 s per question with no context size set) on
  other questions and at another time, and **no ratio is claimed**.

## Regenerate
```
set GEEKBOT_PROFILE=target
geekbot bench --name my-run
geekbot bench --name my-run --report-only
```
Give the GPU to the run (about 30 minutes); the report's first lines say whether it is official and why not.
