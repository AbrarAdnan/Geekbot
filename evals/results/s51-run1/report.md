# Bench report: s51-run1

Chat model `gemma4:12b`, CSV model `qwen2.5-coder:7b`, profile `target`, embedder `nomic-embed-text` and reranker `Xenova/ms-marco-MiniLM-L-6-v2` on the CPU; Ollama 0.34.2; run 2026-10-24T12:16:47+06:00.

**Official: NO (a plumbing check, not a result).**
- the GPU was not idle: the GPU is 21% busy (limit 15%); the driver lists these processes: 21 of them (the names are not saved)
- the CPU gate was not idle before some sweep points

## The machine at the start
GPU idle: **no** (975 of 10240 MiB used with our models unloaded, 21.0% busy; waited 60.0 s). Reasons it was not: the GPU is 21% busy (limit 15%); the driver lists these processes: 21 of them (the names are not saved)
CPU load before the first section: 7.0%.

## VRAM at the product defaults (PLAN.md §6.5 budget, measured)
| state | GPU memory in use (MiB) | free (MiB) |
|---|---|---|
| nothing of ours loaded | 973 | 9267 |
| chat model warm | 9310 | 930 |
| CSV model warm | 6560 | 3680 |

## The sweep (every row: the real answer graph, retrieval on the CPU)
Quality is the mean must-include recall over the answerable golden questions asked (Claude-authored, provisional). The model's first token and the question-to-first-token time are in seconds; p95 over few runs is close to the maximum.

| ctx | KV | batch | top-k | GPU % | load s | peak MiB | quality | 1st token p50 | Q to 1st p50 | Q to 1st p95 | tok/s p50 | whole p95 | retrieval p95 ms | loads | misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | f16 | 256 | 3 | 100 | 4.68 | 9108 | 0.750 | 0.69 | 1.04 | 1.35 | 66.9 | 1.71 | 446 | 0 | - |
| 4096 | f16 | 256 | 5 | 100 | 0.0 | 9108 | 0.708 | 0.92 | 1.17 | 1.69 | 69.7 | 1.93 | 439 | 0 | - |
| 4096 | f16 | 512 | 3 | 100 | 16.59 | 9256 | 0.750 | 0.57 | 0.93 | 1.16 | 70.9 | 1.52 | 416 | 0 | - |
| 4096 | f16 | 512 | 5 | 100 | 0.0 | 9256 | 0.708 | 0.79 | 1.13 | 1.51 | 70.5 | 1.64 | 417 | 0 | - |
| 8192 | f16 | 256 | 3 | 100 | 5.98 | 9256 | 0.688 | 0.66 | 0.99 | 1.31 | 70.6 | 1.65 | 416 | 0 | - |
| 8192 | f16 | 256 | 5 | 100 | 0.0 | 9174 | 0.667 | 0.91 | 1.12 | 1.66 | 69.8 | 1.93 | 405 | 0 | - |
| 8192 | f16 | 512 | 3 | 100 | 5.66 | 9324 | 0.708 | 0.57 | 0.91 | 1.20 | 70.9 | 1.54 | 423 | 0 | - |
| 8192 | f16 | 512 | 5 | 100 | 0.0 | 9324 | 0.708 | 0.80 | 1.15 | 1.51 | 69.7 | 1.67 | 418 | 0 | - |
| 4096 | q8_0 | 256 | 3 | 100 | 4.33 | 8896 | 0.708 | 0.33 | 0.70 | 1.11 | 67.1 | 1.26 | 408 | 0 | - |
| 4096 | q8_0 | 256 | 5 | 100 | 0.0 | 8896 | 0.792 | 0.38 | 0.72 | 0.86 | 66.9 | 1.23 | 393 | 0 | - |
| 4096 | q8_0 | 512 | 3 | 100 | 5.36 | 9004 | 0.688 | 0.33 | 0.69 | 0.94 | 67.0 | 1.13 | 414 | 0 | - |
| 4096 | q8_0 | 512 | 5 | 100 | 0.0 | 9004 | 0.792 | 0.37 | 0.74 | 0.95 | 66.4 | 1.26 | 410 | 0 | - |
| 8192 | q8_0 | 256 | 3 | 100 | 5.65 | 9004 | 0.750 | 0.33 | 0.69 | 1.08 | 66.9 | 1.23 | 395 | 0 | - |
| 8192 | q8_0 | 256 | 5 | 100 | 0.0 | 8932 | 0.792 | 0.38 | 0.72 | 0.88 | 66.1 | 1.29 | 387 | 0 | - |
| 8192 | q8_0 | 512 | 3 | 100 | 5.69 | 9042 | 0.792 | 0.33 | 0.69 | 0.93 | 67.1 | 1.13 | 405 | 0 | - |
| 8192 | q8_0 | 512 | 5 | 100 | 0.0 | 9042 | 0.708 | 0.36 | 0.73 | 0.95 | 67.0 | 1.25 | 408 | 0 | - |

The retrieval target (p95 400 ms) is the CPU's and is the same under every setting, so it takes no part in choosing; the column shows how this sitting did against it.

**Chosen default:** ctx8192-kvq8_0-batch256-k3: 16 of 16 configurations meet every rule; 7 of them are within 0.05 of the best must-include recall (0.792), which is as fine as this question set can tell apart, so the fastest median first token decides (0.69 s), then the smaller VRAM peak
The product ships `ctx8192-kvf16-batch512-k5`.

## Follow-up questions (condense, then retrieval, then the model)
6 runs at `ctx8192-kvf16-batch512-k5`: question to first token p50 1.98 s / p95 2.34 s; whole turn p50 1.26 s / p95 2.07 s.

## Thinking on, as information (thinking is off by default)
3 runs at `ctx8192-kvf16-batch512-k5`: question to first token p50 6.59 s / p95 7.13 s; whole turn p50 6.56 s / p95 7.34 s.

## CSV mode
Model `qwen2.5-coder:7b`; 40 question-runs (hospital, sales): seconds per question **p50 2.76 / p95 6.36 / max 9.56**; model loads during questions: 0; solved 31 of 40 (a check on the copy, not a new claim).
Swaps (3 round trips): chat to csv median 2.51 s (max 2.51), csv to chat median 4.99 s (max 5.51); every residency check passed: True.
Residency: PASS: qwen2.5-coder:7b: 100% GPU, context 16384.
CSV official notes: partial run (not both suites, fewer than 3 runs, or --limit).

## Targets (PLAN.md §6.5), for reading the tables above
Retrieval plus rerank p95 400 ms; question to first token p50 2.5 s and p95 4 s; whole turn p95 15 s; model loads during questions 0; CSV: p50 3 s, p95 10 s, swap 6 s. A target is a target until a sitting on an idle PC measures it.
