# Bench report: s51-run2

Chat model `gemma4:12b`, CSV model `qwen2.5-coder:7b`, profile `target`, embedder `nomic-embed-text` and reranker `Xenova/ms-marco-MiniLM-L-6-v2` on the CPU; Ollama 0.34.2; run 2026-10-24T12:39:49+06:00.

**Official: yes.**

## The machine at the start
GPU idle: **yes** (969 of 10240 MiB used with our models unloaded, 0.0% busy; waited 0.0 s). 
CPU load before the first section: 4.0%.

## Ingest
15 documents, 65 chunks in 6.2 s: **10.5 chunks per second** (embedder on the CPU).

## VRAM at the product defaults (PLAN.md §6.5 budget, measured)
| state | GPU memory in use (MiB) | free (MiB) |
|---|---|---|
| nothing of ours loaded | 969 | 9271 |
| chat model warm | 9306 | 934 |
| CSV model warm | 6556 | 3684 |

## The sweep (every row: the real answer graph, retrieval on the CPU)
Quality is the mean must-include recall over the answerable golden questions asked (Claude-authored, provisional). The model's first token and the question-to-first-token time are in seconds; p95 over few runs is close to the maximum.

| ctx | KV | batch | top-k | GPU % | load s | peak MiB | quality | 1st token p50 | Q to 1st p50 | Q to 1st p95 | tok/s p50 | whole p95 | retrieval p95 ms | loads | misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | f16 | 256 | 3 | 100 | 4.34 | 9104 | 0.708 | 0.66 | 0.97 | 1.27 | 71.1 | 1.63 | 404 | 0 | - |
| 4096 | f16 | 256 | 5 | 100 | 0.0 | 9104 | 0.708 | 0.91 | 1.10 | 1.67 | 70.7 | 1.88 | 402 | 0 | - |
| 4096 | f16 | 512 | 3 | 100 | 5.11 | 9252 | 0.708 | 0.58 | 0.89 | 1.16 | 70.6 | 1.54 | 417 | 0 | - |
| 4096 | f16 | 512 | 5 | 100 | 0.0 | 9252 | 0.708 | 0.79 | 1.14 | 1.50 | 69.5 | 1.66 | 393 | 0 | - |
| 8192 | f16 | 256 | 3 | 100 | 5.98 | 9252 | 0.688 | 0.66 | 0.96 | 1.29 | 70.9 | 1.63 | 410 | 0 | - |
| 8192 | f16 | 256 | 5 | 100 | 0.0 | 9170 | 0.708 | 0.91 | 1.10 | 1.67 | 69.1 | 1.91 | 389 | 0 | - |
| 8192 | f16 | 512 | 3 | 100 | 6.03 | 9320 | 0.750 | 0.58 | 0.90 | 1.16 | 70.4 | 1.52 | 401 | 0 | - |
| 8192 | f16 | 512 | 5 | 100 | 0.0 | 9320 | 0.708 | 0.79 | 1.12 | 1.54 | 69.7 | 1.74 | 417 | 0 | - |
| 4096 | q8_0 | 256 | 3 | 100 | 4.1 | 8892 | 0.708 | 0.33 | 0.70 | 1.07 | 66.9 | 1.23 | 405 | 0 | - |
| 4096 | q8_0 | 256 | 5 | 100 | 0.0 | 8892 | 0.833 | 0.37 | 0.72 | 0.88 | 66.6 | 1.22 | 400 | 0 | - |
| 4096 | q8_0 | 512 | 3 | 100 | 5.35 | 9000 | 0.708 | 0.33 | 0.68 | 0.92 | 67.3 | 1.12 | 396 | 0 | - |
| 4096 | q8_0 | 512 | 5 | 100 | 0.0 | 9000 | 0.750 | 0.37 | 0.74 | 0.97 | 66.1 | 1.26 | 403 | 0 | - |
| 8192 | q8_0 | 256 | 3 | 100 | 5.99 | 9000 | 0.792 | 0.34 | 0.71 | 1.10 | 66.5 | 1.26 | 435 | 0 | - |
| 8192 | q8_0 | 256 | 5 | 100 | 0.0 | 8982 | 0.750 | 0.39 | 0.76 | 1.01 | 64.7 | 1.31 | 430 | 0 | - |
| 8192 | q8_0 | 512 | 3 | 100 | 6.31 | 9071 | 0.708 | 0.34 | 0.71 | 0.99 | 64.4 | 1.19 | 426 | 0 | - |
| 8192 | q8_0 | 512 | 5 | 100 | 0.0 | 9019 | 0.708 | 0.36 | 0.74 | 0.95 | 65.4 | 1.29 | 399 | 0 | - |

The retrieval target (p95 400 ms) is the CPU's and is the same under every setting, so it takes no part in choosing; the column shows how this sitting did against it.

**Chosen default:** ctx8192-kvq8_0-batch256-k3: 16 of 16 configurations meet every rule; 2 of them are within 0.05 of the best must-include recall (0.833), which is as fine as this question set can tell apart, so the fastest median first token decides (0.71 s), then the smaller VRAM peak
The product ships `ctx8192-kvf16-batch512-k5`.

## Follow-up questions (condense, then retrieval, then the model)
6 runs at `ctx8192-kvf16-batch512-k5`: question to first token p50 2.05 s / p95 2.33 s; whole turn p50 1.46 s / p95 2.09 s.

## Thinking on, as information (thinking is off by default)
3 runs at `ctx8192-kvf16-batch512-k5`: question to first token p50 5.89 s / p95 6.11 s; whole turn p50 5.93 s / p95 6.07 s.

## CSV mode
Model `qwen2.5-coder:7b`; 40 question-runs (hospital, sales): seconds per question **p50 2.8 / p95 6.39 / max 9.75**; model loads during questions: 0; solved 31 of 40 (a check on the copy, not a new claim).
Swaps (3 round trips): chat to csv median 2.74 s (max 2.76), csv to chat median 4.97 s (max 5.0); every residency check passed: True.
Residency: PASS: qwen2.5-coder:7b: 100% GPU, context 16384.
CSV official notes: partial run (not both suites, fewer than 3 runs, or --limit).

## Targets (PLAN.md §6.5), for reading the tables above
Retrieval plus rerank p95 400 ms; question to first token p50 2.5 s and p95 4 s; whole turn p95 15 s; model loads during questions 0; CSV: p50 3 s, p95 10 s, swap 6 s. A target is a target until a sitting on an idle PC measures it.
