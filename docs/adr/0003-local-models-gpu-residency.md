# ADR-0003: Local models of 12B or fewer, the chat model kept 100% on the GPU

- **Status:** accepted (PLAN.md §6.5; D-007 to D-009, D-028, D-037, D-042)
- **Date:** 2026-10-07 (measured in S3.3, C3, S5.1)

## Context
The target machine has one 10 GB GPU that is shared with other work. A model that does not fit entirely
in video memory is partly run on the CPU and is several times slower; a model that is loaded and
unloaded between questions makes the first token slow. v1 set no context size, so Ollama ran its model at
262,144 tokens, about 44% on the CPU (measured once at CP2).

## Decision
- Chat model `gemma4:12b`, CSV model `qwen2.5-coder:7b`; **nothing over 12B is used or suggested**.
  A lighter `proto` profile (`gemma3:4b`, `qwen2.5-coder:3b`) exists for building; every reported number
  names the model that produced it.
- **The two chat models are never on the GPU together.** A swap is explicit: unload one (`keep_alive:
  0`), wait, then warm up the other with the context size real questions will use (R-2, R-4).
- **The embedder and the reranker run on the CPU**, so the active mode's model keeps the whole GPU.
- Context sizes are fixed (8,192 chat, 16,384 CSV) and the warm-up sends exactly what queries send, or
  Ollama would reload the model.
- Thinking is off by default; "think harder" is a per-question toggle.
- Official numbers are recorded only when an idle-GPU guard passes; a run that fails it still runs and
  is marked not official.

## Consequences
- Residency and loads are *checked* (`geekbot doctor`, the bench, a live test), not assumed.
- The first question after a change of mode pays a swap (measured in `docs/performance.md`); the UI
  says so.
- Whether to quantize the KV cache (q8_0) is a measured trade, left to the user.

## Evidence
`docs/performance.md` (every number there is from `evals/results/` and names its model),
`project/steps/S5.1-bench-sweep.md`, `C3-swap-and-reproduction.md`.
