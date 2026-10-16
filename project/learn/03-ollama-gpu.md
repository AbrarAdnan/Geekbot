# 03 — How Ollama loads models, and what to check (step S0.3)

## 1. What "loading" means
When you ask Ollama for a model it hasn't loaded, it copies the model's **weights** (the learned numbers, the big file on disk) into memory, preferably the GPU's own memory (**VRAM**). Generating text then runs on the GPU. A model that sits partly on the CPU still works but is much slower, because every token waits for the slow part.

## 2. VRAM math: weights + KV cache + buffers
VRAM use of a loaded model is roughly:
- **Weights:** about the file size (e.g. `gemma4:12b` ≈ 7.6 GB with vision/audio parts).
- **KV cache:** the attention "memory" of the tokens seen so far, kept so it isn't recomputed. It grows with the **context length** (`num_ctx`) and with the number of layers and attention heads.
- **Compute buffers:** scratch space for the calculation (depends on settings such as `num_batch`).

Ollama decides how many layers fit; if weights + cache + buffers exceed free VRAM, the remaining layers go to the CPU. That is the failure Geekbot's rule **R-1** forbids.

**gemma4's cache estimate (PLAN §1):** 48 layers. 40 are *sliding-window* layers that only look back 1,024 tokens (8 KV heads × 256 dims); 8 are *global* layers that look at everything (1 KV head × 512 dims). So at 8k context the cache is about 40 × 1,024 tokens × 8 KiB + 8 × 8,192 tokens × 2 KiB ≈ 448 MiB — small compared with a plain full-attention model. It is an estimate until `doctor` and C3/S5.1 measure it.

## 3. `keep_alive`
After a request, Ollama keeps the model loaded for `keep_alive` (default **5 minutes**), then unloads it. The next request after that pays a cold load of seconds. Geekbot sets a long value (`30m`) and warms the model up at start. `keep_alive: 0` unloads immediately; swaps (R-4) use this on purpose instead of letting Ollama evict a model by itself, which can leave a model half on the CPU.

## 4. The context trap
A model is loaded with a particular `num_ctx`. If a later request asks for a different one, Ollama **reloads** the model. So warm-up and real queries must use the same value: Geekbot reads both from `chat_num_ctx` / `csv_num_ctx` in the settings. Your own `ollama ps` showed the point: after a plain `ollama run gemma3:4b`, the CONTEXT column said **131072**, far more than the 8192 Geekbot plans to use. `doctor` warns when the loaded context differs from the planned one.

## 5. Reading what is loaded
- `ollama ps`: which models are loaded, how they are split (`100% GPU`, `100% CPU`, or `48%/52% CPU/GPU`), their context and how long they stay.
- `nvidia-smi`: the GPU's memory use across *all* programs. Other programs' use matters: a benchmark is only valid on an idle GPU (PLAN §2).

Geekbot parses both (`llm/vram.py`) from text. Parsers are tested on *recorded* output (`tests/fixtures/`), so they are checked against reality without needing a GPU. The weak point of parsing human-oriented text is that its layout can change between versions (your Ollama 0.34.2 has a CONTEXT column that older versions lack), so the parser finds values by **header name**, not by position, and the fixtures say which are real and which are constructed.

## 6. Design pattern: separate reading from deciding
`doctor.py` never runs a command or parses text. It receives already-parsed data and returns `Check` results. Each check is a pure function (data in, verdict out), so it is tested with made-up inputs: a model 52% on the GPU, a context mismatch, a missing model. Only the CLI wires in the real commands.

## Interview-style questions
1. **Why must a model be 100% on the GPU?** Any layers on the CPU make generation wait for the slow part on every token; speed drops sharply, which breaks the latency targets.
2. **What goes into a model's VRAM use?** Weights, KV cache (grows with context), and compute buffers.
3. **Why does Geekbot warm up the model with the same `num_ctx` as real queries?** A different context makes Ollama reload the model, so the first real query would pay a cold load.
4. **Why test a parser on recorded output instead of calling `ollama ps` in the test?** The test runs anywhere, fast and repeatably, and a recorded real sample catches format surprises.
5. **Why does `doctor` take parsed data instead of running commands itself?** So each rule can be tested with made-up cases, including ones that are hard to produce on a real machine.
