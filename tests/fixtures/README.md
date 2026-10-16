# Recorded outputs (test fixtures)

Parsers are tested against real or realistic program output so they don't drift from reality.

| File | Origin |
|---|---|
| `ollama_ps_one_model.txt` | **Real**, captured from Ollama 0.34.2 on 2026-10-15 (`gemma3:4b` just loaded) |
| `ollama_ps_empty.txt` | Constructed: header only (nothing loaded) |
| `ollama_ps_split.txt` | Constructed from Ollama's documented `48%/52% CPU/GPU` form; not yet seen on this PC |
| `ollama_ps_two_models.txt` | Constructed: two models, one on the CPU |
| `ollama_ps_no_context.txt` | Constructed: the older layout without a CONTEXT column |

"Constructed" files copy the real file's column layout. Replace one with a real capture the first time that situation shows up (e.g. the split form when a model doesn't fit).
