# Recorded outputs (test fixtures)

Parsers are tested against real or realistic program output so they don't drift from reality.

| File | Origin |
|---|---|
| `ollama_ps_one_model.txt` | **Real**, captured from Ollama 0.34.2 on 2026-10-15 (`gemma3:4b` just loaded) |
| `ollama_ps_empty.txt` | Constructed: header only (nothing loaded) |
| `ollama_ps_split.txt` | Constructed from Ollama's documented `48%/52% CPU/GPU` form; not yet seen on this PC |
| `ollama_ps_two_models.txt` | Constructed: two models, one on the CPU |
| `ollama_ps_no_context.txt` | Constructed: the older layout without a CONTEXT column |
| `sales_small.csv` | Constructed: 12 rows, four columns (region, month, units, revenue); the small table the CSV tests and the hand checks use |
| `bad_header_only.csv` | Constructed (C5): a header line and no data row. For the hand check of the CSV upload: the vendored loader refuses it with its own reason (its wording is the package's) and nothing must be left in `data\uploads`. Hand check of 2026-10-20: refused with a reason |
| `bad_empty.csv` | Constructed (C5): zero bytes. For the same hand check: the upload is refused as empty |

"Constructed" files copy the real file's column layout. Replace one with a real capture the first time that situation shows up (e.g. the split form when a model doesn't fit).
