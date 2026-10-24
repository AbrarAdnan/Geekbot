# Geekbot v2

A local RAG engine: ask questions about your own documents and get answers that cite where they came
from. Everything runs on your machine. Geekbot v2 is a rebuild of the first version (v1, kept at the git
tag `v1.0`) with three aims: **measure retrieval quality**, **control inference cost**, and **treat
retrieved content as untrusted data**. It has a second conversation mode that answers questions about one
CSV file by writing and running a read-only SQL query.

> **Status: a working prototype with its evidence.** Everything below is measured, names the model that
> produced it, and can be regenerated with one command (see "Regenerate a number"). Several results are
> unflattering and are here as they are. Some evidence is **not complete**: the answer judge has not been
> checked against human labels, the golden questions are written by the system's author and are
> provisional, and the security evaluation is 43 attacks I wrote. "What these results do not show" is part of
> the results.

## What it does

- **Chat mode.** Index PDFs, Word and PowerPoint files, Markdown, text, code and notebooks (CSV and Excel
  files are indexed as a description of their columns, never their rows); ask questions in a browser or
  on the command line; get streamed answers with `[1]`-style citations, follow-ups that resolve against the
  conversation, and a refusal when the documents do not hold the answer.
- **CSV mode.** Start a conversation about one CSV file: the answer shows the SQL that was run and the
  result table; a follow-up ("and by month?") is rewritten into a full question first.
- **Web search, opt-in.** Off by default, and off means the search tool is not even built. When it is on, your
  own words go out as you typed them; a query rewritten from earlier answers that used documents or web
  pages is shown to you, as the exact text that would leave your machine, and is only sent if you say so.
- **Local only.** Models run in [Ollama](https://ollama.com), the index is SQLite (the source of truth) plus
  Chroma (a derived index), and the server listens on `127.0.0.1` only. The browser UI and the server share a
  local password (an API key in `data/api-key`); nothing is sent to any cloud service.

## Run it

You need [Ollama](https://ollama.com/download) running, and [conda](https://docs.conda.io).

```
ollama pull gemma3:4b
ollama pull nomic-embed-text
conda env create -f environment.yml
conda activate geekbot
geekbot doctor
geekbot up
```

`geekbot up` starts the API server and the browser UI (it opens `http://127.0.0.1:8501`). Add documents on the
**Knowledge** page (upload files, or give a folder listed in `GEEKBOT_ALLOWED_ROOTS` in your `.env`), then ask
on the **Chat** page. For CSV mode choose "A CSV file" when you start a conversation. `.env.example` lists every
setting, including the web search.

`gemma3:4b` is the light model used while building (the `proto` profile). The target models are `gemma4:12b`
for chat and `qwen2.5-coder:7b` for CSV mode (`GEEKBOT_PROFILE=target`): every number below comes from them.
Nothing over 12 billion parameters is used.

## Results

All of this is one PC (an RTX 3080 with 10 GB, Windows 11, Ollama 0.34.2) shared with other work; official
timings are taken only when a guard finds the GPU and CPU idle. Details, caveats and the raw files are in
[`docs/performance.md`](docs/performance.md), [`docs/security.md`](docs/security.md) and `evals/results/`.

### Retrieval quality (standard BEIR datasets, standard metrics)

Each change was added one at a time and measured with nDCG@10 on BEIR SciFact (300 test queries) and NFCorpus
(323); differences carry a 95% confidence interval.

| | SciFact | NFCorpus |
|---|---|---|
| v1's pipeline rebuilt in v2 | 0.644 | 0.235 |
| + one query, no re-sort (removing what v1 did wrong) | 0.674 | 0.296 |
| + the embedder's task prefixes, section-aware chunks, contextual header | 0.699 | 0.338 |
| + BM25 fused by rank | 0.738 | 0.349 |
| + cross-encoder rerank (the full system) | 0.712 | 0.350 |

The full system against the v1 replica: **SciFact +0.068 (95% CI +0.035 to +0.104), NFCorpus +0.115 (+0.097 to
+0.133)**, both Holm-adjusted p = 0.0003. **Most of that gain is removing something v1 did wrong** (the single
step "one query instead of five, no re-sort" is +0.030 and +0.061). **The reranker did not improve nDCG@10 on
either set** (SciFact −0.025, CI −0.046 to −0.004) and costs most of the latency budget; whether to keep it is
an open decision. See [`docs/evolution.md`](docs/evolution.md).

### Speed (`gemma4:12b`, 100% on the GPU, official run)

| | target | measured |
|---|---|---|
| question to first token, p50 / p95 | 2.5 s / 4 s | 0.68 to 0.76 s / 0.88 to 1.67 s |
| a follow-up (rewrite, search, first token), p50 | 5 s | 2.05 s |
| model loads during questions | 0 | 0 (all 16 settings) |
| **retrieval plus rerank, p95** | **400 ms** | **389 to 435 ms: met at 5 of 16 points** |
| CSV question, p50 / p95 (`qwen2.5-coder:7b`) | 3 s / 10 s | 2.8 s / 6.4 s |
| swap between the two models | 6 s | median 2.7 s one way, 5.0 s the other |

`gemma4:12b` leaves 934 MiB of the 10 GB free. The one setting that moved speed was the KV-cache type (q8_0
gave a first token of 0.33 to 0.39 s against 0.57 to 0.92 s, with no quality difference visible on 8 easy
questions); nothing was changed and the choice is the author's.

### Answers (golden v0: 15 documents, 100 questions, **written by the system's author, provisional**)

On `gemma4:12b`: **19 of 20 unanswerable questions were refused; 6 of 80 answerable ones were refused wrongly.**
Every answer to an answerable question cited a document the golden item names (74 of 74), and none of 75 carried
a citation the check had to remove. Must-include recall is 0.65 overall (0.22 on multi-hop questions, which is
mostly a strict phrase check being fooled by right answers worded differently). A judge model of another family
graded the answers, but **its agreement with human labels has not been measured, so it supports no claim**.

### CSV mode

- **A check on the copy.** The copied package (from the author's other project) solved 33 of 40 benchmark
  questions through Geekbot's adapter (the source project's own README says 35), refused all 10 unanswerable
  ones, and invented an answer in 0 of 30 attempts.
- **The follow-up rewrite, measured** (15 two-turn items, 3 runs, `qwen2.5-coder:7b`): the conversation solved
  12 of 15 and the hand-written standalone questions solved 11 of 15; **no failure was caused by the rewrite**.
  The three that failed were an answer with one column too many, labels that differ from the expected ones (two
  ambiguities in my items), and one wrong query. 12 against 11 is one unstable item, not a difference.

### Security: guarantees and measurements, kept apart

**Guarantees**, each with a test that can fail ([`docs/security.md`](docs/security.md)): no image and no link to
an address the sources do not name reaches the user (the output filter); a web query built from text the user did
not write is never sent without their approval of the exact text (the egress gate; the web tool is off by
default); a citation mark naming no source is removed; CSV mode refuses writes, `ATTACH` and `PRAGMA`; the
server listens on `127.0.0.1` only behind a key.

**Measurement** (43 planted attacks: poisoned documents and web pages in four vectors, and CSV cells; 3 runs
each; `gemma4:12b` and `qwen2.5-coder:7b`; the poison always in the prompt; a simulated user who denies every
approval): **0 of 129 attempts bypassed a guaranteed control**, and **the model obeyed the planted text in 15 of
129 attempts (5 of 43 attacks)**. The three runs of every attack agreed, so the effective number is 43, not 129.
The evaluation's first sitting **found three holes in the output filter** (a raw HTML anchor, `[x][y](url)`, a
removed link leaving a live definition); they are fixed, and that no hole is left is **not** shown. Answer
manipulation by a poisoned document is not guaranteed against; it is the second number.

### What these results do not show

Answer quality on questions the author did not write; whether the judge model is any good; quality differences
between speed settings (too few, easy questions); retrieval on corpora much larger than 65 chunks; anything on
another machine; attacks I did not think of; a speed comparison with v1 (it was run once, took a median of 400 s
per question with no context size set, and **no ratio is claimed**).

## Regenerate a number

| result | command (in `conda activate geekbot`, `set GEEKBOT_PROFILE=target`) | files |
|---|---|---|
| retrieval ladder | `geekbot eval fetch`, then `geekbot eval retrieval --config evals/configs/ablation.toml`, `geekbot eval report` | `evals/results/s26-run1/` |
| speed, VRAM, residency | `geekbot bench --name my-run` (GPU to yourself, about 30 minutes) | `evals/results/s51-run2/` |
| abstention and answers | `geekbot eval abstention`, `geekbot eval answers --name my-run` | `evals/results/s52-run1/` |
| CSV mode on the benchmark | `geekbot csv repro` (needs the source project's datasets) | `evals/csv/results/` |
| CSV follow-ups | `geekbot eval csv-followups --name my-run` | `evals/results/c6-run1/` |
| attacks | `geekbot eval injection --name my-run` | `evals/results/s54-run2/` |

## How it is built

```
question ─► begin_turn ─► (follow-up? condense) ─► retrieve:  dense ‖ lexical ─► fuse (RRF) ─► rerank ─┐
                                                                                                         ▼
                        (web asked for?) ─► egress_gate ─► web ──────────────────────────────────► generate ─► cite check ─► sanitize ─► remember
 CSV conversation: ─► (follow-up? condense) ─► csv_ask ─► sanitize ─► remember      (a turn can stop at the gate's question)
```

A fixed LangGraph state machine: no agent loop, no tool choice, every edge written down. SQLite is the source of
truth and one process owns it; the UI is an HTTP client. The decisions and their evidence are in
[`docs/adr/`](docs/adr/):

1. [an explicit graph, no agent loop](docs/adr/0001-explicit-graph-no-agent-loop.md)
2. [SQLite as the source of truth, one owner](docs/adr/0002-sqlite-source-of-truth-one-owner.md)
3. [local models of 12B or fewer, kept on the GPU](docs/adr/0003-local-models-gpu-residency.md)
4. [retrieved content is untrusted data](docs/adr/0004-untrusted-content-security-model.md)
5. [evidence in a fixed order, claims that say what they are](docs/adr/0005-evaluation-hierarchy-and-claims.md)
6. [CSV mode: a copied package behind one adapter](docs/adr/0006-csv-mode-copied-package-behind-an-adapter.md)

## Develop

```
ruff check .
ruff format --check .
mypy src
lint-imports
pytest
```

`lint-imports` enforces that the UI never imports the storage (the server is the one process that owns it) and
that only `geekbot.csv_mode` imports the copied CSV package. Tests that need Ollama or the GPU are marked `live`
and run only with `pytest -m live`.

## Where things are

- `src/geekbot/`: the engine (`ingest`, `index`, `retrieval`, `agent`, `tools`, `server`, `ui`, `llm`, `evals`)
- `src/csv_analyst/`: the CSV analyst package copied in from the author's other project (`VENDORED.md`)
- `evals/`: the test corpus, the golden questions, the attack files and every evaluation result
- `docs/`: [performance](docs/performance.md), [security](docs/security.md), [from v1 to v2](docs/evolution.md), the ADRs
- `project/PLAN.md`: the plan; `project/STATUS.md`: where the build stands; `project/DECISIONS.md`: why
