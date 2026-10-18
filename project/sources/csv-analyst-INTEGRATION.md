<!--
Source: C:\Work\Gigalogy\INTEGRATION.md, Gigalogy commit f482455, copied verbatim on 2026-10-11 (step P1).
Why it's here: every Geekbot chat, and any reviewer reading GitHub, sees the same guide the plan builds on (PLAN.md §6.7).
Geekbot's own choices differ from the guide in places: copy-in route (D-036), the CSV model and swaps (D-037),
hosted providers and the 27B tier removed (D-036, D-041). Paths below are relative to the Gigalogy repo.
-->

# Integrating `csv_analyst` into a chatbot

Guide for adding this project to a host chatbot as a built-in **"CSV analyst" mode**: at the start
of a conversation the user picks either normal chat or CSV analysis; in CSV mode every message goes
through this package instead of the normal chat flow.

## What the package does

`csv_analyst` answers a natural-language question about one CSV. The LLM writes one read-only SQLite
query; the query runs on an in-memory copy of the CSV; the answer is written from the real result
and its numbers are checked against it. The LLM never states a number on its own. Questions the
data cannot answer are refused with a reason instead of guessed.

## What to take

| Path | Needed | What it is |
|---|---|---|
| `csv_analyst/` | **yes** | The whole analyst: loading, profiling, prompts (`llm/prompts/*.txt`), SQL engine, checks, answer writing, `session.py` (entry point). Self-contained: reads no file outside itself. |
| `pyproject.toml` | yes, if installed as a package | Dependencies and package data (prompt files). |
| `tests/test_session.py` | recommended | Entry-point tests; no model needed. The other tests cover the internals. |
| `csv_analyst/interface/` | no | This project's own FastAPI backend, Streamlit UI and CLI. Not used by the session; safe to drop or keep. |
| `evaluation/`, `data/`, `scripts/`, `PLAN*.md`, `untracked/` | no | Benchmark, datasets, experiment history. Keep them in this repository for re-measuring. |

Two ways in, both keep the package name `csv_analyst` (chosen so it cannot clash with a host
package such as `app`):
1. **Install** (keeps the projects separate): `pip install -e /path/to/this/repo` (add `[api]`, `[cli]`
   extras only if the host wants those interfaces).
2. **Vendor**: copy the `csv_analyst/` folder into the host repository at the import root, and add
   the dependencies below to the host's requirements.

## Requirements

- Python ≥ 3.11.
- Runtime dependencies: `pandas>=2.2`, `numpy>=1.26`, `pydantic>=2.6`, `pydantic-settings>=2.2`,
  `python-dotenv>=1.0`, `requests>=2.31`, `charset-normalizer>=3.3`, `pyarrow>=15.0`, `sqlglot>=30.0`.
- A model: by default Ollama at `http://localhost:11434` with `qwen2.5-coder:7b` pulled
  (`ollama pull qwen2.5-coder:7b`, 4.7 GB; all measurements are for this model). A hosted model works
  through `provider="anthropic"` or `provider="openai_compatible"` with an API key, but has not been
  benchmarked here.

## The API

```python
from csv_analyst import CsvAnalystSession, IngestionError, Status, to_markdown

session = CsvAnalystSession.from_bytes(data: bytes, filename="sales.csv")   # or .from_path(path)
session.describe()            # -> str, chat message confirming rows / columns / load warnings
record = session.ask(text)    # -> RunRecord (pydantic model)
to_markdown(record)           # -> str, chat reply: answer, confidence, SQL, result table
```

- `CsvAnalystSession(..., model=None, provider="ollama", api_key=None, api_base_url=None, llm=None)`:
  `model` defaults to `CSV_ANALYST_LLM_MODEL_MAIN`; `llm` accepts any object with
  `generate_json(prompt, *, system=None, temperature=0.2, seed=None, schema=None) -> str` (tests use
  a fake). API keys are only used for requests, never stored on a record.
- `IngestionError` (`.code`, `.message`) is raised by `from_bytes` / `from_path` for files that are
  not usable CSVs (empty, binary, too large — default limit 200 MB, unparseable). Show `.message`.
- `ask` does not raise for hard questions. Check `record.status`:

| `record.status` | Meaning | Show |
|---|---|---|
| `Status.ANSWERED` | Answer computed from the data | `record.answer` (+ evidence; `to_markdown` does both) |
| `Status.ABSTAINED` | The data cannot answer it (missing column, undefined term, outside knowledge) | `record.reason` |
| `Status.NEEDS_CLARIFICATION` | The question is ambiguous | `record.reason`, `record.clarifying_question` |
| `Status.FAILED` | No reliable answer after retries, the time limit, or the model backend is down | `record.reason` (says which) |

  Useful fields on an answered record: `confidence` (`high` = 2+ independent samples agreed),
  `evidence.code` (the SQL), `result.columns` / `result.preview` (first rows), `result.shape`,
  `reliability.warnings`, `meta.elapsed_ms`. `record.model_dump_json()` stores the full trace.

## Wiring the mode switch

```python
import asyncio
from csv_analyst import CsvAnalystSession, IngestionError, to_markdown

# conversation state, chosen at the start: state.mode in {"chat", "csv"}; state.analyst = None

async def on_csv_upload(state, data: bytes, filename: str) -> str:
    try:
        state.analyst = await asyncio.to_thread(CsvAnalystSession.from_bytes, data, filename)
    except IngestionError as exc:
        return f"I couldn't read that file: {exc.message}"
    return state.analyst.describe() + "\n\nAsk me anything about this data."

async def on_message(state, text: str) -> str:
    if state.mode != "csv":
        return await normal_chat(state, text)          # the host's existing flow
    if state.analyst is None:
        return "Upload a CSV file to start the analysis."
    record = await asyncio.to_thread(state.analyst.ask, text)   # blocking call, seconds
    return to_markdown(record)
```

Notes for the host:
- **One session per conversation.** The session holds the CSV in memory (a DataFrame); drop it when
  the conversation ends. Separate sessions can run on separate threads at the same time (Ollama
  answers one request at a time by default, so concurrent questions queue there).
- **Blocking and slow-ish:** about 3 s per question with the local 7b on an RTX 3080 (1–15 s seen),
  up to the time limit (below). Run it off the event loop, as above.
- **No conversation memory:** each question is answered on its own. Follow-ups like "and by month?"
  need the host to turn them into a full question first (for example with its normal chat model);
  that step is not part of this package and has not been measured.
- **Replacing the CSV:** create a new session.

## Settings

All settings are environment variables (or a `.env` file in the working directory) with the
`CSV_ANALYST_` prefix, so they do not collide with the host's own; `.env.example` lists every one.
The ones a host is likely to touch:

| Variable | Default | Meaning |
|---|---|---|
| `CSV_ANALYST_OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `CSV_ANALYST_LLM_MODEL_MAIN` | `qwen2.5-coder:7b` | Default model |
| `CSV_ANALYST_LLM_MODEL_FAST` / `_HEAVY` | `qwen2.5-coder:3b` / `qwen3.8:27b` | Optional smaller / larger tiers |
| `CSV_ANALYST_QUESTION_TIME_LIMIT_S_MAIN` / `_HEAVY` / `_FAST` | `600` / `900` / `300` | Per-question time limit by tier; other models get the main one |
| `CSV_ANALYST_LLM_TIMEOUT_S` | `60` | One model call |
| `CSV_ANALYST_MAX_CSV_MB` | `200` | Upload size limit |

In code, `csv_analyst.settings.<name> = value` changes a setting for the whole process. The prompt
and pipeline settings (`SQL_*`, `SCHEMA_FACTS`, `PLANNER_SUFFIX`, `ANSWER_PROMPT`, ...) were each
chosen by measurement; changing them means re-running the benchmark in this repository
(`evaluation/`, README "Does it actually work?").

## Checks after merging

```bash
pytest -q tests/test_session.py
```

```bash
python -c "from csv_analyst import CsvAnalystSession, to_markdown; s = CsvAnalystSession.from_bytes(b'region,revenue\nNorth,100\nSouth,200\n', 'r.csv'); print(to_markdown(s.ask('What is the total revenue by region?')))"
```

The second needs Ollama running with the model pulled; it prints an answer, the SQL and a two-row
table. The answer line may be a plain list of the result ("Result (region, total_revenue): North,
100.0; South, 200.0."): when the model's sentence contains a number that is not in the result
(here it tends to add the grand total, 300), the number check rejects the sentence twice and the
reply falls back to the rows themselves. That is the check working, not an error.

## Limits to keep in mind

Measured on two benchmark datasets plus a dev set (README): 35 of 40 questions solved, 0 invented
answers in 30 attempts on unanswerable ones there, but 6 in 27 on the dev set. Known failure types:
ties shown as one winner, invented meanings for concepts the data lacks ("streams", "profit"),
multi-step "top X per group" questions. The SQL safety checks are prototype-grade (read-only
SELECT, SQLite authorizer, timeout), not a security audit. A fine-tuned model is planned; adopting
it would only change `CSV_ANALYST_LLM_MODEL_MAIN`.
