# Geekbot v2

A local RAG engine: ask questions about your own documents and get answers that cite where they came
from. Everything runs on your machine. Geekbot v2 is a rebuild of the first version (v1, kept at the git
tag `v1.0`) with three aims: **measure retrieval quality**, **control inference cost**, and **treat
retrieved content as untrusted data**.

> **Status: a prototype under construction.** The results section is not written yet: no number is
> quoted here until it has been measured and can be regenerated with one command. The plan, the
> decisions and the step-by-step log are in [`project/`](project/).

## What it does

- **Chat mode.** Index PDFs, Word and PowerPoint files, Markdown, text, code and notebooks (CSV and
  Excel files are indexed as a description of their columns, never their rows); ask questions; get
  streamed answers with `[1]`-style citations, follow-up questions that resolve against the
  conversation, and a model that is asked to say so when the documents do not hold the answer (how
  often it does is measured in `project/`, not claimed here).
- **CSV mode.** Ask questions about one CSV file (`geekbot csv ask`); the answer shows the SQL and the
  result table. It is not in the browser UI yet.
- **Local only.** Models run in [Ollama](https://ollama.com), the index is SQLite (the source of truth)
  plus Chroma (a derived index), and the server listens on `127.0.0.1` only. The browser UI and the
  server share a local password (an API key kept in `data/api-key`); nothing is sent to any cloud
  service.

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

`geekbot up` starts the API server and the browser UI (it opens `http://127.0.0.1:8501`). Add
documents on the **Knowledge** page (upload files, or give a folder listed in `GEEKBOT_ALLOWED_ROOTS`
in your `.env`), then ask on the **Chat** page. `.env.example` lists every setting.

`gemma3:4b` is the light model used while building (the `proto` profile). The target model is
`gemma4:12b` (`GEEKBOT_PROFILE=target`); every reported number will name the model that produced it.

## Develop

```
ruff check .
ruff format --check .
mypy src
lint-imports
pytest
```

`lint-imports` enforces that the UI never imports the storage: the server is the one process that owns
it. Tests that need Ollama or the GPU are marked `live` and run only with `pytest -m live`.

## Where things are

- `src/geekbot/`: the engine (`ingest`, `index`, `retrieval`, `agent`, `server`, `ui`, `llm`)
- `src/csv_analyst/`: the CSV analyst package copied in from the author's other project
- `evals/`: the test corpus, the golden questions and the evaluation results
- `project/PLAN.md`: the plan; `project/STATUS.md`: where the build stands; `project/DECISIONS.md`: why
