# Evaluation data

This folder holds the data Geekbot is tested on. The code that runs the evaluations lives in `src/geekbot/evals/` and is built from step S2.4 on.

| Path | What it is | Filled in |
|---|---|---|
| `corpus/` | the ~15 test documents (the "golden corpus") | by you, before step S1.6, which ingests this folder |
| `datasets/golden-questions.md` | the hand-written test questions, as a fill-in template. **The source of truth** | by you, over several sittings |
| `datasets/golden.jsonl` | the same questions as data, generated from the template | by `geekbot eval import`, never by hand |
| `datasets/golden-draft.md` | questions drafted by the chat model to copy from. Scratch, git-ignored, **not data** | by `geekbot eval generate` |
| `csv/` | results of the CSV analyst mode's evaluation runs (see its own `README.md`) | by the runs themselves, from step C3 |

The public benchmark datasets (BEIR SciFact and NFCorpus) are downloaded at runtime by `geekbot eval fetch` into `data/beir/` and are **not** stored here (PLAN.md §9).

`corpus/` and the questions are still yours to fill in (DECISIONS.md D-032).

## Building the golden set (step S2.4)
In the `geekbot` conda env, once `corpus/` has its documents (see below):

1. `geekbot ingest evals/corpus` indexes the documents. Use this exact folder spelling from the repository root every time: document IDs are hashes of file paths (D-054), so a different spelling is a different set of documents.
2. `geekbot eval generate` (loads the chat model on the GPU) writes `datasets/golden-draft.md`: drafted single-hop and unanswerable questions, one per stored passage, spread over the documents. Read them, copy the good ones into `golden-questions.md` under real IDs and reword each so it does not echo the document. Multi-hop and multi-turn questions are not drafted: write those yourself.
3. `geekbot eval import` reads `golden-questions.md`, checks it and writes `golden.jsonl`. It writes nothing while there are errors, and prints each one with its item ID.
4. `geekbot eval validate --check-index` re-checks the JSONL and confirms every source is a ready document in the index (a wrong folder spelling would otherwise show up later as a retriever that finds nothing).

The questions are split into development (40%) and test (60%) items by a fixed rule from each ID (DECISIONS.md D-065), so finish the whole set before anything is tuned and do not renumber afterwards.

## The golden corpus (`corpus/`)
The folder doesn't exist yet: create it when you add the first file. **Only documents go inside it.** Geekbot ingests everything in that folder, which is why these notes live here, one level up.

**Rules for choosing documents**
- **Publishable.** This repo is public, so every file must be your own or under a license that allows sharing (public domain, CC-BY, MIT, …). Record the license in the table below.
- **Mixed formats**, so every loader gets tested (step S1.1): PDF, Word (`.docx`), PowerPoint (`.pptx`), HTML, Markdown, plain text, code (`.py`), a Jupyter notebook (`.ipynb`), and spreadsheets (`.csv` / `.xlsx`).
- **Spreadsheets:** at least one `.csv` or `.xlsx`. Chat mode indexes only a schema card per table (columns, types, example values), never its rows (D-051, which replaced D-016), so questions about a spreadsheet in the golden set should be about *what it contains*, not about figures read from it.
- **Overlapping topics**, so some questions need facts from two documents (multi-hop questions).
- **Small files** (a few MB each), and **no secrets or personal data**.

**Document list** (fill in as you add files)

| # | File | Format | Topic | Source / license | Why it's included |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |
| 9 | | | | | |
| 10 | | | | | |
| 11 | | | | | |
| 12 | | | | | |
| 13 | | | | | |
| 14 | | | | | |
| 15 | | | | | |
