# Evaluation data

This folder holds the data Geekbot is tested on. The code that runs the evaluations lives in `src/geekbot/evals/` and is built from step S2.4 on.

| Path | What it is | Filled in |
|---|---|---|
| `corpus/` | the ~15 test documents (the "golden corpus") | by you, before step S1.6, which ingests this folder |
| `datasets/golden-questions.md` | the hand-written test questions, as a fill-in template | by you, by step S2.4, which turns it into `golden.jsonl` |

The public benchmark datasets (BEIR SciFact and NFCorpus) are downloaded at runtime and are **not** stored here (PLAN.md §9).

These two are blank templates for now (DECISIONS.md D-032).

## The golden corpus (`corpus/`)
The folder doesn't exist yet: create it when you add the first file. **Only documents go inside it.** Geekbot ingests everything in that folder, which is why these notes live here, one level up.

**Rules for choosing documents**
- **Publishable.** This repo is public, so every file must be your own or under a license that allows sharing (public domain, CC-BY, MIT, …). Record the license in the table below.
- **Mixed formats**, so every loader gets tested (step S1.1): PDF, Word (`.docx`), PowerPoint (`.pptx`), HTML, Markdown, plain text, code (`.py`), a Jupyter notebook (`.ipynb`), and spreadsheets (`.csv` / `.xlsx`).
- **Spreadsheets:** at least one with ≤ 2,000 rows (its rows get embedded) and, ideally, one larger one (only its schema card is indexed). See D-016.
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
