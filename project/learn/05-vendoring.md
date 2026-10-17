# 05 — Working in a codebase you didn't just write (step C1)

## 1. Three ways to reuse someone else's code
- **Install it as a package** (`pip install -e C:\Work\Gigalogy`): the projects stay separate and fixes arrive by updating. Downside: Geekbot would depend on a folder outside its repo, which breaks CI and anyone else's checkout.
- **Git subtree/submodule:** git tracks the other repo inside yours. Powerful, but it adds git machinery that is easy to get wrong.
- **Vendoring (what we did):** copy the code into your repo. It is self-contained, you can change it, and CI sees it. The cost is that you now own the copy; fixes upstream don't arrive by themselves.

## 2. Provenance
Vendored code needs a note saying where it came from. `src/csv_analyst/VENDORED.md` records the source, what we cut, every local edit and how to re-sync. Local edits are marked in the code with `GEEKBOT EDIT (C1)`, so they can be found, and re-applied, after a re-sync.

**An honest provenance note matters.** Our copy came from a working tree with edits that were never committed, so the note says "not a clean commit" and that Gigalogy's published numbers describe the commit they were measured on. Claims about the copy are only claims we measured ourselves.

## 3. Cutting what you don't need
We cut by one rule each:
- **Safety:** the hosted providers would send your data to a remote service (D-041).
- **Our own limits:** the 27B tier is over the 12B cap.
- **Can't run here:** the tests that need Gigalogy's benchmark folder.
- **Not worth the risk:** the old pandas engine is deeply wired into the pipeline. Cutting it would mean larger edits to code we didn't write. So it stays, unused. The fewer lines we change, the easier the re-sync and the lower the chance of new bugs.

## 4. Gradual typing
`mypy --strict` demands type hints everywhere. The copied package wasn't written for that. Instead of rewriting it, we tell mypy to ignore errors in `csv_analyst.*` (recorded as typing debt), and keep strict checks on Geekbot's own code, including the adapter that calls it. Same idea for ruff: the copy is excluded so it stays byte-identical to its source.

## 5. A guided tour: what happens to one question
1. **Load** (`ingestion/`): read the CSV bytes, guess the encoding, check size and shape.
2. **Profile** (`profiling/`): column types, null counts, sample values, a bank of facts about the data.
3. **Plan** (`generation/`): the model gets the schema and the question and writes one read-only SQLite `SELECT` (or says it can't be answered). Two samples; a third only if they disagree.
4. **Check** (`validation/`, `sql_engine/`): the query must be a single read-only `SELECT`; a rewrite drops NULL group keys like pandas would.
5. **Run** (`sql_engine/executor.py`): execute on an in-memory SQLite copy of the CSV.
6. **Agree** (`validation/consensus.py`): do the samples give the same result? Abstentions count as votes.
7. **Explain** (`answer/`): a second model call writes the answer from the real result.
8. **Number check** (`validation/fidelity.py`): every number in the sentence must appear in the result; if not, retry once, then fall back to a template. This is how "the model never states a number on its own" is enforced.

`pipeline.py` wires these steps together and is the only place that builds the final `RunRecord`.

## Interview-style questions
1. **When would you vendor a dependency instead of installing it?** When it lives outside your repo, you need to change it, and reproducibility for others matters more than automatic updates.
2. **What is provenance, and what goes wrong without it?** A record of the source and every local change. Without it, nobody can tell which parts are upstream's, so re-syncing becomes guesswork.
3. **Why did we keep the pandas engine?** Cutting it needs larger edits to code we didn't write; unused code that stays identical to the source is cheaper and safer than a bigger edit.
4. **Why exclude vendored code from ruff and relax mypy for it?** Rewriting it for our style would make the copy diverge from the source and complicate re-syncing, so the debt is recorded and kept in one place.
5. **How is "the model never states a number" enforced?** The answer's numbers are checked against the real query result; mismatches trigger a retry and then a deterministic template.
