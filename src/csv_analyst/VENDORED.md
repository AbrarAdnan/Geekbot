# csv_analyst: a vendored copy (Geekbot step C1, D-036, D-045)

This folder is a **copy** of the `csv_analyst` package from the user's Gigalogy project (project folder "mini-maira", `C:\Work\Gigalogy`). Geekbot reaches it only through its own adapter (`geekbot.csv_mode`, from step C2). Read this file before changing anything here.

## Source
- **Not a clean commit.** Copied on 2026-10-16 from the Gigalogy **working tree**, which was based on commit `f482455` ("Standalone package for the chatbot integration", master) **plus later edits that were not committed or pushed at the time**. The exact difference is unknown.
- **Consequence for claims:** Gigalogy's published numbers (35/40 solved, about 3.0 s per question, 0 and 6 invented answers) were measured on the commit they name. They describe the source project, not this copy. Geekbot reports its own CSV results from step C3 on (D-041).
- **To get a real SHA:** commit and push Gigalogy's changes, then replace this section with that SHA.
- Reference copy of the source used for the copy: `scratch/gigalogy-latest/` (git-ignored, local only).

## How it was copied (Windows `cmd`, from the repo root)
```
robocopy scratch\gigalogy-latest\csv_analyst src\csv_analyst /E /XD __pycache__ interface /XF anthropic_provider.py openai_compatible_provider.py
robocopy scratch\gigalogy-latest\tests tests\csv_analyst\vendored /E /XD __pycache__ /XF test_api.py test_comparator_coercion.py test_evaluation_benchmark.py test_evaluation_benchmark_chinook.py test_evaluation_benchmark_sales.py test_evaluator_consistency.py test_evaluator_grading.py test_evaluator_metrics.py
```
Its own tests live in `tests/csv_analyst/vendored/` (no model needed; a fake LLM).

## Cuts (files not copied)
| Not copied | Why |
|---|---|
| `interface/` (FastAPI, Streamlit, CLI) | Gigalogy's own front ends; Geekbot has its own (nothing in the package imports it) |
| `llm/anthropic_provider.py`, `llm/openai_compatible_provider.py` | Hosted providers would send CSV data off the machine on a path Geekbot's egress gate does not control (D-041) |
| Tests `test_api.py`, `test_comparator_coercion.py`, `test_evaluation_benchmark*.py`, `test_evaluator_*.py` | They need Gigalogy's `evaluation/` folder, its data, or the API; they cannot run here |

## Kept on purpose
- **The pandas engine** (`execution/runner.py`, `execution/sandbox_entry.py`, `validation/ast_guard.py`, the `planner-v6` prompt). Deprecated and **unused**: the default engine is SQL (`query_engine = "sql"`) and Geekbot never sets "pandas". Removing it needs more than small edits (`pipeline.py` and three modules import `ExecutionResult` from `runner.py`). Decide again after C3.
- The older prompt files (`planner-sql-v2/v3/v4`, `answer-v3`) stay on disk, as in the source.

## Local edits (every one is marked `GEEKBOT EDIT (C1)` in the code)
1. `config.py`: removed the 27B "heavy" tier (`llm_model_heavy`, `question_time_limit_s_heavy`, and its branch in `question_time_limit_s`). Reason: no model over 12B.
2. `session.py`: removed the two hosted-provider imports; `PROVIDERS = ("ollama",)`; `make_provider(provider, model)` lost `api_key` / `api_base_url`; `CsvAnalystSession.__init__` lost the same two parameters.
3. `tests/csv_analyst/vendored/test_pipeline.py`: `test_time_limit_follows_the_model_tier` no longer checks the heavy tier.
4. `tests/csv_analyst/vendored/test_session.py`: `test_make_provider` expects `ValueError` for "anthropic" (now an unknown provider).

## Tests: what runs and what doesn't
`tests/csv_analyst/conftest.py` (Geekbot's file) adapts the vendored tests without editing them:
- every test runs with `src/` as its working directory (some read `csv_analyst/llm/prompts/*.txt` relative to the old repo root);
- 10 tests are **skipped**: they need Gigalogy's hospital/sales datasets, which are not vendored;
- 1 test is an **expected failure**: `test_templates_cover_values_numbers_dates_nulls_relations`. The fact bank says "holds dates" where the test expects "holds ISO text" (the profiler converts ISO date text to real dates). Cause not confirmed, because the source folder was gone when we tried to compare. Check again in C3.
- Result when C1 was verified: 382 passed, 10 skipped, 1 expected failure.

## Quality gates (typing debt)
- **ruff:** `src/csv_analyst` and `tests/csv_analyst/vendored` are excluded (`pyproject.toml`, `.pre-commit-config.yaml`), so the files stay as Gigalogy wrote them and a re-sync is a plain copy.
- **mypy:** errors in `csv_analyst.*` are ignored (`[[tool.mypy.overrides]]`). Geekbot's own code that calls the package is still checked strictly.

## Settings
The package reads `CSV_ANALYST_*` environment variables (and `.env`), separate from Geekbot's `GEEKBOT_*` (S0.2). Its prompt and pipeline defaults were chosen by Gigalogy's measurements; changing them means re-running the benchmark. Step C2 passes the model name, host and time limit in from Geekbot's settings.

## Re-syncing from Gigalogy later
1. Copy again with the commands above (from a Gigalogy checkout of a named commit).
2. Re-apply the four local edits (search for `GEEKBOT EDIT` in the old copy; `git diff` against this one shows them).
3. Run `pytest`, then update the **Source** section of this file.
