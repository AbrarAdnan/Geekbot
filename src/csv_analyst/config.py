"""Env-driven settings. No hardcoded model names or limits — see .env.example.

Every setting reads the environment variable `CSV_ANALYST_<NAME>` (e.g. `CSV_ANALYST_LLM_MODEL_MAIN`),
also from a `.env` file in the working directory; the prefix keeps them apart from a host app's own
settings when this package runs inside another application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CSV_ANALYST_", env_file=".env", extra="ignore")

    # LLM backend
    llm_provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    llm_model_main: str = "qwen2.5-coder:7b"
    llm_model_fast: str = "qwen2.5-coder:3b"
    # GEEKBOT EDIT (C1): the "heavy" 27B tier (llm_model_heavy) is removed; Geekbot allows no
    # model over 12B (CLAUDE.md rule 4).
    llm_temperature: float = 0.2
    llm_timeout_s: int = 60
    # 16384 since 2026-10-07: an 8192 window silently cut the planner prompt of
    # a 31-column CSV (Chinook dev set, 8,309 tokens) to its last ~4k tokens.
    # Prompts that fit in 8192 are unaffected. The provider also refuses a
    # prompt that would not fit (PromptTooLongError) instead of sending it.
    llm_num_ctx: int = 16384
    # Per-question time limit by model tier (PLAN_v2.md D5, user decision
    # 2026-10-11): main 10 min, heavy 15 min, fast 5 min; any other model
    # (e.g. a hosted API model) gets the main limit. Checked before each new
    # attempt / sample batch -- a model call already running finishes first.
    question_time_limit_s_main: int = 600
    # GEEKBOT EDIT (C1): question_time_limit_s_heavy removed with the heavy tier.
    question_time_limit_s_fast: int = 300

    # Reliability controls
    consensus_samples: int = 3
    max_retries: int = 2
    # Adaptive consensus (PLAN_v2.md, default since 2026-09-30): generate
    # samples 1-2, and the 3rd only if they disagree. Measured on all 40
    # questions: -26% time, -28% planner calls, no accuracy change.
    # False = always generate all `consensus_samples`.
    consensus_adaptive: bool = True

    # Query engine: "sql" (one generated SQLite SELECT over table `data`) or
    # "pandas" (generated pandas, subprocess sandbox). Default "sql" since
    # 2026-10-07 (user decision, PLAN_v2.md D10); "pandas" is deprecated and
    # kept only as an option / comparison arm.
    query_engine: str = "sql"
    # SQL engine schema layout: "plain" (default since 2026-10-07, card H-C C2:
    # no role tags, nothing the fact bank already says), "profile" (bulleted
    # profiler facts, before C2) or "ddl" (CREATE TABLE with the profile facts
    # as column comments). csv_analyst/sql_engine/prompt.py.
    sql_schema_style: str = "plain"
    # SQL v3 fixes (PLAN_v2.md), on with the SQL default (D10): store integer
    # measure columns as REAL so `/` divides like pandas, and drop NULL
    # GROUP BY keys like pandas groupby(dropna=True).
    sql_real_measures: bool = True
    sql_drop_null_groups: bool = True
    # SQL planner prompt file (csv_analyst/llm/prompts/<name>.txt). "planner-sql-v4-tables"
    # (default since 2026-10-08, card H-B1 D28) = v4 naming the entity tables.
    # v4 (card H-C C2) = v3 with grain and NULL rows explained once. v3 (card C1) moved the worked examples to an invented
    # library-loans table instead of ones that mirrored protected benchmark
    # questions. v2 and v3 kept on disk.
    sql_planner_prompt: str = "planner-sql-v4-tables"
    # Card H-G: add worked examples generated from the loaded CSV's own schema
    # (csv_analyst/sql_engine/examples.py) after the static ones. Default on since
    # 2026-10-07 (user decision; +1.67 correct per run vs v3, below the +2
    # keep rule -- re-test on the dev set).
    sql_generated_examples: bool = True

    # Card H-P (PLAN_v2.md): one sentence appended as the planner prompt's
    # last line. "emphasis" ("Make no mistakes.") is the default since
    # 2026-10-07 (user decision; measured +0.67 q/seed set vs placebo, below
    # the +2 keep rule -- re-test on the dev set). "off" | "placebo" for arms.
    planner_suffix: str = "emphasis"

    # Card H-R (PLAN_v2.md): ingestion-time fact bank added under the schema
    # block. "all" is the default since 2026-10-07 (user decision D12; measured
    # +2.67 correct per seed set on the protected 40, re-test on dev).
    # "all" | "off".
    schema_facts: str = "all"
    # Answer prompt file (csv_analyst/llm/prompts/<name>.txt). "answer-v4" (engine-
    # neutral wording, H-C C0) is the default since 2026-10-07; v3 kept on disk.
    answer_prompt: str = "answer-v4"
    # Numeric-fidelity check version (csv_analyst/validation/fidelity.py). "v2" (card
    # H-F F0) reads hyphens in ids / dates as separators, accepts numbers
    # written inside text cells and rounded percentages; default since
    # 2026-10-07 (D24: template fallbacks 10 -> 0 of 120). "v1" | "v2".
    fidelity_check: str = "v2"
    # Card H-B1 (PLAN_v2.md): one extra SQLite table per repeating entity id
    # (csv_analyst/sql_engine/entities.py), listed in the planner prompt; used with
    # sql_planner_prompt="planner-sql-v4-tables". Default since 2026-10-08
    # (user decision D28, kept against the rule: dev +11.67 correct per run,
    # protected +0.67, one extra invented dev answer).
    sql_entity_tables: bool = True
    # H-B1 fix F1: constant columns in entity tables. "off" leaves them all
    # out; "prefix" keeps one in the table of the entity its name starts with.
    # Default "prefix" since 2026-10-08 (user decision D29: dev +2.33 correct
    # per run; hospital and sales tables unchanged).
    sql_entity_constants: str = "prefix"

    # Execution sandbox
    sandbox_timeout_s: int = 15

    # Ingestion limits
    max_csv_mb: int = 200

    def question_time_limit_s(self, model: str) -> int:
        if model == self.llm_model_fast:
            return self.question_time_limit_s_fast
        return self.question_time_limit_s_main


settings = Settings()
