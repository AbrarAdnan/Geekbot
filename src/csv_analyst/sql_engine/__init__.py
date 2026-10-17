"""SQL query engine (PLAN_v2.md B4): a self-contained alternative to the
pandas path. The model writes one SQLite SELECT over table `data`; everything
SQL-specific lives here -- loading the CSV into SQLite, safe execution, the
SQL schema rendering and prompt, and SQL retry hints -- so it can be tuned
without touching the pandas engine. The pipeline picks one engine per run;
validation results, consensus, and the answer stage are shared.
"""

from csv_analyst.sql_engine.executor import execute_sql, validate_sql
from csv_analyst.sql_engine.hints import sql_retry_hint
from csv_analyst.sql_engine.prompt import PROMPT_VERSION, build_sql_prompt, normalize_sql

__all__ = [
    "PROMPT_VERSION", "build_sql_prompt", "execute_sql", "normalize_sql", "sql_retry_hint", "validate_sql",
]
