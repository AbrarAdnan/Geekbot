"""Evaluation code (PLAN.md §6.6). Today only the CSV reproduction run (step C3); the retrieval
and answer evaluations arrive in Phase 2 and Phase 5.

The defaults live here, in a module that imports nothing heavy, so the CLI can use them as
option defaults without importing pandas (``geekbot --version`` should stay fast).
"""

from pathlib import Path

SUITE_NAMES = ("hospital", "sales")
DEFAULT_SOURCE = Path("scratch/gigalogy-latest")
DEFAULT_RUNS = 3
# The package's own per-question limit for the main model. The reproduction uses it, not
# Geekbot's 60 s, so the cap does not muddy "does the copy match the original?"; the report then
# counts how many questions took longer than Geekbot's cap.
DEFAULT_TIME_LIMIT_S = 600
