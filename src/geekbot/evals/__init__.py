"""Evaluation code (PLAN.md §6.6): the CSV reproduction run (C3), the metrics and statistics
(S2.5), the golden set and the public datasets (S2.4). The ablation harness arrives in S2.6 and the
answer evaluations in Phase 5.

The defaults live here, in a module that imports nothing heavy, so the CLI can use them as
option defaults without importing pandas (``geekbot --version`` should stay fast).
"""

from pathlib import Path

# The golden set (step S2.4). The markdown template is what a person edits; the JSONL file is
# generated from it by ``geekbot eval import``; the draft file is model output to curate from.
DEFAULT_CORPUS = Path("evals/corpus")
DEFAULT_GOLDEN_MARKDOWN = Path("evals/datasets/golden-questions.md")
DEFAULT_GOLDEN_JSONL = Path("evals/datasets/golden.jsonl")
DEFAULT_GOLDEN_DRAFT = Path("evals/datasets/golden-draft.md")

SUITE_NAMES = ("hospital", "sales")
DEFAULT_SOURCE = Path("scratch/gigalogy-latest")
DEFAULT_RUNS = 3
# The package's own per-question limit for the main model. The reproduction uses it, not
# Geekbot's 60 s, so the cap does not muddy "does the copy match the original?"; the report then
# counts how many questions took longer than Geekbot's cap.
DEFAULT_TIME_LIMIT_S = 600
