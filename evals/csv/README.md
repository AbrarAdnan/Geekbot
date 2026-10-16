# CSV-mode evaluation (step C3 onward)

Results of the runs that check the CSV analyst mode (PLAN.md §9, §6.7). The code is in `src/geekbot/evals/csv_repro.py`.

| Path | What it is |
|---|---|
| `results/repro-<profile>-<model>-<time>.json` | one reproduction run: Gigalogy's 40-question benchmark, asked through Geekbot's adapter (C3) |

## Rules
- **The benchmark itself is not stored here.** Its questions, gold answers and CSV files belong to the Gigalogy project and come from third-party datasets (sql-practice.com's hospital database, the Tableau "Superstore" sample), and this repository is public. The run reads them from the Gigalogy folder with `--source` (default `scratch/gigalogy-latest/`, read-only, git-ignored).
- **A result file holds ids and numbers, not text.** It names each question by id, with its difficulty, status, correctness, time and model-load time. The questions and their gold answers can be looked up in the Gigalogy folder by id.
- **Every file names its model and says whether it is official** (target model, idle GPU, full run: D-028, D-043). A file with `"official": false` is a plumbing check and is never quoted as a result.
- **Gigalogy's own numbers** (35/40 solved and so on) are quoted only as the source project's measurements (D-041). Geekbot's numbers come from these files.
- C6 adds its own results here (the follow-up and CSV-cell-attack sets).
