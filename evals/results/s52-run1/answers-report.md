# Answer evaluation: s52-run1

Chat model `gemma4:12b` (profile `target`), retrieval hybrid with the cross-encoder, top-k 5; golden v0 (Claude-authored, provisional: easy mode, never a headline); 2026-10-24T13:06:43+06:00.

## Deterministic checks (PLAN.md §9: the primary evidence)
Counts are shown next to every share. A citation hit says the answer cited a document the golden item names; it does not say the claim is in it.

| group | items | answerable answered | answerable refused wrongly | unanswerable refused | unanswerable answered | must-include recall | citation hit | citation recall | answers with an invalid citation |
|---|---|---|---|---|---|---|---|---|---|
| all | 100 | 74 of 80 | 6 of 80 | 19 of 20 | 1 of 20 | 0.65 (n=80) | 74 of 74 | 0.95 | 0 of 75 |
| single_hop | 50 | 46 of 50 | 4 of 50 | none | none | 0.70 (n=50) | 46 of 46 | 1.00 | 0 of 46 |
| multi_hop | 15 | 13 of 15 | 2 of 15 | none | none | 0.22 (n=15) | 13 of 13 | 0.85 | 0 of 13 |
| multi_turn | 15 | 15 of 15 | 0 of 15 | none | none | 0.93 (n=15) | 15 of 15 | 0.87 | 0 of 15 |
| unanswerable | 20 | none | none | 19 of 20 | 1 of 20 | n/a (n=0) | none | n/a | 0 of 1 |
| dev | 40 | 30 of 32 | 2 of 32 | 7 of 8 | 1 of 8 | 0.70 (n=32) | 30 of 30 | 0.92 | 0 of 31 |
| test | 60 | 44 of 48 | 4 of 48 | 12 of 12 | 0 of 12 | 0.62 (n=48) | 44 of 44 | 0.97 | 0 of 44 |

## The judge (supporting evidence only)
Judge `qwen2.5-coder:7b` (family `qwen`, the generator's is `gemma`); 75 answers judged, 25 refusals not judged. **Supported** (every claim is stated by the sources): 69 of 75; **answers the question**: 67 of 75.
- single_hop: supported 45 of 46, answers the question 46 of 46
- multi_hop: supported 12 of 13, answers the question 13 of 13
- multi_turn: supported 11 of 15, answers the question 7 of 15
- unanswerable: supported 1 of 1, answers the question 1 of 1

## Agreement with the human labels (PLAN.md §9)
no verdict yet: 0 labels, and at least 30 are needed (the plan asks for 60)

These are the numbers of one system on a small, easy, provisional set; a deterministic check is a floor, the judge is a model's opinion, and neither says an answer is true.
