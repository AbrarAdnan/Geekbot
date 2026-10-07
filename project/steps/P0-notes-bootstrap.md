# P0 — Project notes bootstrap

- **Phase / Gate:** pre-Phase 0 (no gate)
- **Chat date(s):** 2026-10-07 (planning chat), 2026-10-07 (verification chat)
- **Status:** green-lit (2026-10-07)
- **Commit:** `f7c2100` (notes) + the verification commit (SHA to be reported by the user)
- **Lesson note:** none (setup step)

## Goal
Move the approved plan into the repo and create the central notes directory and the per-chat bootstrap, so every later step can run in its own chat with full context (PLAN.md §2.1).

## Approved design
As specified in PLAN.md §11, "Step P0". Only files are written; nothing is executed. The verification chat added a self-audit before the Codex review (below).

## Files changed
**Planning chat** (commit `f7c2100`):
- `CLAUDE.md`: standing rules + session protocol, auto-loaded by every Claude Code chat in this repo
- `project/PLAN.md`: the approved plan, rev 7
- `project/STATUS.md`: step tracker + "Next up: S0.1" handoff
- `project/DECISIONS.md`: decision log D-001 … D-029
- `project/reviews/README.md`, `chatgpt-review-1.md`, `chatgpt-review-2.md`: both reviews verbatim + resolution tables
- `project/steps/README.md`, `project/steps/_TEMPLATE.md`, this log
- `project/learn/README.md`: index of the 34 planned lesson notes

**Verification chat:**
- `project/PLAN.md`: rev 8 (see "Verification" below), then rev 9 (one-time v1 reference run, D-034)
- `project/DECISIONS.md`: D-030 … D-034 added; older entries unchanged
- `CLAUDE.md`: rule 2 notes the separate `geekbot-v1` env (D-034)
- `project/STATUS.md`: P0 green-lit, S0.4 dropped, new handoff
- `project/learn/README.md`: lesson 04 dropped; its topic folded into lesson 20
- `project/reviews/chatgpt-review-1.md`, `chatgpt-review-2.md`: resolution tables updated (the reviews themselves are unchanged)
- `evals/README.md`, `evals/datasets/golden-questions.md`: blank templates (D-032)
- this log

v1 files are untouched.

## Commands the user ran + results
**Planning chat**, from the repo root on `main`:
```
git tag v1.0
git checkout -b v2
git add CLAUDE.md project
git commit -m "chore: add v2 plan and project notes (P0)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Result, read from the `.git` files in the verification chat (git itself was not run): tag `v1.0` → `96df82d` (the tip of `main`), branch `v2`, commit `f7c2100`. Nothing had been pushed yet.

**Verification chat:** one commit for the fixes in VS Code (Source Control → stage all → commit), then branch `v2` published and tag `v1.0` pushed to GitHub. _The commit SHA is recorded at the start of the S0.1 chat._

## Verification (self-audit before the Codex review)
Every P0 file was re-read, and every v1 finding in PLAN §5 was checked line by line against the v1 code. Line references that were already correct: #1–#6, #8–#11, #14.

| # | Finding | Resolution |
|---|---|---|
| 1 | S0.4 would run v1 inside the v2 env, but v1 pins LangChain 0.3 and v2 uses 1.x. After S4.4 removes v1, the baseline couldn't be re-run either | User: S0.4 dropped (D-031). Instead, one v1 run at CP2 in its own `geekbot-v1` env, from tag `v1.0` (D-034) |
| 2 | S0.4 used golden questions that only exist after S2.4, and no step created `evals/corpus/`, which S1.6 ingests | User: blank templates now, filled later (D-032); the v1 run waits until CP2, when the questions exist (D-034) |
| 3 | v1 ships with `qwen3:0.6b` and v2 uses `gemma4:12b`, so a v1 → v2 latency comparison mixes in the model size | User: the one v1 run uses `gemma4:12b` (D-034) |
| 4 | The abstain threshold is tuned and graded on the same golden set; BEIR tuning should use the train/dev splits | Deferred to S2.4–S2.6 (D-033) |
| 5 | Holm test family undefined; Recall@5 undefined for unanswerable items; multi-turn items have no `condense` before Phase 3 | Deferred to S2.4–S2.6 (D-033) |
| 6 | §5 #7: "a no-op re-sort" contradicted R2's own note; "~20" embedding calls is really up to 21 | Fixed (D-030) |
| 7 | §5 #9: "~3k tokens" was never measured | Marked as an estimate (D-030) |
| 8 | §5 #12: "undelimited" was wrong (plain separators exist); the real issue is that nothing marks the text as untrusted | Fixed (D-030) |
| 9 | §5 #13: an empty input tries to index the whole repo (and likely crashes first); it doesn't actually index the venv | Fixed (D-030) |
| 10 | §5 #14 / rung R1: L2 and cosine probably rank identically, if Ollama returns unit-length vectors | Reworded; R1 labeled "expected Δ≈0", checked at S1.5 (D-030) |
| 11 | §5 #15: wrong line range (`CHUNK_SIZE` is at app.py:27; `CHUNK_OVERLAP` is unused too); overstated cause of the reset hack | Fixed (D-030) |
| 12 | §5 #17: the swap/offload hypothesis doesn't fit v1's default 0.6B model | Reworded (D-030) |
| 13 | KV estimate: the layer math gives 448 MiB (0.44 GiB), not 0.45 GiB | Fixed in §1 and §6.5 (D-030) |
| 14 | ChatGPT review #2, resolution #8, said Gate A became "~20 steps" (it was 27) | Count removed |
| 15 | §14 chained commands with `&&`, which fails in Windows PowerShell 5.1 | One command per line (D-030) |
| 16 | §15 named a specific Ollama version ("0.40.0") with no source | Made neutral (D-030) |

## Walkthrough Q&A
Not applicable (setup step).

## Decisions
- Planning chat: D-001 … D-029 seeded.
- Verification chat: D-030 (factual corrections), D-031 (S0.4 dropped), D-032 (blank templates), D-033 (eval-design questions deferred), D-034 (one-time v1 reference run at CP2 with `gemma4:12b`, measuring speed + search quality; rung R0 kept).

## Open issues / follow-ups
- **Codex review pending:** save it as `project/reviews/codex-review-1.md` + a resolution table.
- Miniconda/Anaconda isn't verified yet. S0.1 starts by asking the user to run `conda --version`.
- My earlier `ollama list` check auto-started the Ollama desktop app (idle, no models loaded). The user may close it while working on the CSV project.

## Handoff to the next chat (S0.1)
1. Ask for the verification commit's SHA, and record it in STATUS.md's P0 row.
2. If the user has pasted the Codex review, file it first (see STATUS.md).
3. Confirm conda is installed (`conda --version`, run by the user).
4. Start S0.1 with the lesson (packaging, src layout, conda vs pip, lockfiles), then propose the design for `environment.yml` + `pyproject.toml` for approval before writing anything.
