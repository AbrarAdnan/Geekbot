# P0 — Project notes bootstrap

- **Phase / Gate:** pre-Phase 0 (no gate)
- **Chat date(s):** 2026-10-07 (planning chat)
- **Status:** verify (awaiting the user's review + commit)
- **Commit:** _to be reported by the user_
- **Lesson note:** none (setup step)

## Goal
Move the approved plan (rev 7) into the repo and create the central notes directory and the per-chat bootstrap, so every later step can run in its own chat with full context (PLAN.md §2.1).

## Approved design
As specified in PLAN.md §11, "Step P0". Only files are written; nothing is executed.

## Files changed
- `CLAUDE.md`: standing rules + session protocol, auto-loaded by every Claude Code chat in this repo
- `project/PLAN.md`: the approved plan, rev 7, verbatim
- `project/STATUS.md`: step tracker + "Next up: S0.1" handoff
- `project/DECISIONS.md`: decision log D-001 … D-029
- `project/reviews/README.md`, `chatgpt-review-1.md`, `chatgpt-review-2.md`: both reviews verbatim + resolution tables
- `project/steps/README.md`, `project/steps/_TEMPLATE.md`, this log
- `project/learn/README.md`: index of the 34 planned lesson notes

v1 files are untouched.

## Commands the user runs
From the repo root (`C:\Work\Geekbot`), on `main`:
```
git tag v1.0
git checkout -b v2
git add CLAUDE.md project
git commit -m "chore: add v2 plan and project notes (P0)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git log --oneline -1
```
Optionally, publish the branch and tag:
```
git push -u origin v2
git push origin v1.0
```
_Results: to be filled in at the start of the S0.1 chat (commit SHA)._

## Walkthrough Q&A
Not applicable. The plan itself was reviewed over seven revisions and two ChatGPT reviews.

## Decisions
D-001 … D-029 seeded in `DECISIONS.md`.

## Open issues / follow-ups
- Miniconda/Anaconda isn't verified yet. S0.1 starts by asking the user to run `conda --version`.
- My earlier `ollama list` check auto-started the Ollama desktop app (idle, no models loaded). The user may close it while working on the CSV project.

## Handoff to the next chat (S0.1)
1. Ask for the P0 commit SHA; record it in STATUS.md and set P0 to `green-lit`.
2. Confirm conda is installed (`conda --version`, run by the user).
3. Start S0.1 with the lesson (packaging, src layout, conda vs pip, lockfiles), then propose the design for `environment.yml` + `pyproject.toml` for approval before writing anything.
