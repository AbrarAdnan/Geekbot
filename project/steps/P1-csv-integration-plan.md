# P1 — CSV analyst integration plan

- **Phase / Gate:** planning (like P0); no gate
- **Chat date(s):** 2026-10-11
- **Status:** verify (waiting for the user's review and green light)
- **Commit:** <SHA reported by the user>
- **Lesson note:** none (a planning step)

## Goal
- **CSV mode:** merge the user's Gigalogy project (`csv_analyst`, "Mini Maira") into Geekbot as a second conversation mode, "CSV analyst", following Gigalogy's `INTEGRATION.md`.
- **Hands-on learning:** restructure the plan so the user writes code in every step and can follow the whole development lifecycle.
- **Scope:** writes notes only (PLAN rev 11). No code.

## Approved design
Approved by the user on 2026-10-11, in the plain-language version of the plan. A first, more technical version was rejected as too hard to read.

User decisions (asked in this chat):
1. **Copy the code in** (over git subtree, installing from the Gigalogy folder, or rebuilding step by step) → D-036.
2. **CSV mode uses `qwen2.5-coder:7b`** (over "measure both first" or "gemma4 for everything"). It's swapped with `gemma4:12b`, because they don't fit in 10 GB together → D-037.
3. **Track C1–C6, each step placed where its dependencies exist** (over one block after Gate A, or at the end) → D-038.
4. **Pair programming:** one ✍️ piece per step (over "keep as now" or "you write most of it") → D-039.

Also in the approved plan:
- the lifecycle map, a branch + pull request per step from S0.5, and a look-back per phase → D-040
- CSV-mode safety and honesty rules → D-041

**Not in the approved plan, so left open:** whether chat mode should stop embedding row text of small CSVs (D-016). It's recorded as an open question for S1.1's design, with Claude's recommendation (schema card only). It is not a decision.

## ✍️ Your piece
None (a planning step).

## Files changed
- `project/PLAN.md`: rev 11
  - §1 decisions; §2 the pair-programming loop + lifecycle map + hint ladder; §2.1 `sources/`; §3 CSV mode under the three pillars; §4 rev 11
  - §6 diagram; §6.3 csv row; §6.5 qwen VRAM table + residency rules R-1/R-2/R-4 + CSV targets; §6.6 layout; new §6.7 CSV analyst mode
  - §7 mode router; §8 CSV security; §9 CSV evaluation; §10 phases + Parked
  - §11 the ✍️ column, the C1–C6 rows, S6.2 dropped, Gate A CSV items
  - §12 risks 16–21; §13–§15
- `project/DECISIONS.md`: D-036 – D-041
- `project/STATUS.md`: P1 row; C1–C6 rows in execution order; S6.2 dropped; new "Next up" + handoff; legend
- `CLAUDE.md`: second mode in the intro; rules 3, 4, 6 and 8; end-of-step protocol (branch + PR, look-back); code conventions and map for `src/csv_analyst/` and `project/sources/`
- `project/learn/README.md`: renumbered in execution order, with C-track lessons 05, 06, 07, 24, 31, 34
- `project/steps/_TEMPLATE.md`: "✍️ Your piece", "Review + walkthrough Q&A" and "Phase look-back" sections
- `project/steps/README.md`: P1 row
- `project/sources/csv-analyst-INTEGRATION.md`: a verbatim copy of Gigalogy's guide, with a source header

## Commands the user ran + results
_None needed for the notes._ The commit command is given at the end of the chat.

## Review + walkthrough Q&A
_Pending: the user reviews PLAN §2, §6.7 and §11._

## Decisions
D-036, D-037, D-038, D-039, D-040, D-041.

## Open issues / follow-ups
- **No-run slip:** early in this chat, Claude ran one read-only shell command in `C:\Work\Gigalogy` (a file listing, `git log`, `git status`, line counts). It didn't touch the GPU, but it broke CLAUDE.md rule 1. Claude told the user and used file-reading tools only for the rest of the chat.
- **S0.1 is unfinished:** `requirements/lock-win.txt` is missing from the repo; see STATUS "Next up".
- **D-016 open question** for S1.1 (above).
- **For C1's design, from reading the Gigalogy code:**
  - `OllamaProvider` calls `/api/generate` with `num_ctx` 16,384 and JSON-schema `format`.
  - Its fallback to the `thinking` field exists for thinking models; `qwen2.5-coder` isn't one.
  - `to_markdown` escapes `|` and newlines in table cells but not markdown images, so G3 must run on CSV replies.
  - `tests/test_session.py` uses a fake LLM, so it runs without Ollama.
- **Codex review:** still pending from P0.

## Handoff to the next chat
- **Next:** finish S0.1 first (the lock), then S0.2, S0.3, S0.5, then C1.
- **The plan:** every step now has a ✍️ piece (PLAN §11).
- **Commit layout:**
  - This P1 commit also carries S0.1's already-approved rev-10 notes (D-035, PLAN §6.1, the S0.1 row), because they sit in the same files.
  - S0.1's code, its log and lesson 01 stay uncommitted until S0.1 is green-lit.
  - It also carries the notes from 2026-10-14: plan rev 12 (D-042, the prototyping profile) and `project/README.md`.
