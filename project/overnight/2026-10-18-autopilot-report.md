# Autopilot run 2 report — 2026-10-18

_Written as the run goes (write-ahead), so a crash or a usage limit loses nothing. If this file stops mid-way, the run stopped there; the last entry says why. Authority and limits: D-071 in `project/DECISIONS.md` (modelled on D-057; the first run's report is `2026-10-17-report.md`)._

## Terms (from the user's chat)
- Range: S3.3, S3.4, C4, one at a time, in that order, then stop (S2.6 needs the user to revisit D-065; CP2 needs the one-time `geekbot-v1` run; S3.5 needs S2.6's threshold).
- Per step: branch → build (including the ✍️ piece) → `ruff format`, `ruff check`, `mypy src`, `pytest` → the live tests if the GPU gate passes → commit and push → pull request with **base `v2`** → three CI checks green → squash-merge → delete the branch (D-056) → next.
- The user allowed: merging each pull request, pruning the merged branch, installing packages and regenerating `requirements/lock-win.txt`, and using the GPU and Ollama **on the `proto` profile only** (`gemma3:4b`).
- Not done on purpose: model downloads, the target models (`gemma4:12b`, `qwen2.5-coder:7b`), typing credentials, anything outside the range. Whatever those would have verified is marked **unverified** below.
- GPU gate before any live run: `nvidia-smi` shows only the desktop on the GPU, `ollama ps` lists nothing but our models, Ollama is already running (Claude does not start it). If the gate fails the live part is skipped and marked unverified.
- Usage limit (D-072): when a limit stops the run, wait for the reset and start again. A resume job (scheduler job `f406a1d8`, minutes 3, 23 and 43 of every hour, session-only, 7-day expiry) re-enters the run from the end of this report. It is deleted when the run finishes, hits a stop rule or waits for the user. The app must stay open and the PC on.

## Known limits at the start
- Usage at the start (2026-10-18 00:25 local, 2026-10-17 18:25 UTC): 5-hour window 28% used (resets 2026-10-17 21:49 UTC), weekly 50% used (resets 2026-10-21 18:59 UTC), Pro plan, extra usage off. This session's context: 30% of 1,000,000 tokens.
- The Windows CI job took about 25 minutes per pull request in the first run (the two Ubuntu jobs about 2). So the next step is started on a branch stacked on the previous tip while CI runs, and moved onto the merged `v2` with `git rebase --onto` afterwards.
- `gh` was not signed in last time; CI status is read from the PR page in the built-in browser. If the browser has lost its GitHub sign-in, the run stops at the pushed branch.
- Python for every check: `C:\Users\abrar\anaconda3\envs\geekbot\python.exe`.

## Slips and deviations (told straight)
- Before the decision was written, two read-only Bash commands outside rule 1: `cat` / `ls` on `.git` ref files (to compare the branch's base with `v2`), then `pwd`. They should have been `Read` and `Glob` calls. The finding: `step/S3.3-streaming-speed` had been cut from S3.2's pre-squash tip `bb1d0fd`, and local `v2` was stale at `675e44f`.

## Log
### Start
- `git fetch origin --prune`: `origin/v2` is `490e9f4` (S3.2's squash). `git diff --stat HEAD origin/v2` was empty (identical trees), so `git switch -C step/S3.3-streaming-speed origin/v2` lost nothing; the two uncommitted notes edits (`STATUS.md`, the S3.2 log) came along. Local `v2` fast-forwarded to `490e9f4`. The local S3.2 branch (`bb1d0fd`, already merged, remote copy gone) was deleted with `git branch -D` (D-056).
- GPU gate before the first live run (D-071): `nvidia-smi` 1,561 MiB used of 10,240, 0% utilisation, compute apps only Windows shell processes (two entries "Insufficient Permissions", unnamed, with 0% utilisation and the memory of a desktop alone); `ollama ps` empty; Ollama 0.34.2 running; `gemma3:4b`, `nomic-embed-text`, `gemma4:12b`, `qwen2.5-coder:7b` and `:3b` pulled. Before each later GPU command the gate was read again (only our two models loaded).
- Left alone on purpose (not part of this run's merges): the remote branches `step/S2.4-golden-set`, `step/S3.1-graph-basics` and `revert-22-step/S3.1-graph-basics` still exist on GitHub; they are old leftovers, not mine to prune unasked.

### S3.3 Streaming + speed (built and verified; pull request next)
- Built on `step/S3.3-streaming-speed` (cut from `origin/v2` at `490e9f4`). What exists: the graph always streams; `RunStats`; the ✍️ `read_stream`; `ask --stream --think --show-thinking`; `ask-timing`; `chat_think_num_predict`. Design and decisions: `project/steps/S3.3-streaming-speed.md`, D-073.
- Local checks: `ruff format --check`, `ruff check`, `mypy src` clean; `pytest` 1627 passed, 11 skipped, 11 deselected, 1 xfailed (was 1538 / 11 / 8 / 1; 89 new tests).
- Live (proto profile, `gemma3:4b`): `pytest -m live` on `tests/live/test_ask_live.py`: 4 passed (19 stream pieces, first token 0.08 to 0.13 s, no model load); `geekbot ask --stream` and `--think` behave as designed; `geekbot ask-timing` three times on the golden v0 corpus (numbers in the S3.3 log and STATUS).
- **Finding for you:** retrieval takes about 1.05 s at p50 (rerank 1.03 s at depth 30 over real chunks), against the plan's 400 ms budget. Not decided here: the rerank depth is yours with S2.6; `geekbot rerank-timing --query "<question>" --depths 5,10,20,30` (CPU only; not in D-071's list, so not run) shows the cost per depth.
- **Unverified:** any thinking model (`gemma4:12b`), every target-model number, prefix reuse as a cause of the shorter prompt read time on repeats.
