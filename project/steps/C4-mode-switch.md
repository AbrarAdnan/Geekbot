# C4 — Mode switch in the graph

- **Phase / Gate:** Track C (CSV analyst), Gate A. It follows S3.4 and, in the order of PLAN rev 19 (D-068), comes before S2.6 and CP2.
- **Chat date(s):** 2026-10-18 (autopilot run 2, D-071 / D-072: the third and last step of the run)
- **Status:** built and verified by Claude under D-071; **awaiting the user's review** (the green light is still the user's)
- **Commit:** `ed0a28c` (squash commit of PR #28 into `v2`; 3 CI checks passed; branch tip `82bed4b`, identical tree; the branch was deleted afterwards, D-056). Merged by Claude under D-071.
- **Lesson note:** deferred (prototyping mode, D-047)

## Goal
Make the CSV analyst a second branch of the answer graph, chosen by the *thread's mode*, so that a CSV conversation is a thread like any other: saved, with follow-ups, with taint (PLAN.md §6.7, §7, §8.2, §11 row C4).

A little context. The copied `csv_analyst` package answers one question about one file and remembers nothing, so a follow-up such as "and which one is highest?" reaches it as a fragment it cannot use. Two parts of S3.4 now carry over: the rewrite of a follow-up (here with the *CSV* model, the one already on the GPU) and the saved thread. What is new is the decision of which branch a turn takes. It is a plain function of the thread, not a model: a model that chose the branch would cost a call per turn and could be steered by the very text the branch handles. And the mode is **fixed for the thread**, because the model on the GPU (the two never share it), what the memory contains, and what the taint means were all built under one mode: a thread that started on documents cannot start answering from a table's cells half way, and the other way round. The error says so and names the fix (a new thread name).

## Approved design
Built straight after S3.4 under the autopilot terms of D-071 (range S3.3, S3.4, C4). Decisions: D-075 (design); D-071 and D-072 are the run's terms. No plan change; one planned piece (`sanitize`) is left to S3.5 where the plan builds it (see "Open issues").

- **The router** (`agent/mode.py`, the ✍️ piece): `choose_mode(saved, requested)` and the two conditional-edge functions `route_after_begin` and `route_after_condense`.
- **State** (`agent/state.py`): `requested_mode` (input only, cleared once used), `mode` (the thread's, saved), `csv_file` (display only), `csv_status` (the package's verdict for the turn).
- **The graph** (`agent/graph.py`): `begin_turn` settles the mode; from there a first turn goes straight to `retrieve` (chat) or `csv_ask` (CSV), and a follow-up goes through `condense` first, then on to its branch; every branch ends in `remember`.
- **The CSV branch** (`agent/nodes.py::csv_ask`): one call into the package through a `CsvTool` Protocol (`agent/runtime.py`; `csv_mode.adapter.CsvAnalyst` fits it), asked the standalone question; every status the package reports (answered, abstained, needs clarification, failed) is an answer the user reads; an exception becomes a `ChatError`; the answer is labelled `TOOL` and `USER`; no sources, no citations; the readings are the package's time and the model load.
- **Stubs** (`agent/runtime.py`): `UnavailableRetriever` and `UnavailableChat`, which a CSV run passes and which raise if anything touches them.
- **The rewrite in CSV mode** (`llm/chat.py::make_structured_model` gained `num_ctx`): the CSV model with `csv_num_ctx`, the size it was warmed up with (R-2: a different size would reload the model for the rewrite). `prompts/condense.md` now says "to search the user's documents, or to ask a question of a table".
- **Command line** (`cli.py`): `geekbot csv ask FILE QUESTION --thread NAME` takes the storage lock, checks Ollama and the thread's mode, swaps explicitly to the CSV model (the chat model is unloaded), opens the file, runs one turn of the graph in CSV mode, and exits 1 if the package's status is `failed`; without `--thread` the C2 path is untouched. `geekbot ask --thread` refuses a CSV thread before loading any model; `thread show` names the mode and file and prints each message on one line (a CSV answer is a whole table).
- **Output** (`agent/render.py`): `csv status: answered`, `asked as: "..."` (a table is asked, documents are searched), `thread t1 (csv): turn 2, ...`.

## ✍️ Your piece
- **What the plan names:** the mode router (a pure function) + its tests.
- **What was written:** `agent/mode.py` (`choose_mode`, `route_after_begin`, `route_after_condense`). Spec: the docstring of `choose_mode` (five rules: a saved mode is kept; a first turn takes the one asked for; neither means chat; a contradiction is an error, never a switch; an unknown name is an error whichever argument carries it, and `None` and `""` both mean "not given"). Tests: `tests/unit/test_mode.py` (the whole table, the error messages, and three properties over every pair of inputs: the answer is a mode or an error, a saved mode is never overridden, choosing again with the result changes nothing; and a property that a turn never reaches the other mode's branch) and `tests/unit/test_csv_branch.py` (the graph, with a real checkpointer).
- **Result:** all passed on the first run, including the properties; no counterexample, no change after review.
- **Help used:** written by Claude (prototyping mode, D-047). On the ✍️ backlog in STATUS.md.

## Files changed
- `src/geekbot/agent/mode.py` (new, **✍️**).
- `src/geekbot/agent/graph.py`, `nodes.py` (`csv_ask`; `begin_turn` settles the mode; `needs_condense` moved into `mode.py` as `route_after_begin`), `runtime.py` (`CsvTool`, `CsvReply`, the stubs, `AgentContext.csv`), `state.py`, `render.py`, `runner.py` (`mode` and `csv_file` parameters), `prompts/condense.md`.
- `src/geekbot/llm/chat.py` (`make_structured_model(num_ctx=)`), `src/geekbot/cli.py` (`csv ask --thread`, the mode check in `ask`, `thread show`).
- `tests/unit/test_mode.py`, `test_csv_branch.py` (new); `test_csv_cli.py`, `test_ask_cli.py`, `test_agent_graph.py`, `test_conversation.py`, `test_render_stream.py` (extended); `tests/live/test_csv_thread_live.py` (new).
- `project/DECISIONS.md` (D-075), this log, `STATUS.md`, the autopilot report.

## Commands run + results
Run by Claude under D-071 with the `geekbot` env's `python.exe`; the GPU commands only after the gate (`nvidia-smi` 5.6 GB used, 5% utilisation: the desktop plus our own two loaded models; `ollama ps` listed only `gemma3:4b` and `nomic-embed-text`).

- **Checks.** `ruff format` and `ruff check` clean, `mypy src` no issues in 105 files, `pytest` **1790 passed**, 11 skipped, 13 deselected, 1 xfailed in 108 s (before: 1729 passed, so 61 new tests; the live test adds one deselected).
- **Live test** (`pytest -m live -s tests/live/test_csv_thread_live.py`, `qwen2.5-coder:3b`, the unit tests' `sales_small.csv`): passed in 8.1 s. The swap unloaded `gemma3:4b` and loaded the CSV model in 2.7 s. Turn 1 "Which region sells the most?": "South sells the most, with 4 sales" (the package counted rows), 1.75 s, load 0.02 s, labels `TOOL, USER`. Turn 2 "And what is its total revenue?" was rewritten by the real CSV model as "What is the total revenue of the region that sells the most?" in 0.31 s (valid JSON in the schema's shape), and the package then answered 520.0 for **North** (it read "sells the most" as revenue this time): the **two turns disagree about which region "sells the most"**, a reading of an ambiguous question that varies from run to run in the package, not a fault of the rewrite; it is the kind of thing C6's follow-up set exists to measure. The thread was saved with mode `csv`, file `sales_small.csv`, taint `TOOL, USER`.
- **A real conversation** through the command (`geekbot csv ask tests/fixtures/sales_small.csv ... --thread c4-demo`, two invocations): turn 1 "What is the total revenue per region?" gave East 300, North 520, South 380 (1.78 s, load 0.02 s); turn 2 "And which one is highest?" was `asked as: "Which region has the highest total revenue?"` (rewrite 0.46 s) and answered North, 520.0 (1.66 s). `thread show` printed `thread c4-demo (csv, sales_small.csv): 2 turns, 4 messages remembered` with the labels `[TOOL, USER]` and the taint. `geekbot ask "..." --thread c4-demo` was refused at once with "this thread is a csv conversation and cannot become a chat one: a thread keeps the mode it started with (use a new thread name)". The thread was deleted afterwards. One run each on the `proto` profile: **plumbing checks, not quality results and not SLO measurements.**

## Review + walkthrough Q&A
Not done (the green light is the user's). The spots most worth reading: `choose_mode` (the five rules and why a contradiction is never a switch), `route_after_begin` (a follow-up is rewritten in either mode; a first turn makes no model call), `csv_ask` (every status is an answer; what the labels are and why), `make_structured_model(num_ctx=)` (R-2), and `_csv_ask_in_thread` in `cli.py` (the order: lock, thread and mode check, explicit swap, then the file).

## Decisions
D-075.

## Open issues / follow-ups
- **`sanitize` is not in the CSV branch yet.** PLAN.md §7 draws a `sanitize` step after `csv_ask`, and the C4 row lists it; the plan's S3.5 row builds `sanitize.py` (G3: a streaming-safe filter that strips markdown images and links to URLs that are not sources) "also on CSV replies". Building the CSV branch first without it is a choice, not a plan change: the filter is S3.5's piece and it will be added in front of `remember` on **both** branches there. **Until then a CSV reply reaches the terminal as the package renders it**, and a cell that contains a markdown image or link would be shown as written. That is the known gap G3 closes (§8.1), recorded here so nobody assumes the CSV branch is already filtered.
- **The thread does not hold the file.** It remembers the mode and the file's name (display only); each `csv ask --thread` invocation is given the file again and loads it. A different file in the same thread is not refused (C5's upload registry, one session per thread, is where that is decided). The follow-up rewrite then works from a history about the old file.
- **CSV follow-up quality is not measured** (C6's job: about 15 multi-turn items scored 2 of 3). One example worked well; the ambiguity above shows a way it can go wrong that is not the rewrite's.
- **The storage lock** is held for the whole `csv ask --thread` (it writes `threads.db`), so a chat `ask` or an ingest at the same time is refused at once.
- **Not verified:** the target CSV model (`qwen2.5-coder:7b`), the swap time on the target models (C3 measured it; not repeated here), what a markdown image in a cell does end to end (S3.5, S5.4), two threads of different modes alternating with a swap each time (each swap is a few seconds, R-2's one exception), and the history budget on a long CSV conversation (an answer is a whole table; it is cut to one line per message and to the budget).
- **Slip, told straight:** my first click on "Confirm squash and merge" for S3.4's pull request did not register and the page still showed the open request; I re-read the page instead of assuming, found it not merged, and clicked by element reference. Recorded in the autopilot report.

## Handoff to the next chat
The run's range ends here (D-071): S2.6 needs you to revisit D-065 first (the three questions deferred in D-033), CP2 needs the one-time `geekbot-v1` environment, and S3.5 needs S2.6's threshold. S3.5 adds `sanitize.py` and places it in front of `remember` on both branches, and the abstain decision between `retrieve` and `generate`. The graph is `begin_turn → (condense) → retrieve | csv_ask → ... → remember`; `AgentContext` holds the retriever, the chat model, the optional rewrite model, the history limits and (CSV mode) the loaded file; `run_turn(..., mode=..., csv_file=...)` drives it.
