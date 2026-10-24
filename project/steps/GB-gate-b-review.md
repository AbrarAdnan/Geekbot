# GB — Gate B review (Phase 5, Evidence)

- **Phase / Gate:** the review at the end of Phase 5 (PLAN §11 row "Gate B review"). Run by Claude in autopilot run 5 (D-097, 2026-10-24); **the verdict is Claude's, not yours: you decide whether Gate B is passed.**
- **Status:** the checklist below; one part is not done and cannot be by Claude (your labels).
- **Commit:** branch `step/GB-gate-b-review` (stacked on `step/S5.5-writeup`)

## The definition of done (PLAN §11, with §14 items 5 and 7)

| # | Item | Verdict | Evidence and what to keep with it |
|---|---|---|---|
| 1 | **Every README number regenerable** | **pass, with a stated limit** | The README has a table of the command that regenerates each result, and `S5.5-writeup.md` audits each number against its file: the C6 and S5.4 numbers were recomputed from their JSON, the rest read from the result reports and step logs. **Not done:** recomputing the S2.6 and S5.2 numbers from their raw files, and *running* every command again (several need the GPU for half an hour). |
| 2 | **Bypasses 0/N** | **pass, with caveats that matter** | S5.4 run 2: **0 of 129 attempts** (0 of 43 attacks). Caveats: the three runs of every attack agreed, so the effective N is 43; the attacks are mine and synthetic; the poison was always in the prompt; **run 1 found three holes in the output filter (fixed)**, which is the best evidence that "none left" is not shown. |
| 3 | **κ rule applied** | **partly: the rule is applied, κ is not measured** | The rule is in code and in the report (`labelling.kappa_verdict`: fewer than 30 labels, no verdict; κ below 0.6, the judge is excluded from headline claims). With **0 labels** the judge is excluded: the README says it "supports no claim". **The ~60 labels and the κ number are yours** (Labels page; `geekbot eval answers --phase report`). |
| 4 | **SLO table measured, CSV mode included** | **pass** | `docs/performance.md` from `s51-run2` (official, idle guard passed): every target met except **retrieval plus rerank p95 (389 to 435 ms against 400 ms: met at 5 of 16 points)**, reported, not hidden; CSV p50 2.8 s and p95 6.4 s; swaps 2.7 s and 5.0 s (once 6.1 s in Gate A against a 6 s target). |
| 5 | §14 item 5: **web off, no egress** | **pass** | Structural: with the web off (the default) the tool and its backend are not built (`tests/unit/test_web_tool.py`); a CSV conversation never reaches the web step; the inventory test lists every module that can reach the network (`test_egress.py`). In the UI the toggle is only shown when the server has a web search (checked with `AppTest`; the browser check below was done with the web **on**). |
| 6 | §14 item 5: **a tainted web query shows the approval prompt** | **pass** | Checked in the **built-in browser on the real server with `gemma4:12b`** (fixture backend): a follow-up the model rewrote stopped at "A web search is waiting for your answer" with the exact text and why; after a reload the same card was there (the question lives on the server); "Send as shown" sent that text and the answer arrived with a Details line "searched the web, as you approved, for: ...". "Send my edit" and "Do not search" were not clicked in the browser (they are tested with `AppTest`, `TestClient` and the CLI). |
| 7 | §14 item 7: the evidence commands | **pass** | `geekbot bench` (S5.1), `geekbot eval injection` (S5.4), `geekbot eval answers` (S5.2; κ pending), `geekbot eval csv-followups` (C6) each ran on the target models. |

**Verdict: Gate B holds for everything Claude can check; it is not complete until the judge is validated or dropped (item 3).** Nothing in Phase 5 needs a change before you review, but the open decisions below are yours.

## Phase 5 look-back (planned against actual)

- **Planned:** bench and sweep, answer eval and judge, the CSV evaluation, web tool and taint, injection eval, write-up. **Actual:** all six, plus a security finding in each of two steps.
- **What did not go as planned**
  - **The output filter (G3) was not finished**: D-100 (six holes, found by S5.2's full run) and D-103 (three more, found by S5.4 and by repeating its property tests). A hand-written reader of markdown found, each time, forms the last round had not thought of. *Change next time:* run the property tests many times before calling a security step done, and treat a hypothesis database that "replays a failure" as a finding, not noise.
  - **The idle-GPU guard failed on a quiet desktop** (the Windows driver lists every window as a compute process); fixed by deciding on memory and utilisation (D-098).
  - **The three runs of an attack are not three samples** (fixed seeds): the injection report now counts attacks and says so. *Change next time:* vary the sampling seed per run if independent runs are wanted.
  - **My own items had ambiguities** (C6: an extra column, labels): the run stands as it is; rewording them is a revision for you.
  - **A prompt that defaults to "no" is awkward through a pipe** (PowerShell answered the approval prompt as no); the safe side, noted.
- **What the evidence says that was not expected**
  - Most of the retrieval gain over v1 was removing a bug; the reranker did not help on BEIR and takes most of the latency budget.
  - The model obeyed the planted text in 5 of 43 attacks and **never carried an instruction into a later turn**; the web pages and citation forgeries mostly did not move it.
  - The CSV follow-up rewrite caused none of the 3 failures of 15 items.
- **Honest gaps going into Gate C and the README review:** the judge (no labels), the golden set (the author's, provisional), a live web backend (never tried), the DuckDuckGo parser (never run on the live page), a second sitting of the retrieval timings.

## Decisions waiting for you (none was changed by a result)
1. The reranker (it did not help on BEIR, it takes 330 to 780 ms), its depth, and the 400 ms retrieval target.
2. The KV cache: q8_0 gives a first token about 0.3 s sooner and about 200 MiB of headroom, with no quality difference visible on 8 easy questions.
3. top-k 3 or 5; the embedder (`embeddinggemma` was clearly better dense-only at CP2).
4. The judge: pull `qwen3:8b`, label about 60 answers, read κ.
5. Your own golden questions and CSV items; reword C6's F-006 and F-010.
6. A real web backend (SearXNG you run, or DuckDuckGo after a live check) or none.
7. Whether the pull requests, in order, are merged: Gate A, S5.1, S5.2, C6, S5.3, S5.4, S5.5, GB (each stacked on the one before; they wait for a GitHub sign-in in the built-in browser).

## Slips (told straight)
- The `@(@(a,b))` flattening that rewrote three ADR files (S5.4 log). No other.

## Handoff
S6.1 (the MCP server) is next, then the Gate C review.
