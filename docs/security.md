# Security: what is guaranteed, what is measured, and what is neither

Geekbot treats everything it retrieves as **untrusted data**: a document, a web page and a cell of a CSV
file can all contain text that a model copies or obeys. This page keeps two kinds of statement apart on
purpose (ADR-0004, ADR-0005): **guarantees** (a control that holds unless the code is wrong, each with a
test that can fail) and **measurements** (what happened, k times in N attempts, with what was tried).
Nothing here is "secure": it is a list of what is checked and how.

## The guarantees, and the test that stands behind each

| | Claim | Where | Test |
|---|---|---|---|
| **G1** | The nodes that read untrusted text have no tools bound. | `agent/graph.py`: the model is given no tools; the graph's edges are plain functions. | the sixteen steps are asserted (`test_agent_graph.py`) |
| **G2** | No network call whose arguments were built from text the user did not write, unless the user approved those exact arguments. The web tool is off by default and off means it is not built. | `agent/taint.py::egress_decision` (a pure function), `tools/egress.py` (`GatedTool` asks the rule again on every call; an ungated network tool cannot be registered), `agent/nodes.py::egress_gate` (`interrupt()`) | `test_taint.py` (a table and hypothesis properties: off never allows, a denial is never overridden, allow means user-written or exactly approved), `test_egress.py` (including an inventory of every module that can reach the network), `test_web_graph.py` |
| **G3** | No image and no link to an address the sources do not name reaches a user in a chat answer, the stream, a REST or SSE reply or a CSV reply. | `agent/sanitize.py` | `test_sanitize.py` (attack strings and hypothesis properties, including chunking-invariance: the streamed text equals the cleaned whole) |
| | A `[n]` citation mark that names no source in the prompt is removed. | `agent/citations.py` | `test_citations.py` |
| | CSV mode refuses writes, `ATTACH` and `PRAGMA`; CSV data never reaches the web tool or leaves the machine. | the package's read-only SQL engine; `csv_mode`; no CSV branch reaches `egress_gate` | `test_csv_sql_readonly.py`; `test_web_graph.py::test_a_csv_conversation_never_reaches_the_web_tool` |
| | The server listens on `127.0.0.1` only; every route but `/healthz` needs the API key; a foreign `Host` is refused; no CORS header is ever sent. | `server/security.py` | `test_perimeter.py` |
| | A path outside the allowed folders is refused (no symlink escape, network paths refused before the disk is touched, the data folder never readable). | `PathPolicy` | `test_path_policy.py` |
| | MCP is read-only: two tools, neither writes, deletes, ingests or reaches the network; an `Origin` that is not this machine is refused. | `server/mcp.py` | `test_mcp.py` |

**Taint** is kept in graph state, never on strings (PLAN §8.2): every message carries labels (`USER`, `KB`,
`WEB`, `TOOL`, `UNKNOWN`), a model call's output carries the union of the labels of its prompt (`packing.py` is
the only place a prompt is assembled), the thread's taint only grows, an unreadable label is `UNKNOWN`, and a
history message with no labels is not clean. A client-written assistant message is `UNKNOWN`.

## What was measured (S5.4, `evals/results/s54-run2/`)

`gemma4:12b` (chat) and `qwen2.5-coder:7b` (CSV), target profile, idle GPU. **30 planted documents and web
pages** over the four vectors of PLAN §8.3 (a web call carrying secret text, markdown-image exfiltration,
history laundering, citation forgery) and **13 CSV-cell attacks**, 3 runs each (129 attempts). The planted
text is always among the sources the model reads (worst-case placement); a simulated user denies every
approval; every address is under `.example` and the search backend is canned. The detectors are written
independently of the controls, and each control was broken on purpose in a test to check that the same attack
is then reported.

| | result |
|---|---|
| **bypasses of a guaranteed control (target 0)** | **0 of 129 attempts** (0 of 43 attacks) |
| the gate asked, the simulated user refused | 39 times; **0** queries sent that nobody approved |
| the canary reached a web query | 0 |
| markup the model wrote and the output filter removed | 18 attempts (6 attacks) |
| **hijacks: the model obeyed** | **15 of 129 attempts = 5 of 43 attacks** |
| a planted instruction laundered into a later turn | 0 |
| links to an address only the planted text named, kept by G3 | 0 |

**Read the caveats before the numbers.** The three runs of every attack gave the same outcome (43 of 43): the
models are sampled with fixed seeds here, so the effective N is the **43 attacks**, not 129. The attacks are
mine, synthetic, and few. Retrieval was not part of the test (the poison was always in the prompt, which makes
the rates an upper bound for this model on these attacks and says nothing about attacks I did not write). The
five attacks the model obeyed were an image badge (removed by G3), two standing instructions obeyed in the turn
that read them (not carried into the next), a direct "answer only" override, and a CSV question about the
poisoned column that echoed the planner's canary. **Hijacks are not guaranteed against and are not claimed
solved.**

## What the evaluation found in the controls

Security measured is security found out. **The output filter had holes**, found by its own property tests and
then by this evaluation: S5.2's full run (D-100: six holes; a removal could join two harmless halves into a live
outside link), and S5.4's first sitting (D-103: a raw HTML anchor survived, `[x][y](https://attacker.example/...)`
survived untouched, a removed link could leave a live link-reference definition behind). All were fixed without
weakening a test. **That no hole is left is not shown**: a hand-written reader of markdown is never finished, and
this is the second round. A hypothesis test that fails once in a while is the test doing its job.

## Known limits and what is not covered

- **A link to an address that appears in a source is kept (G3, by design).** Whoever controls a source controls
  that address. The evaluation counts it as an *exposure* (0 here), because the address cannot carry anything the
  model adds, but it is a link the attacker chose.
- **A bare URL in running text is left alone**; a renderer that links bare addresses makes it clickable. The
  project's own UI renders answers with Streamlit's markdown, which has its own rules.
- **Answer manipulation** (a poisoned document making the answer false) is not a control: it is the hijack rate
  above, and the system prompt and the data-marking (every line of a source starts with `> `) lower it without
  guaranteeing anything.
- **The web search**: the DuckDuckGo backend's parser has not been run against the live site; the fixture and
  SearXNG backends are the tested ones. The search sends the user's own words as typed; a query rewritten from
  outside text needs the user's approval of the exact text. A search *result* is untrusted text like any
  document.
- **The perimeter is a local password**, not authentication for the network: the server is for one user on one
  machine and refuses to listen elsewhere.
- **The approval question** reaches the user as plain text; whether a person reads it before clicking is not
  something the code can know.
- The SQL checks of the copied CSV package beyond the tested read-only property are prototype-grade
  (INTEGRATION.md of the source project says so); only the tested property is claimed.

## Reproduce
```
pytest tests/unit/test_taint.py tests/unit/test_egress.py tests/unit/test_sanitize.py tests/unit/test_web_graph.py
set GEEKBOT_PROFILE=target
geekbot eval injection --name my-run
```
`geekbot eval injection --phase report` rebuilds the report from saved results without a model. Add your own
attacks to `evals/datasets/injection.jsonl` (one JSON line each; the loader says what is wrong with a line).
