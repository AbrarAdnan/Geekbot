# ADR-0004: Retrieved content is untrusted data: four controls, and a measurement that is kept apart

- **Status:** accepted (PLAN.md §8; D-003, D-014, D-023, D-024, D-041, D-100, D-102)
- **Date:** 2026-10-07 (G3 in S3.5, the gate in S5.3, measured in S5.4)

## Context
A document, a web page or a cell of a CSV file can contain text that a model copies or obeys. The
dangerous cases are not "the answer is wrong" but **data leaving the machine** (a markdown image whose
address carries the question, a web query built from a document) and **the model being steered**
(instructions in a source). A small local model cannot be relied on to resist; a defence that depends on
the model behaving is a hope, not a control.

## Decision
Four controls that do **not** depend on the model behaving, each with a test that can fail, and a
separate, honest measurement of the part nothing guarantees.

- **G1, tool isolation.** The nodes that read untrusted text have no tools bound (ADR-0001).
- **G2, taint-gated egress.** Every piece of text carries labels (`USER`, `KB`, `WEB`, `TOOL`,
  `UNKNOWN`) kept in graph state, never on strings; a model call's output carries the union of its
  prompt's labels; the thread's taint only grows. The one network tool (the web search) sits behind
  `EgressGate`: no call whose arguments were built from text the user did not write unless the user
  approved those exact arguments (a pure function, `agent.taint.egress_decision`). The web is off by
  default and "off" means the tool is not built. The tool also asks the rule on every call, and a test
  lists every module that can reach the network.
- **G3, output sanitization.** Images and links to addresses the sources do not name are removed from
  the stream, the final answer and CSV replies; a removal never joins two halves into a construct.
- **Citation check.** A `[n]` mark that names no source in the prompt is removed.
- **CSV mode:** the package's read-only SQL engine refuses writes, `ATTACH` and `PRAGMA` (a dedicated
  test); CSV data never reaches the web tool.

What is **not** guaranteed is reported as a rate and never claimed solved: that the model ignores
instructions planted in a source. The injection evaluation (S5.4) counts **bypasses** (a control
failed; target 0) and **hijacks** (the model obeyed) separately, with k out of N, and says what its
attacks are and are not.

## Consequences
- A README sentence about security can be one of two kinds, and says which: "this cannot happen unless
  the code is wrong" (a guarantee, with its test) or "this happened k times in N attempts" (a measurement).
- G3 keeps a link to an address that appears in a source, by design; an attacker who controls a source
  controls that address. The evaluation counts it as an *exposure*, not a bypass, and says so.
- A bare URL in running text is left alone by G3 (a renderer that links bare addresses makes it
  clickable): a stated limit.
- The output filter is a hand-written reader of markdown and is never finished; S5.2's full test run found
  six holes in it (D-100), which is the reason its property tests stay.

## Evidence
`docs/security.md`, `evals/results/` (the injection report), `tests/unit/test_taint.py`,
`test_egress.py`, `test_web_graph.py`, `test_sanitize.py`, `test_injection_run.py` (controls broken on
purpose are reported as bypasses).
