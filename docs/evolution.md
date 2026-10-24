# From v1 to v2: what was wrong, what changed, what was measured

Geekbot v1 (kept at the git tag `v1.0`) was a Streamlit app that indexed files, searched them with
Ollama's embedder and answered with a local model. It worked on the author's laptop and made claims
("100% local", "secure and private") that its own code contradicted. v2 is a rebuild whose three aims are
to **measure retrieval quality**, **control inference cost** and **treat retrieved content as untrusted
data**. This page is the story of that rebuild, told from the evidence: every number is from a result
file in `evals/results/` or a step log in `project/steps/`, names its model, and is the number *as
measured*, favourable or not.

## What v1 got wrong (found by reading its code and checked against it)

The 17 findings are in [`project/PLAN.md`](../project/PLAN.md) §5. The ones that shaped v2:

| v1 | Effect | v2 |
|---|---|---|
| A Chroma call that raises and is swallowed (`get(include=["ids"])`) | every file was re-embedded on every run; delete-by-source silently did nothing | content-hash IDs and an outbox: the second ingest of unchanged files embeds nothing (checked at Gate A) |
| The upload block ran on every Streamlit rerun | every chat message re-indexed every uploaded file | the UI is an HTTP client; an upload is sent only from inside a button, tested with Streamlit's own `AppTest` |
| "Rerank" re-embedded chunks with the same embedder; "multi-query" was five string templates | up to 21 embedding calls per question, no new relevance signal | one query, hybrid dense + BM25 fused by rank, and a real cross-encoder on the CPU |
| `nomic-embed-text` without its task prefixes | a worse embedding | the prefixes are part of the embedder's fingerprint |
| Web search hard-coded on | three sequential calls per question; text left the machine | off by default; a gated tool; the user approves the exact text of any query built from outside text |
| Retrieved text pasted into the prompt between plain separators | nothing marked it as untrusted | data-marked sources, labels in graph state, an output filter, a measured injection evaluation |
| Chat history stored, never sent | follow-ups failed | a rewrite step and a thread that remembers its turns and its taint |
| No context size set | Ollama ran the model at 262,144 tokens | fixed context sizes, the chat model kept 100% on the GPU, checked by `geekbot doctor` |

## How v1 behaved when it was run once

At checkpoint CP2 the original code was run once from its tag with the same chat model (`gemma4:12b`) and web
search off, on v2's golden set. In the environment its README describes, **9 of the 15 files could not be
loaded** (its requirements omit the package its Word, PowerPoint, Excel, Markdown and HTML loaders need),
and it found the right document in the top 5 for 40% of the single- and multi-hop questions (100% of those
whose files it could load). A full question took a median of 400 s (range 35 to 755 s, 20 questions): with no
context size set, Ollama ran the model at 262,144 tokens, about 44% on the CPU. Result:
`evals/results/v1-reference.json`. v2's speed on the same model is in [performance.md](performance.md); the two
were measured at different times and on different questions, so **no speed ratio is claimed**.

## The retrieval ladder: what the rebuild gained, and from what

Each change was added one at a time and measured on two public benchmarks (BEIR SciFact and NFCorpus,
standard datasets and metrics, nDCG@10). From `evals/results/s26-run1/`:

| rung | change | SciFact | NFCorpus |
|---|---|---|---|
| R0 | v1 rebuilt in v2 | 0.644 | 0.235 |
| R1 | cosine instead of L2 | 0.644 | 0.236 |
| R2 | one query instead of five, no re-sort | 0.674 | 0.296 |
| R3 | + the embedder's task prefixes | 0.682 | 0.315 |
| R4 | + section-aware 512-token chunks | 0.684 | 0.337 |
| R5 | + contextual header (the title) | 0.699 | 0.338 |
| R6 | + BM25 fused by rank (RRF) | 0.738 | 0.349 |
| R7 | + cross-encoder rerank | 0.712 | 0.350 |

The full system against the v1 replica: SciFact +0.068 (95% CI +0.035 to +0.104), NFCorpus +0.115 (+0.097 to
+0.133), both Holm-adjusted p = 0.0003. **Most of the gain is R2, removing something v1 did wrong** (SciFact
+0.030, NFCorpus +0.061), not something new. **The reranker did not improve nDCG@10 on either set** (SciFact
−0.025, CI −0.046 to −0.004; NFCorpus +0.001, CI −0.008 to +0.009) and takes most of the 400 ms retrieval
budget; it helped on the provisional golden set. Which of these are decisions for the author is listed in
`project/steps/CP2-first-evidence.md`; none was changed by a result.

## What the plan changed on the way (and why)

The plan went through 22 revisions (`project/PLAN.md` §4); the ones a reader should know:

- **A second conversation mode (rev 11).** Numbers in tables cannot be read from retrieved text, so the
  author's CSV analyst was copied in behind one adapter (ADR-0006).
- **Abstention without a score gate (rev 20).** The planned signal, a threshold on the reranker's best
  score, measured weak (F1 0.41 and 0.39), so the model's own refusal plus a citation check is the
  abstention, measured on the unanswerable questions (D-082).
- **No OpenAI-compatible endpoint (rev 22).** It would have been a local door in another program's
  message format; the author does not want an extra API endpoint, so it is parked for good (D-096).
- **Fast prototyping (D-047).** From S1.5 on the author wanted a working prototype before curating data,
  so golden v0 is Claude-authored and *provisional*; the author's own questions are a separate file.

## What building this found about the build

- **The output filter had holes** (D-100). Its own property test failed once in a full run; following it up
  found a removal that joined two harmless halves into a live outside link, among six holes, all fixed
  without weakening a test.
- **A hypothesis run found a tie-order bug** in rank fusion that had passed by luck (S5.3, fixed with an
  exactly rounded sum).
- **A guard that never passes is not a guard.** The first idle-GPU guard called a quiet Windows desktop busy
  because the driver lists every window as a compute process; it now decides on memory and utilisation (D-098).
- **Result files in a public repository must be read before they are committed.** One run had saved the names
  of 21 running programs (caught, scrubbed, and the saved form now keeps only counts).
