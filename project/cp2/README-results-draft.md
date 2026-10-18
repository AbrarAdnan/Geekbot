# Draft: README "Results" section (CP2, 2026-10-18)

_A draft for the user's review, not the README. The README is rewritten in Phase 5 (PLAN.md §11), when every number can be regenerated with one command. Everything below comes from `evals/results/s26-run1/` and `evals/results/v1-reference.json`; nothing is typed from memory. Wording rules (CLAUDE.md rule 8): BEIR is "standard BEIR datasets with standard metrics"; golden v0 is "Claude-authored, provisional"; security guarantees and empirical results are reported separately; service-level targets are targets until measured._

## Results

### How retrieval was measured
Each change to the retrieval pipeline was added one at a time (an *ablation*), and quality was measured after each: nDCG@10 on two public benchmarks, BEIR SciFact (300 test queries) and BEIR NFCorpus (323), which are standard datasets scored with their standard metrics. Differences are reported as the gain with its 95% confidence interval (paired bootstrap, 10,000 resamples); only the three headline comparisons also carry a p-value (Holm-adjusted). The system's own behaviour is checked on a small set of its own documents and questions ("golden v0", 15 documents, 100 questions), which is **written by the same author as the system, is easier than real use, and is reported as provisional, never as a headline**.

Held constant: the embedder `nomic-embed-text` (on the CPU), the reranker `ms-marco-MiniLM-L-6-v2` at depth 10, 50 candidates from each of dense and BM25 search, the top 10 chunks per question. A different embedder or reranker would give different numbers; the choice is compared separately below.

### The ladder
| step | what changed | SciFact nDCG@10 | NFCorpus nDCG@10 |
|---|---|---|---|
| 0 | v1 rebuilt in v2 (800-character chunks, no task prefixes, L2 distance, five template queries, re-sorted by re-embedding) | 0.644 | 0.235 |
| 1 | cosine instead of L2 | 0.644 | 0.236 |
| 2 | one query instead of five, no re-sort | 0.674 | 0.296 |
| 3 | + the embedder's task prefixes | 0.682 | 0.315 |
| 4 | + section-aware 512-token chunks | 0.684 | 0.337 |
| 5 | + contextual header (the title) | 0.699 | 0.338 |
| 6 | + BM25 fused by rank (RRF) | 0.738 | 0.349 |
| 7 | + cross-encoder rerank | 0.712 | 0.350 |

The full system against the v1 replica: SciFact +0.068 nDCG@10 (95% CI +0.035 to +0.104), NFCorpus +0.115 (+0.097 to +0.133); both with Holm-adjusted p = 0.0003. **Most of that gain is step 2, removing something v1 did wrong** (SciFact +0.030, NFCorpus +0.061), not something new. **The reranker did not improve nDCG@10 on either BEIR set** (SciFact −0.025, CI −0.046 to −0.004; NFCorpus +0.001, CI −0.008 to +0.009); it helped on the provisional golden set. Because a system that returns ten *chunks* covers fewer documents when chunks are large, step 4 includes that effect; the report prints documents per query beside every score.

### Comparing the v1 code with v2
The original v1 (git tag `v1.0`) was also run once on the golden set with the same chat model (`gemma4:12b`), web search off. In the environment its README describes, **9 of the 15 files cannot be loaded** (its requirements omit the package its Word, PowerPoint, Excel, Markdown and HTML loaders need), and v1 found the right document in the top 5 for 40% of the single- and multi-hop questions (100% of those whose files it can load). A full question took a median of 400 s (range 35 to 755 s, 20 questions): v1 sets no context size, so Ollama ran the model at 262,144 tokens, about 44% on the CPU. v2's speed on the same model has not been measured yet; no speed comparison is claimed.

### Speed of retrieval
On an idle PC, retrieval including the question's embedding and the rerank takes a median of 350 to 390 ms at depth 10 (95th percentile 528 to 626 ms on the two BEIR sets, 375 to 400 ms on the golden corpus), against a target of 400 ms at the 95th percentile: **the target holds on the small golden corpus and not on BEIR's long abstracts.** Without the reranker, retrieval takes 19 to 48 ms. Rerank depths of 20 and 30 cost two and three times as much and did not improve quality on the tuning questions of the BEIR sets.

### Embedder and reranker choices
Compared on the same questions (gain over `nomic-embed-text`, dense search only): `embeddinggemma` +0.064 nDCG@10 on SciFact and +0.038 on NFCorpus (both intervals exclude zero); `qwen3-embedding:0.6b` no better, and 2 to 5 times slower to embed. After fusion and reranking the gaps shrink. The larger reranker `bge-reranker-base` was not distinguishable from MiniLM and ran 4 to 5 times slower. The prefixes used for the two candidate embedders are their model cards' conventions as recalled, not independently checked.

### Abstention
The planned signal, a threshold on the reranker's best score, is weak on golden v0: the F1-optimal threshold scored F1 0.41 on the development questions and 0.39 on the test questions, and its value is unstable under resampling. Abstention accuracy of the full answering system is not yet measured.

### What these results do not show
Nothing here measures answer quality, groundedness, or the security guarantees (Phase 5). Golden v0 cannot separate close systems. The chat model's speed in v2 is not measured. The numbers are for one PC, one corpus pair and the models named above.
