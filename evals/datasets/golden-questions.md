# Golden questions

The hand-written test questions for the golden corpus (`evals/corpus/`). They check how Geekbot behaves on **our** documents: "system-specific behavioral evaluation" (PLAN.md §9). They say nothing about retrieval quality in general; the BEIR benchmarks cover that.

**This file is the source of truth.** `geekbot eval import` reads it, checks it, and writes `evals/datasets/golden.jsonl` (step S2.4). Edit this file, never the JSONL: the JSONL is overwritten on every import. Blocks left empty are skipped, so you can fill it in over several sittings.

**Target: ~100 questions**

| Type | Target | What it tests |
|---|---|---|
| Single-hop | ~50 | the answer is in one place in one document |
| Multi-hop | ~15 | the answer needs facts from two or more places |
| Multi-turn | ~15 | a follow-up that only makes sense after an earlier question |
| Unanswerable | ~20 | the documents don't contain the answer, so Geekbot should say it doesn't know |

**What each field is for**
- **Question:** written the way a real user would ask it. Don't copy a sentence from the document word for word; that makes the search unrealistically easy.
- **Where the answer is:** the file and, for you, the page or section, for every place needed. This checks whether search finds the right passages, and whether the answer cites them.
- **Must include:** the key facts a correct answer has to contain, as short phrases. This checks the answer itself.
- **Notes:** anything tricky (optional).

**How to write the two list fields** (the importer reads them literally)
- *Where the answer is*: one place per line (a `- ` bullet is fine) or separated by `;`. Write each place as `path/in/corpus.md (page or section)`. The path is **relative to the `evals/corpus/` folder**, with forward slashes, and must be a real file there; the part in parentheses is for you and is optional.
- *Must include*: short phrases, one per line or separated by `;`.
- A field can continue on the next lines. Don't add fields that are not in the blocks below.

**The ID decides the type** (S single-hop, M multi-hop, T multi-turn, U unanswerable), not the section heading. Number them in order and don't reuse an ID: S-002, S-003, ... Copy the block of the matching type for each new question.

**The split.** About 40% of each type becomes *development* items (used to pick thresholds and depths) and the rest *test* items (used for the reported score), decided by a fixed rule from the ID (D-065). So finish the set before anything is tuned, and don't renumber after that.

A worked example (this fenced block is ignored by the importer):

```
### S-017
- **Question:** How long does the router take to finish a factory reset?
- **Where the answer is:** router/manual.md (Reset > Factory reset)
- **Must include:** ten seconds; amber light
- **Notes:** the manual says "about ten seconds"; the question avoids the word "factory".
```

---

## Single-hop (~50)

### S-001
- **Question:**
- **Where the answer is:**
- **Must include:**
- **Notes:**

## Multi-hop (~15)

### M-001
- **Question:**
- **Where the answer is:**
- **Must include:**
- **Notes:**

## Multi-turn (~15)

### T-001
- **Turn 1 (the earlier question):**
- **Turn 2 (the follow-up that gets graded):**
- **Where the answer is:**
- **Must include:**
- **Notes:**

## Unanswerable (~20)

### U-001
- **Question:**
- **Why it can't be answered:**
- **Notes:**
