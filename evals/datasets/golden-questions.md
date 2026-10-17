# Golden questions (blank template)

The hand-written test questions for the golden corpus (`evals/corpus/`). They check how Geekbot behaves on **our** documents: "system-specific behavioral evaluation" (PLAN.md §9). They say nothing about retrieval quality in general; the BEIR benchmarks cover that.

Fill this in later (DECISIONS.md D-032). Step S2.4 turns it into `golden.jsonl` and settles the open details listed in D-033. The format below is a starting point, and S2.4 may adjust it.

**Target: ~100 questions**

| Type | Target | What it tests |
|---|---|---|
| Single-hop | ~50 | the answer is in one place in one document |
| Multi-hop | ~15 | the answer needs facts from two or more places |
| Multi-turn | ~15 | a follow-up that only makes sense after an earlier question |
| Unanswerable | ~20 | the documents don't contain the answer, so Geekbot should say it doesn't know |

**What each field is for**
- **Question:** written the way a real user would ask it. Don't copy a sentence from the document word for word; that makes the search unrealistically easy.
- **Where the answer is:** file name + page or section, for every place needed. This checks whether search finds the right passages, and whether the answer cites them.
- **Must include:** the key facts a correct answer has to contain, as short phrases. This checks the answer itself.
- **Notes:** anything tricky (optional).

Copy the block of the matching type for each new question, and number the IDs in order (S-002, S-003, …).

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
- **Where the answer is:** (two or more places)
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
- **Why it can't be answered:** (what the documents lack; a topic close to the documents makes a better test)
- **Notes:**
