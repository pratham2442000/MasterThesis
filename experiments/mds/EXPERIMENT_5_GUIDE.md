# Experiment 5 — Full RAG Pipeline Cutoff Sweep Guide

> First end-to-end RAG experiment. Measures **answer generation quality**
> (not just retrieval) across BM25 cutoffs.

---

## Goal

**Answer: How does BM25 retrieval cutoff (c) affect the quality of
generated answers in a full Retrieve → Rerank → Generate pipeline?**

Experiments 1–4 measure retrieval quality (R@k, nDCG). This experiment
closes the loop: retrieved passages are fed to an LLM reader that
generates an answer, and that answer is evaluated against gold answers.

The key question: does better retrieval (higher c) always produce better
answers, or does noise from large candidate pools degrade generation?

### Relationship to other experiments

| Experiment | What it shares | What's different |
|---|---|---|
| **Exp 1** | Same BM25 + reranker pipeline, same c sweep | Exp 1 stops at retrieval metrics; Exp 5 adds generation + answer eval |
| **Exp 2** | Same reranker | Exp 2 sweeps N rewrites; Exp 5 sweeps cutoff c |
| **Exp 4** | Same c sweep range | Exp 4 uses ORE graph navigation; Exp 5 uses vanilla BM25 |
| **Exp 6** | Same RAG reader + eval metrics | Exp 6 adds LLM query rewrites + fusion before generation |

---

## What is being compared

### Two systems

| System | Pipeline | Description |
|---|---|---|
| **Full_RAG_with_reranker** | BM25 → GetText → Reranker → Concat top-5 → Reader | Full pipeline with cross-encoder reranking |
| **Full_RAG_without_reranker** | BM25 → GetText → Concat top-5 → Reader | Ablation: skip reranking, feed raw BM25 top-5 to reader |

Both systems sweep the same BM25 cutoff values. The difference isolates
the reranker's contribution to answer quality.

### The RAG pipeline pattern

```
Query → BM25 (cutoff=c) → GetText → [Reranker] → Concatenator (top-5) → Reader → Answer
                                       ↑                                    ↑
                                  Optional step                     Llama-3.2-3B-Instruct
```

1. **Retrieve**: BM25 retrieves `c` candidates
2. **Attach text**: Document text fetched from index/corpus
3. **Rerank** (optional): Cross-encoder scores all `c` candidates
4. **Concatenate**: Top-5 passages merged into prompt context
5. **Generate**: Llama-3.2-3B-Instruct produces a short answer
6. **Evaluate**: EM, F1, LLMasJudge against gold answers

---

## How it works

### Gold answer construction

Unlike Experiments 1–4 which evaluate against standard qrels (docno →
relevance label), Experiment 5 evaluates generated text against gold
answer strings.

```
topics_df → get_gold_answers() → gold_answers_df
                                    │
                                    ├── qid
                                    ├── gold_answer  (ground truth string)
                                    ├── text = gold_answer  (for LLMasJudge)
                                    └── relevance = 3       (LLMasJudge threshold)
```

For DPR-W100 (TriviaQA), multiple gold answers per query are expanded
into separate rows. EM/F1 take max() across rows per qid.

### Backend sharing trick

The RAG reader (Llama-3.2-3B) and LLMasJudge both need an LLM backend.
Loading two copies would OOM a single GPU. Experiment 5 shares the raw
`HuggingFaceBackend` between both:

```python
reader, raw_backend = get_reader(live=True)
eval_metrics = get_experiment_5_eval_metrics(
    shared_backend=raw_backend,   # ← reuse, don't reload
    topics_df=topics,
)
```

### Passage formatting

The `Concatenator` merges the top-5 passages with bullet-point format:

```
- passage 1 text here
- passage 2 text here
- passage 3 text here
- passage 4 text here
- passage 5 text here
```

This formatted context is injected into the Llama prompt template
(`RAG_PROMPT`) which instructs the model to answer in 1–5 words using
only the provided passages.

---

## Parameters

| Parameter | Value | Source |
|---|---|---|
| `c` (sweep) | [10, 16, 27, 46, 77, 129, 215, 359, 599, 1000] | `np.geomspace(10, 1000, 10)` |
| Topic limit | 100 | `EXPERIMENT_5_TOPIC_LIMIT` |
| Top-K passages to reader | 5 | `EXPERIMENT_5_TOP_K` |
| Eval metrics | EM, F1, LLMasJudge | `get_experiment_5_eval_metrics()` |
| Reader model | `meta-llama/Llama-3.2-3B-Instruct` | `RAG_MODEL_NAME` |
| Max new tokens | 64 | `RAG_MODEL_MAX_NEW_TOKENS` |
| Reranker | MiniLM-L6 cross-encoder | `RERANKER_TYPE = "cross_encoder"` |

### Why sweep c for RAG?

- **Small c (10–30)**: Few candidates. Reranker sees limited pool. If
  the relevant passage isn't in the top-c, it can never reach the reader.
  Answer quality bottlenecked by recall.
- **Medium c (50–200)**: Sweet spot. Enough candidates for reranker to
  find relevant passages. Top-5 after reranking likely contains gold.
- **Large c (300–1000)**: Diminishing returns. Reranker sees many
  irrelevant docs. Top-5 quality may plateau or even degrade if reranker
  is confused by noise.

### Why top-K = 5?

The reader prompt is length-limited. 5 passages balance context richness
against noise. More passages dilute signal; fewer miss supporting evidence.
This matches `RAG_CONCATENATOR_TOP_K` in constants.

---

## Pipeline flow (detailed)

```
              ┌──────────────┐
    Query ──→ │   BM25       │ ──→ c candidates (ranked by BM25 score)
              │  cutoff = c  │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   GetText    │ ──→ Attach document text from corpus/index
              └──────────────┘
                     │
            ┌────────┴────────┐
            ▼                 ▼
     [WITH RERANKER]    [WITHOUT RERANKER]
     ┌──────────────┐        │
     │  CrossEncoder │        │
     │  (MiniLM-L6) │        │
     └──────────────┘        │
            │                 │
            ▼                 ▼
     ┌──────────────┐  ┌──────────────┐
     │ Concatenator  │  │ Concatenator  │
     │   top-5       │  │   top-5       │
     └──────────────┘  └──────────────┘
            │                 │
            ▼                 ▼
     ┌──────────────┐  ┌──────────────┐
     │   Reader     │  │   Reader     │
     │  (Llama 3.2) │  │  (Llama 3.2) │
     └──────────────┘  └──────────────┘
            │                 │
            ▼                 ▼
       EM / F1 / LLMasJudge evaluation
```

### Critical detail: top-5 selection

Without the reranker, the Concatenator takes BM25's top-5 directly.
With the reranker, it takes the reranker's top-5 (potentially very
different documents). The reranker's value is measured by whether its
top-5 produces better answers.

---

## Metrics explained

| Metric | What it measures | How it works |
|---|---|---|
| **EM** (Exact Match) | Binary: does generated answer exactly match gold? | Normalized string comparison. Strict — partial matches score 0. |
| **F1** | Token-level overlap between generated and gold answer | Precision × Recall of word tokens. Forgiving of minor differences. |
| **LLMasJudge** | Semantic correctness judged by an LLM | Llama-3.2-3B rates whether generated answer is correct given gold answer. Handles paraphrases, synonyms. |

### What to watch

- **EM is harsh** — expect low absolute scores. Small wording differences
  → 0. Track *relative* differences between systems.
- **F1 tracks EM loosely** — if EM jumps, F1 jumps. F1 gives credit for
  partial matches.
- **LLMasJudge is the most informative** — handles "42" vs "forty-two"
  and similar semantic equivalences that EM/F1 miss.
- **Reranker gap** — difference between with/without reranker shows
  whether reranking actually helps answer quality, not just retrieval.

---

## Cost model

| Component | Cost per query | Scales with |
|---|---|---|
| BM25 retrieval | ~1ms | c |
| Text lookup | ~0.1ms × c | c |
| Reranker (MiniLM-L6) | ~10ms × c | c (scores all c candidates) |
| Concatenator | ~0.01ms | Fixed (always selects 5) |
| Reader (Llama-3.2-3B) | ~200–500ms | Fixed (always processes 5 passages) |
| LLMasJudge eval | ~200–500ms | Fixed per query |
| EM / F1 eval | ~0.01ms | Fixed per query |

Reader and LLMasJudge are the dominant costs, but they're **fixed** —
they always process 5 passages regardless of c. The reranker cost scales
with c. Total: reranker-dominated at large c, reader-dominated at small c.

At c=1000 with reranker: ~10s reranking + ~0.5s generation = ~10.5s per query.
At c=10 with reranker: ~0.1s reranking + ~0.5s generation = ~0.6s per query.

---

## What the curves tell you

### Scenario 1: Reranker helps answer quality

```
     EM
      │        ╭────── With Reranker
      │     ╭──╯
      │   ╭─╯
      │  ╭╯──────── Without Reranker
      │──╯
      └──────────────── c
        10   100   1000
```

Reranker selects better passages for the reader. Gap = reranker's value
for end-to-end RAG. This is the expected scenario.

### Scenario 2: Answer quality plateaus early

```
     F1
      │     ╭──────────── With Reranker (flat after c~50)
      │   ╭─╯
      │  ╭╯
      │ ╭╯──────────── Without Reranker (flat after c~100)
      │╭╯
      └──────────────── c
        10   100   1000
```

Once the top-5 contains the right passages, more candidates don't help.
The reader only sees 5 passages — extra retrieval is wasted. This signals
that small c + reranker matches large c alone.

### Scenario 3: Large c hurts without reranker

```
     EM
      │    ╭────────── With Reranker (stable)
      │   ╭╯
      │  ╭╯
      │╭─╯╮
      │╯   ╰──────── Without Reranker (degrades!)
      └──────────────── c
        10   100   1000
```

Without reranking, BM25's top-5 at large c includes noise from deeper
ranks. The reader gets confused by irrelevant passages. The reranker
prevents this by promoting relevant docs to top-5.

---

## Datasets

| Dataset | Corpus | Answer type | Expected behavior |
|---|---|---|---|
| **Wiki-MuSiQue** | ~500K | Multi-hop compositional | Hard — multiple passages needed for answer. Reranker crucial. |
| **MSMARCO** | 8.8M | Factoid web queries | Moderate — single passage often sufficient. Reader should do well at moderate c. |
| **DPR-w100** | 21M | TriviaQA factoid (multi-answer) | EM/F1 take max() over gold answers. Lexical match often sufficient. |

---

## Key invariant

> **Experiment 5 is the simplest RAG pipeline.**
>
> No query rewrites, no fusion, no graph navigation. Just
> BM25 → (Reranker) → top-5 → Reader → Answer.
>
> The curves are the **RAG baseline** that Experiment 6 (RAG + rewrites)
> tries to beat by adding query expansion before retrieval.
