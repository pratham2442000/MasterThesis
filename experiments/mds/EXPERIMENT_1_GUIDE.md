# Experiment 1 — BM25 & Dense Cutoff Sweep Guide

> Baseline experiment. Establishes retrieval quality curves that all
> other experiments (2, 2a, 3, 4, 7) are compared against.

---

## Goal

**Answer: How does retrieval cutoff (c) affect downstream quality, and
does reranking change the shape of that curve?**

Cutoff `c` = number of documents initially retrieved before any reranking.
Small `c` is cheap but may miss relevant docs. Large `c` retrieves more
candidates but adds noise and compute cost.

This experiment sweeps `c` from 10 → 1000 (log-spaced) and measures
Recall and nDCG at fixed evaluation depths (R@10, R@50, nDCG@10) plus
a dynamic R@c that tracks ceiling recall at each budget.

The resulting curves are the **reference baselines** for the thesis:
- Experiment 2/2a: "Do LLM rewrites + fusion beat these baselines?"
- Experiment 4: "Does ORE Adaptive's graph navigation beat these curves?"
- Experiment 7: "Does ReformIR's reformulation beat these curves?"

---

## What is being compared

Six systems in a 2×3 grid:

| | No Reranker | + MonoT5 | + MiniLM-L6 |
|---|---|---|---|
| **BM25** | BM25 | BM25 → MonoT5 | BM25 → MiniLM-L6 |
| **Dense (TCT-ColBERT)** | Dense | Dense → MonoT5 | Dense → MiniLM-L6 |

### First-stage retrievers

| Retriever | Model | What it does |
|---|---|---|
| **BM25** | Terrier default | Lexical bag-of-words. Fast, no GPU needed. |
| **Dense** | `castorini/tct_colbert-v2-hnp-msmarco` | Bi-encoder dense retrieval from pre-built FlexIndex. |

BM25 and Dense retrieve fundamentally different candidate sets:
- BM25 finds docs with exact term overlap → strong on keyword queries
- Dense finds docs with semantic similarity → strong on paraphrase queries
- Their union is often better than either alone (motivates Exp 2a)

### Second-stage rerankers

| Reranker | Model | Type | Batch Size |
|---|---|---|---|
| **MonoT5** | `castorini/monot5-base-msmarco` | Seq2seq pointwise | 64 |
| **MiniLM-L6** | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder pairwise | 64 |

Both rerankers score all `c` candidates. No approximation, no budget
tricks — pure brute-force reranking. This is the **upper bound** for
what a reranker can achieve at budget `c`.

---

## How it works

### Pre-computation trick

Experiment 1 avoids redundant retrieval by pre-computing everything
at `MAX_C = 1000` once:

```
1. Run BM25 at MAX_C → bm25_full_df (1000 docs per query)
2. Run Dense at MAX_C → dense_full_df (1000 docs per query)
3. Get text for all retrieved docs → text_lookup dict
4. For each c in sweep:
   - Truncate bm25_full_df to top-c per query
   - Truncate dense_full_df to top-c per query
   - Attach text, run rerankers
```

This means BM25 and Dense retrieval happens **once**, not 10× per sweep.
Only reranking repeats per `c` (with smaller inputs at small `c`).

### What rerankers see

At each cutoff `c`, both rerankers receive:
- The same `c` documents from the first-stage retriever
- The document text attached from the pre-computed `text_lookup`
- No query reformulation, no expansion, no graph — just raw `c` candidates

This is the simplest possible pipeline: **retrieve → rerank → evaluate**.

---

## Parameters

| Parameter | Value | Source |
|---|---|---|
| `c` (sweep) | [10, 16, 27, 46, 77, 129, 215, 359, 599, 1000] | `np.geomspace(10, 1000, 10)` |
| Topic limit | 100 | `EXPERIMENT_1_TOPIC_LIMIT` |
| Eval metrics | R@10, R@50, nDCG@10 | `EXPERIMENT_1_EVAL_METRICS` |
| Dynamic metric | R@c (varies per sweep step) | `extra_metrics_fn` |
| MonoT5 batch | 64 | `T5_BATCH_SIZE` |
| CrossEncoder batch | 64 | `CROSS_ENCODER_BATCH_SIZE` |

### Why log-spaced c values?

`geomspace(10, 1000, 10)` gives denser samples at small `c` where the
curve changes fastest (steep recall gains from 10→100) and sparser
samples at large `c` where diminishing returns flatten the curve.

This is deliberate: the interesting region for cost-quality tradeoffs
is c ∈ [10, 200], not c ∈ [800, 1000].

---

## Metrics explained

| Metric | What it measures | Why it matters |
|---|---|---|
| **R@10** | Fraction of relevant docs in top 10 | RAG systems feed ~5–10 docs to the reader. This is the "useful recall" |
| **R@50** | Fraction of relevant docs in top 50 | Captures broader retrieval quality |
| **nDCG@10** | Ranking quality of top 10 (position-weighted) | Rerankers primarily improve *ranking* within fixed candidate sets |
| **R@c** | Fraction of relevant docs in top c | Ceiling recall — "if we kept everything, how much did we find?" |

### Key relationships to watch

- **R@c grows monotonically** with c (more candidates → more recall, by definition)
- **R@10 can decrease** if reranker pushes noise into top 10 at large c
- **nDCG@10 gap** between BM25 and BM25+reranker shows reranker value
- **Dense vs BM25 gap** shows whether semantic retrieval helps for this dataset

---

## Cost model

| Component | Cost per query | Scales with |
|---|---|---|
| BM25 retrieval | ~1ms | c (but pre-computed) |
| Dense retrieval | ~5ms | c (but pre-computed) |
| Text lookup | ~0.1ms × c | c |
| MonoT5 reranking | ~50ms × c | c (dominant at large c) |
| MiniLM-L6 reranking | ~10ms × c | c |

At c=1000, MonoT5 scores 1000 docs per query — ~50 seconds per query.
At c=10, it scores 10 docs — ~0.5 seconds. The 100× difference is why
cutoff matters.

This is exactly what Exp 4 (ORE) and Exp 7 (ReformIR) try to improve:
**achieve large-c quality at small-c cost** via selective scoring.

---

## What the curves tell you

### Scenario 1: Reranker helps at all c

```
     R@10
      │    ╭──── BM25 + MonoT5
      │   ╭╯
      │  ╭╯──── BM25
      │ ╭╯
      └──────────── c
```

Reranker consistently improves ranking quality. Gap = reranker's value.
This is the typical case for well-matched rerankers (trained on same domain).

### Scenario 2: Reranker hurts at small c

```
     nDCG@10
      │      ╭── BM25 + MonoT5
      │  ╮  ╭╯
      │  │╰╯ ── BM25
      │  ╰╮
      └──────────── c
```

At very small c (10–20), the reranker may shuffle a few good docs out of
top 10 because it only sees a tiny, noisy candidate set. Quality dips
then recovers as c grows. This signals the reranker needs minimum `c` to
be useful.

### Scenario 3: Dense dominates BM25

```
     R@10
      │  ╭────── Dense
      │ ╭╯
      │╭╯─────── BM25
      └──────────── c
```

Dense retriever finds semantically relevant docs that BM25 misses.
Strong signal to use Dense as first-stage in Exp 2a.

---

## Datasets

| Dataset | Corpus size | Query type | Expected behavior |
|---|---|---|---|
| **Wiki-MuSiQue** | ~500K passages | Multi-hop compositional | BM25 struggles; rerankers help significantly |
| **MSMARCO-Passage** | 8.8M passages | Web search (factoid) | BM25 strong baseline; rerankers give modest gains |
| **DPR-w100** | 21M Wikipedia passages | TriviaQA (factoid) | Dense may match BM25; large corpus stresses retrieval |

---

## Key invariant

> **Experiment 1 is pure brute-force.**
>
> Every doc in the top-c gets a real reranker score. No approximation,
> no graph, no reformulation. The curves are the ceiling that
> budget-constrained methods (ORE, ReformIR) try to match cheaply.
