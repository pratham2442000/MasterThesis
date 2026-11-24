# Experiment 2 — Merge-then-Rerank (BM25 + LLM Fusion) Guide

> Core fusion experiment. Tests whether LLM query rewrites improve
> retrieval when merged *before* reranking.

---

## Goal

**Answer: Do LLM query rewrites improve recall, and which fusion method
best combines multi-rewrite result lists?**

The thesis hypothesis: a single query often misses relevant documents
because it uses only one phrasing. LLM rewrites paraphrase the query N
different ways, each retrieving slightly different candidate sets. Fusing
these sets should increase recall — but *how* you fuse matters.

This experiment sweeps `N` (number of rewrites) from 1→30 and compares
six fusion strategies, all followed by a reranker.

### Relationship to other experiments

| Experiment | What it shares | What's different |
|---|---|---|
| **Exp 1** | Same BM25 baseline, same reranker | Exp 1 sweeps cutoff c; Exp 2 sweeps N rewrites at fixed c=50 |
| **Exp 2a** | Same fusion methods, same N sweep | Exp 2a uses Dense retriever instead of BM25 |
| **Exp 2b** | Same rewrites, same fusion | Exp 2b reranks *per rewrite first*, then fuses (opposite order) |
| **Exp 3** | Same fusion methods | Exp 3 sweeps cutoff c at fixed N=10 (combines Exp 1 + Exp 2 axes) |

---

## What is being compared

### The Merge-then-Rerank (MTR) pattern

All fusion pipelines follow the same three-stage pattern:

```
Query → [N rewrites] → BM25 (per rewrite) → FUSE → Reranker → Evaluate
                                               ↑
                                        This step varies
```

1. **Expand**: `QueryRewriteTransformer` turns each query into N sub-queries
   (1 original + N-1 LLM rewrites)
2. **Retrieve**: BM25 retrieves `cutoff` docs per sub-query independently
3. **Fuse**: A fusion method merges N ranked lists into one candidate set
4. **Rerank**: Cross-encoder or MonoT5 scores the fused set
5. **Evaluate**: R@10, R@50, nDCG@10 on the reranked output

### Systems

| System | Fusion Method | How it merges N lists |
|---|---|---|
| **BM25** | None | Single query, no rewrites. Flat baseline. |
| **BM25 + Reranker** | None | Single query → rerank. Reranker-only baseline. |
| **DedupCutoff** | Pool + dedup + truncate | Union all docs, keep highest BM25 score per doc, cut to top-50 |
| **Borda** | Borda count voting | Each list votes: rank → normalized score. Scores summed across lists. |
| **BordaWeighted** | Weighted Borda | Same as Borda but `score = rank × exp(rank × 0.1)` — amplifies top positions |
| **CombMAX** | Max-score pooling | For each doc, keep maximum BM25 score across all lists |
| **RRF** | Reciprocal Rank Fusion | `score(d) = Σ 1/(k + rank_i)` across lists. k=60. |
| **Exp4Fuse** | Modified RRF (ACL 2025) | RRF + adaptive boost: `(n/10 + 1) × Σ w_i/(k + r_i)` where n = # lists containing doc |

All fusion systems pass their output through the same reranker.
The baselines (BM25, BM25+Reranker) are independent of N — they
appear as flat horizontal lines on the plots.

---

## How each fusion method works

### DedupCutoff — simple pooling

```
rewrite_0 → BM25 → [d1, d3, d7, d9, ...]
rewrite_1 → BM25 → [d2, d3, d5, d7, ...]
rewrite_2 → BM25 → [d1, d4, d6, d8, ...]
                      ↓
              Pool all → deduplicate (keep max BM25 score)
                      ↓
              Truncate to top-50 by BM25 score
```

Simple and fast. No voting, no rank aggregation. Just "grab everything,
keep the best BM25 score, cut to budget."

### Borda — rank-based voting

Each document gets a score based on its position in each list:

```
List 1: [d1, d3, d7] → d1=3/3, d3=2/3, d7=1/3
List 2: [d3, d2, d5] → d3=3/3, d2=2/3, d5=1/3
List 3: [d1, d4, d6] → d1=3/3, d4=2/3, d6=1/3
                          ↓
Final:  d1=2.0, d3=1.67, d2=0.67, d4=0.67, ...
```

Rewards documents ranked highly by *multiple* rewrites. Ignores raw BM25
scores — only positions matter.

### BordaWeighted — exponential Borda

Same as Borda but applies `weight = exp(normalized_rank × 0.1)` before
summing. Top-ranked docs get disproportionately higher scores. Amplifies
consensus at the top of each list.

### CombMAX — max-score pooling

```
d3 appears in list 1 (score 12.5) and list 2 (score 14.2)
→ CombMAX keeps score = 14.2
```

Conservative method. A document only needs to rank well in *one* rewrite
to survive. Useful when rewrites have low overlap.

### RRF — Reciprocal Rank Fusion

Standard fusion from Cormack et al. (2009):

```
score(d) = Σ_i  1 / (60 + rank_i(d))
```

k=60 dampens the impact of outlier ranks. Documents appearing in more
lists accumulate higher scores. Robust across diverse retrieval systems.

### Exp4Fuse — Adaptive Modified RRF

From Liu & Zhang (ACL 2025, arXiv:2506.04760):

```
score(d) = (n/10 + 1) × Σ_i  w_i / (k + rank_i(d))
```

Where `n` = number of distinct rewrite lists containing document `d`.
The `(n/10 + 1)` multiplier rewards documents found by multiple rewrites.
Route 0 (original query) gets weight `w1`, expanded routes get `w2`.

---

## Parameters

| Parameter | Value | Source |
|---|---|---|
| `N` (sweep) | [1, 4, 7, 10, 14, 17, 20, 23, 27, 30] | `np.linspace(1, 30, 10)` |
| Retrieval cutoff | 50 (fixed) | `EXPERIMENT_2_RETRIEVAL_CUTOFF` |
| Topic limit | 100 | `EXPERIMENT_2_TOPIC_LIMIT` |
| Eval metrics | R@50, R@10, nDCG@10, R@cutoff | `EXPERIMENT_2_EVAL_METRICS` |
| RRF k | 60 | Hardcoded in transformer |
| BordaWeighted weight | `exp(rank × 0.1)` | Hardcoded in pipeline |

### Why N=1→30?

- **N=1**: Original query only. Fusion has no effect. Should match BM25+Reranker baseline.
- **N=4–10**: Sweet spot. More diverse rewrites, diminishing returns not yet hit.
- **N=20–30**: Testing saturation. Does quality plateau, degrade, or keep improving?

### Why cutoff=50?

Fixed at 50 to isolate the N-rewrites variable. At c=50, BM25 retrieves
a moderate candidate set per sub-query. The fusion step can pool up to
`50 × N` unique docs before dedup/truncation.

Exp 3 later sweeps cutoff while holding N fixed — separating the two axes.

---

## Pipeline flow (detailed)

```
              ┌──────────────┐
    Query ──→ │ QueryRewrite │ ──→ N sub-queries (qid_0, qid_1, ..., qid_{N-1})
              │ Transformer  │     qid_0 = original query
              └──────────────┘     qid_1..N-1 = LLM rewrites
                     │
                     ▼
              ┌──────────────┐
              │   BM25 (×N)  │ ──→ N ranked lists, 50 docs each
              │   cutoff=50  │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   FUSION     │ ──→ 1 merged list per query (≤50 docs)
              │  (varies)    │
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │ AttachQuery  │ ──→ Re-attach original query text
              │ + GetText    │     (reranker needs original, not rewrites)
              └──────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   Reranker   │ ──→ Final reranked output
              │ (MonoT5 etc) │
              └──────────────┘
```

### Critical detail: AttachOriginalQuery

The reranker must score `(original_query, document)` pairs — NOT
`(rewrite, document)`. `AttachOriginalQuery` replaces the rewrite text
with the original query before passing to the reranker. Without this,
the reranker scores against a paraphrase, which biases results.

---

## Metrics explained

| Metric | What it measures in this context |
|---|---|
| **R@50** | Ceiling recall. "Did fusion find more relevant docs in the top-50 pool?" |
| **R@10** | Practical recall. "After reranking, are relevant docs in top 10?" |
| **nDCG@10** | Ranking quality. "Are the top-10 docs in the right order?" |
| **R@cutoff** | Same as R@50 here (cutoff=50). Redundant but consistent with Exp 1. |

### What to watch

- **R@50 should increase with N** — more rewrites = more unique docs in pool
- **R@10 may peak then plateau** — reranker can only promote from what fusion provides
- **nDCG@10 should benefit from better R@10** — more relevant docs available to rank
- **Flat lines** = fusion method not helping (or rewrites too similar)

---

## Cost model

| Component | Cost per query | Scales with |
|---|---|---|
| Query rewriting | Pre-computed (CSV lookup) | N (but free at runtime) |
| BM25 retrieval | ~1ms × N | N (one BM25 call per rewrite) |
| Fusion | ~0.1ms | N (in-memory operations) |
| Reranker | ~50ms × (unique docs after fusion) | Fusion output size |

BM25 cost is negligible. Reranker cost is dominant but bounded by
cutoff (≤50 docs after fusion truncation).

Total per query at N=30: ~30 BM25 calls + 1 rerank of 50 docs ≈ same
cost as reranking 50 docs alone. The extra BM25 calls are cheap.

---

## What the curves tell you

### Scenario 1: Fusion helps, plateau at N~10

```
     R@50
      │         ╭────────── RRF
      │     ╭──╯
      │   ╭─╯──────────── BordaW
      │  ╭╯
      │──╯─────────────── BM25+Reranker (flat)
      └──────────────────── N
        1    10    20    30
```

Fusion adds value up to ~10 rewrites, then saturates. Rewrites become
too similar to add novel documents. Best fusion method wins by N=10.

### Scenario 2: Diminishing returns with noise

```
     nDCG@10
      │   ╭╮
      │  ╭╯╰──╮
      │ ╭╯    ╰──────── RRF (peaks then drops)
      │╭╯
      │╯───────────────── BM25+Reranker (flat)
      └──────────────────── N
```

Too many rewrites dilute the candidate pool with noise. nDCG drops
because reranker sees more irrelevant docs. Sweet spot is clear.

### Scenario 3: Method-dependent behavior

```
     R@10
      │    ╭──────────── RRF (strong, stable)
      │   ╭╯
      │  ╭╯──────────── CombMAX (moderate)
      │──╯
      │────────────────── DedupCutoff (weak — no rank info)
      └──────────────────── N
```

Methods using rank information (RRF, Borda) outperform naive pooling
(DedupCutoff). Shows that *how* you combine matters, not just *that*
you combine.

---

## Datasets

| Dataset | Corpus | Query behavior | Expected fusion value |
|---|---|---|---|
| **Wiki-MuSiQue** | ~500K | Multi-hop, compositional | High — rewrites capture sub-questions |
| **MSMARCO** | 8.8M | Factoid web queries | Moderate — single phrasing often sufficient |
| **DPR-w100** | 21M | TriviaQA factoid | Moderate — lexical overlap already good |

---

## Key invariant

> **All fusion methods see the same BM25 results per rewrite.**
>
> The only variable is *how* they merge N ranked lists into one.
> The reranker is applied *after* fusion (Merge-then-Rerank = MTR).
> Exp 2b tests the opposite order (Rerank-then-Merge = RTM).
