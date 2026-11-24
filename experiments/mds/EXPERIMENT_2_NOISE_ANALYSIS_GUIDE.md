# Experiment 2 — Noise Analysis (Oracle & Dedup Oracle) Guide

> Oracle baselines for the rewrite-fusion experiments. Answers: "What's
> the ceiling if we could pick the perfect rewrite every time?"

---

## Goal

**Answer: How much of the fusion methods' performance gap is due to
imperfect merging vs. inherent limitations of the rewrites themselves?**

This experiment constructs two upper-bound baselines that bracket what
fusion methods *should* be achieving, then overlays Experiment 2's fusion
results for direct comparison.

1. **Oracle** — for each query, picks the single best rewrite (per metric).
   "If we had a perfect rewrite selector, how good would we be?"
2. **Dedup Oracle** — merges all N sub-query result lists, deduplicates,
   and feeds the *entire* growing pool to the reranker with no truncation.
   "If we gave the reranker every document any rewrite found, how good?"

### Relationship to other experiments

| Experiment | What it shares | What's different |
|---|---|---|
| **Exp 2** | Same N sweep, same cutoff, same BM25 + reranker | Exp 2 tests fusion methods; this experiment tests oracle baselines |
| **Exp 2b** | Same rewrites | Exp 2b reranks per-rewrite first (RTM); this uses MTR |
| **Exp 1** | Same reranker, same metrics | Exp 1 sweeps cutoff c; this sweeps N rewrites |
| **Exp 3** | Same fusion methods | Exp 3 sweeps c at fixed N; this sweeps N at fixed c |

---

## What is being compared

### Three system types on the same plot

| System | What it does | Expectation |
|---|---|---|
| **Oracle** | Best *single* rewrite per query per metric | Upper bound for rewrite *selection* strategies. Should be ≥ any fusion method. |
| **Dedup Oracle** | All docs from all rewrites → dedup (no cutoff) → reranker | Upper bound for *pool size*. Reranker sees max unique docs. Should be ≥ Oracle. |
| **Fusion methods** (from Exp 2) | RRF, Borda, CombMAX, etc. overlay | Real-world fusion performance. Gap to Oracle = room for improvement. |

### Why two oracles?

```
Oracle:       Pick best single rewrite → BM25 top-50 → reranker
              Ceiling: ~50 docs per query (bounded by cutoff)

Dedup Oracle: All N rewrites → BM25 top-50 each → merge + dedup → reranker
              Ceiling: up to 50×N unique docs per query (unbounded)
```

The **Oracle** answers: "Do we even need fusion? Could a smart selector
just pick the right rewrite?"

The **Dedup Oracle** answers: "Does the reranker benefit from seeing
more documents, even without smart fusion scoring?"

---

## How each baseline works

### Oracle — best single rewrite

For each original query and each metric (R@10, R@50, nDCG@10):

```
Query Q → N rewrites (Q_0, Q_1, ..., Q_{N-1})
              │
              ▼
     Each rewrite → BM25 (cutoff=50) → AttachOriginalQuery → GetText → Reranker
              │
              ▼
     Evaluate each rewrite independently against qrels
              │
              ▼
     Oracle score = MAX across all N rewrites for this query
```

Key details:
- `Q_0` is always the original query
- `AttachOriginalQuery` is applied BEFORE reranking so the reranker
  scores `(original_query, document)` — not `(rewrite, document)`
- Oracle recall: for R@k, compute hits/relevant per rewrite, take max
- Oracle nDCG: run `pt.Experiment` per rewrite per query, take max

### Dedup Oracle — unconstrained pool

```
Query Q → N rewrites → BM25 (cutoff=50 each)
                              │
                              ▼
                    MergeRewrites (strip _0, _1, ...)
                              │
                              ▼
                    AttachOriginalQuery
                              │
                              ▼
                    Dedup (keep max score, NO cutoff)
                              │
                              ▼
                    GetText → Reranker → Evaluate
```

The pipeline reports `avg_docs_per_query` — how many unique docs the
reranker sees after dedup. This number grows with N:
- N=1: ~50 docs (one BM25 call)
- N=10: ~200–350 docs (depending on overlap between rewrites)
- N=30: ~400–800 docs (diminishing marginal unique docs)

The doc count is annotated on plots for each N value.

---

## Pipeline flow (detailed)

```
              ┌──────────────────┐
    Query ──→ │ QueryRewrite     │ ──→ N sub-queries (qid_0 ... qid_{N-1})
              │ Transformer      │
              └──────────────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   BM25 (×N)      │ ──→ N ranked lists, 50 docs each
              │   cutoff=50      │
              └──────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    [ORACLE PATH]             [DEDUP ORACLE PATH]
          │                         │
    AttachOriginalQuery       MergeRewrites
          │                   (strip _0 suffix)
          ▼                         │
    GetText                   AttachOriginalQuery
          │                         │
    Reranker                  Dedup (no cutoff!)
          │                         │
    Eval per rewrite          GetText
    Take MAX                        │
                              Reranker
                                    │
                              pt.Experiment eval
```

### Critical: AttachOriginalQuery placement

In the **Oracle** path, `AttachOriginalQuery` runs BEFORE GetText and
reranker — each rewrite's BM25 results are reranked using the original
query. This matches Experiment 2's pipeline.

In the **Dedup Oracle** path, `MergeRewrites` runs first (collapses
sub-query IDs), then `AttachOriginalQuery` maps the merged qid back
to the original query text.

---

## Parameters

| Parameter | Value | Source |
|---|---|---|
| `N` (sweep) | [1, 4, 7, 10, 14, 17, 20, 23, 27, 30] | `EXPERIMENT_2_N_REWRITES` |
| Retrieval cutoff | 50 (fixed) | `EXPERIMENT_2_RETRIEVAL_CUTOFF` |
| Topic limit | 100 | `EXPERIMENT_2_TOPIC_LIMIT` |
| Eval metrics | R@50, R@10, nDCG@10, R@cutoff | `EXPERIMENT_2_EVAL_METRICS + [R@cutoff]` |

### Why same parameters as Exp 2?

Direct comparability. Oracle and Dedup Oracle results are plotted on the
same axes as Exp 2 fusion methods. Different parameters would make
the overlay meaningless.

---

## Metrics explained

| Metric | Oracle interpretation | Dedup Oracle interpretation |
|---|---|---|
| **R@10** | Best rewrite's top-10 recall | Reranker's top-10 from full pool |
| **R@50** | Best rewrite's top-50 recall | Reranker's top-50 from full pool |
| **nDCG@10** | Best rewrite's ranking quality | Ranking quality over full pool |
| **R@cutoff** | Same as R@50 (cutoff=50) | Same as R@50 |

### What to watch

- **Gap between Oracle and best fusion method** = how much room exists
  for smarter rewrite selection. Large gap → fusion is suboptimal.
- **Gap between Dedup Oracle and Oracle** = value of pool expansion.
  Large gap → more unique docs help the reranker.
- **Dedup Oracle doc count growth** = how much overlap exists between
  rewrites. Slow growth → rewrites are too similar.
- **Fusion methods approaching Oracle** = fusion is doing well.
  Exceeding Oracle → fusion combines *complementary* docs that no single
  rewrite alone could provide.

---

## Reranker consistency check

Before overlaying Exp 2 results, the code validates that the loaded
results used the **same reranker** as the current configuration. If
Exp 2 was run with MonoT5 but constants now say MiniLM-L6, it raises
`RuntimeError`. This prevents silently mixing results from different
rerankers.

The check reads `experiment_2_results_metadata.json` from the latest
Exp 2 run directory.

---

## Cost model

| Component | Cost per query | Scales with |
|---|---|---|
| Query rewriting | Pre-computed (CSV) | Free at runtime |
| BM25 retrieval | ~1ms × N | N |
| Dedup Oracle reranking | ~10ms × (unique docs after dedup) | Grows with N (unbounded!) |
| Oracle reranking | ~10ms × 50 × N | N (one rerank per rewrite) |
| Oracle nDCG eval | ~5ms × N × (pt.Experiment overhead) | N (one eval per rewrite per query) |

**Oracle is expensive.** For each N and each query, it runs `pt.Experiment`
per sub-query to compute nDCG. At N=30, 100 queries → 3000 mini-experiments.

**Dedup Oracle gets expensive at large N.** At N=30, the reranker may
score 500–800 docs per query instead of the usual 50.

---

## What the curves tell you

### Scenario 1: Oracle well above fusion methods

```
     R@10
      │    ★──────────── Oracle (ceiling)
      │
      │        ╭──────── RRF (best fusion)
      │     ╭──╯
      │   ╭─╯
      │  ╭╯──────────── DedupCutoff (worst fusion)
      └──────────────────── N
        1    10    20    30
```

Large gap between Oracle and fusion. Fusion methods are leaving
performance on the table. A smarter rewrite selection strategy
could outperform naive fusion.

### Scenario 2: Dedup Oracle dominates both

```
     R@50
      │  ✕────────────── Dedup Oracle (unconstrained pool)
      │
      │    ★──────────── Oracle
      │        ╭──────── RRF
      │     ╭──╯
      └──────────────────── N
        1    10    20    30
```

The reranker benefits enormously from seeing all documents, even
without smart scoring. Pool size matters more than fusion strategy.
Implication: increase cutoff budget before optimizing fusion.

### Scenario 3: All converge at large N

```
     nDCG@10
      │  ✕  ★  ╭─────── All converge
      │  │  │ ╭╯
      │  │ ╭╯╯
      │  ╭╯╯
      │ ╭╯╯
      └──────────────────── N
        1    10    20    30
```

At large N, rewrites saturate — all find similar doc pools. Fusion
method choice stops mattering. The interesting region is N < 15.

---

## Plot annotations

The Dedup Oracle line includes annotations of the format `Xd` at each
data point, where X is the average number of unique documents per query
after dedup. For example, `247d` means the reranker scored ~247 docs
per query at that N value.

This connects pool size to metric values: if R@10 improves from
`150d` → `300d`, pool expansion is driving the gain.

---

## Datasets

| Dataset | Expected Oracle gap | Expected Dedup Oracle behavior |
|---|---|---|
| **Wiki-MuSiQue** | Large — multi-hop queries have diverse rewrites | Strong growth — rewrites find different evidence passages |
| **MSMARCO** | Moderate — factoid queries have limited paraphrases | Moderate growth — rewrites overlap heavily |
| **DPR-w100** | Moderate — lexical queries have limited diversity | Slow growth — BM25 finds similar docs for similar phrasings |

---

## Key invariant

> **Oracle and Dedup Oracle are upper bounds, not practical systems.**
>
> Oracle requires knowing which rewrite is best *a priori* — impossible
> without running all rewrites and evaluating. Dedup Oracle has unbounded
> reranker cost — impractical at scale.
>
> Their purpose is to diagnose *why* fusion methods succeed or fail.
> If fusion matches Oracle → fusion is near-optimal.
> If fusion is far below Dedup Oracle → the bottleneck is pool size, not
> fusion scoring.
