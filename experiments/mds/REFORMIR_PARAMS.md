# ReformIR Parameter Guide — Experiment 7

> Paper: ReformIR (DC-92), code in `ReformIR-DC92/`

---

## What each parameter does

### `cross_enc_budget` (ce, default 7)

Controls how many **batches** of real cross-encoder calls are made per query.

The loop condition in `reformir.py` line 291:
```python
if num_batch < self.cross_enc_budget:
    # real scorer call — expensive
    reranked_batch = self.scorer(batch)
    # fit λ parameters (alpha, delta, bm25_lambdas) via lsq_linear
else:
    # estimate only — just sort by estimated_scores, no CE call
```

At `budget=100, ce=7, batch_size=16`:  
`ce × batch_size = 7 × 16 = 112` docs get real CE scores — already ≥ budget,  
so at c=100 **all documents are real-scored**. This is the calibration phase.

At `budget=50, ce=4, batch_size=16`:  
`4 × 16 = 64` docs get real CE scores — the README explicitly shows:
```
--ce 4 --budget 50   (DL22)
--ce 7 --budget 100  (DL22)
```
The paper authors chose ce=4 at budget=50 and ce=7 at budget=100. The **ratio is identical**: `ce / budget ≈ 0.08`, meaning ~8% of budget determines the real-scored fraction. This is NOT a coincidence — it's the design point.

**Key difference vs ORE**: In ReformIR, `num_batch` counts total loop iterations (not just real-scored ones). Each iteration scores exactly `batch_size` docs. So `ce` controls real CE calls as a fraction of the total iterations needed to exhaust the budget.

### `batch_size` (REFORMIR_BATCH_SIZE = 16)

**The number of documents processed per iteration of the active-learning loop.**

> **CRITICAL**: This is NOT the scorer's inference batch size (`T5_BATCH_SIZE = 64`
> or `CROSS_ENCODER_BATCH_SIZE = 64`). Those control how the model internally
> batches forward passes. `REFORMIR_BATCH_SIZE` controls how many documents are
> selected, scored (or estimated), and added to the result set per loop iteration.

The paper used `batch_size=16` in all experiments. Each iteration:
1. Selects the top `batch_size` candidates by estimated score
2. If `num_batch < ce`: runs real CE scoring on those docs
3. If `num_batch >= ce`: uses estimated scores directly (no CE call)
4. Updates regression weights (alpha, delta, bm25_lambdas) via `lsq_linear`

### `top_s` (default 15 in code, `--s 30` in run scripts)

**The number of highest-scored results used as RM3 feedback documents.**

```python
cluster_heads = [doc for doc, _ in Counter(scores).most_common(self.top_s)]
rm3neighbors, rm3neighbor_scores = self.generate_rm3_query(
    qid, query, Counter(scores).most_common(self.top_s)
)
```

RM3 (pseudo-relevance feedback) takes the top-`top_s` docs seen so far, extracts the top-10 terms from those `top_s` docs, and creates an expanded query to retrieve new candidate documents via BM25. This expanded candidate pool is then used to prioritise the next batch.

**Important**: `top_s` only kicks in from `num_batch > 0` (after the first iteration). The first batch is scored purely by initial BM25 scores.

**Discrepancy**: The ctor default is `top_s=15` but the run scripts use `--s 30`. In `run_dl22.py`, `--s` is parsed but **never passed to `ReformIR`** — the ctor default of 15 is silently used. This means the paper's own run scripts have a bug and always used `top_s=15`.

### `reform_bm25` (num_results — scales with c)

In `experiment_7.py`, the internal BM25 retriever used for RM3 expansion uses
`num_results=max(128, c)`, scaled per sweep step so the RM3 pool is proportional
to the budget.

In `generate_rm3_query` (line 70): `documents = results.docno.tolist()[:128]` — also hardcoded to cap at 128.

---

## Speed vs Quality: cost per parameter

| Parameter | Operation | Cost |
|---|---|---|
| `ce` (cross_enc_budget) | Real cross-encoder calls × batch_size | **Expensive — dominant cost** |
| `top_s` | `Counter.most_common(n)` + RM3 query expansion | **Moderate** — RM3 fires a full BM25 query each iteration after batch 0 |
| `reform_bm25 num_results` | BM25 retrieval for reformulated queries | Moderate — one BM25 call per reformulation |

**RM3 cost caveat**: Unlike ORE's LAFF lookup (precomputed array), ReformIR's `top_s` triggers an actual BM25 RM3 query retrieval every iteration (from `num_batch=1` onwards). So `top_s` is **not** as cheap as ORE's `top_s`. Growing `top_s` means RM3 feeds from more feedback docs, which may improve term selection quality but doesn't change the retrieval cost significantly (RM3 retrieves a fixed 128 docs regardless of `top_s`).

---

## What the paper actually used (from README)

| budget (c) | ce | top_s | batch_size |
|---|---|---|---|
| 50 | 4 | 15 (ctor default) | 16 |
| 100 | 7 | 15 (ctor default) | 16 |

The ratio `ce / (budget / batch_size) = ce × batch_size / budget`:
- c=50: `4 × 16 / 50 = 1.28` → ~128% real-scored (all docs + a little over)
- c=100: `7 × 16 / 100 = 1.12` → ~112% (all docs)

**Both settings score ALL budget docs in real CE calls.** The paper never went below full coverage at the tested budgets.

---

## ce_mode: two scaling strategies (used in Experiment 7)

Experiment 7 runs the sweep **twice** with two different `ce_mode` values to test
two separate hypotheses:

### `ce_mode="linear"` — Full CE Coverage

```python
ce = math.ceil(c / batch_size)  # every budget doc gets a real CE score
```

- At c=100: ce=7 → 112 docs scored (matches paper)
- At c=1000: ce=63 → 1008 docs scored (all)
- The `else` branch (estimation) never fires
- Tests: **Does ReformIR's reformulation find better candidates than BM25 alone?**
- Equivalent to brute-force reranking on ReformIR's expanded candidate pool

### `ce_mode="fixed"` — Partial Coverage (Tests Approximation)

```python
ce = min(7, math.ceil(c / batch_size))  # cap at 7, paper's baseline
```

- At c=100: ce=7 → 112 docs scored (same as linear, full coverage)
- At c=1000: ce=7 → 112 docs scored, 888 docs **estimated** via lsq_linear
- The `else` branch fires for batches 8–63, testing the regression mechanics
- Tests: **Can lsq_linear extrapolate CE scores beyond its training set?**

### Why both?

The gap between linear and fixed curves on the same plot reveals exactly how
much quality the approximation sacrifices at each budget. If they overlap,
full coverage is wasteful. If they diverge, the regression can't extrapolate.

---

## Scaling rules (code in constants.py)

```python
def reformir_params_for_budget(
    c: int,
    batch_size: int = 16,
    ce_mode: str = "linear",
) -> dict:
    import math
    max_batches = math.ceil(c / batch_size)

    if ce_mode == "fixed":
        ce = min(EXPERIMENT_7_CROSS_ENC_BUDGET, max_batches)  # cap at 7
    else:  # "linear"
        ce = max_batches  # full coverage

    return {
        "cross_enc_budget":        ce,
        "top_s":                   15,      # fixed — RM3 plateaus
        "reform_bm25_num_results": max(128, c),
    }
```

### Concrete values (batch_size=16)

#### ce_mode="linear" (full coverage)

| c | ce | CE calls | budget | Coverage | top_s | rm3 pool |
|---|---|---|---|---|---|---|
| 10 | 1 | 16 | 10 | 160% (full) | 15 | 128 |
| 50 | 4 | 64 | 50 | 128% (full) ✓ paper | 15 | 128 |
| 100 | 7 | 112 | 100 | 112% (full) ✓ paper | 15 | 128 |
| 215 | 14 | 224 | 215 | 104% (full) | 15 | 215 |
| 500 | 32 | 512 | 500 | 102% (full) | 15 | 500 |
| 1000 | 63 | 1008 | 1000 | 101% (full) | 15 | 1000 |

#### ce_mode="fixed" (partial coverage)

| c | ce | CE calls | budget | Coverage | top_s | rm3 pool |
|---|---|---|---|---|---|---|
| 10 | 1 | 16 | 10 | 160% (full — cap) | 15 | 128 |
| 50 | 4 | 64 | 50 | 128% (full — cap) | 15 | 128 |
| 100 | 7 | 112 | 100 | 112% (full — cap) | 15 | 128 |
| 215 | 7 | 112 | 215 | **52% (partial)** | 15 | 215 |
| 500 | 7 | 112 | 500 | **22% (partial)** | 15 | 500 |
| 1000 | 7 | 112 | 1000 | **11% (partial)** | 15 | 1000 |

---

## Key difference from ORE

| | ORE Adaptive | ReformIR (linear) | ReformIR (fixed) |
|---|---|---|---|
| ce at large c | **Fixed at 7** | **Scales with c** | **Fixed at 7** |
| top_s operation | Cheap LAFF lookup | Moderate RM3 expansion | Moderate RM3 expansion |
| top_s scaling | Scales (`c // 4`) | Fixed | Fixed |
| "efficiency" claim | Sublinear CE calls | Full CE — efficiency from reformulation | Tests lsq_linear approximation |

ReformIR's efficiency claim is not about reducing CE calls per budget — it's about getting **better recall** from the same budget by using LLM reformulations to expand the candidate pool beyond what BM25 alone retrieves.

---

## Timing instrumentation

Experiment 7 records wall-clock time per (c, ce_mode) step. This captures:
- CE scoring time (dominant at large c in linear mode)
- RM3 expansion time (fires once per iteration after batch 0)
- Regression fitting time (lsq_linear per CE batch)

The timing plot reveals the compute-quality tradeoff between linear and fixed modes.
