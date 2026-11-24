# ORE Parameter Guide — Experiment 4

> Reference: [arXiv:2504.09353](https://arxiv.org/abs/2504.09353) — "Breaking the Lens of the Telescope"

---

## What each parameter does

### `cross_enc_budget` (ce, default 7)

Controls how many **real cross-encoder calls** ORE makes per query.

The loop condition is:
```python
if len(results) < min(batch_size × cross_enc_budget, budget):
    # use real scorer (expensive)
else:
    # use λ-regression estimate (cheap)
```

At `c=100, batch=16, ce=7`: `min(112, 100) = 100` → **all 100 docs** get real CE scores.  
`ce=7` was tuned to give full coverage exactly at `c=100`. This is also the calibration
phase — the λ parameters for the regression are fitted from these real-scored docs.

### `top_s` (cluster head pool, default 25)

After each scoring batch, the top-`top_s` highest-scored results become the
**"relevance cluster"**. The LAFF graph is queried from these cluster heads to find
neighbours worth scoring next.

Think of it as: *"how wide is our current best-guess relevance region?"*

Concretely: `cluster_heads = [doc for doc, _ in Counter(results).most_common(top_s)]`

### `top_s2` (expansion frontier, default 15)

After scoring a batch, the top-`top_s2` docs from that batch are used to pull **new
candidate documents** from the LAFF graph into the arm pool.

Think of it as: *"how aggressively do we expand the frontier?"*

Concretely:
```python
S_2 = Counter(results).most_common(top_s2)
for docno in S_2:
    neighbors = laff_graph.neighbours(docno)
    # add neighbors to arms pool
```

### `--s` (30, run_adaptive.py only)

Legacy argument in the old `run_adaptive.py` script. **Not used** in the current
`OREAdaptive` constructor. Ignore for Experiment 4.

---

## Why fixed defaults break at large `c`

| c | CE calls (fixed ce=7) | top_s | top_s2 | Problem |
|---|---|---|---|---|
| 100 | 100 (full coverage) | 25 | 15 | Fine — tuned for this |
| 500 | 112 (22% coverage) | 25 | 15 | Pool still sized for 100 docs; navigation misses structure |
| 1000 | 112 (11% coverage) | 25 | 15 | Cluster and frontier far too small for the candidate space |

---

## Speed vs Quality: cost per parameter

| Parameter | Operation | Expensive? |
|---|---|---|
| `ce` | Cross-encoder forward pass × batch_size | **YES — dominant cost** |
| `top_s` | `Counter.most_common(n)` + LAFF precomputed array lookup | No — negligible |
| `top_s2` | `heapq.nlargest` on in-memory scores | No — negligible |

---

## ce_mode: two scaling strategies (used in Experiment 4)

Experiment 4 runs the sweep **twice** with two different `ce_mode` values to test
two separate hypotheses (mirrors ReformIR's approach — see `REFORMIR_PARAMS.md`):

### `ce_mode="linear"` — Full CE Coverage

```python
ce = math.ceil(c / batch_size)  # every budget doc gets a real CE score
```

- At c=100: ce=7 → 112 docs scored (matches paper)
- At c=1000: ce=63 → 1008 docs scored (all)
- The `else` branch (λ estimation) never fires
- Tests: **Does ORE's LAFF graph navigation find better candidates than BM25 alone?**
- Equivalent to brute-force reranking on ORE's expanded candidate pool

### `ce_mode="fixed"` — Partial Coverage (Tests Approximation)

```python
ce = min(7, math.ceil(c / batch_size))  # cap at 7, paper's baseline
```

- At c=100: ce=7 → 112 docs scored (same as linear, full coverage)
- At c=1000: ce=7 → 112 docs scored, 888 docs **estimated** via λ-regression
- The `else` branch fires for batches 8–63, testing the regression mechanics
- Tests: **Can the λ-regression extrapolate CE scores beyond its training set?**

### Why both?

The gap between linear and fixed curves on the same plot reveals exactly how
much quality the approximation sacrifices at each budget. If they overlap,
full coverage is wasteful. If they diverge, the regression can't extrapolate.

---

## Scaling rules (code in constants.py)

```python
def ore_params_for_budget(
    c: int,
    batch_size: int = 16,
    ce_mode: str = "linear",
) -> dict:
    import math
    max_batches = math.ceil(c / batch_size)

    if ce_mode == "fixed":
        ce = min(EXPERIMENT_4_CROSS_ENC_BUDGET, max_batches)  # cap at 7
    else:  # "linear"
        ce = max_batches  # full coverage

    return {
        "cross_enc_budget":  ce,
        "top_s":  max(25, c // 4),   # cheap; scales cluster navigation
        "top_s2": max(15, c // 6),   # cheap; scales frontier expansion
    }
```

### Concrete values (batch_size=16)

#### ce_mode="linear" (full coverage)

| c | ce | CE calls | budget | Coverage | top_s | top_s2 |
|---|---|---|---|---|---|---|
| 10 | 1 | 16 | 10 | 160% (full) | 25 | 15 |
| 100 | 7 | 112 | 100 | 112% (full) ✓ paper | 25 | 16 |
| 215 | 14 | 224 | 215 | 104% (full) | 53 | 35 |
| 500 | 32 | 512 | 500 | 102% (full) | 125 | 83 |
| 1000 | 63 | 1008 | 1000 | 101% (full) | 250 | 166 |

#### ce_mode="fixed" (partial coverage)

| c | ce | CE calls | budget | Coverage | top_s | top_s2 |
|---|---|---|---|---|---|---|
| 10 | 1 | 16 | 10 | 160% (full — cap) | 25 | 15 |
| 100 | 7 | 112 | 100 | 112% (full — cap) | 25 | 16 |
| 215 | 7 | 112 | 215 | **52% (partial)** | 53 | 35 |
| 500 | 7 | 112 | 500 | **22% (partial)** | 125 | 83 |
| 1000 | 7 | 112 | 1000 | **11% (partial)** | 250 | 166 |

---

## Key invariant to remember

> **`ce` controls quality/cost. `top_s`/`top_s2` control quality only.**
>
> In `linear` mode, ce scales with c — full coverage, isolates navigation quality.
> In `fixed` mode, ce stays at 7 — tests λ-regression approximation at large c.
> Scale `top_s` and `top_s2` freely — they're cheap array operations.

---

## Corpus graph for dpr-w100

- HuggingFace artifact: `astappiev/ragwiki-corpusgraph`
- Built for: `rag:nq_wiki` Wikipedia corpus, BM25, k=16 neighbours
- Load: `pta.Artifact.from_hf("astappiev/ragwiki-corpusgraph")`
- Docno format: sequential integers matching dpr-w100 index
- Verify: spot-check that docnos in graph match docnos in local Terrier index
