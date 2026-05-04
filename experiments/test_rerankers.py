"""
Reranker Comparison Test
========================
Compare different rerankers on a small sample of queries.
Measures: nDCG@10, R@10, R@50, and per-query timing.

Rerankers tested:
  - MonoT5 (castorini/monot5-base-msmarco) — T5-based, 220M params
  - CrossEncoder MiniLM-L6 (cross-encoder/ms-marco-MiniLM-L6-v2) — 22M params
  - CrossEncoder MiniLM-L12 (cross-encoder/ms-marco-MiniLM-L12-v2) — 33M params
  - Passthrough (no reranking baseline)

Usage:
    python experiments/test_rerankers.py                # all rerankers
    python experiments/test_rerankers.py --rerankers minilm6 monot5  # subset
    python experiments/test_rerankers.py --n-queries 20 --cutoff 50
"""

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG

import constants
from code_base import dataset_factory
from code_base.experiment_utils import get_text_getter
from code_base.transformers import preprocess_query


# ─── Available Rerankers ──────────────────────────────────

def load_reranker(name: str):
    """Load reranker by short name.

    Args:
        name: One of 'monot5', 'minilm6', 'minilm12', 'passthrough'.

    Returns:
        (display_name, pt.Transformer)
    """
    if name == "monot5":
        from pyterrier_t5 import MonoT5ReRanker
        model_id = constants.RERANKER_MODEL_NAME
        # Check offline cache
        if getattr(constants, "OFFLINE_MODE", False):
            local = os.path.join(constants.MODELS_CACHE_DIR, model_id.split("/")[-1])
            if os.path.exists(local):
                model_id = local
        return "MonoT5", MonoT5ReRanker(
            model=model_id, verbose=False,
            batch_size=constants.T5_BATCH_SIZE,
        )

    elif name == "minilm6":
        from code_base.cross_encoder_reranker import CrossEncoderReRanker
        return "MiniLM-L6", CrossEncoderReRanker(
            model="cross-encoder/ms-marco-MiniLM-L6-v2",
            batch_size=64, verbose=False,
        )

    elif name == "minilm12":
        from code_base.cross_encoder_reranker import CrossEncoderReRanker
        return "MiniLM-L12", CrossEncoderReRanker(
            model="cross-encoder/ms-marco-MiniLM-L12-v2",
            batch_size=64, verbose=False,
        )

    elif name == "passthrough":
        class Passthrough(pt.Transformer):
            def transform(self, inp):
                return inp

        return "NoRerank", Passthrough()

    else:
        raise ValueError(f"Unknown reranker: {name}")


AVAILABLE_RERANKERS = ["passthrough", "minilm6", "minilm12", "monot5"]


# ─── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare rerankers")
    parser.add_argument(
        "--rerankers", nargs="+", default=AVAILABLE_RERANKERS,
        choices=AVAILABLE_RERANKERS,
        help=f"Rerankers to test (default: all)",
    )
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--cutoff", type=int, default=50,
                        help="BM25 retrieval cutoff")
    args = parser.parse_args()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME,
        "reranker_comparison", f"run_{timestamp}",
    )
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    _, topics_df, qrels_df = dataset_factory.get_dataset(
        constants.DATASET_NAME, constants.DATA_SPLIT
    )
    topics_df["query"] = topics_df["query"].apply(preprocess_query)
    topics = topics_df.head(args.n_queries)
    print(f"Topics: {len(topics)}, Qrels: {len(qrels_df)}")

    # Index
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % args.cutoff
    get_text = get_text_getter(index_ref)

    # BM25 baseline results (shared across all rerankers)
    print("\nRunning BM25 retrieval...")
    bm25_results = (bm25 >> get_text).transform(topics)
    print(f"Retrieved {len(bm25_results)} results")

    # Eval metrics
    eval_metrics = [nDCG @ 10, R @ 10, R @ 50]

    # Run each reranker
    all_rows = []
    pipelines = {}
    names = []

    # Add BM25 baseline (no reranking)
    pipelines["BM25_only"] = bm25
    names.append("BM25_only")

    for rr_name in args.rerankers:
        print(f"\n{'─' * 60}")
        print(f"Loading reranker: {rr_name}")
        print(f"{'─' * 60}")

        try:
            display_name, reranker = load_reranker(rr_name)
        except Exception as e:
            print(f"  !! Failed to load {rr_name}: {e}")
            continue

        pipeline = bm25 >> get_text >> reranker
        pipe_name = f"BM25+{display_name}"
        pipelines[pipe_name] = pipeline
        names.append(pipe_name)

        # Time it
        print(f"  Running {display_name}...")
        t0 = time.time()
        reranked = reranker.transform(bm25_results.copy())
        elapsed = time.time() - t0
        print(f"  {display_name}: {elapsed:.2f}s for {len(bm25_results)} docs "
              f"({len(bm25_results) / elapsed:.0f} docs/s)")

        all_rows.append({
            "reranker": display_name,
            "time_seconds": round(elapsed, 2),
            "docs_per_sec": round(len(bm25_results) / elapsed, 1),
            "total_docs": len(bm25_results),
        })

    # Run pt.Experiment for proper evaluation
    print(f"\n{'=' * 60}")
    print("RUNNING EVALUATION")
    print(f"{'=' * 60}")

    exp_df = pt.Experiment(
        list(pipelines.values()),
        topics,
        qrels_df,
        eval_metrics=eval_metrics,
        names=list(pipelines.keys()),
        verbose=True,
        batch_size=constants.EXPERIMENT_BATCH_SIZE,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(exp_df.to_string(index=False))

    # Save
    exp_df.to_csv(os.path.join(results_dir, "reranker_comparison.csv"), index=False)
    print(f"\nResults saved to {results_dir}")

    # Save timing
    if all_rows:
        timing_df = pd.DataFrame(all_rows)
        timing_df.to_csv(os.path.join(results_dir, "reranker_timing.csv"), index=False)
        print("\nTiming:")
        print(timing_df.to_string(index=False))


if __name__ == "__main__":
    main()
