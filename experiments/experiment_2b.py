"""
Experiment 2b — Rerank-then-Merge (Sparse BM25)
=================================================
Same rewrite-sweep as Experiment 2, but the MonoT5 re-ranker is applied
to *each* rewrite individually **before** the fusion step.

This evaluates whether per-rewrite re-ranking produces better final
rankings than the merge-then-rerank order of Experiment 2.
"""

import os
import sys

import numpy as np
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_base.experiment_utils import get_text_getter


import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.transformers import (
    BordaVotingTransformerV2,
    CombMAXTransformer,
    Dedup,
    MergeRewrites,
    QueryRewriteTransformer,
    RRFTransformer,
)


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(n, *, index_ref, reranker, cutoff):
    """Construct rerank-then-merge pipelines for *n* rewrites.

    Args:
        n: Number of query rewrites.
        index_ref: PyTerrier index reference.
        reranker: MonoT5ReRanker instance.
        cutoff: BM25 retrieval cutoff.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    get_text = get_text_getter(index_ref)

    # Shared rerank-per-rewrite prefix
    rewrite_bm25_rerank = rewrite >> bm25 >> get_text >> reranker

    return {
        "BM25": bm25,
        "BM25_MonoT5": bm25 >> get_text >> reranker,
        "Pipeline_Dedup": (
                rewrite_bm25_rerank >> MergeRewrites() >> Dedup()
        ),
        "Pipeline_Borda": (
                rewrite_bm25_rerank
                >> BordaVotingTransformerV2(cutoff=cutoff)
        ),
        "Pipeline_BordaWeighted": (
                rewrite_bm25_rerank
                >> BordaVotingTransformerV2(
            cutoff=cutoff, weight=lambda x: np.exp(x * 0.1)
        )
        ),
        "Pipeline_CombMAX": (
                rewrite_bm25_rerank >> MergeRewrites() >> CombMAXTransformer()
        ),
        "Pipeline_RRF": (
                rewrite_bm25_rerank >> MergeRewrites() >> RRFTransformer()
        ),
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    ("BM25_MonoT5", "BM25 + MonoT5"),
    ("Pipeline_Dedup", "Dedup Method"),
    ("Pipeline_Borda", "Borda Voting"),
    ("Pipeline_BordaWeighted", "Borda Weighted"),
    ("Pipeline_CombMAX", "CombMAX"),
    ("Pipeline_RRF", "RRF"),
]

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2b: Rerank-then-Merge (Recall)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_R_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2b: Rerank-then-Merge (nDCG@10)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_nDCG_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "regression",
        "x_col": "n",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2b: R@10 vs Rewrites (All Methods)",
        "xlabel": "Number of Rewrites",
        "ylabel": "R@10",
        "filename": "n_rewrites_vs_R_regression_all_methods.png",
        "use_log_scale": False,
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 2b."""
    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

    runner = ExperimentRunner(
        name="experiment_2b",
        results_subdir="exp2b",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff],
    )

    results_dir = runner.setup()
    topics, qrels, _ = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_2_N_REWRITES,
        sweep_param="n",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "reranker": mono_t5,
            "cutoff": cutoff,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
