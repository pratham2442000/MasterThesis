"""
Experiment 2 — Merge-then-Rerank (Sparse BM25)
================================================
Varies the number of LLM query rewrites (N) and compares six fusion
methods with BM25 retrieval and MonoT5 re-ranking.

Fusion methods: Dedup, DedupCutoff, Borda, BordaWeighted, CombMAX, RRF.
"""

import os
import sys

import numpy as np
import pyterrier as pt
from pyterrier.measures import R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.transformers import (
    AttachOriginalQuery,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    DedupAndCutoff,
    MergeRewrites,
    QueryRewriteTransformer,
)
from code_base.experiment_utils import get_text_getter, get_reranker


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(n, *, index_ref, reranker, query_lookup, cutoff):
    """Construct all fusion pipelines for *n* rewrites.

    Args:
        n: Number of query rewrites (including original).
        index_ref: PyTerrier index reference.
        reranker: MonoT5ReRanker instance.
        query_lookup: ``{qid: query_text}`` mapping.
        cutoff: BM25 retrieval cutoff.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach = AttachOriginalQuery(query_lookup)
    get_text = get_text_getter(index_ref)

    return {
        "BM25": bm25,
        "BM25_MonoT5": bm25 >> get_text >> reranker,
        # "Pipeline_Dedup": (
        #         rewrite >> bm25 >> MergeRewrites() >> attach
        #         >> Dedup() >> get_text >> reranker
        # ),
        "Pipeline_DedupCutoff": (
                rewrite >> bm25 >> MergeRewrites() >> attach
                >> DedupAndCutoff(cutoff) >> get_text >> reranker
        ),
        "Pipeline_Borda": (
                rewrite >> bm25
                >> BordaVotingTransformerV2(cutoff=cutoff)
                >> attach >> get_text >> reranker
        ),
        "Pipeline_BordaWeighted": (
                rewrite >> bm25
                >> BordaVotingTransformerV2(
            cutoff=cutoff, weight=lambda x: np.exp(x * 0.1)
        )
                >> attach >> get_text >> reranker
        ),
        "Pipeline_CombMAX": (
                rewrite >> bm25 >> MergeRewrites() >> attach
                >> CombMAXTransformer() >> get_text >> reranker
        ),
        # "Pipeline_RRF": (
        #         rewrite >> bm25 >> MergeRewrites() >> attach
        #         >> RRFTransformer() >> get_text >> reranker
        # ),
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    ("BM25_MonoT5", "BM25 + MonoT5"),
    # ("Pipeline_Dedup", "Dedup Method"),
    ("Pipeline_DedupCutoff", "Dedup Cutoff Method"),
    ("Pipeline_Borda", "Borda Voting"),
    ("Pipeline_BordaWeighted", "Borda Weighted"),
    ("Pipeline_CombMAX", "CombMAX"),
    # ("Pipeline_RRF", "RRF"),
]

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_R@50" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2: Fusion Methods Comparison (R@50)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_R_at_50_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2: Fusion Methods Comparison (R@10)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_R_at_10_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2: Fusion Methods Comparison (nDCG@10)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_nDCG_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "regression",
        "x_col": "n",
        "metric_cols": [f"{s}_R@50" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2: R@50 vs Rewrites (All Methods)",
        "xlabel": "Number of Rewrites",
        "ylabel": "R@50",
        "filename": "n_rewrites_vs_R_regression_all_methods.png",
        "use_log_scale": False,
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 2."""
    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

    runner = ExperimentRunner(
        name="experiment_2",
        results_subdir="exp2",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff],
    )

    results_dir = runner.setup()
    topics, qrels, query_lookup = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    reranker = get_reranker()

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_2_N_REWRITES,
        sweep_param="n",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "reranker": reranker,
            "query_lookup": query_lookup,
            "cutoff": cutoff,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
