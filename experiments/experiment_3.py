"""
Experiment 3 — Cutoff Sweep with Fixed Rewrites
=================================================
Fixes the number of rewrites (best from exp2/2b) and sweeps the BM25
retrieval cutoff.  Compares merge-then-rerank (MTR) vs rerank-then-merge
(RTM) strategies across Dedup, CombMAX, and RRF.
"""

import os
import sys

import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.transformers import (
    AttachOriginalQuery,
    CombMAXTransformer,
    Dedup,
    MergeRewrites,
    QueryRewriteTransformer,
    RRFTransformer,
)

N_REWRITES = constants.EXPERIMENT_3_BEST_N_REWRITES


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(cutoff, *, index_ref, reranker, query_lookup):
    """Construct MTR and RTM pipelines for a given *cutoff*.

    Args:
        cutoff: BM25 retrieval cutoff.
        index_ref: PyTerrier index reference.
        reranker: MonoT5ReRanker instance.
        query_lookup: ``{qid: query_text}`` mapping.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(
        N_REWRITES, dataset_name=constants.DATASET_NAME
    )
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    # Merge-then-Rerank (MTR)
    p_dedup_mtr = (
            rewrite >> bm25 >> MergeRewrites() >> attach
            >> Dedup() >> get_text >> reranker
    )
    p_combmax_mtr = (
            rewrite >> bm25 >> MergeRewrites() >> attach
            >> CombMAXTransformer() >> get_text >> reranker
    )
    p_rrf_mtr = (
            rewrite >> bm25 >> MergeRewrites() >> attach
            >> RRFTransformer() >> get_text >> reranker
    )

    # Rerank-then-Merge (RTM)
    rewrite_bm25_rerank = rewrite >> bm25 >> get_text >> reranker
    p_dedup_rtm = rewrite_bm25_rerank >> MergeRewrites() >> Dedup()
    p_combmax_rtm = (
            rewrite_bm25_rerank >> MergeRewrites() >> CombMAXTransformer()
    )
    p_rrf_rtm = rewrite_bm25_rerank >> MergeRewrites() >> RRFTransformer()

    return {
        "BM25": bm25,
        "BM25_MonoT5": bm25 >> get_text >> reranker,
        "Dedup": p_dedup_mtr,
        "CombMAX": p_combmax_mtr,
        "Dedup_RTM": p_dedup_rtm,
        "CombMAX_RTM": p_combmax_rtm,
        "RRF": p_rrf_mtr,
        "RRF_RTM": p_rrf_rtm,
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    ("BM25_MonoT5", "BM25 + MonoT5"),
    ("Dedup", "Dedup (Merge→Rerank)"),
    ("CombMAX", "CombMAX (Merge→Rerank)"),
    ("Dedup_RTM", "Dedup (Rerank→Merge)"),
    ("CombMAX_RTM", "CombMAX (Rerank→Merge)"),
    ("RRF", "RRF (Merge→Rerank)"),
    ("RRF_RTM", "RRF (Rerank→Merge)"),
]

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "cutoff",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": f"Experiment 3: Cutoff vs R@10 (n_rewrites={N_REWRITES})",
        "xlabel": "Retrieval Cutoff",
        "filename": "cutoff_vs_R_comparison.png",
        "use_log_scale": True,
    },
    {
        "type": "metrics",
        "x_col": "cutoff",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": f"Experiment 3: Cutoff vs nDCG@10 (n_rewrites={N_REWRITES})",
        "xlabel": "Retrieval Cutoff",
        "filename": "cutoff_vs_nDCG_comparison.png",
        "use_log_scale": True,
    },
    {
        "type": "regression",
        "x_col": "cutoff",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": f"Experiment 3: R@10 vs Cutoff Regression (n={N_REWRITES})",
        "xlabel": "Retrieval Cutoff",
        "ylabel": "R@10",
        "filename": "cutoff_vs_R_regression_all_methods.png",
        "use_log_scale": True,
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 3."""
    runner = ExperimentRunner(
        name="experiment_3",
        results_subdir="exp3",
        topic_limit=constants.EXPERIMENT_3_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_3_EVAL_METRICS,
        extra_metrics_fn=lambda c: [R @ c],
    )

    results_dir = runner.setup()
    topics, qrels, query_lookup = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_3_C_VALUES,
        sweep_param="cutoff",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "reranker": mono_t5,
            "query_lookup": query_lookup,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
