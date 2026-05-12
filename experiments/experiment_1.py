"""
Experiment 1 — BM25 Cutoff Sweep
=================================
Measures how initial BM25 retrieval cutoff (c) affects Recall and nDCG
with and without MonoT5 re-ranking.

Sweeps ``constants.EXPERIMENT_1_C_VALUES``.
"""

import os
import sys

import pyterrier as pt
from pyterrier.measures import R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import get_text_getter, get_reranker


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, reranker):
    """Construct retrieval pipelines for a given cutoff *c*.

    Args:
        c: BM25 retrieval cutoff.
        index_ref: PyTerrier index reference.
        reranker: MonoT5ReRanker instance.

    Returns:
        Dict mapping system names to pipelines.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    pipeline_t5 = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )
    return {
        "BM25": bm25,
        "Pipeline": pipeline_t5,
    }


# ─── Plot Configuration ──────────────────────────────────

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["BM25_R@10", "Pipeline_R@10"],
        "labels": ["BM25 Baseline", f"BM25 + {constants.RERANKER_NAME} Re-ranker"],
        "title": "Effect of Initial Cutoff (c) on Recall (R@10)",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "filename": "cutoff_vs_R.png",
        "use_log_scale": True,
    },
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["BM25_nDCG@10", "Pipeline_nDCG@10"],
        "labels": ["BM25 Baseline", f"BM25 + {constants.RERANKER_NAME} Re-ranker"],
        "title": "Effect of Initial Cutoff (c) on NDCG@10",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "NDCG@10",
        "filename": "cutoff_vs_NDCG_at_10.png",
        "use_log_scale": True,
    },
    {
        "type": "regression",
        "x_col": "c",
        "metric_cols": ["BM25_R@10", "Pipeline_R@10"],
        "labels": ["BM25 R@10", f"{constants.RERANKER_NAME} R@10"],
        "use_log_scale": True,
        "title": "R@10 vs. Initial Cutoff (c) Regression",
        "xlabel": "Initial Cutoff (c) (log scale)",
        "ylabel": "Recall (R@10)",
        "filename": "cutoff_R_at_10_log_regression.png",
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 1."""
    runner = ExperimentRunner(
        name="experiment_1",
        results_subdir="exp1",
        topic_limit=constants.EXPERIMENT_1_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_1_EVAL_METRICS,
        extra_metrics_fn=lambda c: [R @ c],
    )

    results_dir = runner.setup()
    topics, qrels, _ = runner.load_data()

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
    reranker = get_reranker()

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_1_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={"index_ref": index_ref, "reranker": reranker},
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
