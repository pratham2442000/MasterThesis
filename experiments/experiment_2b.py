"""
Experiment 2b — Rerank-then-Merge (Sparse BM25)
=================================================
Same rewrite-sweep as Experiment 2, but the reranker is applied
to *each* rewrite individually **before** the fusion step.

This evaluates whether per-rewrite reranking produces better final
rankings than the merge-then-rerank order of Experiment 2.
"""

import os
import sys

import numpy as np
import pyterrier as pt
import torch
from pyterrier.measures import R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import get_text_getter, get_reranker
from code_base.transformers import (
    AttachOriginalQuery,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    DedupAndCutoff,
    Exp4FuseTransformer,
    MergeRewrites,
    QueryRewriteTransformer,
    RRFTransformer,
)


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(n, *, index_ref, reranker, query_lookup, cutoff):
    """Construct rerank-then-merge pipelines for *n* rewrites.

    Key difference from Experiment 2 (MTR): the reranker scores each
    rewrite's BM25 results individually, then fusion merges the
    reranked lists.  The reranker sees rewrite text, not original query.

    Args:
        n: Number of query rewrites (including original).
        index_ref: PyTerrier index reference.
        reranker: Reranker instance.
        query_lookup: ``{qid: query_text}`` mapping.
        cutoff: BM25 retrieval cutoff.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach = AttachOriginalQuery(query_lookup)
    get_text = get_text_getter(index_ref)

    # Shared prefix: rerank each rewrite's BM25 results independently
    rewrite_bm25_rerank = rewrite >> bm25 >> get_text >> reranker

    return {
        "BM25": bm25,
        "BM25_MonoT5": bm25 >> get_text >> reranker,
        "Pipeline_DedupCutoff": (
                rewrite_bm25_rerank >> MergeRewrites() >> attach
                >> DedupAndCutoff(cutoff)
        ),
        "Pipeline_Borda": (
                rewrite_bm25_rerank
                >> BordaVotingTransformerV2(cutoff=cutoff)
                >> attach
        ),
        "Pipeline_BordaWeighted": (
                rewrite_bm25_rerank
                >> BordaVotingTransformerV2(
            cutoff=cutoff, weight=lambda x: np.exp(x * 0.1)
        )
                >> attach
        ),
        "Pipeline_CombMAX": (
                rewrite_bm25_rerank >> MergeRewrites() >> attach
                >> CombMAXTransformer() % cutoff
        ),
        "Pipeline_RRF": (
                rewrite_bm25_rerank >> attach >> RRFTransformer(k=60) % cutoff
        ),
        "Pipeline_Exp4Fuse": (
                rewrite_bm25_rerank >> attach >> Exp4FuseTransformer(k=60) % cutoff
        ),
    }


# ─── Plot Configuration ──────────────────────────────────

def _system_label(key):
    return {
        "BM25": "BM25 Baseline",
        f"BM25_{constants.RERANKER_NAME}": f"BM25 + {constants.RERANKER_NAME}",
        "Pipeline_DedupCutoff": "Dedup Cutoff Method",
        "Pipeline_Borda": "Borda Voting",
        "Pipeline_BordaWeighted": "Borda Weighted",
        "Pipeline_CombMAX": "CombMAX",
        "Pipeline_RRF": "RRF",
        "Pipeline_Exp4Fuse": "Exp4Fuse",
    }.get(key, key)


def _build_plot_configs(system_keys):
    """Build PLOT_CONFIGS list from active system keys."""
    systems = [(k, _system_label(k)) for k in system_keys]

    return [
        {
            "type": "metrics",
            "x_col": "n",
            "metric_cols": [f"{s}_R@50" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2b: Rerank-then-Merge (R@50)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_R_at_50_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "metrics",
            "x_col": "n",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2b: Rerank-then-Merge (R@10)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_R_at_10_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "metrics",
            "x_col": "n",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2b: Rerank-then-Merge (nDCG@10)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_nDCG_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "regression",
            "x_col": "n",
            "metric_cols": [f"{s}_R@50" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2b: R@50 vs Rewrites (All Methods)",
            "xlabel": "Number of Rewrites",
            "ylabel": "R@50",
            "filename": "n_rewrites_vs_R_regression_all_methods.png",
            "use_log_scale": False,
        },
    ]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 2b."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())

    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

    runner = ExperimentRunner(
        name="experiment_2b",
        results_subdir="exp2b",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff],
    )

    results_dir = runner.setup()
    topics, qrels, query_lookup = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
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

    # Determine which systems were actually run
    system_keys = [col.replace("_R@10", "") for col in results_df.columns
                   if col.endswith("_R@10")]
    plot_configs = _build_plot_configs(system_keys)
    runner.plot(results_df, results_dir, plot_configs)


if __name__ == "__main__":
    main()
