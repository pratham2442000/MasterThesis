"""
Experiment 2a — Dense Retrieval + Fusion
=========================================
Same rewrite-sweep as Experiment 2 but using TCT-ColBERT dense retrieval
instead of (or combined with) BM25.

Requires a pre-built dense index (see ``scripts/dataset_setup.py --steps dense``).
"""

import os
import sys

import numpy as np
import pyterrier as pt
import pyterrier_dr as pt_dr
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.transformers import (
    AttachOriginalQuery,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    Dedup,
    MergeRewrites,
    QueryRewriteTransformer,
    RRFTransformer,
)


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(
        n,
        *,
        index_ref,
        dense_retriever,
        mono_t5,
        query_lookup,
        cutoff,
):
    """Construct dense and hybrid pipelines for *n* rewrites.

    Args:
        n: Number of query rewrites (including original).
        index_ref: Sparse (Terrier) index reference.
        dense_retriever: TCT-ColBERT retriever pipeline.
        mono_t5: MonoT5ReRanker instance.
        query_lookup: ``{qid: query_text}`` mapping.
        cutoff: Retrieval cutoff.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    attach = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    dense_base = dense_retriever % cutoff

    return {
        "Dense": dense_base,
        "Dense_MonoT5": dense_base >> get_text >> mono_t5,
        "Pipeline_Dedup": (
                rewrite >> dense_base >> MergeRewrites() >> attach
                >> Dedup() >> get_text >> mono_t5
        ),
        "Pipeline_Borda": (
                rewrite >> dense_base
                >> BordaVotingTransformerV2(cutoff=cutoff)
                >> attach >> get_text >> mono_t5
        ),
        "Pipeline_BordaWeighted": (
                rewrite >> dense_base
                >> BordaVotingTransformerV2(
            cutoff=cutoff, weight=lambda x: np.exp(x * 0.1)
        )
                >> attach >> get_text >> mono_t5
        ),
        "Pipeline_CombMAX": (
                rewrite >> dense_base >> MergeRewrites() >> attach
                >> CombMAXTransformer() >> get_text >> mono_t5
        ),
        "Pipeline_RRF": (
                rewrite >> dense_base >> MergeRewrites() >> attach
                >> RRFTransformer() >> get_text >> mono_t5
        ),
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("Dense", "Dense Baseline"),
    ("Dense_MonoT5", "Dense + MonoT5"),
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
        "title": "Experiment 2a: Dense Fusion Methods (Recall)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_R_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "metrics",
        "x_col": "n",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2a: Dense Fusion Methods (nDCG@10)",
        "xlabel": "Number of Rewrites",
        "filename": "n_rewrites_vs_nDCG_comparison.png",
        "use_log_scale": False,
    },
    {
        "type": "regression",
        "x_col": "n",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Experiment 2a: R@10 vs Rewrites (All Methods)",
        "xlabel": "Number of Rewrites",
        "ylabel": "R@10",
        "filename": "n_rewrites_vs_R_regression_all_methods.png",
        "use_log_scale": False,
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 2a."""
    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

    runner = ExperimentRunner(
        name="experiment_2a",
        results_subdir="exp2a",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff],
    )

    results_dir = runner.setup()

    # Dense retrieval setup
    model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME)
    index = pt_dr.FlexIndex(constants.DENSE_INDEX_PATH)
    dense_retriever = model.query_encoder() >> index.retriever()

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    topics, qrels, query_lookup = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_2_N_REWRITES,
        sweep_param="n",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "dense_retriever": dense_retriever,
            "mono_t5": mono_t5,
            "query_lookup": query_lookup,
            "cutoff": cutoff,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
