"""
Experiment 2a — Dense Retrieval + Fusion
=========================================
Same rewrite-sweep as Experiment 2 but using TCT-ColBERT dense retrieval
instead of (or combined with) BM25.

Requires a pre-built dense index (see ``scripts/dataset_setup.py --steps dense``).
"""

import os
import sys

import pyterrier as pt
import pyterrier_dr as pt_dr
import torch
from pyterrier.measures import R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code_base.experiment_utils import get_text_getter, get_reranker

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.transformers import (
    AttachOriginalQuery,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    Exp4FuseTransformer,
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
        reranker,
        query_lookup,
        cutoff,
):
    """Construct dense and hybrid pipelines for *n* rewrites.

    Args:
        n: Number of query rewrites (including original).
        index_ref: Sparse (Terrier) index reference.
        dense_retriever: TCT-ColBERT retriever pipeline.
        reranker: MonoT5ReRanker instance.
        query_lookup: ``{qid: query_text}`` mapping.
        cutoff: Retrieval cutoff.

    Returns:
        Dict of ``{system_name: pipeline}``.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    attach = AttachOriginalQuery(query_lookup)
    get_text = get_text_getter(index_ref)

    dense_base = dense_retriever % cutoff

    return {
        "Dense": dense_base,
        "Dense_MonoT5": dense_base >> get_text >> reranker,
        "Pipeline_Borda": (
                rewrite >> dense_base
                >> BordaVotingTransformerV2(cutoff=cutoff)
                >> attach >> get_text >> reranker
        ),
        "Pipeline_CombMAX": (
                rewrite >> dense_base >> MergeRewrites() >> attach
                >> CombMAXTransformer() % cutoff >> get_text >> reranker
        ),
        "Pipeline_RRF": (
                rewrite >> dense_base >> attach >> RRFTransformer(k=60) % cutoff >> get_text >> reranker
        ),
        "Pipeline_Exp4Fuse": (
                rewrite >> dense_base >> attach >> Exp4FuseTransformer(k=60) % cutoff >> get_text >> reranker
        ),
    }


# ─── Plot Configuration ──────────────────────────────────

def _system_label(key):
    return {
        "Dense": "Dense Baseline",
        "Dense_MonoT5": f"Dense + {constants.RERANKER_NAME}",
        "Pipeline_Borda": "Borda Voting",
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
            "title": "Experiment 2a: Dense Fusion Methods (R@50)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_R_at_50_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "metrics",
            "x_col": "n",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2a: Dense Fusion Methods (R@10)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_R_at_10_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "metrics",
            "x_col": "n",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2a: Dense Fusion Methods (nDCG@10)",
            "xlabel": "Number of Rewrites",
            "filename": "n_rewrites_vs_nDCG_comparison.png",
            "use_log_scale": False,
        },
        {
            "type": "regression",
            "x_col": "n",
            "metric_cols": [f"{s}_R@50" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Experiment 2a: R@50 vs Rewrites (All Methods)",
            "xlabel": "Number of Rewrites",
            "ylabel": "R@50",
            "filename": "n_rewrites_vs_R_regression_all_methods.png",
            "use_log_scale": False,
        },
    ]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 2a."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

    runner = ExperimentRunner(
        name="experiment_2a",
        results_subdir="exp2a",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff],
    )

    results_dir = runner.setup()

    # Dense retrieval setup
    dense_index_path = constants.DENSE_INDEX_PATH
    if not dense_index_path or not os.path.exists(
            os.path.join(dense_index_path, "pt_meta.json")
    ):
        print(f"ERROR: Dense index not found at '{dense_index_path}' — cannot run experiment 2a.")
        return

    model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME, batch_size=512, device=device)
    index = pt_dr.FlexIndex(dense_index_path)
    dense_retriever = model.query_encoder() >> index.retriever()
    print(f"Dense retriever loaded from {dense_index_path}")

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
    reranker = get_reranker()

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
