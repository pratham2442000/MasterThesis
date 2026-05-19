"""
Experiment 1 — BM25 & Dense Cutoff Sweep (All Rerankers)
==========================================================
Measures how initial retrieval cutoff (c) affects Recall and nDCG
with rerankers, for both BM25 and TCT-ColBERT dense retrieval:

  - BM25 (baseline)
  - BM25 + MonoT5
  - BM25 + MiniLM-L6 (CrossEncoder)
  - Dense (TCT-ColBERT)
  - Dense + MonoT5
  - Dense + MiniLM-L6

Sweeps ``constants.EXPERIMENT_1_C_VALUES``.
"""

import os
import sys

import pyterrier as pt
import pyterrier_dr as pt_dr
from pyterrier.measures import R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import get_text_getter


# ─── Reranker Loaders ────────────────────────────────────

def _load_monot5():
    from pyterrier_t5 import MonoT5ReRanker

    model_id = constants.RERANKER_MODEL_NAME
    tok_model = None

    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(constants.MODELS_CACHE_DIR,
                                  model_id.replace("/", os.sep))
        if os.path.exists(local_path):
            model_id = local_path
            t5_tok_path = os.path.join(constants.MODELS_CACHE_DIR, "t5-base")
            if os.path.exists(t5_tok_path):
                tok_model = t5_tok_path

    print(f"Loading MonoT5ReRanker ({model_id})")
    return MonoT5ReRanker(
        model=model_id,
        tok_model=tok_model,
        verbose=False,
        batch_size=constants.T5_BATCH_SIZE,
    )


def _load_cross_encoder():
    from code_base.cross_encoder_reranker import CrossEncoderReRanker

    model_id = constants.CROSS_ENCODER_MODEL_NAME
    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(constants.MODELS_CACHE_DIR,
                                  model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

    print(f"Loading CrossEncoderReRanker ({model_id})")
    return CrossEncoderReRanker(
        model=model_id,
        batch_size=constants.CROSS_ENCODER_BATCH_SIZE,
        verbose=False,
    )


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, monot5, cross_encoder, dense_retriever):
    """Construct retrieval pipelines for a given cutoff *c*.

    Args:
        c: Retrieval cutoff.
        index_ref: PyTerrier index reference.
        monot5: MonoT5ReRanker instance (or None if unavailable).
        cross_encoder: CrossEncoderReRanker instance (or None).
        dense_retriever: TCT-ColBERT retriever pipeline (or None).

    Returns:
        Dict mapping system names to pipelines.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    text_getter = get_text_getter(index_ref)
    bm25_with_text = bm25 >> text_getter

    pipelines = {"BM25": bm25}

    if monot5 is not None:
        # Use the shared prefix
        pipelines["BM25_MonoT5"] = bm25_with_text >> monot5

    if cross_encoder is not None:
        # Use the shared prefix
        pipelines["BM25_MiniLM-L6"] = bm25_with_text >> cross_encoder

    if dense_retriever is not None:
        dense = dense_retriever % c
        # If your text_getter works for dense too, create a shared dense prefix:
        dense_with_text = dense >> text_getter

        pipelines["Dense"] = dense
        if monot5 is not None:
            pipelines["Dense_MonoT5"] = dense_with_text >> monot5
        if cross_encoder is not None:
            pipelines["Dense_MiniLM-L6"] = dense_with_text >> cross_encoder

    return pipelines


# ─── Plot Configuration ──────────────────────────────────

# Built dynamically in main() once we know which rerankers loaded.

def _build_plot_configs(system_keys):
    """Build PLOT_CONFIGS list from active system keys."""
    systems = [(k, _system_label(k)) for k in system_keys]

    return [
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on Recall@10",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "filename": "cutoff_vs_R10.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@50" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on Recall@50",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "Recall@50",
            "filename": "cutoff_vs_R50.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on nDCG@10",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "nDCG@10",
            "filename": "cutoff_vs_nDCG10.png",
            "use_log_scale": True,
        },
    ]


def _system_label(key):
    return {
        "BM25": "BM25 Baseline",
        "BM25_MonoT5": "BM25 + MonoT5",
        "BM25_MiniLM-L6": "BM25 + MiniLM-L6",
        "Dense": "Dense (TCT-ColBERT)",
        "Dense_MonoT5": "Dense + MonoT5",
        "Dense_MiniLM-L6": "Dense + MiniLM-L6",
    }.get(key, key)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 1 with BM25 and Dense rerankers."""
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

    # ── Load rerankers (skip ones that fail) ──────────────
    monot5, cross_encoder = None, None

    try:
        monot5 = _load_monot5()
    except Exception as e:
        print(f"WARNING: MonoT5 not loaded — {e}")

    try:
        cross_encoder = _load_cross_encoder()
    except Exception as e:
        print(f"WARNING: CrossEncoder not loaded — {e}")

    # ── Load dense retriever (skip if index not built yet) ─
    dense_retriever = None
    dense_index_path = constants.DENSE_INDEX_PATH

    if dense_index_path and os.path.exists(
            os.path.join(dense_index_path, "pt_meta.json")
    ):
        try:
            model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME, batch_size=64)
            index = pt_dr.FlexIndex(dense_index_path)
            dense_retriever = model.query_encoder() >> index.retriever()
            print(f"Dense retriever loaded from {dense_index_path}")
        except Exception as e:
            print(f"WARNING: Dense retriever not loaded — {e}")
    else:
        print(f"WARNING: Dense index not found at '{dense_index_path}' — skipping dense pipelines.")

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_1_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "monot5": monot5,
            "cross_encoder": cross_encoder,
            "dense_retriever": dense_retriever,
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
