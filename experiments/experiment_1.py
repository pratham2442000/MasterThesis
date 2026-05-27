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
import torch
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
    kwargs = {
        "model": model_id,
        "verbose": False,
        "batch_size": constants.T5_BATCH_SIZE,
    }
    if tok_model is not None:
        kwargs["tok_model"] = tok_model

    # ─── MONKEY PATCH FOR THE TOKENIZER BUG ───
    reranker = MonoT5ReRanker(**kwargs)

    if hasattr(reranker, "tokenizer") and not hasattr(reranker.tokenizer, "batch_encode_plus"):
        # If the top-level tokenizer object lacks the method, route it to the underlying HF tokenizer
        if hasattr(reranker.tokenizer, "tokenizer"):
            reranker.tokenizer.batch_encode_plus = reranker.tokenizer.tokenizer.batch_encode_plus

    return reranker


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

def _truncate_df(df, c):
    """Keep only top-c rows per query from a pre-sorted DataFrame."""
    return df.groupby("qid").head(c).reset_index(drop=True)


def _attach_text(df, text_lookup):
    """Add 'text' column from cached lookup dict."""
    out = df.copy()
    out["text"] = out["docno"].map(text_lookup)
    return out


def build_pipelines(
        c,
        *,
        bm25_full_df,
        dense_full_df,
        monot5,
        cross_encoder,
        text_lookup,
):
    """Construct retrieval pipelines for a given cutoff *c*.

    Uses pre-materialized DataFrames (retrieved at MAX_C) so each
    sweep step only truncates — no redundant retrieval or text lookups.

    Args:
        c: Retrieval cutoff.
        bm25_full_df: Pre-computed BM25 results DataFrame at MAX_C.
        dense_full_df: Pre-computed dense results DataFrame (or None).
        monot5: MonoT5ReRanker instance (or None).
        cross_encoder: CrossEncoderReRanker instance (or None).
        text_lookup: dict mapping docno → text.

    Returns:
        Dict mapping system names to pipelines.
    """
    bm25_df = _truncate_df(bm25_full_df, c)
    bm25 = pt.Transformer.from_df(bm25_df)
    bm25_text_df = _attach_text(bm25_df, text_lookup)
    bm25_text = pt.Transformer.from_df(bm25_text_df)

    pipelines = {"BM25": bm25}

    if monot5 is not None:
        pipelines["BM25_MonoT5"] = bm25_text >> monot5

    if cross_encoder is not None:
        pipelines["BM25_MiniLM-L6"] = bm25_text >> cross_encoder

    if dense_full_df is not None:
        dense_df = _truncate_df(dense_full_df, c)
        dense = pt.Transformer.from_df(dense_df)
        dense_text_df = _attach_text(dense_df, text_lookup)
        dense_text = pt.Transformer.from_df(dense_text_df)

        pipelines["Dense"] = dense
        if monot5 is not None:
            pipelines["Dense_MonoT5"] = dense_text >> monot5
        if cross_encoder is not None:
            pipelines["Dense_MiniLM-L6"] = dense_text >> cross_encoder

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
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


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
            model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME, batch_size=512, device=device)
            index = pt_dr.FlexIndex(dense_index_path)
            dense_retriever = model.query_encoder() >> index.retriever()
            print(f"Dense retriever loaded from {dense_index_path}")
        except Exception as e:
            print(f"WARNING: Dense retriever not loaded — {e}")
    else:
        print(f"WARNING: Dense index not found at '{dense_index_path}' — skipping dense pipelines.")

    # ── Pre-compute retrieval at MAX_C ─────────────────────
    # Run BM25 and dense once at largest cutoff. Store results
    # as DataFrames. Each sweep step truncates to top-c per query.
    MAX_C = max(constants.EXPERIMENT_1_C_VALUES)
    print(f"\nPre-computing retrieval at MAX_C={MAX_C} ...")

    bm25_retriever = pt.terrier.Retriever(index_ref, wmodel="BM25") % MAX_C
    bm25_full_df = bm25_retriever.transform(topics)
    print(f"  BM25 cached: {len(bm25_full_df)} rows")

    dense_full_df = None
    if dense_retriever is not None:
        dense_full_df = (dense_retriever % MAX_C).transform(topics)
        print(f"  Dense cached: {len(dense_full_df)} rows")

    # Pre-compute text for all retrieved docs (union of BM25 + dense)
    text_getter = get_text_getter(index_ref)
    all_docs = bm25_full_df
    if dense_full_df is not None:
        import pandas as pd
        all_docs = pd.concat([bm25_full_df, dense_full_df]).drop_duplicates(
            subset=["qid", "docno"]
        )
    all_text_df = text_getter.transform(all_docs)
    # Build docno→text lookup for fast join
    _text_lookup = dict(zip(all_text_df["docno"], all_text_df["text"]))
    print(f"  Text cached: {len(_text_lookup)} unique docs")

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_1_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "bm25_full_df": bm25_full_df,
            "dense_full_df": dense_full_df,
            "monot5": monot5,
            "cross_encoder": cross_encoder,
            "text_lookup": _text_lookup,
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
