"""
Experiment 5 — Full RAG Pipeline Cutoff Sweep
==============================================
Evaluates the full Retrieve → Rerank → Generate pipeline.

Stage 1: Retrieval quality (R@10, R@100) across BM25 cutoffs.
Stage 2: Answer generation quality (EM, F1) using pyterrier-rag.

Sweeps ``constants.EXPERIMENT_5_C_VALUES``.
"""

import os
import sys

import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.wiki_loader_pyt import get_gold_answers
from code_base.experiment_utils import get_text_getter

# pyterrier-rag imports
from pyterrier_rag.prompt import Concatenator
import pyterrier_rag.measures as rag_measures


# ─── Pipeline Factories ──────────────────────────────────

def build_retrieval_pipelines(c, *, index_ref, reranker):
    """Retrieval-only pipelines for a given cutoff *c*.

    Returns:
        Dict mapping system names to retrieval pipelines.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    pipeline_rerank = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )
    return {
        "BM25": bm25,
        "BM25+Reranker": pipeline_rerank,
    }


def build_rag_pipelines(c, *, index_ref, reranker, reader):
    """Full RAG pipeline for a given cutoff *c*.

    Returns:
        Dict mapping system name to RAG pipeline.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    retrieval_pipe = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )
    concat = Concatenator(max_elements=constants.EXPERIMENT_5_TOP_K)
    rag_pipe = retrieval_pipe >> concat >> reader
    return {
        "Full_RAG": rag_pipe,
    }


# ─── Plot Configuration ──────────────────────────────────

RETRIEVAL_PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["BM25_R@10", "BM25+Reranker_R@10"],
        "labels": ["BM25 Baseline", "BM25 + MonoT5"],
        "title": "Effect of Cutoff (c) on Recall (R@10) — RAG Retrieval",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "filename": "exp5_cutoff_vs_R10.png",
        "use_log_scale": True,
    },
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["BM25_R@100", "BM25+Reranker_R@100"],
        "labels": ["BM25 Baseline", "BM25 + MonoT5"],
        "title": "Effect of Cutoff (c) on Recall (R@100) — RAG Retrieval",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "filename": "exp5_cutoff_vs_R100.png",
        "use_log_scale": True,
    },
]

RAG_PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["Full_RAG_EM", "Full_RAG_F1"],
        "labels": ["Exact Match", "F1 Score"],
        "title": "Effect of Cutoff (c) on RAG Answer Quality",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "Score",
        "filename": "exp5_cutoff_vs_EM_F1.png",
        "use_log_scale": True,
    },
]


# ─── Reader Setup ─────────────────────────────────────────

def get_reader():
    """Load HuggingFace RAG reader, respecting offline mode."""
    model_id = constants.RAG_MODEL_NAME
    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(
            constants.MODELS_CACHE_DIR, model_id.split("/")[-1]
        )
        if os.path.exists(local_path):
            model_id = local_path
            print(f"Loading offline model: {model_id}")
        else:
            print(f"Loading model {model_id} (offline env set, relying on HF cache)")
    else:
        print(f"Loading HuggingFace model: {model_id}")

    from pyterrier_rag import HuggingFaceBackend
    from pyterrier_rag.readers import Reader

    backend = HuggingFaceBackend(
        model_id,
        max_new_tokens=constants.RAG_MODEL_MAX_NEW_TOKENS,
        generation_args={"temperature": 0.1},
    )
    return Reader(backend)


# ─── Reranker Setup ───────────────────────────────────────

def get_reranker():
    """Load MonoT5 reranker, respecting offline mode."""
    model_id = constants.RERANKER_MODEL_NAME
    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(
            constants.MODELS_CACHE_DIR, model_id.split("/")[-1]
        )
        if os.path.exists(local_path):
            model_id = local_path
            print(f"Loading offline reranker: {model_id}")

    return MonoT5ReRanker(model=model_id, verbose=False, batch_size=constants.T5_BATCH_SIZE)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 5."""

    # ── Stage 1: Retrieval Evaluation ──
    print("=" * 60)
    print("EXPERIMENT 5 — STAGE 1: RETRIEVAL EVALUATION")
    print("=" * 60)

    retrieval_runner = ExperimentRunner(
        name="experiment_5_retrieval",
        results_subdir="exp5",
        topic_limit=constants.EXPERIMENT_5_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_5_RETRIEVAL_METRICS,
    )

    results_dir = retrieval_runner.setup()
    topics, qrels, _ = retrieval_runner.load_data()

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    reranker = get_reranker()

    ret_df = retrieval_runner.run_sweep(
        sweep_values=constants.EXPERIMENT_5_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_retrieval_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={"index_ref": index_ref, "reranker": reranker},
    )

    retrieval_runner.save(ret_df, results_dir, filename="experiment_5_retrieval_results")
    retrieval_runner.plot(ret_df, results_dir, RETRIEVAL_PLOT_CONFIGS)

    # ── Stage 2: Generation Evaluation (EM / F1) ──
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 — STAGE 2: GENERATION EVALUATION (EM/F1)")
    print("=" * 60)

    gold_answers = get_gold_answers(topics)
    reader = get_reader()

    rag_runner = ExperimentRunner(
        name="experiment_5_rag",
        results_subdir="exp5",
        topic_limit=constants.EXPERIMENT_5_TOPIC_LIMIT,
        eval_metrics=[rag_measures.EM, rag_measures.F1],
    )

    rag_df = rag_runner.run_sweep(
        sweep_values=constants.EXPERIMENT_5_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_rag_pipelines,
        topics=topics,
        qrels=gold_answers,
        factory_kwargs={"index_ref": index_ref, "reranker": reranker, "reader": reader},
    )

    rag_runner.save(rag_df, results_dir, filename="experiment_5_rag_results")
    rag_runner.plot(rag_df, results_dir, RAG_PLOT_CONFIGS)

    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
