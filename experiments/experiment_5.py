"""
Experiment 5 — Full RAG Pipeline Cutoff Sweep
==============================================
Evaluates the full Retrieve → Rerank → Generate pipeline.

Measures answer generation quality (EM, F1, LLMasJudge) across BM25 cutoffs.

Sweeps ``constants.EXPERIMENT_5_C_VALUES``.
"""

import os
import sys

import pyterrier as pt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.dataset_factory import get_gold_answers
from code_base.experiment_utils import get_text_getter, get_reranker, get_reader, ColumnPreservingReader



# pyterrier-rag imports
from pyterrier_rag.prompt import Concatenator


def _passage_formatter(record: dict) -> str:
    """Number each passage so the model can distinguish them."""
    # record is a dict with keys from in_fields (e.g. 'text')
    return record.get("text", "")


# ─── Pipeline Factory ────────────────────────────────────


def build_rag_pipelines(c, *, index_ref, reranker, reader):
    """Full RAG pipeline for a given cutoff *c*.

    Returns:
        Dict mapping system name to RAG pipeline.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", threads=1) % c
    retrieval_pipe = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )
    concat = Concatenator(
        max_elements=constants.EXPERIMENT_5_TOP_K,
        intermediate_format=lambda text="", **kw: f"- {text}",
    )
    rag_pipe = retrieval_pipe >> concat >> reader
    rag_pipe_without_reranker = bm25 >> get_text_getter(index_ref) >> concat >> reader

    return {
        "Full_RAG_with_reranker": rag_pipe,
        "Full_RAG_without_reranker": rag_pipe_without_reranker,
    }


# ─── Plot Configuration ──────────────────────────────────

RAG_PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["Full_RAG_with_reranker_EM", "Full_RAG_with_reranker_F1",
                        "Full_RAG_without_reranker_EM", "Full_RAG_without_reranker_F1"],
        "labels": ["EM (w/ Reranker)", "F1 (w/ Reranker)",
                   "EM (w/o Reranker)", "F1 (w/o Reranker)"],
        "title": "Effect of Cutoff (c) on RAG Answer Quality",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "Score",
        "filename": "exp5_cutoff_vs_EM_F1.png",
        "use_log_scale": True,
    },
]


# ─── Reranker Setup ───────────────────────────────────────

def _get_reranker():
    """Load reranker via centralized factory."""
    return get_reranker()


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 5 — RAG pipeline only."""
    print("=" * 60)
    print("EXPERIMENT 5 — RAG PIPELINE (EM / F1 / LLMasJudge)")
    print("=" * 60)

    rag_runner = ExperimentRunner(
        name="experiment_5_rag",
        results_subdir="exp5",
        topic_limit=constants.EXPERIMENT_5_TOPIC_LIMIT,
        eval_metrics=[],  # placeholder — replaced after reader loads
    )

    results_dir = rag_runner.setup()
    topics, qrels, _ = rag_runner.load_data()

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    reranker = _get_reranker()

    gold_answers = get_gold_answers(topics).reset_index(drop=True)

    # Add unique docno and label to be a valid qrels dataframe
    gold_answers["docno"] = [f"gold_{i}" for i in range(len(gold_answers))]
    gold_answers["label"] = 3  # becomes 'relevance' via ir_measures conversion; must match LLMasJudge rel threshold
    # LLMasJudge expects qrels with columns: query_id, relevance, text
    # PyTerrier internally renames label -> relevance, so do NOT set 'relevance' separately
    gold_answers["text"] = gold_answers["gold_answer"]

    reader, raw_backend = get_reader(live=True)
    wrapped_reader = ColumnPreservingReader(reader)

    # Share raw backend with LLMasJudge to avoid loading model twice (OOM)
    eval_metrics = constants.get_experiment_5_eval_metrics(
        shared_backend=raw_backend,
        topics_df=topics,
    )
    rag_runner.eval_metrics = eval_metrics

    rag_df = rag_runner.run_sweep(
        sweep_values=constants.EXPERIMENT_5_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_rag_pipelines,
        topics=topics,
        qrels=gold_answers,
        factory_kwargs={"index_ref": index_ref, "reranker": reranker, "reader": wrapped_reader},
    )

    rag_runner.save(rag_df, results_dir, filename="experiment_5_rag_results")
    rag_runner.plot(rag_df, results_dir, RAG_PLOT_CONFIGS)

    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
