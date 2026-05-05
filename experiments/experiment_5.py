"""
Experiment 5 — Full RAG Pipeline Cutoff Sweep
==============================================
Evaluates the full Retrieve → Rerank → Generate pipeline.

Stage 1: Retrieval quality (R@10, R@100) across BM25 cutoffs.
Stage 2: Answer generation quality (EM, F1) using pyterrier-rag.
Stage 3: Per-query trace logging (context, LLM output, gold answer).

Sweeps ``constants.EXPERIMENT_5_C_VALUES``.
"""

import os
import sys

import pyterrier as pt
from pyterrier.measures import R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.dataset_factory import get_gold_answers
from code_base.experiment_utils import get_text_getter, get_reranker

# pyterrier-rag imports
from pyterrier_rag.prompt import Concatenator
import pyterrier_rag.measures as rag_measures


RAG_PROMPT = (
    "You are given passages and a question. "
    "Answer the question using ONLY information from the passages below. "
    "Give a short answer (1-5 words). Do not explain. Do not add extra text.\n\n"
    "Passages:\n{{ qcontext }}\n\n"
    "Question: {{ query }}\n"
    "Answer:"
)


def _passage_formatter(record: dict) -> str:
    """Number each passage so the model can distinguish them."""
    # record is a dict with keys from in_fields (e.g. 'text')
    return record.get("text", "")


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
        "metric_cols": ["Full_RAG_EM", "Full_RAG_F1", "Full_RAG_without_reranker_EM", "Full_RAG_without_reranker_F1"],
        "labels": ["EM_Reranker", "F1_Reranker", "EM_without_reranker", "F1_without_reranker"],
        "title": "Effect of Cutoff (c) on RAG Answer Quality",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "Score",
        "filename": "exp5_cutoff_vs_EM_F1.png",
        "use_log_scale": True,
    },
]


def save_per_query_traces(
        topics,
        gold_answers,
        pipelines: dict,
        output_dir: str,
        sweep_param: str,
        sweep_value,
) -> None:
    """Run each RAG pipeline and save per-query output.

    Saves CSV with: qid, query, qcontext, qanswer, gold_answer.

    Args:
        topics: Topic DataFrame.
        gold_answers: Gold answers DataFrame (qid, gold_answer).
        pipelines: Dict of {system_name: pipeline}.
        output_dir: Directory to save trace CSVs.
        sweep_param: Swept parameter name.
        sweep_value: Current sweep value.
    """
    import pandas as pd

    ga_map = dict(zip(
        gold_answers["qid"].astype(str),
        gold_answers["gold_answer"],
    ))

    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    for sys_name, pipeline in pipelines.items():
        print(f"  Saving traces for {sys_name} ({sweep_param}={sweep_value})...")
        try:
            raw_output = pipeline.transform(topics)
        except Exception as e:
            print(f"  !! Trace failed for {sys_name}: {e}")
            continue

        trace_rows = []
        for _, row in raw_output.iterrows():
            qid = str(row.get("qid", ""))
            trace_rows.append({
                "qid": qid,
                "query": row.get("query", ""),
                "qcontext": row.get("qcontext", ""),
                "qanswer": row.get("qanswer", ""),
                "gold_answer": ga_map.get(qid, ""),
            })

        trace_df = pd.DataFrame(trace_rows)
        fname = f"trace_{sys_name}_{sweep_param}_{sweep_value}.csv"
        path = os.path.join(traces_dir, fname)
        trace_df.to_csv(path, index=False)
        print(f"  Trace saved: {path} ({len(trace_df)} rows)")


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
        generation_args={"temperature": 0.1, "do_sample": False},
    )
    return Reader(backend, prompt=RAG_PROMPT)


# ─── Reranker Setup ───────────────────────────────────────

def _get_reranker():
    """Load reranker via centralized factory."""
    return get_reranker()


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
    reranker = _get_reranker()

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
        eval_metrics=[rag_measures.EM, rag_measures.F1, rag_measures.LLMasJudge],
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

    # ── Stage 3: Per-query trace logging ──
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 — STAGE 3: SAVING PER-QUERY TRACES")
    print("=" * 60)

    for c in constants.EXPERIMENT_5_C_VALUES:
        rag_pipelines = build_rag_pipelines(
            c, index_ref=index_ref, reranker=reranker, reader=reader
        )
        save_per_query_traces(
            topics=topics,
            gold_answers=gold_answers,
            pipelines=rag_pipelines,
            output_dir=results_dir,
            sweep_param="c",
            sweep_value=c,
        )

    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
