"""
Test file for the full RAG pipeline.
Tests: BM25 retrieval → T5 Reranking → Context Concatenation → Answer Generation

Uses pyterrier-rag for the generation stage and evaluation metrics.
By default, runs in DUMMY mode (no model loading, no API calls).
Pass --live to use the real HuggingFace backend with the configured model.

Usage:
    python experiments/test_rag_pipeline.py          # dummy mode
    python experiments/test_rag_pipeline.py --live    # real model
"""
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Enforce offline mode if configured
import constants
if getattr(constants, "OFFLINE_MODE", False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG

from code_base import dataset_factory
from code_base.transformers import preprocess_query
from code_base.wiki_loader_pyt import get_gold_answers
from code_base.experiment_runner import ExperimentRunner

# pyterrier-rag imports
from pyterrier_rag.prompt import Concatenator
import pyterrier_rag.measures as rag_measures


# ---------------------------------------------------------
# Dummy Reader (test mode — no model needed)
# ---------------------------------------------------------
class DummyReader(pt.Transformer):
    """
    A mock reader that returns a dummy answer derived from the context.
    Used for testing the pipeline wiring without loading a real model.
    """

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result = inp.copy()
        if "qcontext" in result.columns:
            result["qanswer"] = result["qcontext"].apply(
                lambda ctx: f"Dummy answer based on: {str(ctx)[:50]}"
            )
        elif "text" in result.columns:
            result["qanswer"] = result["text"].apply(
                lambda txt: f"Dummy answer based on: {str(txt)[:50]}"
            )
        else:
            result["qanswer"] = "no context available"
        return result


# ---------------------------------------------------------
# Simulated Reranker (passes through scores unchanged)
# ---------------------------------------------------------
class PassthroughReranker(pt.Transformer):
    """
    A no-op reranker that passes through results unchanged.
    Avoids loading MonoT5 during test runs.
    """

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return inp


# ---------------------------------------------------------
# Build pipelines
# ---------------------------------------------------------
def build_retrieval_pipeline(index_ref, reranker, cutoff):
    """BM25 + T5 reranker (retrieval-only, no generation)."""
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_retrieved = bm25 % cutoff

    pipeline = (
        bm25_retrieved
        >> pt.text.get_text(index_ref, "text")
        >> reranker
    )
    return bm25_retrieved, pipeline


def build_rag_pipeline(retrieval_pipeline, reader, top_k):
    """
    Full RAG: Retrieve + Rerank → Concatenate top-k → Generate answer.
    """
    concat = Concatenator(max_elements=top_k)
    rag_pipeline = retrieval_pipeline >> concat >> reader
    return rag_pipeline


def get_reader(live: bool):
    """
    Returns the RAG reader.
    - live=False: DummyReader (no model)
    - live=True: pyterrier_rag Reader with HuggingFaceBackend
    """
    if not live:
        print("Using DummyReader (test mode — no model loaded)")
        return DummyReader()

    model_id = constants.RAG_MODEL_NAME
    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(constants.MODELS_CACHE_DIR, model_id.split('/')[-1])
        if os.path.exists(local_path):
            model_id = local_path
            print(f"Loading offline HuggingFace model from local cache: {model_id}")
        else:
            print(f"Loading HF model {model_id} (offline mode env set, relying on HF cache)")
    else:
        print(f"Loading HuggingFace model: {model_id}")

    from pyterrier_rag import HuggingFaceBackend
    from pyterrier_rag.readers import Reader

    # Llama models often require device_map="auto" to fit across GPUs
    backend = HuggingFaceBackend(
        model_id,
        max_new_tokens=constants.RAG_MODEL_MAX_NEW_TOKENS,
        generation_args={"temperature": 0.1},
    )
    reader = Reader(backend)
    return reader


# ---------------------------------------------------------
# Pipeline factories for ExperimentRunner sweeps
# ---------------------------------------------------------
def retrieval_pipeline_factory(cutoff, *, index_ref, reranker):
    bm25_baseline, retrieval_pipe = build_retrieval_pipeline(
        index_ref, reranker, cutoff
    )
    return {
        "BM25": bm25_baseline,
        "BM25+Reranker": retrieval_pipe,
    }

def rag_pipeline_factory(cutoff, *, index_ref, reranker, reader):
    _, retrieval_pipe = build_retrieval_pipeline(
        index_ref, reranker, cutoff
    )
    rag_pipe = build_rag_pipeline(
        retrieval_pipe, reader, constants.RAG_CONCATENATOR_TOP_K
    )
    return {
        "Full_RAG": rag_pipe,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test full RAG pipeline")
    parser.add_argument(
        "--live", action="store_true",
        help="Use real HuggingFace model instead of dummy reader"
    )
    parser.add_argument(
        "--n-queries", type=int, default=10,
        help="Number of queries to test with (default: 10)"
    )
    parser.add_argument(
        "--cutoff", type=int, default=20,
        help="BM25 retrieval cutoff (default: 20)"
    )
    args = parser.parse_args()

    # ---- 1. Load Data via ExperimentRunner ----
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Use a dummy ExperimentRunner just to load data easily
    data_runner = ExperimentRunner(
        name="data_loader",
        results_subdir="test_rag_pipeline",
        topic_limit=args.n_queries,
        eval_metrics=[],
    )
    topics, qrels, query_lookup = data_runner.load_data()
    gold_answers = get_gold_answers(topics)

    # ---- 2. Setup Index + Models ----
    print("\n" + "=" * 60)
    print("SETTING UP INDEX & MODELS")
    print("=" * 60)

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)

    if args.live:
        model_id = constants.RERANKER_MODEL_NAME
        if getattr(constants, "OFFLINE_MODE", False):
            local_path = os.path.join(constants.MODELS_CACHE_DIR, model_id.split('/')[-1])
            if os.path.exists(local_path):
                model_id = local_path

        from pyterrier_t5 import MonoT5ReRanker
        reranker = MonoT5ReRanker(model=model_id, verbose=False, batch_size=constants.T5_BATCH_SIZE)
        print(f"Using MonoT5ReRanker with {model_id}")
    else:
        reranker = PassthroughReranker()
        print("Using PassthroughReranker (test mode)")

    reader = get_reader(args.live)

    # ---- 3. Retrieval Sweep ----
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL PIPELINE")
    print("=" * 60)

    retrieval_runner = ExperimentRunner(
        name="test_rag_retrieval",
        results_subdir="test_rag_pipeline",
        topic_limit=args.n_queries,
        eval_metrics=[R @ 10, R @ 100],  # Recall at K!
    )
    retrieval_runner.setup()

    ret_df = retrieval_runner.run_sweep(
        sweep_values=constants.EXPERIMENT_1_C_VALUES,
        sweep_param="cutoff",
        pipeline_factory=retrieval_pipeline_factory,
        topics=topics,
        qrels=qrels,
        factory_kwargs={"index_ref": index_ref, "reranker": reranker}
    )
    print("\nRetrieval Evaluation Results:")
    print(ret_df)

    # ---- 4. Generation Pipeline Setup and Eval ----
    print("\n" + "=" * 60)
    print("TESTING FULL RAG PIPELINE (EM/F1)")
    print("=" * 60)

    rag_eval_metrics = [
        rag_measures.EM,
        rag_measures.F1,
    ]

    rag_runner = ExperimentRunner(
        name="test_rag_generation",
        results_subdir="test_rag_pipeline",
        topic_limit=args.n_queries,
        eval_metrics=rag_eval_metrics,
    )
    # use gold_answers as the 'qrels' input
    try:
        rag_df = rag_runner.run_sweep(
            sweep_values=constants.EXPERIMENT_1_C_VALUES,
            sweep_param="cutoff",
            pipeline_factory=rag_pipeline_factory,
            topics=topics,
            qrels=gold_answers,
            factory_kwargs={"index_ref": index_ref, "reranker": reranker, "reader": reader}
        )
        print("\nRAG Generation Evaluation Results:")
        print(rag_df)
    except Exception as e:
        print(f"\nRAG Evaluation failed: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
