"""
Test file for the full RAG pipeline.
Tests: BM25 retrieval → T5 Reranking → Context Concatenation → Answer Generation

Uses pyterrier-rag for the generation stage and evaluation metrics.
By default, runs in DUMMY mode (no model loading, no API calls).
Pass --live to use the real HuggingFace backend with the configured model.

Saves per-query traces (context, LLM output, gold answer) for inspection.

Usage:
    python experiments/test_rag_pipeline.py          # dummy mode
    python experiments/test_rag_pipeline.py --live    # real model
"""
import argparse
import os
import sys
from datetime import datetime

from code_base.experiment_utils import get_text_getter

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

from code_base.wiki_loader_pyt import get_gold_answers
from code_base.experiment_runner import ExperimentRunner
from code_base import dataset_factory

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
        >> get_text_getter(index_ref)
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
# Single-query diagnostic test
# ---------------------------------------------------------
def run_single_query_test(live: bool, cutoff: int = 20, split_override: str = None):
    """
    Run ONE query through every pipeline stage.
    Prints intermediate output at each step to diagnose issues.

    Args:
        live: Use real models (True) or dummy/passthrough (False).
        cutoff: BM25 retrieval cutoff.
        split_override: Force a specific data split (e.g. "dev").
    """
    from code_base.transformers import preprocess_query

    print("=" * 60)
    print("SINGLE-QUERY DIAGNOSTIC TEST")
    print("=" * 60)

    # --- Step 0: Determine split and warn about answers ---
    active_split = split_override or constants.DATA_SPLIT
    print(f"\n[STEP 0] Data split: '{active_split}'")

    if active_split == "test":
        print("  !! WARNING: 'test' split has EMPTY answers !!")
        print("  !! RAG metrics (EM/F1) will always be 0   !!")
        print("  !! Use --split dev for real answers        !!")

    # --- Step 1: Load data ---
    print(f"\n[STEP 1] Loading data (split={active_split})...")
    _, topics_df, qrels_df = dataset_factory.get_dataset(
        constants.DATASET_NAME, active_split
    )
    topics_df["query"] = topics_df["query"].apply(preprocess_query)

    # Take first query only
    single_topic = topics_df.iloc[[3]]
    qid = single_topic.iloc[0]["qid"]
    query = single_topic.iloc[0]["query"]
    answer = single_topic.iloc[0].get("answer", "")

    print(f"  qid:    {qid}")
    print(f"  query:  {query}")
    print(f"  answer: '{answer}'")

    if not answer or answer.strip() == "":
        print("  !! PROBLEM: Gold answer is EMPTY !!")
        print("  !! EM/F1 will be meaningless     !!")
    else:
        print("  ✓ Gold answer present")

    # Gold answers DF
    gold_answers = get_gold_answers(single_topic)
    print(f"\n  Gold answers DataFrame:")
    print(gold_answers.to_string(index=False))

    # Qrels for this query
    query_qrels = qrels_df[qrels_df["qid"] == qid]
    print(f"\n  Qrels for qid={qid}: {len(query_qrels)} relevant docs")
    if len(query_qrels) > 0:
        print(query_qrels.head(5).to_string(index=False))
    else:
        print("  !! WARNING: No qrels found for this query !!")

    # --- Step 2: Setup index ---
    print(f"\n[STEP 2] Loading index from {constants.INDEX_PATH}...")
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    index = pt.IndexFactory.of(constants.INDEX_PATH)
    print(f"  Index docs: {index.getCollectionStatistics().getNumberOfDocuments()}")
    print(f"  ✓ Index loaded")

    # --- Step 3: BM25 retrieval ---
    print(f"\n[STEP 3] BM25 retrieval (cutoff={cutoff})...")
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_cut = bm25 % cutoff

    bm25_results = bm25_cut.transform(single_topic)
    print(f"  Retrieved {len(bm25_results)} docs")
    if len(bm25_results) == 0:
        print("  !! PROBLEM: BM25 returned 0 results !!")
        return
    print(f"  Top-3 docnos: {bm25_results['docno'].head(3).tolist()}")
    print(f"  Score range: [{bm25_results['score'].min():.4f}, {bm25_results['score'].max():.4f}]")
    print(f"  Columns: {list(bm25_results.columns)}")

    # --- Step 4: Get text ---
    print(f"\n[STEP 4] Fetching document text...")
    text_pipe = get_text_getter(index_ref)
    with_text = text_pipe.transform(bm25_results)

    if "text" not in with_text.columns:
        print("  !! PROBLEM: 'text' column missing after get_text !!")
        print(f"  Columns: {list(with_text.columns)}")
        return

    empty_text_count = (with_text["text"].str.strip() == "").sum()
    null_text_count = with_text["text"].isna().sum()
    print(f"  Docs with text: {len(with_text)}")
    print(f"  Empty text: {empty_text_count}, Null text: {null_text_count}")
    print(f"  Sample text (first 200 chars): {with_text.iloc[0]['text'][:200]}")
    print(f"  ✓ Text fetched")

    # --- Step 5: Reranking ---
    print(f"\n[STEP 5] Reranking...")
    if live:
        from pyterrier_t5 import MonoT5ReRanker
        reranker = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)
        print("  Using MonoT5ReRanker")
    else:
        reranker = PassthroughReranker()
        print("  Using PassthroughReranker (test mode)")

    reranked = reranker.transform(with_text)
    print(f"  Reranked {len(reranked)} docs")
    print(f"  Top-3 scores: {reranked['score'].head(3).tolist()}")
    print(f"  ✓ Reranking done")

    # --- Step 6: Concatenation ---
    print(f"\n[STEP 6] Context concatenation (top_k={constants.RAG_CONCATENATOR_TOP_K})...")
    concat = Concatenator(max_elements=constants.RAG_CONCATENATOR_TOP_K)
    concatenated = concat.transform(reranked)

    print(f"  Output rows: {len(concatenated)}")
    print(f"  Columns: {list(concatenated.columns)}")

    if "qcontext" in concatenated.columns:
        ctx = concatenated.iloc[0]["qcontext"]
        print(f"  Context length: {len(str(ctx))} chars")
        print(f"  Context preview (first 300 chars):")
        print(f"    {str(ctx)[:300]}")
        print(f"  ✓ Context built")
    else:
        print("  !! PROBLEM: 'qcontext' column missing !!")
        print("  !! Concatenator output unexpected     !!")

    # --- Step 7: Generation ---
    print(f"\n[STEP 7] Answer generation...")
    reader = get_reader(live)
    generated = reader.transform(concatenated)

    if "qanswer" in generated.columns:
        gen_answer = generated.iloc[0]["qanswer"]
        print(f"  Generated answer: '{gen_answer}'")
        print(f"  ✓ Generation complete")
    else:
        print("  !! PROBLEM: 'qanswer' column missing !!")
        print(f"  Columns: {list(generated.columns)}")

    # --- Step 8: Compare with gold ---
    print(f"\n[STEP 8] Comparison...")
    print(f"  Gold answer:      '{answer}'")
    if "qanswer" in generated.columns:
        print(f"  Generated answer: '{generated.iloc[0]['qanswer']}'")
    else:
        print("  No generated answer to compare")

    # --- Step 9: Save trace ---
    print(f"\n[STEP 9] Saving per-query trace...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traces_dir = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME,
        "test_rag_pipeline", f"run_{timestamp}", "traces"
    )
    os.makedirs(traces_dir, exist_ok=True)

    trace_data = {
        "qid": [qid],
        "query": [query],
        "qcontext": [str(concatenated.iloc[0].get("qcontext", "")) if "qcontext" in concatenated.columns else ""],
        "qanswer": [str(generated.iloc[0].get("qanswer", "")) if "qanswer" in generated.columns else ""],
        "gold_answer": [answer],
    }
    trace_df = pd.DataFrame(trace_data)
    trace_path = os.path.join(traces_dir, "single_query_trace.csv")
    trace_df.to_csv(trace_path, index=False)
    print(f"  Trace saved: {trace_path}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


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
    parser.add_argument(
        "--single-query", action="store_true",
        help="Run single-query diagnostic test instead of full sweep"
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Override data split (e.g. 'dev' for real answers)"
    )
    args = parser.parse_args()

    # Single-query diagnostic mode
    if args.single_query:
        run_single_query_test(
            live=args.live,
            cutoff=args.cutoff,
            split_override=args.split,
        )
        return

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

    # ---- 5. Save Per-Query Traces ----
    print("\n" + "=" * 60)
    print("SAVING PER-QUERY TRACES")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    traces_dir = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME,
        "test_rag_pipeline", f"run_{timestamp}", "traces"
    )
    os.makedirs(traces_dir, exist_ok=True)

    ga_map = dict(zip(
        gold_answers["qid"].astype(str),
        gold_answers["gold_answer"],
    ))

    for cutoff_val in constants.EXPERIMENT_1_C_VALUES:
        pipelines = rag_pipeline_factory(
            cutoff_val, index_ref=index_ref, reranker=reranker, reader=reader
        )
        for sys_name, pipeline in pipelines.items():
            print(f"  Tracing {sys_name} (cutoff={cutoff_val})...")
            try:
                raw_output = pipeline.transform(topics)
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
                fname = f"trace_{sys_name}_cutoff_{cutoff_val}.csv"
                trace_df.to_csv(os.path.join(traces_dir, fname), index=False)
                print(f"  Saved: {fname} ({len(trace_df)} rows)")
            except Exception as e:
                print(f"  !! Trace failed for {sys_name}: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
