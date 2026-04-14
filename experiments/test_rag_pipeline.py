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

import pandas as pd
import pyterrier as pt

import constants
from code_base.experiment_utils import preprocess_query
from code_base.wiki_loader_pyt import wiki_loader_pyt, get_gold_answers

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
    
    Expects input from Concatenator, which produces one row per qid
    with a 'qcontext' column containing the concatenated document text.
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
# Build pipelines
# ---------------------------------------------------------
def build_retrieval_pipeline(index, reranker, cutoff):
    """BM25 + T5 reranker (retrieval-only, no generation)."""
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    bm25_retrieved = bm25 % cutoff

    pipeline = (
            bm25_retrieved
            >> pt.text.get_text(index, "text")
            >> reranker
    )
    return bm25_retrieved, pipeline


def build_rag_pipeline(retrieval_pipeline, reader, top_k):
    """
    Full RAG: Retrieve + Rerank → Concatenate top-k → Generate answer.

    Uses pyterrier_rag.prompt.Concatenator to aggregate top-k docs into
    a 'qcontext' field, then feeds into the reader.
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

    print(f"Loading HuggingFace model: {constants.RAG_MODEL_NAME}")
    from pyterrier_rag import HuggingFaceBackend
    from pyterrier_rag.readers import Reader

    backend = HuggingFaceBackend(
        constants.RAG_MODEL_NAME,
        max_new_tokens=constants.RAG_MODEL_MAX_NEW_TOKENS,
        generation_args={"temperature": 0.1},
    )
    reader = Reader(backend)
    return reader


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
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test full RAG pipeline")
    parser.add_argument(
        "--live", action="store_true",
        help="Use real HuggingFace model instead of dummy reader"
    )
    parser.add_argument(
        "--n-queries", type=int, default=3,
        help="Number of queries to test with (default: 3)"
    )
    parser.add_argument(
        "--cutoff", type=int, default=20,
        help="BM25 retrieval cutoff (default: 20)"
    )
    args = parser.parse_args()

    # ---- 1. Load Data (dev split — has ground truth answers) ----
    print("=" * 60)
    print("LOADING DATA (dev split)")
    print("=" * 60)

    # Use dev split because test.json has no answers
    docs_df, topics_df, qrels_df = wiki_loader_pyt(constants.DATA_PATH, "dev")
    topics_df["query"] = topics_df["query"].apply(preprocess_query)

    # Subset for testing
    topics = topics_df[:args.n_queries].copy()

    # Filter for topics that have non-empty answers
    topics = topics[topics["answer"].str.strip() != ""]
    if topics.empty:
        print("ERROR: No topics with non-empty answers found in dev split!")
        return

    print(f"\nTest topics ({len(topics)}):")
    for _, row in topics.iterrows():
        print(f"  qid={row['qid']}: {row['query'][:60]}...")
        print(f"    answer: {row['answer']}")

    # Build gold answers DataFrame (pyterrier-rag GA format)
    gold_answers = get_gold_answers(topics)
    print(f"\nGold answers DataFrame: {len(gold_answers)} rows")
    print(gold_answers.head())

    # ---- 2. Setup Index + Reranker ----
    print("\n" + "=" * 60)
    print("SETTING UP INDEX & MODELS")
    print("=" * 60)

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)

    # Use passthrough reranker in dummy mode, real T5 in live mode
    if args.live:
        from pyterrier_t5 import MonoT5ReRanker
        reranker = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)
        print("Using MonoT5ReRanker")
    else:
        reranker = PassthroughReranker()
        print("Using PassthroughReranker (test mode)")

    # ---- 3. Build Pipelines ----
    print("\n" + "=" * 60)
    print("BUILDING PIPELINES")
    print("=" * 60)

    bm25_baseline, retrieval_pipeline = build_retrieval_pipeline(
        index_ref, reranker, args.cutoff
    )

    reader = get_reader(args.live)
    rag_pipeline = build_rag_pipeline(
        retrieval_pipeline, reader, constants.RAG_CONCATENATOR_TOP_K
    )

    print(f"\nPipeline architecture:")
    print(
        f"  BM25 % {args.cutoff} → get_text → Reranker → Concatenator(top_k={constants.RAG_CONCATENATOR_TOP_K}) → Reader")

    # ---- 4. Test retrieval pipeline first ----
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL PIPELINE")
    print("=" * 60)

    from pyterrier.measures import R, nDCG
    retrieval_metrics = [R @ 10, nDCG @ 10]

    retrieval_results = pt.Experiment(
        [bm25_baseline, retrieval_pipeline],
        topics,
        qrels_df,
        eval_metrics=retrieval_metrics,
        names=["BM25", "BM25+Reranker"],
        verbose=True,
    )
    print("\nRetrieval Results:")
    print(retrieval_results.to_string())

    # ---- 5. Test full RAG pipeline ----
    print("\n" + "=" * 60)
    print("TESTING FULL RAG PIPELINE")
    print("=" * 60)

    # Run the RAG pipeline to get generated answers
    rag_output = rag_pipeline.transform(topics)
    print(f"\nRAG output shape: {rag_output.shape}")
    print(f"RAG output columns: {list(rag_output.columns)}")

    if "qanswer" in rag_output.columns:
        print("\nGenerated Answers:")
        for _, row in rag_output.iterrows():
            qid = row["qid"]
            generated = row.get("qanswer", "N/A")
            gt_row = gold_answers[gold_answers["qid"] == qid]
            ground_truth = gt_row.iloc[0]["gold_answer"] if not gt_row.empty else "N/A"
            print(f"  qid={qid}:")
            print(f"    Generated:    {generated[:100]}")
            print(f"    Ground Truth: {ground_truth}")
    else:
        print("WARNING: 'qanswer' column not found in RAG output!")
        print(f"  Available columns: {list(rag_output.columns)}")
        print(rag_output.head())

    # ---- 6. Evaluate with RAG metrics ----
    print("\n" + "=" * 60)
    print("EVALUATING RAG METRICS")
    print("=" * 60)

    rag_eval_metrics = [
        rag_measures.EM,
        rag_measures.F1,
        rag_measures.ROUGE1F,
        rag_measures.ROUGELF,
        rag_measures.AnswerLen,
        rag_measures.AnswerZeroLen,
    ]

    # BERTScore requires qrels with text — skip in dummy mode to keep test fast
    if args.live:
        rag_eval_metrics.append(
            rag_measures.BERTScore(rel=1, submeasure='f1', agg='max')
        )

    try:
        rag_results = pt.Experiment(
            [rag_pipeline],
            topics,
            gold_answers,
            eval_metrics=rag_eval_metrics,
            names=["Full RAG Pipeline"],
            verbose=True,
        )
        print("\nRAG Evaluation Results:")
        print(rag_results.to_string())
    except Exception as e:
        print(f"\nRAG Evaluation failed: {e}")
        print("This may happen if pyterrier-rag measures expect a specific DataFrame format.")
        print("Falling back to manual evaluation...")

        # Manual metric computation as fallback
        _manual_evaluation(rag_output, gold_answers)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def _manual_evaluation(rag_output, gold_answers):
    """Fallback: compute metrics manually if pt.Experiment fails."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    if "qanswer" not in rag_output.columns:
        print("Cannot evaluate: no 'qanswer' column")
        return

    merged = rag_output.merge(gold_answers, on="qid", how="inner")

    ems, f1s, rouge_ls = [], [], []
    for _, row in merged.iterrows():
        pred = str(row["qanswer"]).strip().lower()
        ref = str(row["gold_answer"]).strip().lower()
        if not ref:
            continue

        # Exact Match
        ems.append(1.0 if pred == ref else 0.0)

        # Token F1
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        if not pred_tokens or not ref_tokens:
            f1s.append(0.0)
        else:
            precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
            recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)

        # ROUGE-L
        scores = scorer.score(ref, pred)
        rouge_ls.append(scores['rougeL'].fmeasure)

    print(f"\n  Manual Evaluation ({len(ems)} queries):")
    print(f"  Exact Match:  {sum(ems) / len(ems):.4f}" if ems else "  Exact Match:  N/A")
    print(f"  Token F1:     {sum(f1s) / len(f1s):.4f}" if f1s else "  Token F1:     N/A")
    print(f"  ROUGE-L F:    {sum(rouge_ls) / len(rouge_ls):.4f}" if rouge_ls else "  ROUGE-L F:    N/A")


if __name__ == "__main__":
    main()
