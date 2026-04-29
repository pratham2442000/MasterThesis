"""
Experiment 6 — Full RAG Pipeline with Query Rewrites (MPI)
============================================================
Combines the rewriting + fusion approach from Experiment 2 with the
full RAG generation pipeline from Experiment 5.

Sweeps over N (number of rewrites) with best fusion methods, then
evaluates generation quality with EM and F1.
"""

import logging
import os
import sys
import traceback
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI

# Add parent directory to path (repo root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    get_text_getter,
    get_reader,
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
)
from code_base.wiki_loader_pyt import get_gold_answers
from code_base.transformers import (
    preprocess_query,
    QueryRewriteTransformer,
    MergeRewrites,
    AttachOriginalQuery,
    Dedup,
    DedupAndCutoff,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    RRFTransformer,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_6_mpi.log"


def setup_environment(rank: int):
    """Configures environment variables and PyTerrier for offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    pt.init()
    if rank == 0:
        logging.info("PyTerrier initialized (Offline Mode).")


def setup_logging(rank: int, comm: MPI.Comm):
    """Sets up rank-aware logging."""
    file_mode = 'w' if rank == 0 else 'a'

    if rank != 0:
        comm.Barrier()

    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format=f'%(asctime)s - Rank {rank} - %(message)s',
        filemode=file_mode
    )

    if rank == 0:
        comm.Barrier()


def load_t5_reranker(batch_size: int, rank: int):
    """Loads the MonoT5 reranker, preferring local paths for cluster execution."""
    from pyterrier_t5 import MonoT5ReRanker

    mono_t5_path = os.path.join(LOCAL_MODEL_BASE, "castorini/monot5-base-msmarco")
    t5_tok_path = os.path.join(LOCAL_MODEL_BASE, "t5-base")

    if os.path.exists(mono_t5_path):
        logging.info(f"Rank {rank}: Loading local model from {mono_t5_path}")
        tok_model = t5_tok_path if os.path.exists(t5_tok_path) else None
        if not tok_model:
            logging.warning(f"Rank {rank}: Local t5-base tokenizer not found.")

        return MonoT5ReRanker(
            model=mono_t5_path,
            tok_model=tok_model,
            verbose=False,
            batch_size=batch_size,
        )
    else:
        logging.warning(f"Rank {rank}: Local model not found. Attempting HF download.")
        return MonoT5ReRanker(verbose=False, batch_size=batch_size)


def get_rag_pipelines(
        n_rewrites: int,
        index_ref: Any,
        reranker: Any,
        reader: Any,
        query_lookup: Dict[str, str],
        cutoff: int,
        top_k: int,
) -> Dict[str, Any]:
    """Constructs RAG pipelines with query rewrites and fusion methods.

    Each pipeline: Rewrite → Retrieve → Fuse → Rerank → Concatenate → Generate.

    Args:
        n_rewrites: Number of query rewrites.
        index_ref: PyTerrier index reference.
        reranker: MonoT5ReRanker instance.
        reader: RAG reader (pyterrier_rag Reader).
        query_lookup: {qid: query_text} mapping.
        cutoff: BM25 retrieval cutoff.
        top_k: Number of passages for concatenation.

    Returns:
        Dict of {system_name: pipeline}.
    """
    from pyterrier_rag.prompt import Concatenator

    rewrite = QueryRewriteTransformer(n_rewrites, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    text_getter = get_text_getter(index_ref)
    attach_query = AttachOriginalQuery(query_lookup)
    concat = Concatenator(max_elements=top_k)

    # Baseline: BM25 + Rerank → RAG (no rewrites)
    base_retrieval = bm25 >> text_getter >> reranker
    pipe_baseline = base_retrieval >> concat >> reader

    # Dedup: Rewrite → BM25 → Merge → Dedup → Rerank → RAG
    pipe_dedup = (
            rewrite >> bm25 >> MergeRewrites() >> attach_query
            >> Dedup() >> text_getter >> reranker
            >> concat >> reader
    )

    # DedupCutoff: Rewrite → BM25 → Merge → DedupCutoff → Rerank → RAG
    pipe_dedup_cutoff = (
            rewrite >> bm25 >> MergeRewrites() >> attach_query
            >> DedupAndCutoff(cutoff) >> text_getter >> reranker
            >> concat >> reader
    )

    # CombMAX: Rewrite → BM25 → Merge → CombMAX → Rerank → RAG
    pipe_combmax = (
            rewrite >> bm25 >> MergeRewrites() >> attach_query
            >> CombMAXTransformer() >> text_getter >> reranker
            >> concat >> reader
    )

    # RRF: Rewrite → BM25 → Merge → RRF → Rerank → RAG
    pipe_rrf = (
            rewrite >> bm25 >> MergeRewrites() >> attach_query
            >> RRFTransformer() >> text_getter >> reranker
            >> concat >> reader
    )

    # Borda: Rewrite → BM25 → Borda → Rerank → RAG
    pipe_borda = (
            rewrite >> bm25
            >> BordaVotingTransformerV2(cutoff=cutoff)
            >> attach_query >> text_getter >> reranker
            >> concat >> reader
    )

    return {
        "Baseline_RAG": pipe_baseline,
        "Dedup_RAG": pipe_dedup,
        "DedupCutoff_RAG": pipe_dedup_cutoff,
        "CombMAX_RAG": pipe_combmax,
        "RRF_RAG": pipe_rrf,
        "Borda_RAG": pipe_borda,
    }


def process_results(
        all_results_lists: List[List[pd.DataFrame]],
        n_values: List[int],
) -> pd.DataFrame:
    """Aggregates per-query results from all MPI ranks."""
    final_rows = []

    for i, n in enumerate(n_values):
        dfs = [
            rank_res[i] for rank_res in all_results_lists
            if i < len(rank_res) and not rank_res[i].empty
        ]
        if not dfs:
            logging.warning(f"No results found for n={n}")
            continue

        combined = pd.concat(dfs, ignore_index=True)

        row_data = {"n": n}
        systems = combined['name'].unique()

        for system in systems:
            sys_df = combined[combined['name'] == system]
            measures = sys_df['measure'].unique()
            for measure in measures:
                mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                row_data[f"{system}_{measure}"] = mean_val

        final_rows.append(row_data)

    return pd.DataFrame(final_rows)


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for experiment 6 results."""
    common_args = {
        "x_col": "n",
        "output_dir": output_dir,
        "use_log_scale": False,
    }

    systems = [
        ("Baseline_RAG", "Baseline RAG"),
        ("Dedup_RAG", "Dedup + RAG"),
        ("DedupCutoff_RAG", "DedupCutoff + RAG"),
        ("CombMAX_RAG", "CombMAX + RAG"),
        ("RRF_RAG", "RRF + RAG"),
        ("Borda_RAG", "Borda + RAG"),
    ]

    # 1. EM plot
    em_cols = [f"{s}_EM" for s, _ in systems if f"{s}_EM" in df.columns]
    em_labels = [lbl for s, lbl in systems if f"{s}_EM" in df.columns]
    if em_cols:
        plot_metrics(
            df,
            metric_cols=em_cols,
            labels=em_labels,
            title="Experiment 6: RAG EM vs Number of Rewrites",
            xlabel="Number of Rewrites",
            ylabel="Exact Match",
            filename="n_rewrites_vs_EM.png",
            **common_args,
        )

    # 2. F1 plot
    f1_cols = [f"{s}_F1" for s, _ in systems if f"{s}_F1" in df.columns]
    f1_labels = [lbl for s, lbl in systems if f"{s}_F1" in df.columns]
    if f1_cols:
        plot_metrics(
            df,
            metric_cols=f1_cols,
            labels=f1_labels,
            title="Experiment 6: RAG F1 vs Number of Rewrites",
            xlabel="Number of Rewrites",
            ylabel="F1",
            filename="n_rewrites_vs_F1.png",
            **common_args,
        )

    # 3. EM Regression plot
    if em_cols:
        plot_with_regression(
            df,
            metric_cols=em_cols,
            labels=em_labels,
            title="Experiment 6: EM vs Rewrites Regression",
            xlabel="Number of Rewrites",
            ylabel="Exact Match",
            filename="n_rewrites_vs_EM_regression.png",
            **common_args,
        )

    # 4. F1 Regression plot
    if f1_cols:
        plot_with_regression(
            df,
            metric_cols=f1_cols,
            labels=f1_labels,
            title="Experiment 6: F1 vs Rewrites Regression",
            xlabel="Number of Rewrites",
            ylabel="F1",
            filename="n_rewrites_vs_F1_regression.png",
            **common_args,
        )


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    setup_logging(rank, comm)

    try:
        logging.info(f"Starting process. Rank {rank}/{size}")

        # 1. Environment & Data Setup
        setup_environment(rank)

        RESULTS_PATH = os.path.join(constants.RESULTS_PATH, "exp6_rag_rewrites")
        if rank == 0 and not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH, exist_ok=True)
            logging.info(f"Created results dir: {RESULTS_PATH}")

        # Load Dataset
        _, topics_df, _ = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)

        # Load rewritten queries
        filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
        filepath = os.path.join(constants.DATA_PATH, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rewrites file not found: {filepath}")

        rewritten_queries = pd.read_csv(filepath)

        # Filter topics to match rewrites and apply limit
        all_topics = topics_df[topics_df["qid"].isin(rewritten_queries["q_id"])].reset_index(drop=True)
        all_topics = all_topics.tail(constants.EXPERIMENT_6_TOPIC_LIMIT)

        # Create Lookup (needed by all ranks)
        query_lookup = {str(row["qid"]): row["query"] for _, row in all_topics.iterrows()}

        # Distribute Work
        my_topics = np.array_split(all_topics, size)[rank]
        gold_answers = get_gold_answers(my_topics)
        logging.info(f"Rank {rank}: Assigned {len(my_topics)} topics.")

        # 2. Initialize Models & Index
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
        reranker = load_t5_reranker(constants.T5_BATCH_SIZE, rank)
        reader = get_reader(live=True)
        logging.info(f"Rank {rank}: Models loaded (reranker + RAG reader).")

        # 3. Experiment Loop (varying n_rewrites)
        local_results_list = []
        N_VALUES = constants.EXPERIMENT_6_N_REWRITES
        cutoff = constants.EXPERIMENT_6_RETRIEVAL_CUTOFF

        if rank == 0:
            logging.info(f"Starting loop for N_REWRITES: {N_VALUES}")

        for N in N_VALUES:
            systems_dict = get_rag_pipelines(
                N, index_ref, reranker, reader, query_lookup,
                cutoff, constants.EXPERIMENT_6_TOP_K,
            )

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    list(systems_dict.values()),
                    my_topics,
                    gold_answers,
                    eval_metrics=constants.EXPERIMENT_6_EVAL_METRICS,
                    names=list(systems_dict.keys()),
                    verbose=(rank == 0),
                    perquery=True,
                    batch_size=constants.EXPERIMENT_BATCH_SIZE,
                    precompute_prefix=True,
                )
                exp_results['n'] = N
                local_results_list.append(exp_results)
            else:
                local_results_list.append(pd.DataFrame())

        # 4. Gather & Aggregate
        logging.info(f"Rank {rank}: Finished loop. Waiting for gather.")
        all_results_lists = comm.gather(local_results_list, root=0)

        if rank == 0:
            logging.info("Rank 0: Aggregating results...")
            results_df = process_results(all_results_lists, N_VALUES)

            logging.info("\n" + results_df.to_string())

            save_experiment_results(
                results_df,
                experiment_name="experiment_6",
                output_dir=RESULTS_PATH,
                filename="experiment_6_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            generate_plots(results_df, RESULTS_PATH)

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        comm.Abort(1)


if __name__ == "__main__":
    main()
