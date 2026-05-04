import logging
import os
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    get_text_getter,
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
)
from code_base.transformers import (
    preprocess_query,
    QueryRewriteTransformer,
    MergeRewrites,
    Dedup,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    RRFTransformer,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_2b_mpi.log"


def setup_environment(rank: int):
    """Configures environment variables and PyTerrier for offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Initialize PyTerrier
    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    pt.init()
    if rank == 0:
        logging.info("PyTerrier initialized (Offline Mode).")


def setup_logging(rank: int, comm: MPI.Comm):
    """Sets up rank-aware logging."""
    file_mode = 'w' if rank == 0 else 'a'

    # Barrier to ensure rank 0 clears the file first
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


def load_t5_reranker(batch_size: int, rank: int) -> MonoT5ReRanker:
    """Loads the MonoT5 reranker, preferring local paths for cluster execution."""
    mono_t5_path = os.path.join(LOCAL_MODEL_BASE, "castorini/monot5-base-msmarco")
    t5_tok_path = os.path.join(LOCAL_MODEL_BASE, "t5-base")

    if os.path.exists(mono_t5_path):
        logging.info(f"Rank {rank}: Loading local model from {mono_t5_path}")

        # Check for local tokenizer/config fallback
        tok_model = t5_tok_path if os.path.exists(t5_tok_path) else None
        if not tok_model:
            logging.warning(f"Rank {rank}: Local t5-base tokenizer not found. Using model path only.")

        return MonoT5ReRanker(
            model=mono_t5_path,
            tok_model=tok_model,
            verbose=False,
            batch_size=batch_size
        )
    else:
        logging.warning(f"Rank {rank}: Local model not found. Attempting HF download.")
        return MonoT5ReRanker(verbose=False, batch_size=batch_size)


def get_pipelines(
        n_rewrites: int,
        index_ref: Any,
        reranker: MonoT5ReRanker,
        query_lookup: Dict[str, str],
        cutoff: int
) -> Dict[str, Any]:
    """Constructs the list of retrieval pipelines for a specific N (Experiment 2b: Rerank-then-Merge)."""

    # Transformers
    rewrite = QueryRewriteTransformer(n_rewrites, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    text_getter = get_text_getter(index_ref)
    merge_rewrites = MergeRewrites()

    # Components
    dedup = Dedup()
    borda_voting = BordaVotingTransformerV2(cutoff=cutoff, weight=None, method="borda")
    borda_weighted = BordaVotingTransformerV2(
        cutoff=cutoff,
        weight=lambda x: np.exp(x * 0.1),
        method="borda"
    )
    combmax = CombMAXTransformer()

    # 1. BM25 Baseline (Original Query)
    # Using 'bm25' directly on original topics
    pipeline_bm25_base = bm25

    # 2. Base BM25 + Rerank (Original Query)
    pipeline_bm25_t5 = (
            bm25
            >> text_getter
            >> reranker
    )

    # 3. Dedup Method (Rerank EACH rewrite, then merge)
    # rewrite -> bm25 -> text -> monoT5 -> merge_rewrites -> dedup
    pipeline_dedup = (
            rewrite
            >> bm25
            >> text_getter
            >> reranker
            >> merge_rewrites
            >> dedup
    )

    # 4. Borda Voting (Rerank EACH rewrite, then Borda merge)
    # rewrite -> bm25 -> text -> monoT5 -> borda (borda handles merge)
    pipeline_borda = (
            rewrite
            >> bm25
            >> text_getter
            >> reranker
            >> borda_voting
    )

    # 5. Borda Weighted
    pipeline_borda_weighted = (
            rewrite
            >> bm25
            >> text_getter
            >> reranker
            >> borda_weighted
    )

    # 6. CombMAX (Rerank EACH rewrite, then CombMAX merge)
    # rewrite -> bm25 -> text -> monoT5 -> merge_rewrites -> combmax
    pipeline_combmax = (
            rewrite
            >> bm25
            >> text_getter
            >> reranker
            >> merge_rewrites
            >> combmax
    )

    # 7. RRF (Rerank EACH rewrite, then RRF merge)
    pipeline_rrf = (
            rewrite
            >> bm25
            >> text_getter
            >> reranker
            >> MergeRewrites()
            >> RRFTransformer()
    )

    return {
        "BM25": pipeline_bm25_base,
        "BM25_MonoT5": pipeline_bm25_t5,
        "Pipeline_Dedup": pipeline_dedup,
        "Pipeline_Borda": pipeline_borda,
        "Pipeline_BordaWeighted": pipeline_borda_weighted,
        "Pipeline_CombMAX": pipeline_combmax,
        "Pipeline_RRF": pipeline_rrf
    }


def process_results(all_results_lists: List[List[pd.DataFrame]], n_values: List[int], metrics: List[Any],
                    cutoff: int) -> pd.DataFrame:
    """Aggregates results from all MPI ranks and calculates means."""
    final_rows = []

    for i, N in enumerate(n_values):
        # Gather all dataframes for this specific N (i-th iteration) from all ranks
        dfs_for_n = [
            rank_res[i] for rank_res in all_results_lists
            if i < len(rank_res) and not rank_res[i].empty
        ]

        if not dfs_for_n:
            logging.warning(f"No results found for n={N}")
            continue

        combined_df = pd.concat(dfs_for_n, ignore_index=True)

        row_data = {"n": N}
        systems = combined_df['name'].unique()

        for system in systems:
            sys_df = combined_df[combined_df['name'] == system]
            measures = sys_df['measure'].unique()
            for measure in measures:
                mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                if str(measure) == f"R@{cutoff}":
                    key = f"{system}_R@c"
                else:
                    key = f"{system}_{measure}"
                row_data[key] = mean_val

        final_rows.append(row_data)

    results_df = pd.DataFrame(final_rows)
    return results_df


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Encapsulate plotting logic."""
    common_args = {
        "x_col": "n",
        "output_dir": output_dir,
        "use_log_scale": False
    }

    systems = [
        ("BM25", "BM25 Baseline"),
        ("BM25_MonoT5", "BM25 + MonoT5"),
        ("Pipeline_Dedup", "Dedup Method"),
        ("Pipeline_Borda", "Borda Voting"),
        ("Pipeline_BordaWeighted", "Borda Weighted"),
        ("Pipeline_CombMAX", "CombMAX"),
        ("Pipeline_RRF", "RRF")
    ]

    # 1. Recall Plot
    recall_metrics = []
    recall_labels = []

    for sys, label in systems:
        col_c = f"{sys}_R@c"
        col_10 = f"{sys}_R@10"

        if col_c in df.columns:
            recall_metrics.append(col_c)
            recall_labels.append(label)
        elif col_10 in df.columns:
            recall_metrics.append(col_10)
            recall_labels.append(label)
        else:
            logging.warning(f"Skipping Recall plot for {sys}: Neither {col_c} nor {col_10} found.")

    if recall_metrics:
        plot_metrics(
            df,
            metric_cols=recall_metrics,
            labels=recall_labels,
            title="Experiment 2b: Document Combination Methods Comparison (Recall)",
            xlabel="Number of Rewrites",
            filename="/n_rewrites_vs_R_comparison_2b.png",
            **common_args
        )

    # 2. NDCG Plot
    ndcg_metrics = []
    ndcg_labels = []
    for sys, label in systems:
        col = f"{sys}_nDCG@10"
        if col in df.columns:
            ndcg_metrics.append(col)
            ndcg_labels.append(label)
        else:
            logging.warning(f"Skipping nDCG plot for {sys}: {col} not found.")

    if ndcg_metrics:
        plot_metrics(
            df,
            metric_cols=ndcg_metrics,
            labels=ndcg_labels,
            title="Experiment 2b: Document Combination Methods Comparison (nDCG@10)",
            xlabel="Number of Rewrites",
            filename="/n_rewrites_vs_nDCG_comparison_2b.png",
            **common_args
        )

    # 3. Regression Plot
    if recall_metrics:
        plot_with_regression(
            df,
            metric_cols=recall_metrics,
            labels=recall_labels,
            title="Experiment 2b: R@10 vs Number of Rewrites (All Methods)",
            xlabel="Number of Rewrites",
            ylabel="R@10",
            filename="/n_rewrites_vs_R_regression_all_methods_2b.png",
            **common_args
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

        # Results path for experiment 2b
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        timestamp = comm.bcast(timestamp, root=0)
        RESULTS_PATH_2B = os.path.join(
            constants.RESULTS_PATH, constants.DATASET_NAME, "exp2b", f"run_{timestamp}"
        )
        if rank == 0:
            os.makedirs(RESULTS_PATH_2B, exist_ok=True)
            logging.info(f"Created results dir: {RESULTS_PATH_2B}")

        # Load Dataset
        docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)

        # Load rewritten queries
        filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
        filepath = os.path.join(constants.DATA_PATH, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rewrites file not found: {filepath}")

        rewritten_queries = pd.read_csv(filepath)

        # Filter topics to match rewrites and apply limit
        all_topics = topics_df[topics_df["qid"].isin(rewritten_queries["q_id"])].reset_index(drop=True)
        all_topics = all_topics.tail(constants.EXPERIMENT_2_TOPIC_LIMIT)

        # Create Lookup (needed by all ranks)
        query_lookup = {str(row["qid"]): row["query"] for _, row in all_topics.iterrows()}

        # Distribute Work
        my_topics = np.array_split(all_topics, size)[rank]
        logging.info(f"Rank {rank}: Assigned {len(my_topics)} topics.")

        # 2. Initialize Models & Index
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
        reranker = load_t5_reranker(constants.T5_BATCH_SIZE, rank)

        # 3. Experiment Loop
        local_results_list = []
        metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ constants.EXPERIMENT_2_RETRIEVAL_CUTOFF]

        if rank == 0:
            logging.info(f"Starting loop for N_REWRITES: {constants.EXPERIMENT_2_N_REWRITES}")

        for N in constants.EXPERIMENT_2_N_REWRITES:
            systems_dict = get_pipelines(
                N, index_ref, reranker, query_lookup,
                constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
            )

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    list(systems_dict.values()),
                    my_topics,
                    qrels_df,
                    eval_metrics=metrics,
                    names=list(systems_dict.keys()),
                    verbose=(rank == 0),
                    batch_size=constants.EXPERIMENT_BATCH_SIZE,
                    perquery=True,
                    precompute_prefix=True
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
            results_df = process_results(
                all_results_lists,
                constants.EXPERIMENT_2_N_REWRITES,
                metrics,
                constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
            )

            logging.info("\n" + results_df.to_string())

            save_experiment_results(
                results_df,
                experiment_name="experiment_2b",
                output_dir=RESULTS_PATH_2B,
                filename="experiment_2b_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            generate_plots(results_df, RESULTS_PATH_2B)

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        # Crucial for MPI: If one fails, kill all to prevent hanging
        comm.Abort(1)


if __name__ == "__main__":
    main()
