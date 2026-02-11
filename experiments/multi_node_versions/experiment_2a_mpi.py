import logging
import os
import sys
import traceback
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_dr as pt_dr
from mpi4py import MPI
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query,
    QueryRewriteTransformer,
    MergeRewrites,
    AttachOriginalQuery
)
from code_base.query_trasnformers import (
    Dedup,
    BordaVotingTransformerV2,
    CombMAXTransformer
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_2a_mpi.log"


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


def load_dense_model_and_index(rank: int):
    """Loads the dense retrieval components (Model and Index)."""
    logging.info(f"Rank {rank}: Loading Dense Retrieval models...")

    # Define local path for the dense model
    dense_model_path = os.path.join(LOCAL_MODEL_BASE, constants.DENSE_MODEL_NAME)

    # Check if local model exists
    if os.path.exists(dense_model_path):
        logging.info(f"Rank {rank}: Loading local dense model from {dense_model_path}")
        model = pt_dr.TctColBert(dense_model_path)
    else:
        logging.warning(f"Rank {rank}: Local dense model not found at {dense_model_path}. Attempting HF download.")
        model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME)

    # Initialize FlexIndex
    # Note: Check if we need to set absolute path validation?
    index = pt_dr.FlexIndex(constants.DENSE_INDEX_PATH)

    logging.info(f"Rank {rank}: Loaded dense index from {constants.DENSE_INDEX_PATH}.")

    return model, index


def get_pipelines(
        n_rewrites: int,
        index_ref: Any,
        mono_t5: MonoT5ReRanker,
        dense_retriever: Any,
        dense_reranker: Any,
        sparse_retriever: Any,
        query_lookup: Dict[str, str],
        cutoff: int
) -> Dict[str, Any]:
    """Constructs the list of retrieval pipelines for a specific N using Dense Retrieval."""

    # Common Components
    rewrite = QueryRewriteTransformer(n_rewrites, dataset_name=constants.DATASET_NAME)
    attach_query = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    # 1. Dense Baseline
    dense_base = dense_retriever % cutoff

    # 2. Dense + MonoT5
    pipeline_dense_t5 = dense_base >> get_text >> mono_t5

    # 3. BM25 + Dense Rerank
    pipeline_bm25_dense = (
            sparse_retriever % cutoff
            >> dense_reranker
            >> pt.apply.generic(lambda df: df.sort_values(["qid", "score"], ascending=[True, False]))
    )

    # Pipeline 4: Dedup
    p_dedup = (rewrite
               >> dense_retriever % cutoff
               >> MergeRewrites()
               >> attach_query
               >> Dedup()
               >> get_text
               >> mono_t5)

    # Pipeline 5: Borda Voting
    borda_voting = BordaVotingTransformerV2(cutoff=cutoff, weight=None, method="borda")
    p_borda = (rewrite
               >> dense_retriever % cutoff
               >> borda_voting
               >> attach_query
               >> get_text
               >> mono_t5)

    # Pipeline 6: Weighted Borda
    borda_weighted = BordaVotingTransformerV2(
        cutoff=cutoff,
        weight=lambda x: np.exp(x * 0.1),
        method="borda"
    )
    p_borda_weighted = (rewrite
                        >> dense_retriever % cutoff
                        >> borda_weighted
                        >> attach_query
                        >> get_text
                        >> mono_t5)

    # Pipeline 7: CombMAX
    p_combmax = (rewrite
                 >> dense_retriever % cutoff
                 >> MergeRewrites()
                 >> attach_query
                 >> CombMAXTransformer()
                 >> get_text
                 >> mono_t5)

    return {
        "Dense": dense_base,
        "Dense_MonoT5": pipeline_dense_t5,
        "BM25_DenseRerank": pipeline_bm25_dense,
        "Pipeline_Dedup": p_dedup,
        "Pipeline_Borda": p_borda,
        "Pipeline_BordaWeighted": p_borda_weighted,
        "Pipeline_CombMAX": p_combmax
    }


def process_results(all_results_lists: List[List[pd.DataFrame]], n_values: List[int], metrics: List[Any],
                    cutoff: int) -> pd.DataFrame:
    """Aggregates results from all MPI ranks and calculates means."""
    final_rows = []

    for i, N in enumerate(n_values):
        # Gather all dataframes for this specific N (i-th iteration) from all ranks
        # Filter out empty dataframes (ranks that might have had 0 topics)
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

            # Helper to safely get metric
            def get_metric_mean(df, metric_key):
                if metric_key in df.columns:
                    return df[metric_key].mean()
                return None

            for metric in metrics:
                metric_str = str(metric)
                # Handle R@cutoff specifically for naming consistency
                if metric_str == f"R@{cutoff}":
                    # In pyterrier results, it's typically "R@20" string
                    # We need to find the column that matches
                    val = get_metric_mean(sys_df, metric_str)
                    if val is not None:
                        row_data[f"{system}_R@c"] = val
                    else:
                        logging.warning(f"Metric {metric_str} not found in results for system {system} (n={N})")
                else:
                    val = get_metric_mean(sys_df, metric_str)
                    if val is not None:
                        row_data[f"{system}_{metric_str}"] = val
                    else:
                        logging.warning(f"Metric {metric_str} not found in results for system {system} (n={N})")

        final_rows.append(row_data)

        # Log quick summary
        logging.info(f"n={N} Summary: Dedup R@c={row_data.get('Pipeline_Dedup_R@c', 0):.4f}")

    return pd.DataFrame(final_rows)


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Encapsulate plotting logic."""
    common_args = {
        "x_col": "n",
        "output_dir": output_dir,
        "use_log_scale": False
    }

    systems = [
        ("Dense", "Dense Baseline"),
        ("Dense_MonoT5", "Dense + MonoT5"),
        ("BM25_DenseRerank", "BM25 + Dense Rerank"),
        ("Pipeline_Dedup", "Dedup Method"),
        ("Pipeline_Borda", "Borda Voting"),
        ("Pipeline_BordaWeighted", "Borda Weighted"),
        ("Pipeline_CombMAX", "CombMAX")
    ]

    # 1. Recall Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_R@10" if "R@10" in df.columns else f"{sys}_R@c" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2a: Document Combination Methods Comparison (Recall) (Dense)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_R_comparison.png",
        **common_args
    )

    # 2. NDCG Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_nDCG@10" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2a: Document Combination Methods Comparison (nDCG@10) (Dense)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_nDCG_comparison.png",
        **common_args
    )

    # 3. Regression Plot
    plot_with_regression(
        df,
        metric_cols=[f"{sys}_R@c" if f"{sys}_R@c" in df.columns else f"{sys}_R@10" for sys, _ in systems],
        title="Experiment 2a: R@10 vs Number of Rewrites (All Methods) (Dense)",
        xlabel="Number of Rewrites",
        ylabel="R@10",
        filename="/n_rewrites_vs_R_regression_all_methods.png",
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

        # Ensure results directory exists
        RESULTS_PATH = os.path.join(constants.RESULTS_PATH, "experiment_2a")
        if rank == 0 and not os.path.exists(RESULTS_PATH):
            os.makedirs(RESULTS_PATH)
            logging.info(f"Created results dir: {RESULTS_PATH}")

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
        # We need the sparse index for specific pipelines and get_text
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
        sparse_retriever = pt.terrier.Retriever(index_ref, wmodel="BM25")

        # Dense Setup
        model, dense_index = load_dense_model_and_index(rank)
        dense_retriever = model.query_encoder() >> dense_index.retriever()
        dense_reranker = model.query_encoder() >> dense_index.scorer()

        # Reranker
        mono_t5 = load_t5_reranker(constants.T5_BATCH_SIZE, rank)

        # 3. Experiment Loop
        local_results_list = []
        metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ constants.EXPERIMENT_2_RETRIEVAL_CUTOFF]

        if rank == 0:
            logging.info(f"Starting loop for N_REWRITES: {constants.EXPERIMENT_2_N_REWRITES}")

        for N in constants.EXPERIMENT_2_N_REWRITES:
            if rank == 0:
                logging.info(f"Running pipelines for N={N}...")
                
            systems_dict = get_pipelines(
                N, index_ref, mono_t5, dense_retriever, dense_reranker, sparse_retriever,
                query_lookup, constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
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
                experiment_name="experiment_2a",
                output_dir=RESULTS_PATH,
                filename="experiment_2a_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            generate_plots(results_df, RESULTS_PATH)
            logging.info("Done.")

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        # Crucial for MPI: If one fails, kill all to prevent hanging
        comm.Abort(1)


if __name__ == "__main__":
    main()
