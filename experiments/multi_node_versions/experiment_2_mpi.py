import logging
import os
import sys
import traceback
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
    DedupAndCutoff,
    BordaVotingTransformerV2,
    CombMAXTransformer
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_2_mpi.log"


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
) -> List[Any]:
    """Constructs the list of retrieval pipelines for a specific N."""

    # Transformers
    rewrite = QueryRewriteTransformer(n_rewrites, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    text_getter = pt.text.get_text(index_ref, "text")
    attach_query = AttachOriginalQuery(query_lookup)
    merge_rewrites = MergeRewrites()

    # Pipeline Components
    dedup_topk = DedupAndCutoff(cutoff)
    dedup = Dedup()
    borda_voting = BordaVotingTransformerV2(cutoff=cutoff, weight=None, method="borda")
    borda_weighted = BordaVotingTransformerV2(cutoff=cutoff, weight=lambda x: np.exp(x * 0.1))
    comb_max = CombMAXTransformer()

    # 1. Base BM25 + Rerank
    pipe_base = bm25 >> text_getter >> reranker

    # 2. Dedup
    pipe_dedup = (
            rewrite >> bm25 >> merge_rewrites >> attach_query
            >> dedup >> text_getter >> reranker
    )

    # 3. Dedup + Cutoff
    pipe_dedup_cutoff = (
            rewrite >> bm25 >> merge_rewrites >> attach_query
            >> dedup_topk >> text_getter >> reranker
    )

    # 4. Borda
    pipe_borda = (
            rewrite >> bm25 >> borda_voting >> attach_query
            >> text_getter >> reranker
    )

    # 5. Borda Weighted
    pipe_borda_weighted = (
            rewrite >> bm25 >> borda_weighted >> attach_query
            >> text_getter >> reranker
    )

    # 6. CombMAX
    pipe_combmax = (
            rewrite >> bm25 >> merge_rewrites >> attach_query
            >> comb_max >> text_getter >> reranker
    )

    return [
        pipe_base,
        pipe_dedup,
        pipe_dedup_cutoff,
        pipe_borda,
        pipe_borda_weighted,
        pipe_combmax
    ]


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

            for metric in metrics:
                metric_str = str(metric)
                # Handle R@cutoff specifically for naming consistency if needed
                if metric_str in sys_df.columns:
                    mean_val = sys_df[metric_str].mean()

                    # Naming clean-up
                    clean_metric_name = "R@c" if metric_str == f"R@{cutoff}" else metric_str
                    row_data[f"{system}_{clean_metric_name}"] = mean_val

        final_rows.append(row_data)

        # Log quick summary
        logging.info(f"n={N} Summary: Dedup R@10={row_data.get('Pipeline_Dedup_R@10', 0):.4f}")

    return pd.DataFrame(final_rows)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    setup_logging(rank, comm)

    try:
        logging.info(f"Starting process. Rank {rank}/{size}")

        # 1. Environment & Data Setup
        setup_environment(rank)

        if rank == 0 and not os.path.exists(constants.RESULTS_PATH):
            os.makedirs(constants.RESULTS_PATH)
            logging.info(f"Created results dir: {constants.RESULTS_PATH}")

        # Load Dataset
        docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)
        all_topics = topics_df[:constants.EXPERIMENT_2_TOPIC_LIMIT]

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
        pipeline_names = [
            "BM25_MonoT5", "Pipeline_Dedup", "Pipeline_DedupCutoff",
            "Pipeline_Borda", "Pipeline_BordaWeighted", "Pipeline_CombMAX"
        ]

        if rank == 0:
            logging.info(f"Starting loop for N_REWRITES: {constants.EXPERIMENT_2_N_REWRITES}")

        for N in constants.EXPERIMENT_2_N_REWRITES:
            pipelines = get_pipelines(
                N, index_ref, reranker, query_lookup,
                constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
            )

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    pipelines,
                    my_topics,
                    qrels_df,
                    eval_metrics=metrics,
                    names=pipeline_names,
                    verbose=(rank == 0),
                    batch_size=constants.EXPERIMENT_BATCH_SIZE,
                    perquery=True,
                    precompute_prefix=True
                    # Note: baselines without rewrite dependency will be re-run here. 
                    # If heavily optimized, you could run baseline once outside loop and merge, 
                    # but running inside ensures dataframe alignment.
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
                experiment_name="experiment_2",
                output_dir=constants.RESULTS_PATH,
                filename="experiment_2_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            # (Plotting calls remain largely same, just ensuring correct column names)
            plot_metrics(
                results_df, x_col="n",
                metric_cols=[f"{name}_R@10" for name in pipeline_names],
                labels=pipeline_names,
                title="Experiment 2: R@10 Comparison",
                filename="/n_rewrites_vs_R_comparison.png",
                output_dir=constants.RESULTS_PATH,
            )

            # Regression Plot
            plot_with_regression(
                results_df, x_col="n",
                metric_cols=[f"{name}_R@10" for name in pipeline_names],
                title="Experiment 2: R@10 Regression",
                filename="/n_rewrites_vs_R_regression.png",
                output_dir=constants.RESULTS_PATH,
            )

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        # Crucial for MPI: If one fails, kill all to prevent hanging
        comm.Abort(1) 

if __name__ == "__main__":
    main()