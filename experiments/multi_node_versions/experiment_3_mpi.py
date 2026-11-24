import logging
import os
import sys
import traceback
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import pyterrier as pt
from mpi4py import MPI
from pyterrier.measures import R, nDCG

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    get_text_getter,
    get_reranker,
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
)
from code_base.transformers import (
    preprocess_query,
    QueryRewriteTransformer,
    MergeRewrites,
    AttachOriginalQuery,
    Dedup,
    CombMAXTransformer,
    RRFTransformer,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = os.environ.get("LOCAL_MODEL_BASE", os.path.abspath(constants.MODELS_CACHE_DIR))
LOG_FILENAME = "experiment_3_mpi.log"

# Best n_rewrites from exp2/exp2b results
N_REWRITES = constants.EXPERIMENT_3_BEST_N_REWRITES


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





def get_pipelines(
        cutoff: int,
        index_ref: Any,
        reranker: pt.Transformer,
        query_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """
    Constructs the dictionary of pipelines for a specific cutoff value.
    Uses fixed N_REWRITES=10 (best from exp2/exp2b results).
    """

    # Common Components
    rewrite = QueryRewriteTransformer(N_REWRITES, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", threads=10) % cutoff
    text_getter = get_text_getter(index_ref)
    attach_query = AttachOriginalQuery(query_lookup)

    # Baselines
    pipeline_bm25_base = bm25
    pipeline_bm25_t5 = bm25 >> text_getter >> reranker

    # --- Merge-then-Rerank pipelines ---

    # Pipeline 1: Dedup (merge-then-rerank)
    p_dedup = (rewrite
               >> bm25
               >> MergeRewrites()
               >> attach_query
               >> Dedup()
               >> text_getter
               >> reranker)

    # Pipeline 2: CombMAX (merge-then-rerank) — best fusion from exp2
    p_combmax = (rewrite
                 >> bm25
                 >> MergeRewrites()
                 >> attach_query
                 >> CombMAXTransformer()
                 >> text_getter
                 >> reranker)

    # --- Rerank-then-Merge pipelines (from exp2b, best overall nDCG) ---

    # Pipeline 3: Dedup RTM (rerank-then-merge)
    p_dedup_rtm = (rewrite
                   >> bm25
                   >> text_getter
                   >> reranker
                   >> MergeRewrites()
                   >> Dedup())

    # Pipeline 4: CombMAX RTM (rerank-then-merge)
    p_combmax_rtm = (rewrite
                     >> bm25
                     >> text_getter
                     >> reranker
                     >> MergeRewrites()
                     >> CombMAXTransformer())

    # Pipeline 5: RRF (merge-then-rerank)
    p_rrf = (rewrite
             >> bm25
             >> MergeRewrites()
             >> attach_query
             >> RRFTransformer()
             >> text_getter
             >> reranker)

    # Pipeline 6: RRF RTM (rerank-then-merge)
    p_rrf_rtm = (rewrite
                 >> bm25
                 >> text_getter
                 >> reranker
                 >> MergeRewrites()
                 >> RRFTransformer())

    return {
        "BM25": pipeline_bm25_base,
        "BM25_MonoT5": pipeline_bm25_t5,
        "Dedup": p_dedup,
        "CombMAX": p_combmax,
        "Dedup_RTM": p_dedup_rtm,
        "CombMAX_RTM": p_combmax_rtm,
        "RRF": p_rrf,
        "RRF_RTM": p_rrf_rtm,
    }


def process_results(all_results_lists: List[List[pd.DataFrame]], c_values: List[int]) -> pd.DataFrame:
    """Aggregates results from all MPI ranks and calculates means."""
    final_rows = []

    for i, cutoff in enumerate(c_values):
        # Gather all dataframes for this specific cutoff (i-th iteration) from all ranks
        # Filter out empty dataframes (ranks that might have had 0 topics)
        dfs_for_c = [
            rank_res[i] for rank_res in all_results_lists
            if i < len(rank_res) and not rank_res[i].empty
        ]

        if not dfs_for_c:
            logging.warning(f"No results found for cutoff={cutoff}")
            continue

        combined_df = pd.concat(dfs_for_c, ignore_index=True)

        row_data = {"cutoff": cutoff}
        systems = combined_df['name'].unique()

        for system in systems:
            sys_df = combined_df[combined_df['name'] == system]
            measures = sys_df['measure'].unique()
            for measure in measures:
                mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                if measure == f"R@{cutoff}":
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
        "x_col": "cutoff",
        "output_dir": output_dir,
        "use_log_scale": True
    }

    systems = [
        ("BM25", "BM25 Baseline"),
        ("BM25_MonoT5", f"BM25 + {constants.RERANKER_NAME}"),
        ("Dedup", "Dedup (Merge→Rerank)"),
        ("CombMAX", "CombMAX (Merge→Rerank)"),
        ("Dedup_RTM", "Dedup (Rerank→Merge)"),
        ("CombMAX_RTM", "CombMAX (Rerank→Merge)"),
        ("RRF", "RRF (Merge→Rerank)"),
        ("RRF_RTM", "RRF (Rerank→Merge)"),
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
            title="Experiment 3: Effect of Cutoff on R@10 (n_rewrites=10)",
            xlabel="Retrieval Cutoff",
            filename="/cutoff_vs_R_comparison.png",
            **common_args
        )

    # 2. NDCG Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_nDCG@10" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 3: Effect of Cutoff on nDCG@10 (n_rewrites=10)",
        xlabel="Retrieval Cutoff",
        filename="/cutoff_vs_nDCG_comparison.png",
        **common_args
    )

    # 3. Regression Plot
    if recall_metrics:
        plot_with_regression(
            df,
            metric_cols=recall_metrics,
            labels=recall_labels,
            title="Experiment 3: R@10 vs Cutoff Regression (n_rewrites=10)",
            xlabel="Retrieval Cutoff",
            ylabel="R@10",
            filename="/cutoff_vs_R_regression_all_methods.png",
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        timestamp = comm.bcast(timestamp, root=0)
        results_path = os.path.join(
            constants.RESULTS_PATH, constants.DATASET_NAME, "exp3", f"run_{timestamp}"
        )
        if rank == 0:
            os.makedirs(results_path, exist_ok=True)
            logging.info(f"Created results dir: {results_path}")

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
        all_topics = all_topics.tail(constants.EXPERIMENT_3_TOPIC_LIMIT)

        # Create Lookup (needed by all ranks)
        query_lookup = {str(row["qid"]): row["query"] for _, row in all_topics.iterrows()}

        # Distribute Work
        import numpy as np
        my_topics = np.array_split(all_topics, size)[rank]
        logging.info(f"Rank {rank}: Assigned {len(my_topics)} topics.")

        # 2. Initialize Models & Index
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
        reranker = get_reranker(models_base_dir=LOCAL_MODEL_BASE)

        # 3. Experiment Loop (varying cutoff)
        local_results_list = []
        c_values = constants.EXPERIMENT_3_C_VALUES

        if rank == 0:
            logging.info(f"Starting loop for C_VALUES: {c_values}, fixed N_REWRITES={N_REWRITES}")

        for cutoff in c_values:
            # Define metrics for this cutoff
            metrics = constants.EXPERIMENT_3_EVAL_METRICS + [R @ cutoff]

            systems_dict = get_pipelines(cutoff, index_ref, reranker, query_lookup)

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
                exp_results['cutoff'] = cutoff
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
                c_values,
            )

            logging.info("\n" + results_df.to_string())

            save_experiment_results(
                results_df,
                experiment_name="experiment_3",
                output_dir=results_path,
                filename="experiment_3_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            generate_plots(results_df, results_path)

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        # Crucial for MPI: If one fails, kill all to prevent hanging
        comm.Abort(1)


if __name__ == "__main__":
    main()
