import logging
import os
import sys

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path to allow importing constants and code_base
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
from code_base.query_trasnformers import Dedup, DedupAndCutoff, BordaVotingTransformerV2


def main():
    # -----------------------------
    # 1. Force Offline Mode
    # -----------------------------
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configure logging
    log_filename = "experiment_2_mpi.log"
    # Clear log file on rank 0, append on others
    file_mode = 'w' if rank == 0 else 'a'

    # Ensure rank 0 writes first to clear the file
    if rank != 0:
        comm.Barrier()

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format=f'%(asctime)s - Rank {rank} - %(message)s',
        filemode=file_mode
    )

    if rank == 0:
        comm.Barrier()

    try:
        logging.info(f"Starting process. Rank {rank}/{size}")

        # -----------------------------
        # Configuration
        # -----------------------------
        DATA_PATH = constants.DATA_PATH
        RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_2"  # Matches experiment_2.py logic roughly or use constants path
        # In experiment_1_mpi it was constants.RESULTS_PATH. In experiment_2.py it was explicitly constructed.
        # We will use the explicit construction to match experiment_2 output structure or constants?
        # experiment_1_mpi used: RESULTS_PATH = constants.RESULTS_PATH
        # experiment_2 used: RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_2"
        # I'll stick to experiment_2's path structure for consistency with that experiment.

        if rank == 0:
            if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)
                logging.info(f"Created results directory at {RESULTS_PATH}.")
            else:
                logging.info(f"Results directory {RESULTS_PATH} already exists.")

        DATA_SPLIT = constants.DATA_SPLIT
        INDEX_PATH = constants.INDEX_PATH

        # Experiment controls
        TOPIC_LIMIT = constants.EXPERIMENT_2_TOPIC_LIMIT
        N_REWRITES = constants.EXPERIMENT_2_N_REWRITES
        EVAL_METRICS = constants.EXPERIMENT_2_EVAL_METRICS
        RETRIEVAL_CUTOFF = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

        # Reranker
        T5_BATCH_SIZE = constants.T5_BATCH_SIZE
        EXPERIMENT_BATCH_SIZE = constants.EXPERIMENT_BATCH_SIZE

        if rank == 0:
            logging.info(f"Running on {size} MPI processes.")
            logging.info("Loading data...")

        # Load data
        docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)
        all_topics = topics_df[:TOPIC_LIMIT]

        # Create global lookup for AttachOriginalQuery (needed by all ranks)
        QUERY_LOOKUP = {str(row["qid"]): row["query"] for _, row in all_topics.iterrows()}

        # Distribute topics
        my_topics = np.array_split(all_topics, size)[rank]

        if rank == 0:
            logging.info(f"Total topics: {len(all_topics)}")

        logging.info(f"Rank {rank}: Processing {len(my_topics)} topics.")

        # -----------------------------
        # 2. Initialize PyTerrier (Offline Config)
        # -----------------------------
        pt.terrier.set_version("5.11")
        pt.terrier.set_helper_version("0.0.8")
        pt.init()
        logging.info(f"Rank {rank}: PyTerrier initialized (Offline Mode, forced version).")

        index_ref = pt.IndexFactory.of(INDEX_PATH)

        # Initialize T5 (Uses local cache logic from experiment_1_mpi)
        local_model_path = "/scratch/pjohari/MasterThesis/models"
        mono_t5_model_path = os.path.join(local_model_path, "castorini/monot5-base-msmarco")
        t5_base_path = os.path.join(local_model_path, "t5-base")

        if os.path.exists(mono_t5_model_path):
            logging.info(f"Rank {rank}: Found local model at {mono_t5_model_path}. Using it.")
            if os.path.exists(t5_base_path):
                logging.info(
                    f"Rank {rank}: Found local t5-base at {t5_base_path}. Setting as tokenizer/config fallback.")
                mono_t5_reranker = MonoT5ReRanker(
                    model=mono_t5_model_path,
                    tok_model=t5_base_path,
                    verbose=False,
                    batch_size=T5_BATCH_SIZE
                )
            else:
                logging.warning(f"Rank {rank}: Local t5-base not found. Using monoT5 model path only.")
                mono_t5_reranker = MonoT5ReRanker(model=mono_t5_model_path, verbose=False, batch_size=T5_BATCH_SIZE)
        else:
            logging.warning(
                f"Rank {rank}: Local model not found at {mono_t5_model_path}. Attempting download (will fail if offline).")
            mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

        local_results_list = []
        attach_query = AttachOriginalQuery(QUERY_LOOKUP)
        bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")

        if rank == 0:
            logging.info(f"Starting experiment loop for N_REWRITES: {N_REWRITES}")

        for N in N_REWRITES:
            # Reconstruct pipelines for this N
            rewrite = QueryRewriteTransformer(N, dataset_name=constants.DATASET_NAME)
            dedup_topk = DedupAndCutoff(RETRIEVAL_CUTOFF)
            dedup = Dedup()

            borda_voting = BordaVotingTransformerV2(
                cutoff=RETRIEVAL_CUTOFF, weight=None, method="borda"
            )
            borda_voting_weighted = BordaVotingTransformerV2(
                cutoff=RETRIEVAL_CUTOFF,
                weight=lambda x: np.exp(x * 0.1),
                method="borda",
            )

            bm25_base = bm25 % RETRIEVAL_CUTOFF

            pipeline_bm25_t5 = (
                    bm25 % RETRIEVAL_CUTOFF
                    >> pt.text.get_text(index_ref, "text")
                    >> mono_t5_reranker
            )

            pipeline_dedup = (
                    rewrite
                    >> bm25 % RETRIEVAL_CUTOFF
                    >> MergeRewrites()
                    >> attach_query
                    >> dedup
                    >> pt.text.get_text(index_ref, "text")
                    >> mono_t5_reranker
            )

            pipeline_dedup_cutoff = (
                    rewrite
                    >> bm25 % RETRIEVAL_CUTOFF
                    >> MergeRewrites()
                    >> attach_query
                    >> dedup_topk
                    >> pt.text.get_text(index_ref, "text")
                    >> mono_t5_reranker
            )

            pipeline_borda = (
                    rewrite
                    >> bm25 % RETRIEVAL_CUTOFF
                    >> borda_voting
                    >> attach_query
                    >> pt.text.get_text(index_ref, "text")
                    >> mono_t5_reranker
            )

            pipeline_borda_weighted = (
                    rewrite
                    >> bm25 % RETRIEVAL_CUTOFF
                    >> borda_voting_weighted
                    >> attach_query
                    >> pt.text.get_text(index_ref, "text")
                    >> mono_t5_reranker
            )

            current_metrics = EVAL_METRICS + [R @ RETRIEVAL_CUTOFF]

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    [
                        bm25_base,
                        pipeline_bm25_t5,
                        pipeline_dedup,
                        pipeline_dedup_cutoff,
                        pipeline_borda,
                        pipeline_borda_weighted,
                    ],
                    my_topics,
                    qrels_df,
                    eval_metrics=current_metrics,
                    names=[
                        "BM25",
                        "BM25_MonoT5",
                        "Pipeline_Dedup",
                        "Pipeline_DedupCutoff",
                        "Pipeline_Borda",
                        "Pipeline_BordaWeighted",
                    ],
                    verbose=(rank == 0),
                    batch_size=EXPERIMENT_BATCH_SIZE,
                    perquery=True,
                    precompute_prefix=True
                )
                exp_results['n'] = N
                local_results_list.append(exp_results)
            else:
                local_results_list.append(pd.DataFrame())

        # Gather results to Rank 0
        all_results_lists = comm.gather(local_results_list, root=0)

        if rank == 0:
            logging.info("Aggregating results...")

            final_rows_list = []

            # Iterate through the N values (corresponding to the loop above)
            for i, N in enumerate(N_REWRITES):
                # Collect dataframe for this N from all ranks
                dfs_for_n = [rank_res[i] for rank_res in all_results_lists if not rank_res[i].empty]

                if not dfs_for_n:
                    logging.warning(f"Warning: No results for n={N}")
                    continue

                combined_df = pd.concat(dfs_for_n, ignore_index=True)

                # Calculate means for this N
                row_data = {"n": N}
                systems = combined_df['name'].unique()

                for system in systems:
                    sys_df = combined_df[combined_df['name'] == system]

                    # We need to compute mean for each metric in EVAL_METRICS + [R@CUTOFF]
                    # Note: combined_df has columns like 'R@100', 'nDCG@10' etc.
                    # We can inspect the columns or iterate expected metrics

                    for metric in current_metrics:
                        metric_str = str(metric)
                        if metric_str in sys_df.columns:
                            mean_val = sys_df[metric_str].mean()

                            # Fix naming convention for columns
                            if metric_str == f"R@{RETRIEVAL_CUTOFF}":
                                key = f"{system}_R@c"
                            else:
                                key = f"{system}_{metric_str}"

                            row_data[key] = mean_val

                final_rows_list.append(row_data)

                # Log summary for this N
                logging.info(
                    f"n={N}: "
                    f"BM25 R@10={row_data.get('BM25_R@10', 0):.4f}, "
                    f"Dedup R@10={row_data.get('Pipeline_Dedup_R@10', 0):.4f}, "
                    f"Borda R@10={row_data.get('Pipeline_Borda_R@10', 0):.4f}"
                )

            results_df = pd.DataFrame(final_rows_list)

            save_experiment_results(
                results_df,
                experiment_name="experiment_2",
                output_dir=RESULTS_PATH,
                filename="experiment_2_results",
            )

            logging.info("\nFinal Results DataFrame:")
            logging.info(results_df.to_string())

            # CHART GENERATION
            # Comparison Plots
            plot_metrics(
                results_df,
                x_col="n",
                metric_cols=[
                    "BM25_R@10",
                    "BM25_MonoT5_R@10",
                    "Pipeline_Dedup_R@10",
                    "Pipeline_DedupCutoff_R@10",
                    "Pipeline_Borda_R@10",
                    "Pipeline_BordaWeighted_R@10",
                ],
                labels=[
                    "BM25 Baseline",
                    "BM25 + MonoT5",
                    "Dedup Method",
                    "Dedup Cutoff Method",
                    "Borda Voting",
                    "Borda Weighted",
                ],
                title="Experiment 2: Document Combination Methods Comparison",
                xlabel="Number of Rewrites",
                filename="/n_rewrites_vs_R_comparison.png",
                use_log_scale=False,
                output_dir=RESULTS_PATH,
            )

            plot_metrics(
                results_df,
                x_col="n",
                metric_cols=[
                    "BM25_nDCG@10",
                    "BM25_MonoT5_nDCG@10",
                    "Pipeline_Dedup_nDCG@10",
                    "Pipeline_DedupCutoff_nDCG@10",
                    "Pipeline_Borda_nDCG@10",
                    "Pipeline_BordaWeighted_nDCG@10",
                ],
                labels=[
                    "BM25 Baseline",
                    "BM25+MonoT5",
                    "Dedup Method",
                    "Dedup Cutoff Method",
                    "Borda Voting",
                    "Borda Weighted",
                ],
                title="Experiment 2: Document Combination Methods Comparison (nDCG@10)",
                xlabel="Number of Rewrites",
                filename="/n_rewrites_vs_nDCG_comparison.png",
                use_log_scale=False,
                output_dir=RESULTS_PATH,
            )

            plot_with_regression(
                results_df,
                x_col="n",
                metric_cols=[
                    "BM25_R@10",
                    "BM25_MonoT5_R@10",
                    "Pipeline_Dedup_R@10",
                    "Pipeline_DedupCutoff_R@10",
                    "Pipeline_Borda_R@10",
                    "Pipeline_BordaWeighted_R@10",
                ],
                title="Experiment 2: R@10 vs Number of Rewrites (All Methods)",
                xlabel="Number of Rewrites",
                ylabel="R@10",
                filename="/n_rewrites_vs_R_regression_all_methods.png",
                use_log_scale=False,
                output_dir=RESULTS_PATH,
            )

    except Exception as e:
        import traceback
        logging.error(f"Rank {rank}: An error occurred: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
