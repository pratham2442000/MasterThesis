import os
import sys
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker
from mpi4py import MPI
import numpy as np
import logging

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query
)

def main():
    # -----------------------------
    # 1. Force Offline Mode for HuggingFace (T5)
    # -----------------------------
    # Ideally set these before importing transformers, but definitely before using them.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configure logging
    log_filename = "experiment_1_mpi.log"
    if rank == 0:
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format=f'%(asctime)s - Rank {rank} - %(message)s',
            filemode='w'
        )
    else:
        comm.Barrier()
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format=f'%(asctime)s - Rank {rank} - %(message)s',
            filemode='a'
        )
    
    if rank == 0:
        comm.Barrier()

    try:
        logging.info(f"Starting process. Rank {rank}/{size}")

        # -----------------------------
        # Configuration
        # -----------------------------
        DATA_PATH = constants.DATA_PATH
        RESULTS_PATH = constants.RESULTS_PATH
        
        if rank == 0:     
            if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)
                logging.info(f"Created results directory at {RESULTS_PATH}.")
            else:
                 logging.info(f"Results directory {RESULTS_PATH} already exists.")

        DATA_SPLIT = constants.DATA_SPLIT
        INDEX_PATH = constants.INDEX_PATH
        TOPIC_LIMIT = constants.EXPERIMENT_1_TOPIC_LIMIT
        C_VALUES = constants.EXPERIMENT_1_C_VALUES
        if type(C_VALUES) is int:
            C_VALUES = [C_VALUES]
        T5_BATCH_SIZE = constants.T5_BATCH_SIZE
        EXPERIMENT_BATCH_SIZE = constants.EXPERIMENT_BATCH_SIZE
        EVAL_METRICS = constants.EXPERIMENT_1_EVAL_METRICS

        if rank == 0:
            logging.info(f"Running on {size} MPI processes.")
            logging.info("Loading data...")

        # Load data
        docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)
        all_topics = topics_df[:TOPIC_LIMIT]
        
        # Distribute topics
        my_topics = np.array_split(all_topics, size)[rank]
        
        if rank == 0:
            logging.info(f"Total topics: {len(all_topics)}")
        
        logging.info(f"Rank {rank}: Processing {len(my_topics)} topics.")

        # -----------------------------
        # 2. Initialize PyTerrier (Offline Config)
        # -----------------------------
        if not pt.java.started():
            # version='5.11' prevents checking the internet for updates
            # mem='8g' ensures enough heap space for retrieval
            pt.java.init(version='5.11', mem='8g')
            logging.info(f"Rank {rank}: PyTerrier initialized (Offline Mode, v5.11).")
            
        index_ref = pt.IndexFactory.of(INDEX_PATH)
        
        # This will now use the local cache because of the os.environ flags set above
        mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

        local_results_list = []

        if rank == 0:
            logging.info(f"Starting experiment for initial cutoff (c) in: {C_VALUES}")

        for c in C_VALUES:
            bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
            bm25_retrieved = bm25 % c
            pipeline = bm25_retrieved >> pt.text.get_text(index_ref, "text") >> mono_t5_reranker

            current_metrics = EVAL_METRICS + [R @ c]

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    [bm25_retrieved, pipeline],
                    my_topics,
                    qrels_df,
                    eval_metrics=current_metrics,
                    names=["BM25", "Pipeline"],
                    verbose=(rank == 0),
                    batch_size=EXPERIMENT_BATCH_SIZE,
                    perquery=True,
                    precompute_prefix=True
                )
                exp_results['c'] = c
                local_results_list.append(exp_results)
            else:
                local_results_list.append(pd.DataFrame())

        # Gather results to Rank 0
        all_results_lists = comm.gather(local_results_list, root=0)

        if rank == 0:
            logging.info("Aggregating results...")
            
            final_summary_list = []
            
            for i, c in enumerate(C_VALUES):
                dfs_for_c = [rank_res[i] for rank_res in all_results_lists if not rank_res[i].empty]
                
                if not dfs_for_c:
                    logging.warning(f"Warning: No results for c={c}")
                    continue
                    
                combined_df = pd.concat(dfs_for_c, ignore_index=True)
                
                summary_row = {"c": c}
                systems = combined_df['name'].unique()
                
                for system in systems:
                    sys_df = combined_df[combined_df['name'] == system]
                    measures = sys_df['measure'].unique()
                    
                    for measure in measures:
                        mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                        if measure == f"R@{c}":
                            key = f"{system}_R@c"
                        else:
                            key = f"{system}_{measure}"
                        summary_row[key] = mean_val
                
                final_summary_list.append(summary_row)
                
                logging.info(
                    f"c={c}: "
                    f"BM25 R@10={summary_row.get('BM25_R@10', 0):.4f}, "
                    f"Pipeline R@10={summary_row.get('Pipeline_R@10', 0):.4f}, "
                    f"BM25 NDCG@10={summary_row.get('BM25_nDCG@10', 0):.4f}, "
                    f"Pipeline NDCG@10={summary_row.get('Pipeline_nDCG@10', 0):.4f}"
                )

            results_df = pd.DataFrame(final_summary_list)

            save_experiment_results(
                results_df,
                experiment_name="experiment_1",
                output_dir=RESULTS_PATH,
                filename="experiment_1_results",
            )

            logging.info("\nFinal Results DataFrame:")
            logging.info(results_df.to_string())

            # CHART GENERATION
            plot_metrics(
                results_df,
                x_col="c",
                metric_cols=["BM25_R@10", "Pipeline_R@10"],
                labels=["BM25 Baseline", "BM25 + MonoT5 Re-ranker"],
                title="Effect of Initial Cutoff (c) on Recall (R@10)",
                xlabel="Initial Cutoff (c) for BM25 Retrieval",
                filename="/cutoff_vs_R.png",
                output_dir=RESULTS_PATH,
                use_log_scale=True
            )

            plot_metrics(
                results_df,
                x_col="c",
                metric_cols=["BM25_nDCG@10", "Pipeline_nDCG@10"],
                labels=["BM25 Baseline", "BM25 + MonoT5 Re-ranker"],
                title="Effect of Initial Cutoff (c) on NDCG@10",
                xlabel="Initial Cutoff (c) for BM25 Retrieval",
                ylabel="NDCG@10 (Discounted Cumulative Gain)",
                filename="/cutoff_vs_NDCG_at_10.png",
                output_dir=RESULTS_PATH,
                use_log_scale=True
            )

            logging.info("--- Linear Regression Analysis (log(c) vs R@10) ---")
            plot_with_regression(
                results_df,
                x_col="c",
                metric_cols=["BM25_R@10", "Pipeline_R@10"],
                labels=["BM25 (Baseline) R@10", "Pipeline (BM25+MonoT5) R@10"],
                use_log_scale=True,
                title="R (@ 10) vs. Initial Cutoff (c) with Log(c) Linear Regression",
                xlabel="Initial Cutoff (c) for BM25 Retrieval (log scale)",
                ylabel="Recall (R@10)",
                filename="/cutoff_R_at_10_log_regression.png",
                output_dir=RESULTS_PATH,
            )

    except Exception as e:
        logging.error(f"Rank {rank}: An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()