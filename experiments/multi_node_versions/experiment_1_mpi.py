import os
import sys
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker
from mpi4py import MPI
import numpy as np

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query
)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -----------------------------
    # Configuration (from constants.py)
    # -----------------------------
    DATA_PATH = constants.DATA_PATH
    RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_1"
    
    if rank == 0:
        if os.path.exists(RESULTS_PATH):
            print(f"Results directory {RESULTS_PATH} already exists.")
        else:
            os.makedirs(RESULTS_PATH)
            print(f"Created results directory at {RESULTS_PATH}.")

    DATA_SPLIT = constants.DATA_SPLIT
    INDEX_PATH = constants.INDEX_PATH

    # Experiment Configuration
    TOPIC_LIMIT = constants.EXPERIMENT_1_TOPIC_LIMIT
    C_VALUES = constants.EXPERIMENT_1_C_VALUES
    if type(C_VALUES) is int:
        C_VALUES = [C_VALUES]
    T5_BATCH_SIZE = constants.T5_BATCH_SIZE
    EXPERIMENT_BATCH_SIZE = constants.EXPERIMENT_BATCH_SIZE
    EVAL_METRICS = constants.EXPERIMENT_1_EVAL_METRICS

    if rank == 0:
        print(f"Running on {size} MPI processes.")
        print("Loading data...")

    # Load data on all ranks (simplest approach for now)
    # If memory is an issue, could load on rank 0 and scatter, but topics_df is small.
    docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, DATA_SPLIT)
    
    topics_df["query"] = topics_df["query"].apply(preprocess_query)
    
    all_topics = topics_df[:TOPIC_LIMIT]
    
    # Distribute topics
    # np.array_split handles uneven splits gracefully
    my_topics = np.array_split(all_topics, size)[rank]
    
    if rank == 0:
        print(f"Total topics: {len(all_topics)}")
    
    print(f"Rank {rank}: Processing {len(my_topics)} topics.")

    # Initialize PyTerrier and components on all ranks
    if not pt.started():
        pt.init()
        
    index_ref = pt.IndexFactory.of(INDEX_PATH)
    mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

    local_results_list = []

    if rank == 0:
        print(f"\nStarting experiment for initial cutoff (c) in: {C_VALUES}")

    for c in C_VALUES:
        bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
        bm25_retrieved = bm25 % c
        pipeline = bm25_retrieved >> pt.text.get_text(index_ref, "text") >> mono_t5_reranker

        # Define metrics for this iteration
        current_metrics = EVAL_METRICS + [R @ c]

        # Run experiment on local topics
        # perquery=True to get results for each query, which we can then aggregate
        if len(my_topics) > 0:
            exp_results = pt.Experiment(
                [bm25_retrieved, pipeline],
                my_topics,
                qrels_df,
                eval_metrics=current_metrics,
                names=["BM25", "Pipeline"],
                verbose=(rank == 0), # Only rank 0 prints progress bar
                batch_size=EXPERIMENT_BATCH_SIZE,
                perquery=True,
                precompute_prefix=True
            )
            # Add 'c' column to identify this run
            exp_results['c'] = c
            local_results_list.append(exp_results)
        else:
            local_results_list.append(pd.DataFrame())

    # Gather results to Rank 0
    all_results_lists = comm.gather(local_results_list, root=0)

    if rank == 0:
        print("Aggregating results...")
        
        # Flatten the list of lists
        # Structure: [[rank0_c1, rank0_c2], [rank1_c1, rank1_c2], ...]
        # We want to group by 'c'
        
        final_summary_list = []
        
        for i, c in enumerate(C_VALUES):
            # Collect all DFs for this 'c' from all ranks
            dfs_for_c = [rank_res[i] for rank_res in all_results_lists if not rank_res[i].empty]
            
            if not dfs_for_c:
                print(f"Warning: No results for c={c}")
                continue
                
            combined_df = pd.concat(dfs_for_c, ignore_index=True)
            
            # Calculate averages per system
            # Group by 'name' and calculate mean of value for each measure
            # The DF has columns: name, qid, measure, value, c
            
            # Pivot to get metrics as columns: name, qid, metric1, metric2...
            # But pt.Experiment perquery format is long.
            # Let's aggregate directly.
            
            summary_row = {"c": c}
            
            systems = combined_df['name'].unique()
            
            for system in systems:
                sys_df = combined_df[combined_df['name'] == system]
                
                # Get all unique measures
                measures = sys_df['measure'].unique()
                
                for measure in measures:
                    # Calculate mean
                    mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                    
                    # Format key name
                    # Map dynamic R@c to fixed column name "R@c" for consistency
                    if measure == f"R@{c}":
                        key = f"{system}_R@c"
                    else:
                        key = f"{system}_{measure}"
                        
                    summary_row[key] = mean_val
            
            final_summary_list.append(summary_row)
            
            # Print summary for this c (similar to original script)
            print(
                f"c={c}: "
                f"BM25 R@10={summary_row.get('BM25_R@10', 0):.4f}, "
                f"Pipeline R@10={summary_row.get('Pipeline_R@10', 0):.4f}, "
                f"BM25 NDCG@10={summary_row.get('BM25_nDCG@10', 0):.4f}, "
                f"Pipeline NDCG@10={summary_row.get('Pipeline_nDCG@10', 0):.4f}, "
                f"BM25 R@{c}={summary_row.get('BM25_R@c', 0):.4f}, "
                f"Pipeline R@{c}={summary_row.get('Pipeline_R@c', 0):.4f}"
            )

        results_df = pd.DataFrame(final_summary_list)

        # Save results with metadata
        save_experiment_results(
            results_df,
            experiment_name="experiment_1",
            output_dir=RESULTS_PATH,
            filename="experiment_1_results",
        )

        print("\nFinal Results DataFrame:")
        print(results_df)

        # CHART GENERATION
        # R vs. Cutoff
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

        # NDCG@10 vs. Cutoff
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

        # LINEAR REGRESSION ANALYSIS (R@10)
        print("\n--- Linear Regression Analysis (log(c) vs R@10) ---")

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

if __name__ == "__main__":
    main()
