import os
import sys

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    save_experiment_results,
    preprocess_query,
    plot_with_regression,
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


# ---------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------
def setup_directories():
    """Ensure result directories exist."""
    results_path = constants.RESULTS_PATH
    os.makedirs(results_path, exist_ok=True)
    print(f"Results directory ensured at: {results_path}")
    return results_path


def load_data():
    """Load and preprocess dataset, topics, and qrels."""
    print("Loading data...")
    docs_df, topics_df, qrels_df = dataset_factory.get_dataset(
        constants.DATASET_NAME, constants.DATA_SPLIT
    )

    # Preprocess queries
    topics_df["query"] = topics_df["query"].apply(preprocess_query)

    # Load rewritten queries
    filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
    filepath = os.path.join(constants.DATA_PATH, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Rewrites file not found: {filepath}")

    rewritten_queries = pd.read_csv(filepath)

    # Filter topics to match rewrites and apply limit
    topics = topics_df[topics_df["qid"].isin(rewritten_queries["q_id"])].reset_index(drop=True)
    topics = topics.tail(constants.EXPERIMENT_2_TOPIC_LIMIT)


    # Create lookup for AttachOriginalQuery
    query_lookup = {str(row["qid"]): row["query"] for _, row in topics.iterrows()}

    return topics, qrels_df, query_lookup


# ---------------------------------------------------------
# Pipeline Definitions
# ---------------------------------------------------------
def get_pipelines(n, index_ref, reranker, query_lookup, cutoff):
    """
    Constructs the dictionary of pipelines for a specific number of rewrites (N).
    """
    # 

    # Common Components
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach_query = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    # Baselines
    pipeline_bm25_base = bm25
    pipeline_bm25_t5 = bm25 >> get_text >> reranker

    # Pipeline 1: Dedup
    p_dedup = (rewrite
               >> bm25
               >> MergeRewrites()
               >> attach_query
               >> Dedup()
               >> get_text
               >> reranker)

    # Pipeline 2: Dedup + Cutoff
    p_dedup_cutoff = (rewrite
                      >> bm25
                      >> MergeRewrites()
                      >> attach_query
                      >> DedupAndCutoff(cutoff)
                      >> get_text
                      >> reranker)

    # Pipeline 3: Borda Voting
    borda_voting = BordaVotingTransformerV2(cutoff=cutoff, weight=None, method="borda")
    p_borda = (rewrite
               >> bm25
               >> borda_voting
               >> attach_query
               >> get_text
               >> reranker)

    # Pipeline 4: Weighted Borda
    borda_weighted = BordaVotingTransformerV2(
        cutoff=cutoff,
        weight=lambda x: np.exp(x * 0.1),
        method="borda"
    )
    p_borda_weighted = (rewrite
                        >> bm25
                        >> borda_weighted
                        >> attach_query
                        >> get_text
                        >> reranker)

    # Pipeline 5: CombMAX
    p_combmax = (rewrite
                 >> bm25
                 >> MergeRewrites()
                 >> attach_query
                 >> CombMAXTransformer()
                 >> get_text
                 >> reranker)

    return {
        "BM25": pipeline_bm25_base,
        "BM25_MonoT5": pipeline_bm25_t5,
        "Pipeline_Dedup": p_dedup,
        "Pipeline_DedupCutoff": p_dedup_cutoff,
        "Pipeline_Borda": p_borda,
        "Pipeline_BordaWeighted": p_borda_weighted,
        "Pipeline_CombMAX": p_combmax
    }


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. Setup
    results_path = setup_directories()
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)

    # Initialize Reranker once
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    # 2. Data
    topics, qrels, query_lookup = load_data()

    # 3. Experiment Loop
    all_results = []
    eval_metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ constants.EXPERIMENT_2_RETRIEVAL_CUTOFF]

    print(f"Starting experiment loop for N={constants.EXPERIMENT_2_N_REWRITES} topics... {len(topics)}")

    for n in constants.EXPERIMENT_2_N_REWRITES:
        print(f"Running pipeline for N={n}...")

        # Define systems
        systems_dict = get_pipelines(
            n, index_ref, mono_t5, query_lookup, constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
        )

        # Run Experiment
        exp_df = pt.Experiment(
            list(systems_dict.values()),
            topics,
            qrels,
            eval_metrics=eval_metrics,
            names=list(systems_dict.keys()),
            verbose=True,
            batch_size=constants.EXPERIMENT_BATCH_SIZE,
            precompute_prefix=True
        )

        # Post-process: Add 'n' column and formatting
        exp_df['n'] = n
        print(exp_df)
        row_data = {"n": n}
        for _, row in exp_df.iterrows():
            system_name = row['name']
            for metric in eval_metrics:
                metric_str = str(metric)
                val = row[metric_str]

                # Standardize metric string for CSV column headers
                if metric_str == f"R@{constants.EXPERIMENT_2_RETRIEVAL_CUTOFF}":
                    key = f"{system_name}_R@c"  # Maps specific cutoff to generic key
                    metric_str = "R@10"  # For print display only
                else:
                    key = f"{system_name}_{metric_str}"

                row_data[key] = val

        print("\n")
        print(f"Experiment for N={n} complete. Results head:")
        print(row_data)
        print("\n")

        all_results.append(row_data)

    # 4. Save and Visualize
    results_df = pd.DataFrame(all_results)

    print("\nExperiment Complete. Results head:")
    print(results_df.head())

    save_experiment_results(
        results_df,
        experiment_name="experiment_2",
        output_dir=results_path,
        filename="experiment_2_results",
    )

    # Generate Plots
    generate_plots(results_df, results_path)


def generate_plots(df, output_dir):
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
        ("Pipeline_DedupCutoff", "Dedup Cutoff Method"),
        ("Pipeline_Borda", "Borda Voting"),
        ("Pipeline_BordaWeighted", "Borda Weighted"),
        ("Pipeline_CombMAX", "CombMAX")
    ]

    # 1. Recall Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_R@10" if "R@10" in df.columns else f"{sys}_R@c" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2: Document Combination Methods Comparison (Recall)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_R_comparison.png",
        **common_args
    )

    # 2. NDCG Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_nDCG@10" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2: Document Combination Methods Comparison (nDCG@10)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_nDCG_comparison.png",
        **common_args
    )

    # 3. Regression Plot
    plot_with_regression(
        df,
        metric_cols=[f"{sys}_R@c" if f"{sys}_R@c" in df.columns else f"{sys}_R@10" for sys, _ in systems],
        title="Experiment 2: R@10 vs Number of Rewrites (All Methods)",
        xlabel="Number of Rewrites",
        ylabel="R@10",
        filename="/n_rewrites_vs_R_regression_all_methods.png",
        **common_args
    )


if __name__ == "__main__":
    main()
