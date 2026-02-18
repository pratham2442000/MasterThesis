import os
import sys

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
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
from code_base.query_trasnformers import Dedup, CombMAXTransformer

# Best n_rewrites from exp2/exp2b results
N_REWRITES = constants.EXPERIMENT_3_BEST_N_REWRITES


# ---------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------
def setup_directories():
    """Ensure result directories exist."""
    results_path = constants.RESULTS_PATH + "/exp3"
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
    topics = topics.tail(constants.EXPERIMENT_3_TOPIC_LIMIT)

    # Create lookup for AttachOriginalQuery
    query_lookup = {str(row["qid"]): row["query"] for _, row in topics.iterrows()}

    return topics, qrels_df, query_lookup


# ---------------------------------------------------------
# Pipeline Definitions
# ---------------------------------------------------------
def get_pipelines(cutoff, index_ref, reranker, query_lookup):
    """
    Constructs the dictionary of pipelines for a specific cutoff value.
    Uses fixed N_REWRITES=10 (best from exp2/exp2b results).
    """

    # Common Components
    rewrite = QueryRewriteTransformer(N_REWRITES, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach_query = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    # Baselines
    pipeline_bm25_base = bm25
    pipeline_bm25_t5 = bm25 >> get_text >> reranker

    # --- Merge-then-Rerank pipelines ---

    # Pipeline 1: Dedup (merge-then-rerank)
    p_dedup = (rewrite
               >> bm25
               >> MergeRewrites()
               >> attach_query
               >> Dedup()
               >> get_text
               >> reranker)

    # Pipeline 2: CombMAX (merge-then-rerank) — best fusion from exp2
    p_combmax = (rewrite
                 >> bm25
                 >> MergeRewrites()
                 >> attach_query
                 >> CombMAXTransformer()
                 >> get_text
                 >> reranker)

    # --- Rerank-then-Merge pipelines (from exp2b, best overall nDCG) ---

    # Pipeline 3: Dedup RTM (rerank-then-merge)
    p_dedup_rtm = (rewrite
                   >> bm25
                   >> get_text
                   >> reranker
                   >> MergeRewrites()
                   >> Dedup())

    # Pipeline 4: CombMAX RTM (rerank-then-merge)
    p_combmax_rtm = (rewrite
                     >> bm25
                     >> get_text
                     >> reranker
                     >> MergeRewrites()
                     >> CombMAXTransformer())

    return {
        "BM25": pipeline_bm25_base,
        "BM25_MonoT5": pipeline_bm25_t5,
        "Dedup": p_dedup,
        "CombMAX": p_combmax,
        "Dedup_RTM": p_dedup_rtm,
        "CombMAX_RTM": p_combmax_rtm,
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
    c_values = constants.EXPERIMENT_3_C_VALUES

    print(
        f"Starting experiment 3 with fixed n_rewrites={N_REWRITES}, varying cutoff={c_values}... topics={len(topics)}")

    for cutoff in c_values:
        print(f"Running pipeline for cutoff={cutoff}...")

        # Define metrics for this cutoff
        eval_metrics = constants.EXPERIMENT_3_EVAL_METRICS + [R @ cutoff]

        # Define systems
        systems_dict = get_pipelines(cutoff, index_ref, mono_t5, query_lookup)

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

        # Post-process: collect results
        exp_df['cutoff'] = cutoff
        print(exp_df)
        row_data = {"cutoff": cutoff}
        for _, row in exp_df.iterrows():
            system_name = row['name']
            for metric in eval_metrics:
                metric_str = str(metric)
                val = row[metric_str]

                # Standardize metric string for CSV column headers
                if metric_str == f"R@{cutoff}":
                    key = f"{system_name}_R@c"
                else:
                    key = f"{system_name}_{metric_str}"

                row_data[key] = val

        print("\n")
        print(f"Experiment for cutoff={cutoff} complete. Results head:")
        print(row_data)
        print("\n")

        all_results.append(row_data)

    # 4. Save and Visualize
    results_df = pd.DataFrame(all_results)

    print("\nExperiment Complete. Results head:")
    print(results_df.head())

    save_experiment_results(
        results_df,
        experiment_name="experiment_3",
        output_dir=results_path,
        filename="experiment_3_results",
    )

    # Generate Plots
    generate_plots(results_df, results_path)


def generate_plots(df, output_dir):
    """Encapsulate plotting logic."""
    common_args = {
        "x_col": "cutoff",
        "output_dir": output_dir,
        "use_log_scale": True
    }

    systems = [
        ("BM25", "BM25 Baseline"),
        ("BM25_MonoT5", "BM25 + MonoT5"),
        ("Dedup", "Dedup (Merge→Rerank)"),
        ("CombMAX", "CombMAX (Merge→Rerank)"),
        ("Dedup_RTM", "Dedup (Rerank→Merge)"),
        ("CombMAX_RTM", "CombMAX (Rerank→Merge)"),
    ]

    # 1. Recall Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_R@10" if f"{sys}_R@10" in df.columns else f"{sys}_R@c" for sys, _ in systems],
        labels=[label for _, label in systems],
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
    plot_with_regression(
        df,
        metric_cols=[f"{sys}_R@c" if f"{sys}_R@c" in df.columns else f"{sys}_R@10" for sys, _ in systems],
        title="Experiment 3: R@10 vs Cutoff Regression (n_rewrites=10)",
        xlabel="Retrieval Cutoff",
        ylabel="R@10",
        filename="/cutoff_vs_R_regression_all_methods.png",
        **common_args
    )


if __name__ == "__main__":
    main()
