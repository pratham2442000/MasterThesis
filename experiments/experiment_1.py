import os
import sys

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query
)


# ---------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------
def ensure_results_dir(base_path):
    """Creates the experiment results directory if it doesn't exist."""
    results_path = os.path.join(base_path, "experiment_results", "experiment_1")
    os.makedirs(results_path, exist_ok=True)
    print(f"Results directory ensured at: {results_path}")
    return results_path


def load_and_prep_data():
    """Loads dataset and preprocesses queries."""
    print("Loading data...")
    docs_df, topics_df, qrels_df = dataset_factory.get_dataset(
        constants.DATASET_NAME, constants.DATA_SPLIT
    )

    topics_df["query"] = topics_df["query"].apply(preprocess_query)
    topics = topics_df[:constants.EXPERIMENT_1_TOPIC_LIMIT]

    return topics, qrels_df


# ---------------------------------------------------------
# Experiment Logic
# ---------------------------------------------------------
def get_pipelines(index, reranker, c):
    """
    Constructs the retrieval pipelines for a specific cutoff 'c'.
    """
    # 
    # 1. First stage: BM25 retrieval with cutoff
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    bm25_retrieved = bm25 % c

    # 2. Second stage: T5 Reranking
    pipeline_t5 = (
            bm25_retrieved
            >> pt.text.get_text(index, "text")
            >> reranker
    )

    return {
        "BM25": bm25_retrieved,
        "Pipeline": pipeline_t5
    }


def run_experiment_loop(topics, qrels, index, reranker):
    """
    Iterates through cutoff values (C), runs experiments, and aggregates results.
    """
    results_list = []
    c_values = constants.EXPERIMENT_1_C_VALUES
    if isinstance(c_values, int):
        c_values = [c_values]

    base_metrics = constants.EXPERIMENT_1_EVAL_METRICS
    print(f"\nStarting experiment for cutoffs (c): {c_values}")

    for c in c_values:
        # Define dynamic metric: Recall at cutoff c
        current_metrics = base_metrics + [R @ c]

        # Get systems for this cutoff
        systems = get_pipelines(index, reranker, c)

        # Run PyTerrier Experiment
        exp_df = pt.Experiment(
            list(systems.values()),
            topics,
            qrels,
            eval_metrics=current_metrics,
            names=list(systems.keys()),
            verbose=True,
            batch_size=constants.EXPERIMENT_BATCH_SIZE,
            precompute_prefix=True
        )

        print("\nExperiment results for c =", c)
        print(exp_df)

        # Process Results for this 'c'
        row_data = {"c": c}
        for _, row in exp_df.iterrows():
            system_name = row["name"]
            for metric in current_metrics:
                metric_str = str(metric)
                val = row[metric_str]

                # Normalize the dynamic Recall metric name for the CSV
                if metric_str == f"R@{c}":
                    key = f"{system_name}_R@c"
                else:
                    key = f"{system_name}_{metric_str}"

                row_data[key] = val

        results_list.append(row_data)
        print_summary(c, row_data)

    return pd.DataFrame(results_list)


def print_summary(c, data):
    """Helper to print a readable summary of the iteration."""
    print(
        f"c={c}: "
        f"BM25 R@10={data.get('BM25_R@10', 0):.4f}, "
        f"Pipeline R@10={data.get('Pipeline_R@10', 0):.4f}, "
        f"BM25 NDCG@10={data.get('BM25_nDCG@10', 0):.4f}, "
        f"Pipeline NDCG@10={data.get('Pipeline_nDCG@10', 0):.4f}"
    )


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def generate_visualizations(results_df, output_dir):
    """Generates all plots for the experiment."""
    # 

    # 1. R@10 vs Cutoff
    plot_metrics(
        results_df,
        x_col="c",
        metric_cols=["BM25_R@10", "Pipeline_R@10"],
        labels=["BM25 Baseline", "BM25 + MonoT5 Re-ranker"],
        title="Effect of Initial Cutoff (c) on Recall (R@10)",
        xlabel="Initial Cutoff (c) for BM25 Retrieval",
        filename="/cutoff_vs_R.png",
        output_dir=output_dir,
        use_log_scale=True
    )

    # 2. NDCG@10 vs Cutoff
    plot_metrics(
        results_df,
        x_col="c",
        metric_cols=["BM25_nDCG@10", "Pipeline_nDCG@10"],
        labels=["BM25 Baseline", "BM25 + MonoT5 Re-ranker"],
        title="Effect of Initial Cutoff (c) on NDCG@10",
        xlabel="Initial Cutoff (c) for BM25 Retrieval",
        ylabel="NDCG@10",
        filename="/cutoff_vs_NDCG_at_10.png",
        output_dir=output_dir,
        use_log_scale=True
    )

    # 3. Regression Analysis
    print("\n--- Linear Regression Analysis (log(c) vs R@10) ---")
    plot_with_regression(
        results_df,
        x_col="c",
        metric_cols=["BM25_R@10", "Pipeline_R@10"],
        labels=["BM25 R@10", "Pipeline R@10"],
        use_log_scale=True,
        title="R (@ 10) vs. Initial Cutoff (c) Regression",
        xlabel="Initial Cutoff (c) (log scale)",
        ylabel="Recall (R@10)",
        filename="/cutoff_R_at_10_log_regression.png",
        output_dir=output_dir,
    )


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. Setup
    results_path = ensure_results_dir(constants.DATA_PATH)
    topics, qrels = load_and_prep_data()

    # 2. Initialize Resources
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    # 3. Run Experiment
    results_df = run_experiment_loop(topics, qrels, index_ref, mono_t5)

    # 4. Save Data
    print("\nFinal Results DataFrame:")
    print(results_df)

    save_experiment_results(
        results_df,
        experiment_name="experiment_1",
        output_dir=results_path,
        filename="experiment_1_results",
    )

    # 5. Visualize
    generate_visualizations(results_df, results_path)


if __name__ == "__main__":
    main()
