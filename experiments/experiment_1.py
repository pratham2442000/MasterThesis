import os
import sys

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query
)

# -----------------------------
# Configuration (from constants.py)
# -----------------------------
DATA_PATH = constants.DATA_PATH
RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_1"

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

print("Loading data...")
docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, DATA_SPLIT)

topics_df["query"] = topics_df["query"].apply(preprocess_query)

topics = topics_df[:TOPIC_LIMIT]
index_ref = pt.IndexFactory.of(INDEX_PATH)

mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

results_list = []

print(f"\nStarting experiment for initial cutoff (c) in: {C_VALUES}")

for c in C_VALUES:
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_retrieved = bm25 % c
    pipeline = bm25_retrieved >> pt.text.get_text(index_ref, "text") >> mono_t5_reranker

    # Define metrics for this iteration
    current_metrics = EVAL_METRICS + [R @ c]

    experiment_results = pt.Experiment(
        [bm25_retrieved, pipeline],
        topics,
        qrels_df,
        eval_metrics=current_metrics,
        names=["BM25", "Pipeline"],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    # Collect results for this 'c'
    row_data = {"c": c}

    for _, exp_row in experiment_results.iterrows():
        system_name = exp_row["name"]
        for metric in current_metrics:
            metric_str = str(metric)
            val = exp_row[metric_str]

            # Map dynamic R@c to fixed column name "R@c" for consistency
            if metric_str == f"R@{c}":
                key = f"{system_name}_R@c"
            else:
                key = f"{system_name}_{metric_str}"

            row_data[key] = val

    results_list.append(row_data)

    # Print summary
    print(
        f"c={c}: "
        f"BM25 R@10={row_data.get('BM25_R@10', 0):.4f}, "
        f"Pipeline R@10={row_data.get('Pipeline_R@10', 0):.4f}, "
        f"BM25 NDCG@10={row_data.get('BM25_nDCG@10', 0):.4f}, "
        f"Pipeline NDCG@10={row_data.get('Pipeline_nDCG@10', 0):.4f}, "
        f"BM25 R@{c}={row_data.get('BM25_R@c', 0):.4f}, "
        f"Pipeline R@{c}={row_data.get('Pipeline_R@c', 0):.4f}"
    )

results_df = pd.DataFrame(results_list)

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
