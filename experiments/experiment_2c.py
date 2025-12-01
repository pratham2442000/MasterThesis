import os
import sys

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

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
from code_base.query_trasnformers import Dedup, BordaVotingTransformerV2

# -----------------------------
# Configuration (from constants.py)
# -----------------------------
DATA_PATH = constants.DATA_PATH
RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_2c"

if os.path.exists(RESULTS_PATH):
    print(f"Results directory {RESULTS_PATH} already exists.")
else:
    os.makedirs(RESULTS_PATH)
    print(f"Created results directory at {RESULTS_PATH}.")

DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH

# Experiment controls
TOPIC_LIMIT = constants.EXPERIMENT_2_TOPIC_LIMIT
N_REWRITES = constants.EXPERIMENT_2C_REWRITING_COUNT
EVAL_METRICS = constants.EXPERIMENT_2_EVAL_METRICS
C_VALUES = constants.EXPERIMENT_2C_C_VALUES

# Reranker
T5_BATCH_SIZE = constants.T5_BATCH_SIZE
EXPERIMENT_BATCH_SIZE = constants.EXPERIMENT_BATCH_SIZE

# -----------------------------
# Data
# -----------------------------
print("Loading data...")
docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, DATA_SPLIT)

topics_df["query"] = topics_df["query"].apply(preprocess_query)
topics = topics_df[:TOPIC_LIMIT]
QUERY_LOOKUP = {str(row["qid"]): row["query"] for _, row in topics.iterrows()}

index_ref = pt.IndexFactory.of(INDEX_PATH)
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

# -----------------------------
# Experiment
# -----------------------------
print(f"Running experiment 2c with fixed rewrites={N_REWRITES} and varying cutoff...")

results_list = []
attach_query = AttachOriginalQuery(QUERY_LOOKUP)

# Fixed rewrite transformer since N is constant
rewrite = QueryRewriteTransformer(N_REWRITES, dataset_name=constants.DATASET_NAME)

for cutoff in C_VALUES:
    print(f"Running for cutoff={cutoff}")

    dedup = Dedup()

    # Borda voting transformers
    borda_voting = BordaVotingTransformerV2(
        cutoff=cutoff, weight=None, method="borda"
    )
    borda_voting_weighted = BordaVotingTransformerV2(
        cutoff=cutoff,
        weight=lambda x: np.exp(x * 0.1),  # Exponential weighting
        method="borda",
    )

    # Baseline BM25 on original queries
    bm25_base = bm25 % cutoff

    # Baseline BM25 + MonoT5
    pipeline_bm25_t5 = (
            bm25 % cutoff
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 1: Dedup method (original)
    pipeline_dedup = (
            rewrite
            >> bm25 % cutoff
            >> MergeRewrites()
            >> attach_query
            >> dedup
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 2: Borda voting method
    pipeline_borda = (
            rewrite
            >> bm25 % cutoff
            >> borda_voting
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 3: Borda voting with weighted scores
    pipeline_borda_weighted = (
            rewrite
            >> bm25 % cutoff
            >> borda_voting_weighted
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Define metrics for this iteration
    current_metrics = EVAL_METRICS + [R @ cutoff]

    experiment_results = pt.Experiment(
        [
            bm25_base,
            pipeline_bm25_t5,
            pipeline_dedup,

            pipeline_borda,
            pipeline_borda_weighted,
        ],
        topics,
        qrels_df,
        eval_metrics=current_metrics,
        names=[
            "BM25",
            "BM25_MonoT5",
            "Pipeline_Dedup",

            "Pipeline_Borda",
            "Pipeline_BordaWeighted",
        ],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    # Collect results for this 'cutoff'
    row_data = {"cutoff": cutoff}

    for _, exp_row in experiment_results.iterrows():
        system_name = exp_row["name"]
        for metric in current_metrics:
            metric_str = str(metric)
            val = exp_row[metric_str]

            # Map dynamic R@c to fixed column name "R@c" for consistency
            if metric_str == f"R@{cutoff}":
                key = f"{system_name}_R@c"
            else:
                key = f"{system_name}_{metric_str}"

            row_data[key] = val

    results_list.append(row_data)

    print("\n\n")

    print(
        f"cutoff={cutoff}: BM25 R@10={row_data.get('BM25_R@10', 0):.4f}, "
        f"BM25 + MonoT5 R@10={row_data.get('BM25_MonoT5_R@10', 0):.4f},"
        f"Dedup R@10={row_data.get('Pipeline_Dedup_R@10', 0):.4f}, "

        f"Borda R@10={row_data.get('Pipeline_Borda_R@10', 0):.4f}, "
        f"BordaWeighted R@10={row_data.get('Pipeline_BordaWeighted_R@10', 0):.4f}, "
        f"BM25 NDCG@10={row_data.get('BM25_nDCG@10', 0):.4f}, "
        f"BM25 + MonoT5 NDCG@10={row_data.get('BM25_MonoT5_nDCG@10', 0):.4f}, "
        f"Dedup NDCG@10={row_data.get('Pipeline_Dedup_nDCG@10', 0):.4f}, "

        f"Borda NDCG@10={row_data.get('Pipeline_Borda_nDCG@10', 0):.4f}, "
        f"BordaWeighted NDCG@10={row_data.get('Pipeline_BordaWeighted_nDCG@10', 0):.4f}"
    )

results_df = pd.DataFrame(results_list)

save_experiment_results(
    results_df,
    experiment_name="experiment_2c",
    output_dir=RESULTS_PATH,
    filename="experiment_2c_results",
)

print(results_df)

plot_metrics(
    results_df,
    x_col="cutoff",
    metric_cols=[
        "BM25_R@10",
        "BM25_MonoT5_R@10",
        "Pipeline_Dedup_R@10",

        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    labels=[
        "BM25 Baseline",
        "BM25 + MonoT5",
        "Dedup Method",

        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2c: Document Combination Methods Comparison (Varying Cutoff)",
    xlabel="Retrieval Cutoff",
    filename="/cutoff_vs_R_comparison.png",
    use_log_scale=True,
    output_dir=RESULTS_PATH,
)

plot_metrics(
    results_df,
    x_col="cutoff",
    metric_cols=[
        "BM25_nDCG@10",
        "BM25_MonoT5_nDCG@10",
        "Pipeline_Dedup_nDCG@10",

        "Pipeline_Borda_nDCG@10",
        "Pipeline_BordaWeighted_nDCG@10",
    ],
    labels=[
        "BM25 Baseline",
        "BM25+MonoT5",
        "Dedup Method",

        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2c: Document Combination Methods Comparison (nDCG@10)",
    xlabel="Retrieval Cutoff",
    filename="/cutoff_vs_nDCG_comparison.png",
    use_log_scale=True,
    output_dir=RESULTS_PATH,
)

plot_with_regression(
    results_df,
    x_col="cutoff",
    metric_cols=[
        "BM25_R@10",
        "BM25_MonoT5_R@10",
        "Pipeline_Dedup_R@10",

        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    title="Experiment 2c: R@10 vs Retrieval Cutoff (All Methods)",
    xlabel="Retrieval Cutoff",
    ylabel="R@10",
    filename="/cutoff_vs_R_regression_all_methods.png",
    use_log_scale=True,
    output_dir=RESULTS_PATH,
)
