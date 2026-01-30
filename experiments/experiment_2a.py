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
import pyterrier_dr as pt_dr

# -----------------------------
# Configuration (from constants.py)
# -----------------------------
DATA_PATH = constants.DATA_PATH
RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_2a"

if os.path.exists(RESULTS_PATH):
    print(f"Results directory {RESULTS_PATH} already exists.")
else:
    os.makedirs(RESULTS_PATH)
    print(f"Created results directory at {RESULTS_PATH}.")

DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH
DENSE_INDEX_PATH = constants.DENSE_INDEX_PATH
DENSE_MODEL_NAME = constants.DENSE_MODEL_NAME

# Experiment controls
TOPIC_LIMIT = constants.EXPERIMENT_2_TOPIC_LIMIT
N_REWRITES = constants.EXPERIMENT_2_N_REWRITES
EVAL_METRICS = constants.EXPERIMENT_2_EVAL_METRICS
RETRIEVAL_CUTOFF = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

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

# -----------------------------
# Dense Retrieval Setup
# -----------------------------
print("Setting up dense retrieval...")

# Initialize TCT-ColBERT model
model = pt_dr.TctColBert(DENSE_MODEL_NAME)

# Initialize Faiss Index
# We expect the index to be created by experiment_2a_setup.py
if not os.path.exists(os.path.join(DENSE_INDEX_PATH, "index.dpr")):
    raise FileNotFoundError(
        f"Dense index not found at {DENSE_INDEX_PATH}. "
        "Please run experiments/experiment_2a_setup.py first."
    )

index = pt_dr.FlexIndex(DENSE_INDEX_PATH, model.doc_encoder())
print(f"Loaded dense index from {DENSE_INDEX_PATH}.")

# Create Dense Retriever and Reranker
dense_retriever = model.query_encoder() >> index.retriever()
dense_reranker = model.query_encoder() >> index.scorer()

# Sparse Index and Retriever for Reranking Pipeline
index_ref = pt.IndexFactory.of(INDEX_PATH)
sparse_retriever = pt.BatchRetrieve(index_ref, wmodel="BM25")

mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)


# -----------------------------
# Transformers
# -----------------------------


# -----------------------------
# Experiment
# -----------------------------
print("Running experiment 2a with dense retrieval and query rewriting pipeline...")

results_list = []
attach_query = AttachOriginalQuery(QUERY_LOOKUP)

for N in N_REWRITES:
    rewrite = QueryRewriteTransformer(N, dataset_name=constants.DATASET_NAME)

    dedup = Dedup()

    # Borda voting transformers
    borda_voting = BordaVotingTransformerV2(
        cutoff=RETRIEVAL_CUTOFF, weight=None, method="borda"
    )
    borda_voting_weighted = BordaVotingTransformerV2(
        cutoff=RETRIEVAL_CUTOFF,
        weight=lambda x: np.exp(x * 0.1),  # Exponential weighting
        method="borda",
    )

    # Baseline Dense on original queries
    dense_base = dense_retriever % RETRIEVAL_CUTOFF

    # Baseline Dense + MonoT5
    pipeline_dense_t5 = (
            dense_retriever % RETRIEVAL_CUTOFF
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline: BM25 + Dense Rerank
    # Retrieve using BM25, then rescale using Dense scores, then re-sort
    pipeline_bm25_dense = (
            sparse_retriever % RETRIEVAL_CUTOFF
            >> dense_reranker
            >> pt.apply.generic(lambda df: df.sort_values(["qid", "score"], ascending=[True, False]))
    )

    # Pipeline 1: Dedup method (original)
    pipeline_dedup = (
            rewrite
            >> dense_retriever % RETRIEVAL_CUTOFF
            >> MergeRewrites()
            >> attach_query
            >> dedup
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 2: Borda voting method
    pipeline_borda = (
            rewrite
            >> dense_retriever % RETRIEVAL_CUTOFF
            >> borda_voting
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 3: Borda voting with weighted scores
    pipeline_borda_weighted = (
            rewrite
            >> dense_retriever % RETRIEVAL_CUTOFF
            >> borda_voting_weighted
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Define metrics for this iteration
    current_metrics = EVAL_METRICS + [R @ RETRIEVAL_CUTOFF]

    experiment_results = pt.Experiment(
        [
            dense_base,
            pipeline_dense_t5,
            pipeline_bm25_dense,
            pipeline_dedup,
            pipeline_borda,
            pipeline_borda_weighted,
        ],
        topics,
        qrels_df,
        eval_metrics=current_metrics,
        names=[
            "Dense",
            "Dense_MonoT5",
            "BM25_DenseRerank",
            "Pipeline_Dedup",
            "Pipeline_Borda",
            "Pipeline_BordaWeighted",
        ],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
        precompute_prefix=True
    )

    # Collect results for this 'N'
    row_data = {"n": N}

    for _, exp_row in experiment_results.iterrows():
        system_name = exp_row["name"]
        for metric in current_metrics:
            metric_str = str(metric)
            val = exp_row[metric_str]

            # Map dynamic R@c to fixed column name "R@c" for consistency
            if metric_str == f"R@{RETRIEVAL_CUTOFF}":
                key = f"{system_name}_R@c"
            else:
                key = f"{system_name}_{metric_str}"

            row_data[key] = val

    results_list.append(row_data)

    print("\n\n")

    print(
        f"n={N}: Dense R@10={row_data.get('Dense_R@10', 0):.4f}, "
        f"Dense + MonoT5 R@10={row_data.get('Dense_MonoT5_R@10', 0):.4f}, "
        f"BM25 + Dense R@10={row_data.get('BM25_DenseRerank_R@10', 0):.4f}, "
        f"Dedup R@10={row_data.get('Pipeline_Dedup_R@10', 0):.4f}, "

        f"Borda R@10={row_data.get('Pipeline_Borda_R@10', 0):.4f}, "
        f"BordaWeighted R@10={row_data.get('Pipeline_BordaWeighted_R@10', 0):.4f}, "
        f"Dense NDCG@10={row_data.get('Dense_nDCG@10', 0):.4f}, "
        f"Dense + MonoT5 NDCG@10={row_data.get('Dense_MonoT5_nDCG@10', 0):.4f}, "
        f"Dedup NDCG@10={row_data.get('Pipeline_Dedup_nDCG@10', 0):.4f}, "

        f"Borda NDCG@10={row_data.get('Pipeline_Borda_nDCG@10', 0):.4f}, "
        f"BordaWeighted NDCG@10={row_data.get('Pipeline_BordaWeighted_nDCG@10', 0):.4f}"
    )

results_df = pd.DataFrame(results_list)

save_experiment_results(
    results_df,
    experiment_name="experiment_2a",
    output_dir=RESULTS_PATH,
    filename="experiment_2a_results",
)

print(results_df)

plot_metrics(
    results_df,
    x_col="n",
    metric_cols=[
        "Dense_R@10",
        "Dense_MonoT5_R@10",
        "BM25_DenseRerank_R@10",
        "Pipeline_Dedup_R@10",

        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    labels=[
        "Dense Baseline",
        "Dense + MonoT5",
        "BM25 + Dense Rerank",
        "Dedup Method",

        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2a: Document Combination Methods Comparison (Dense)",
    xlabel="Number of Rewrites",
    filename="/n_rewrites_vs_R_comparison.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)

plot_metrics(
    results_df,
    x_col="n",
    metric_cols=[
        "Dense_nDCG@10",
        "Dense_MonoT5_nDCG@10",
        "BM25_DenseRerank_nDCG@10",
        "Pipeline_Dedup_nDCG@10",

        "Pipeline_Borda_nDCG@10",
        "Pipeline_BordaWeighted_nDCG@10",
    ],
    labels=[
        "Dense Baseline",
        "Dense+MonoT5",
        "BM25 + Dense Rerank",
        "Dedup Method",

        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2a: Document Combination Methods Comparison (nDCG@10) (Dense)",
    xlabel="Number of Rewrites",
    filename="/n_rewrites_vs_nDCG_comparison.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)

plot_with_regression(
    results_df,
    x_col="n",
    metric_cols=[
        "Dense_R@10",
        "Dense_MonoT5_R@10",
        "BM25_DenseRerank_R@10",
        "Pipeline_Dedup_R@10",

        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    title="Experiment 2a: R@10 vs Number of Rewrites (All Methods) (Dense)",
    xlabel="Number of Rewrites",
    ylabel="R@10",
    filename="/n_rewrites_vs_R_regression_all_methods.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)
