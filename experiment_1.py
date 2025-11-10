import os

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

import constants
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
    preprocess_query
)
from code_base.wiki_loader_pyt import wiki_loader_pyt

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
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)

topics_df["query"] = topics_df["query"].apply(preprocess_query)

topics = topics_df[:TOPIC_LIMIT]
index_ref = pt.IndexFactory.of(INDEX_PATH)

mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)

results = {}
results["c"] = C_VALUES
for i in EVAL_METRICS:
    results["BM25_" + str(i)] = []
    results["Pipeline_" + str(i)] = []

results["BM25_R@c"] = []
results["Pipeline_R@c"] = []



print(f"\nStarting experiment for initial cutoff (c) in: {C_VALUES}")


for c in C_VALUES:
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_retrieved = bm25 % c
    pipeline = bm25_retrieved >> pt.text.get_text(index_ref, "text") >> mono_t5_reranker

    experiment_results = pt.Experiment(
        [bm25_retrieved, pipeline],
        topics,
        qrels_df,
        eval_metrics=EVAL_METRICS + [R @ c],
        names=["BM25", "Pipeline"],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    results["BM25_R@10"].append(float(experiment_results["R@10"][0]))
    results["Pipeline_R@10"].append(float(experiment_results["R@10"][1]))
    results["BM25_nDCG@10"].append(float(experiment_results["nDCG@10"][0]))
    results["Pipeline_nDCG@10"].append(float(experiment_results["nDCG@10"][1]))
    results["BM25_R@c"].append(float(experiment_results[f"R@{c}"][0]))
    results["Pipeline_R@c"].append(float(experiment_results[f"R@{c}"][1]))

    bm25_r10 = results["BM25_R@10"][-1]
    pipe_r10 = results["Pipeline_R@10"][-1]
    bm25_ndcg = results["BM25_nDCG@10"][-1]
    pipe_ndcg = results["Pipeline_nDCG@10"][-1]
    bm25_rc = results["BM25_R@c"][-1]
    pipe_rc = results["Pipeline_R@c"][-1]

    print(
        f"c={c}: BM25 R@10={bm25_r10:.4f}, "
        f"Pipeline R@10={pipe_r10:.4f}, "
        f"BM25 NDCG@10={bm25_ndcg:.4f}, "
        f"Pipeline NDCG@10={pipe_ndcg:.4f}, "
        f"BM25 R@{c}={bm25_rc:.4f}, "
        f"Pipeline R@{c}={pipe_rc:.4f}"
    )

results_df = pd.DataFrame(results)

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
    metric_cols=["BM25_R@10", "Pipeline_R@10", "BM25_R@c", "Pipeline_R@c"],
    labels=["BM25 Baseline", "BM25 + MonoT5 Re-ranker"],
    title="Effect of Initial Cutoff (c) on Recall (R@10 and R@c)",
    xlabel="Initial Cutoff (c) for BM25 Retrieval",
    filename="/cutoff_vs_R.png",
    output_dir=RESULTS_PATH,
    use_log_scale=False
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
    use_log_scale=False
)

# LINEAR REGRESSION ANALYSIS (R@10)

print("\n--- Linear Regression Analysis (log(c) vs R@10) ---")

plot_with_regression(
    results_df,
    x_col="c",
    metric_cols=["BM25_R@10", "Pipeline_R@10", "BM25_R@c", "Pipeline_R@c"],
    labels=["BM25 (Baseline) R@10", "Pipeline (BM25+MonoT5) R@10"],
    use_log_scale=False,
    title="R (@ 10, c) vs. Initial Cutoff (c) with Log(c) Linear Regression",
    xlabel="Initial Cutoff (c) for BM25 Retrieval (log scale)",
    ylabel="Recall (R@10 and R@c)",
    filename="/cutoff_R_at_10_log_regression.png",
    output_dir=RESULTS_PATH,
)
