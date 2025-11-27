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
)
from code_base.llm_q import rewrite_single_query
from code_base.query_trasnformers import Dedup, DedupAndCutoff, BordaVotingTransformerV2

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
C_VALUES = constants.EXPERIMENT_2_C_VALUES
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
# print("Setting up dense retrieval...")
# # Initialize encoder
# encoder = pt_dr.TctColBertDocumentEncoder(DENSE_MODEL_NAME)
# # Initialize query encoder (TCT-ColBERT uses the same model for queries, but wrapped differently usually,
# # but pyterrier_dr handles it via the retriever or we use TctColBertQueryEncoder)
# query_encoder = pt_dr.TctColBertQueryEncoder(DENSE_MODEL_NAME)
#
# # Initialize Faiss Index
# # We use FaissFlat for exact search if dataset is small, or FaissHnsw for larger.
# # Given 267MB corpus, FaissFlat is fine and accurate.
# index = pt_dr.FaissFlat(DENSE_INDEX_PATH, encoder)
#
# # Check if index exists, if not index the documents
# if not os.path.exists(os.path.join(DENSE_INDEX_PATH, "index.dpr")): # Check for specific file created by FaissFlat
#     print("Dense index not found. Indexing documents...")
#     # Ensure docs_df has 'docno' and 'text'
#     if "docno" not in docs_df.columns or "text" not in docs_df.columns:
#         raise ValueError("docs_df must contain 'docno' and 'text' columns")
#
#     index.index(docs_df)
#     print("Indexing complete.")
# else:
#     print(f"Loading dense index from {DENSE_INDEX_PATH}...")

# Create Dense Retriever


dense_retriever = query_encoder >> index

# We also need the sparse index for text lookup for the reranker
index_ref = pt.IndexFactory.of(INDEX_PATH)

mono_t5_reranker = MonoT5ReRanker(verbose=False, batch_size=T5_BATCH_SIZE)


# -----------------------------
# Transformers
# -----------------------------
class QueryRewriteTransformer(pt.Transformer):
    """
    Transformer that reads preprocessed query rewrites from a dataset file.
    Uses rewrite_single_query to get k rewrites for each query ID.
    """

    def __init__(self, n: int, dataset_name: str = None):
        self.n = n
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = constants.DATASET_NAME

    def transform(self, topics_like: pd.DataFrame) -> pd.DataFrame:
        out_rows = []
        for _, row in topics_like.iterrows():
            qid = row["qid"]
            query = row["query"]
            try:
                rewrites = [query] + rewrite_single_query(
                    self.dataset_name, str(qid), self.n - 1
                )
            except (ValueError, FileNotFoundError, KeyError) as e:
                print(
                    f"Warning: Could not load rewrites for qid {qid}: {e}. Using original query."
                )
                rewrites = [query] * self.n
            for i, rw in enumerate(rewrites):
                out_rows.append(
                    {
                        "qid": f"{qid}_{i}",  # unique sub-qid for each rewrite
                        "orig_qid": qid,  # store the original qid
                        "query": preprocess_query(rw),
                    }
                )
        return pd.DataFrame(out_rows)


class MergeRewrites(pt.Transformer):
    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        # restore original qid for further dedup/ranking
        res["qid"] = res["qid"].apply(lambda x: x.split("_")[0])
        return res


class AttachOriginalQuery(pt.Transformer):
    """
    Ensure the 'query' column contains the original topic query (needed by MonoT5).
    """

    def __init__(self, query_lookup: dict):
        self.query_lookup = {str(k): v for k, v in query_lookup.items()}

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        if res.empty:
            return res

        res = res.copy()
        mapped = res["qid"].astype(str).map(self.query_lookup)

        # Fallback to orig_qid if available (e.g., before MergeRewrites)
        if mapped.isna().any() and "orig_qid" in res.columns:
            fallback = res.loc[mapped.isna(), "orig_qid"].astype(str).map(
                self.query_lookup
            )
            mapped.fillna(fallback, inplace=True)

        if mapped.isna().any():
            missing_qids = res.loc[mapped.isna(), "qid"].unique()
            raise KeyError(
                f"Could not attach original queries for qids: {missing_qids}. "
                "Ensure topics include these qids."
            )

        res["query"] = mapped
        return res


# -----------------------------
# Experiment
# -----------------------------
print("Running experiment 2a with dense retrieval and query rewriting pipeline...")

results = {"n": N_REWRITES}
for i in EVAL_METRICS:
    results["Dense_" + str(i)] = []
    results["Dense_MonoT5_" + str(i)] = []
    results["Pipeline_Dedup_" + str(i)] = []
    results["Pipeline_DedupCutoff_" + str(i)] = []
    results["Pipeline_Borda_" + str(i)] = []
    results["Pipeline_BordaWeighted_" + str(i)] = []

results["Dense_R@c"] = []
results["Dense_MonoT5_R@c"] = []
results["Pipeline_Dedup_R@c"] = []
results["Pipeline_DedupCutoff_R@c"] = []
results["Pipeline_Borda_R@c"] = []
results["Pipeline_BordaWeighted_R@c"] = []

attach_query = AttachOriginalQuery(QUERY_LOOKUP)

for N in N_REWRITES:
    rewrite = QueryRewriteTransformer(N, dataset_name=constants.DATASET_NAME)
    dedup_topk = DedupAndCutoff(RETRIEVAL_CUTOFF)
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

    # Pipline 1.1: Dedup method with cutoff
    pipeline_dedup_cutoff = (
            rewrite
            >> dense_retriever % RETRIEVAL_CUTOFF
            >> MergeRewrites()
            >> attach_query
            >> dedup_topk
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

    experiment_results = pt.Experiment(
        [
            dense_base,
            pipeline_dense_t5,
            pipeline_dedup,
            pipeline_dedup_cutoff,
            pipeline_borda,
            pipeline_borda_weighted,
        ],
        topics,
        qrels_df,
        eval_metrics=EVAL_METRICS + [R @ RETRIEVAL_CUTOFF],
        names=[
            "Dense",
            "Dense+MonoT5",
            "Pipeline_Dedup",
            "Pipeline_DedupCutoff",
            "Pipeline_Borda",
            "Pipeline_BordaWeighted",
        ],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    # Store results for all methods
    results["Dense_R@10"].append(float(experiment_results["R@10"].iloc[0]))
    results["Dense_MonoT5_R@10"].append(float(experiment_results["R@10"].iloc[1]))
    results["Pipeline_Dedup_R@10"].append(float(experiment_results["R@10"].iloc[2]))
    results["Pipeline_DedupCutoff_R@10"].append(float(experiment_results["R@10"].iloc[3]))
    results["Pipeline_Borda_R@10"].append(float(experiment_results["R@10"].iloc[4]))
    results["Pipeline_BordaWeighted_R@10"].append(float(experiment_results["R@10"].iloc[5]))

    results["Dense_nDCG@10"].append(float(experiment_results["nDCG@10"].iloc[0]))
    results["Dense_MonoT5_nDCG@10"].append(float(experiment_results["nDCG@10"].iloc[1]))
    results["Pipeline_Dedup_nDCG@10"].append(float(experiment_results["nDCG@10"].iloc[2]))
    results["Pipeline_DedupCutoff_nDCG@10"].append(
        float(experiment_results["nDCG@10"].iloc[3])
    )
    results["Pipeline_Borda_nDCG@10"].append(float(experiment_results["nDCG@10"].iloc[4]))
    results["Pipeline_BordaWeighted_nDCG@10"].append(
        float(experiment_results["nDCG@10"].iloc[5])
    )

    results["Dense_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[0]))
    results["Dense_MonoT5_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[1]))
    results["Pipeline_Dedup_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[2])
    )
    results["Pipeline_DedupCutoff_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[3])
    )
    results["Pipeline_Borda_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[4])
    )
    results["Pipeline_BordaWeighted_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"].iloc[5])
    )

    dense_r10 = results["Dense_R@10"][-1]
    dense_monot5_r10 = results["Dense_MonoT5_R@10"][-1]
    dedup_r10 = results["Pipeline_Dedup_R@10"][-1]
    dedupCutoff = results["Pipeline_DedupCutoff_R@10"][-1]
    borda_r10 = results["Pipeline_Borda_R@10"][-1]
    borda_w_r10 = results["Pipeline_BordaWeighted_R@10"][-1]

    dense_ndcg = results["Dense_nDCG@10"][-1]
    dense_monot5_ndcg = results["Dense_MonoT5_nDCG@10"][-1]
    dedup_ndcg = results["Pipeline_Dedup_nDCG@10"][-1]
    dedupCutoff_ndcg = results["Pipeline_DedupCutoff_nDCG@10"][-1]
    borda_ndcg = results["Pipeline_Borda_nDCG@10"][-1]
    borda_w_ndcg = results["Pipeline_BordaWeighted_nDCG@10"][-1]

    print("\n\n")

    print(
        f"n={N}: Dense R@10={dense_r10:.4f}, "
        f"Dense + MonoT5 R@10={dense_monot5_r10:.4f},"
        f"Dedup R@10={dedup_r10:.4f}, "
        f"DedupCutoff R@10={dedupCutoff:.4f}, "
        f"Borda R@10={borda_r10:.4f}, "
        f"BordaWeighted R@10={borda_w_r10:.4f}, "
        f"Dense NDCG@10={dense_ndcg:.4f}, "
        f"Dense + MonoT5 NDCG@10={dense_monot5_ndcg:.4f}, "
        f"Dedup NDCG@10={dedup_ndcg:.4f}, "
        f"DedupCutoff NDCG@10={dedupCutoff_ndcg:.4f}, "
        f"Borda NDCG@10={borda_ndcg:.4f}, "
        f"BordaWeighted NDCG@10={borda_w_ndcg:.4f}"
    )

results_df = pd.DataFrame(results)

save_experiment_results(
    results_df,
    experiment_name="experiment_2a",
    output_dir=RESULTS_PATH,
    filename="experiment_2a_results",
)

print(results)

plot_metrics(
    results_df,
    x_col="n",
    metric_cols=[
        "Dense_R@10",
        "Dense_MonoT5_R@10",
        "Pipeline_Dedup_R@10",
        "Pipeline_DedupCutoff_R@10",
        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    labels=[
        "Dense Baseline",
        "Dense + MonoT5",
        "Dedup Method",
        "Dedup Cutoff Method",
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
        "Pipeline_Dedup_nDCG@10",
        "Pipeline_DedupCutoff_nDCG@10",
        "Pipeline_Borda_nDCG@10",
        "Pipeline_BordaWeighted_nDCG@10",
    ],
    labels=[
        "Dense Baseline",
        "Dense+MonoT5",
        "Dedup Method",
        "Dedup Cutoff Method",
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
        "Pipeline_Dedup_R@10",
        "Pipeline_DedupCutoff_R@10",
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
