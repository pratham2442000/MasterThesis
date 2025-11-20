import os

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

import constants
from code_base.experiment_utils import (
    plot_metrics,
    save_experiment_results,
    preprocess_query,
    plot_with_regression,
)
from code_base.llm_q import rewrite_single_query
from code_base.query_trasnformers import Dedup, DedupAndCutoff, BordaVotingTransformerV2
from code_base.wiki_loader_pyt import wiki_loader_pyt

# -----------------------------
# Configuration (from constants.py)
# -----------------------------
DATA_PATH = constants.DATA_PATH
RESULTS_PATH = DATA_PATH + "/experiment_results/experiment_2"

if os.path.exists(RESULTS_PATH):
    print(f"Results directory {RESULTS_PATH} already exists.")
else:
    os.makedirs(RESULTS_PATH)
    print(f"Created results directory at {RESULTS_PATH}.")

DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH

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
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)  # TODO change to dateset fatory


topics_df["query"] = topics_df["query"].apply(preprocess_query)
topics = topics_df[:TOPIC_LIMIT]
QUERY_LOOKUP = {str(row["qid"]): row["query"] for _, row in topics.iterrows()}

index_ref = pt.IndexFactory.of(INDEX_PATH)
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
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


#
# Doc ranking merge strategy: keep only the highest-scoring instance of each (qid, docno) pair
#


# -----------------------------
# Experiment
# -----------------------------
print("Running experiment with query rewriting pipeline...")

results = {"n": N_REWRITES}
for i in EVAL_METRICS:
    results["BM25_" + str(i)] = []
    results["BM25_MonoT5_" + str(i)] = []
    results["Pipeline_Dedup_" + str(i)] = []
    results["Pipeline_DedupCutoff_" + str(i)] = []
    results["Pipeline_Borda_" + str(i)] = []
    results["Pipeline_BordaWeighted_" + str(i)] = []

results["BM25_R@c"] = []
results["BM25_MonoT5_R@c"] = []
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

    # Baseline BM25 on original queries
    bm25_base = bm25 % RETRIEVAL_CUTOFF

    # Baseline BM25 + MonoT5
    pipeline_bm25_t5 = (
            bm25 % RETRIEVAL_CUTOFF
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 1: Dedup method (original)
    pipeline_dedup = (
            rewrite
            >> bm25 % RETRIEVAL_CUTOFF
            >> MergeRewrites()
            >> attach_query
            >> dedup
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipline 1.1: Dedup method with cutoff
    pipeline_dedup_cutoff = (
            rewrite
            >> bm25 % RETRIEVAL_CUTOFF
            >> MergeRewrites()
            >> attach_query
            >> dedup_topk
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 2: Borda voting method
    pipeline_borda = (
            rewrite
            >> bm25 % RETRIEVAL_CUTOFF
            >> borda_voting
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    # Pipeline 3: Borda voting with weighted scores
    pipeline_borda_weighted = (
            rewrite
            >> bm25 % RETRIEVAL_CUTOFF
            >> borda_voting_weighted
            >> attach_query
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    experiment_results = pt.Experiment(
        [
            bm25_base,
            pipeline_bm25_t5,
            pipeline_dedup,
            pipeline_dedup_cutoff,
            pipeline_borda,
            pipeline_borda_weighted,
        ],
        topics,
        qrels_df,
        eval_metrics=EVAL_METRICS + [R @ RETRIEVAL_CUTOFF],
        names=[
            "BM25",
            "BM25+MonoT5"
            "Pipeline_Dedup",
            "Pipeline_DedupCutoff",
            "Pipeline_Borda",
            "Pipeline_BordaWeighted",
        ],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    # Store results for all methods
    results["BM25_R@10"].append(float(experiment_results["R@10"][0]))
    results["BM25_MonoT5_R@10"].append(float(experiment_results["MonoT5@10"][1]))
    results["Pipeline_Dedup_R@10"].append(float(experiment_results["R@10"][2]))
    results["Pipeline_DedupCutoff_R@10"].append(float(experiment_results["R@10"][3]))
    results["Pipeline_Borda_R@10"].append(float(experiment_results["R@10"][4]))
    results["Pipeline_BordaWeighted_R@10"].append(float(experiment_results["R@10"][5]))

    results["BM25_nDCG@10"].append(float(experiment_results["nDCG@10"][0]))
    results["BM25_MonoT5_nDCG@10"].append(float(experiment_results["nDCG@10"][1]))
    results["Pipeline_Dedup_nDCG@10"].append(float(experiment_results["nDCG@10"][2]))
    results["Pipeline_DedupCutoff_nDCG@10"].append(
        float(experiment_results["nDCG@10"][3])
    )
    results["Pipeline_Borda_nDCG@10"].append(float(experiment_results["nDCG@10"][4]))
    results["Pipeline_BordaWeighted_nDCG@10"].append(
        float(experiment_results["nDCG@10"][5])
    )

    results["BM25_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][0]))
    results["BM25_MonoT5_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][1]))
    results["Pipeline_Dedup_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][2])
    )
    results["Pipeline_DedupCutoff_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][3])
    )
    results["Pipeline_Borda_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][4])
    )
    results["Pipeline_BordaWeighted_R@c"].append(
        float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][5])
    )

    bm25_r10 = results["BM25_R@10"][-1]
    bm25_monot5_r10 = results["BM25_MonoT5_R@10"][-1]
    dedup_r10 = results["Pipeline_Dedup_R@10"][-1]
    dedupCutoff = results["Pipeline_DedupCutoff_R@10"][-1]
    borda_r10 = results["Pipeline_Borda_R@10"][-1]
    borda_w_r10 = results["Pipeline_BordaWeighted_R@10"][-1]

    bm25_ndcg = results["BM25_nDCG@10"][-1]
    bm25_monot5_ndcg = results["BM25_monot5_nDCG@10"][-1]
    dedup_ndcg = results["Pipeline_Dedup_nDCG@10"][-1]
    dedupCutoff_ndcg = results["Pipeline_DedupCutoff_nDCG@10"][-1]
    borda_ndcg = results["Pipeline_Borda_nDCG@10"][-1]
    borda_w_ndcg = results["Pipeline_BordaWeighted_nDCG@10"][-1]

    bm25_rc = results["BM25_R@c"][-1]
    bm25_monot5_rc = results["BM25_MonoT5_R@c"][-1]
    dedup_rc = results["Pipeline_Dedup_R@c"][-1]
    dedupCutoff_rc = results["Pipeline_DedupCutoff_R@c"][-1]
    borda_rc = results["Pipeline_Borda_R@c"][-1]
    borda_w_rc = results["Pipeline_BordaWeighted_R@c"][-1]

    print("\n\n")

    print(
        f"n={N}: BM25 R@10={bm25_r10:.4f}, "
        f"BM25 + MonoT5 R@10={bm25_monot5_r10:.4f},"
        f"Dedup R@10={dedup_r10:.4f}, "
        f"DedupCutoff R@10={dedupCutoff:.4f}, "
        f"Borda R@10={borda_r10:.4f}, "
        f"BordaWeighted R@10={borda_w_r10:.4f}, "
        f"BM25 NDCG@10={bm25_ndcg:.4f}, "
        f"BM25 + MonoT5 NDCG@10={bm25_monot5_ndcg:.4f}, "
        f"Dedup NDCG@10={dedup_ndcg:.4f}, "
        f"DedupCutoff NDCG@10={dedupCutoff_ndcg:.4f}, "
        f"Borda NDCG@10={borda_ndcg:.4f}, "
        f"BordaWeighted NDCG@10={borda_w_ndcg:.4f}"
    )

results_df = pd.DataFrame(results)

save_experiment_results(
    results_df,
    experiment_name="experiment_2",
    output_dir=RESULTS_PATH,
    filename="experiment_2_results",
)

print(results)

plot_metrics(
    results_df,
    x_col="n",
    metric_cols=[
        "BM25_R@10",
        "BM25_MonoT5_R@10"
        "Pipeline_Dedup_R@10",
        "Pipeline_DedupCutoff_R@10",
        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    labels=[
        "BM25 Baseline",
        "BM25 + MonoT5"
        "Dedup Method",
        "Dedup Cutoff Method",
        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2: Document Combination Methods Comparison",
    xlabel="Number of Rewrites",
    filename="/n_rewrites_vs_R_comparison.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)

plot_metrics(
    results_df,
    x_col="n",
    metric_cols=[
        "BM25_nDCG@10",
        "BM25_MonoT5_nDCG@10"
        "Pipeline_Dedup_nDCG@10",
        "Pipeline_DedupCutoff_nDCG@10",
        "Pipeline_Borda_nDCG@10",
        "Pipeline_BordaWeighted_nDCG@10",
    ],
    labels=[
        "BM25 Baseline",
        "BM25+MonoT5",
        "Dedup Method",
        "Dedup Cutoff Method",
        "Borda Voting",
        "Borda Weighted",
    ],
    title="Experiment 2: Document Combination Methods Comparison (nDCG@10)",
    xlabel="Number of Rewrites",
    filename="/n_rewrites_vs_nDCG_comparison.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)

plot_with_regression(
    results_df,
    x_col="n",
    metric_cols=[
        "BM25_R@10",
        "BM25_MonoT5_R@10"
        "Pipeline_Dedup_R@10",
        "Pipeline_DedupCutoff_R@10",
        "Pipeline_Borda_R@10",
        "Pipeline_BordaWeighted_R@10",
    ],
    title="Experiment 2: R@10 vs Number of Rewrites (All Methods)",
    xlabel="Number of Rewrites",
    ylabel="R@10",
    filename="/n_rewrites_vs_R_regression_all_methods.png",
    use_log_scale=False,
    output_dir=RESULTS_PATH,
)
