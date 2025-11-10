import os

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R
from pyterrier_t5 import MonoT5ReRanker

import constants
from code_base.experiment_utils import (
    plot_metrics,
    save_experiment_results,
    preprocess_query
)
from code_base.llm_q import rewrite_single_query
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
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)


topics_df["query"] = topics_df["query"].apply(preprocess_query)
topics = topics_df[:TOPIC_LIMIT]

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
                rewrites = rewrite_single_query(self.dataset_name, str(qid), self.n)
            except (ValueError, FileNotFoundError, KeyError) as e:
                print(f"Warning: Could not load rewrites for qid {qid}: {e}. Using original query.")
                rewrites = [query] * self.n
            for i, rw in enumerate(rewrites):
                out_rows.append({
                    "qid": f"{qid}_{i}",  # unique sub-qid for each rewrite
                    "orig_qid": qid,  # store the original qid
                    "query": preprocess_query(rw)
                })
        return pd.DataFrame(out_rows)


class MergeRewrites(pt.Transformer):
    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        # restore original qid for further dedup/ranking
        res["qid"] = res["qid"].apply(lambda x: x.split("_")[0])
        return res


#
# Doc ranking merge strategy: keep only the highest-scoring instance of each (qid, docno) pair
#


class Dedup(pt.Transformer):
    def __init__(self, ):
        pass

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        # Keep max score per (qid, docno)
        print("count before dedup:")
        counts = res["qid"].value_counts()
        print(counts)

        res = res.sort_values(["qid", "docno", "score"], ascending=[True, True, False])
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")

        print("count after dedup:")
        counts = res["qid"].value_counts()
        print(counts)

        return res




class DedupAndCutoff(pt.Transformer):
    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        # Keep max score per (qid, docno)
        res = res.sort_values(["qid", "docno", "score"], ascending=[True, True, False])
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        # Then keep top-k per qid
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res["rank"] = res.groupby("qid").cumcount() + 1
        return res[res["rank"] <= self.cutoff]


# -----------------------------
# Experiment
# -----------------------------
print("Running experiment with query rewriting pipeline...")

results = {"n": N_REWRITES}
for i in EVAL_METRICS:
    results["BM25_" + str(i)] = []
    results["Pipeline_" + str(i)] = []

results["BM25_R@c"] = []
results["Pipeline_R@c"] = []

for N in N_REWRITES:
    rewrite = QueryRewriteTransformer(N, dataset_name=constants.DATASET_NAME)
    dedup_topk = DedupAndCutoff(RETRIEVAL_CUTOFF)
    dedup = Dedup()

    # Baseline BM25 on original queries
    bm25_base = bm25 % RETRIEVAL_CUTOFF

    expanded_pipeline = (
            rewrite
            >> bm25 % RETRIEVAL_CUTOFF
            >> MergeRewrites()
            >> dedup
            >> pt.text.get_text(index_ref, "text")
            >> mono_t5_reranker
    )

    experiment_results = pt.Experiment(
        [bm25_base, expanded_pipeline],
        topics,
        qrels_df,
        eval_metrics=EVAL_METRICS + [R @ RETRIEVAL_CUTOFF],
        names=["BM25", "Pipline"],
        verbose=True,
        batch_size=EXPERIMENT_BATCH_SIZE,
    )

    results["BM25_R@10"].append(float(experiment_results["R@10"][0]))
    results["Pipeline_R@10"].append(float(experiment_results["R@10"][1]))
    results["BM25_nDCG@10"].append(float(experiment_results["nDCG@10"][0]))
    results["Pipeline_nDCG@10"].append(float(experiment_results["nDCG@10"][1]))
    results["BM25_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][0]))
    results["Pipeline_R@c"].append(float(experiment_results[f"R@{RETRIEVAL_CUTOFF}"][1]))

    bm25_r10 = results["BM25_R@10"][-1]
    pipe_r10 = results["Pipeline_R@10"][-1]
    bm25_ndcg = results["BM25_nDCG@10"][-1]
    pipe_ndcg = results["Pipeline_nDCG@10"][-1]
    bm25_rc = results["BM25_R@c"][-1]
    pipe_rc = results["Pipeline_R@c"][-1]

    print(
        f"n={N}: BM25 R@10={bm25_r10:.4f}, "
        f"Pipeline R@10={pipe_r10:.4f}, "
        f"BM25 NDCG@10={bm25_ndcg:.4f}, "
        f"Pipeline NDCG@10={pipe_ndcg:.4f}, "
        f"BM25 R@{RETRIEVAL_CUTOFF}={bm25_rc:.4f}, "
        f"Pipeline R@{RETRIEVAL_CUTOFF}={pipe_rc:.4f}"
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
    metric_cols=["BM25_R@10", "Pipeline_R@10"],
    labels=["BM25 Baseline", "Expanded Pipeline with Rewrites"],
    title="Experiment 2: Query Rewriting Pipeline Performance",
    xlabel="Number of Queries",
    filename="/n_rewrites_vs_R.png",
    output_dir=RESULTS_PATH,
)
