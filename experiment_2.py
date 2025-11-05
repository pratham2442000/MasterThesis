import pyterrier as pt
import pandas as pd
from datetime import datetime

from pyterrier_t5 import MonoT5ReRanker

import constants
from code_base.wiki_loader_pyt import wiki_loader_pyt
from code_base.llm_q import rewrite_single_query


if not pt.started():
    pt.init()


# -----------------------------
# Configuration (from constants.py)
# -----------------------------
DATA_PATH = constants.DATA_PATH
DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH

# Experiment controls
TOPIC_LIMIT = constants.EXPERIMENT_2_TOPIC_LIMIT
N_REWRITES = constants.EXPERIMENT_2_N_REWRITES
RETRIEVAL_CUTOFF = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
RECALL_C = constants.EXPERIMENT_2_RECALL_C

# Reranker
T5_BATCH_SIZE = constants.T5_BATCH_SIZE
EXPERIMENT_BATCH_SIZE = constants.EXPERIMENT_BATCH_SIZE


# -----------------------------
# Data
# -----------------------------
print("Loading data...")
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)


def preprocess_query(query: str) -> str:
    query = "".join(e for e in query if e.isalnum() or e.isspace())
    return query.lower()


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
    Transformer that reads preprocessed query rewrites from dataset file.
    Uses rewrite_single_query to get k rewrites for each query ID.
    """

    def __init__(self, n: int, dataset_name: str = None):
        self.n = n
        self.dataset_name = (
            dataset_name if dataset_name is not None else constants.DATASET_NAME
        )

    def transform(self, topics_like: pd.DataFrame) -> pd.DataFrame:
        # Expect columns: ['qid', 'query']
        out_rows = []
        for _, row in topics_like.iterrows():
            qid = row["qid"]
            query = row["query"]
            try:
                # Read preprocessed rewrites from dataset file
                rewrites = rewrite_single_query(self.dataset_name, qid, self.n)
            except (ValueError, FileNotFoundError, KeyError) as e:
                # Fallback to original query if rewrite not found
                print(
                    f"Warning: Could not load rewrites for qid {qid}: {e}. Using original query."
                )
                rewrites = [query] * self.n
            for rw in rewrites:
                out_rows.append({"qid": qid, "query": rw})
        return pd.DataFrame(out_rows)


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
# Pipelines
# -----------------------------
rewrite = QueryRewriteTransformer(N_REWRITES, dataset_name=constants.DATASET_NAME)
dedup_topk = DedupAndCutoff(RETRIEVAL_CUTOFF)

# Baseline BM25 on original queries
bm25_base = bm25 % RETRIEVAL_CUTOFF

# Expanded pipeline: LLM rewrites -> retrieve -> dedup -> rerank
expanded_pipeline = (
    rewrite
    >> bm25
    >> dedup_topk
    >> pt.text.get_text(index_ref, "text")
    >> mono_t5_reranker
)


# -----------------------------
# Experiment
# -----------------------------
eval_metrics = constants.EXPERIMENT_2_EVAL_METRICS

print("Running experiment with query rewriting pipeline...")
results = pt.Experiment(
    [bm25_base, expanded_pipeline],
    topics,
    qrels_df,
    eval_metrics=eval_metrics,
    names=["BM25", f"Expand({N_REWRITES})+BM25+MonoT5"],
    verbose=True,
    batch_size=EXPERIMENT_BATCH_SIZE,
)

print("\nResults:\n", results)


# -----------------------------
# Persist results
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = (
    f"{DATA_PATH}/experiment_2_results_{timestamp}_n{N_REWRITES}_c"
    f"{RETRIEVAL_CUTOFF}.csv"
)
results.to_csv(out_path, index=False)
print(f"Saved results to: {out_path}")
