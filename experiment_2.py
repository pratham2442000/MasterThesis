import pyterrier as pt
import pandas as pd
from datetime import datetime

from pyterrier_t5 import MonoT5ReRanker

from code_base.wiki_loader_pyt import wiki_loader_pyt
from code_base.llm_q import query_llm


if not pt.started():
    pt.init()


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = (
    "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/" "MasterThesis/data/2wikimultihopQA"
)
DATA_SPLIT = "test"

# Terrier index metadata file
INDEX_PATH = "./index/wiki_musique_index/data.properties"

# Experiment controls
TOPIC_LIMIT = 1000
N_REWRITES = 4  # number of rewritten queries per original query
RETRIEVAL_CUTOFF = 200  # docs per query (or per expanded query)
RECALL_C = 100  # additional recall@c metric; c is configurable

# Reranker
T5_BATCH_SIZE = 16
EXPERIMENT_BATCH_SIZE = 20


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
    def __init__(self, n: int):
        self.n = n

    def transform(self, topics_like: pd.DataFrame) -> pd.DataFrame:
        # Expect columns: ['qid', 'query']
        out_rows = []
        for _, row in topics_like.iterrows():
            qid = row["qid"]
            query = row["query"]
            try:
                rewrites = query_llm(query, self.n)
            except Exception:
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
rewrite = QueryRewriteTransformer(N_REWRITES)
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
eval_metrics = ["R@10", f"R@{RECALL_C}", "nDCG@10"]

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
