"""
Test file for Experiment 2 transformers.
Tests document counts at each stage of the pipeline for different methods.
"""
import numpy as np
import pandas as pd
import pyterrier as pt

import constants
from code_base.experiment_utils import preprocess_query
from code_base.llm_q import rewrite_single_query
from code_base.query_trasnformers import (
    Dedup,
    DedupAndCutoff,
    BordaVotingTransformerV2,
)
from code_base.wiki_loader_pyt import wiki_loader_pyt

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = constants.DATA_PATH
DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH
RETRIEVAL_CUTOFF = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF


# -----------------------------
# Transformers (same as in experiment_2.py)
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
                    f"Warning: Could not load rewrites for qid {qid}: {e}. "
                    "Using original query."
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


# -----------------------------
# Setup
# -----------------------------
print("Loading data...")
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)

topics_df["query"] = topics_df["query"].apply(preprocess_query)

index_ref = pt.IndexFactory.of(INDEX_PATH)
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")

# -----------------------------
# Test Transformers and count documents reaching each stage
# -----------------------------
print("\n" + "=" * 60)
print("TESTING TRANSFORMERS - Document Counts at Each Stage")
print("=" * 60 + "\n")

# Test 1: Dedup Method (Original)
print("--- Test 1: Dedup Method ---")
rewrite = QueryRewriteTransformer(n=5).transform(topics_df[:2])
print(f"Number of rewrites: {len(rewrite)}")
bm25_results = (bm25 % RETRIEVAL_CUTOFF).transform(rewrite)
print(f"After BM25 % {RETRIEVAL_CUTOFF}: {len(bm25_results)}")
merged = MergeRewrites().transform(bm25_results)
print(f"After merging: {len(merged)}")
deduped = Dedup().transform(merged)
print(f"After Deduplication: {len(deduped)}")
print(f"Unique queries in deduped: {deduped['qid'].nunique()}")
print(f"Unique documents in deduped: {deduped['docno'].nunique()}\n")

# Test 2: DedupAndCutoff Method
print("--- Test 2: DedupAndCutoff Method ---")
rewrite2 = QueryRewriteTransformer(n=5).transform(topics_df[:1])
print(f"Number of rewrites: {len(rewrite2)}")
bm25_results2 = (bm25 % RETRIEVAL_CUTOFF).transform(rewrite2)
print(f"After BM25 % {RETRIEVAL_CUTOFF}: {len(bm25_results2)}")
merged2 = MergeRewrites().transform(bm25_results2)
print(f"After merging: {len(merged2)}")
deduped_cutoff = DedupAndCutoff(RETRIEVAL_CUTOFF).transform(merged2)
print(f"After Deduplication and Cutoff: {len(deduped_cutoff)}")
print(f"Unique queries in deduped_cutoff: " f"{deduped_cutoff['qid'].nunique()}")
print(f"Unique documents in deduped_cutoff: " f"{deduped_cutoff['docno'].nunique()}")
max_rank_dedup_cutoff = deduped_cutoff.groupby("qid")["rank"].max().tolist()
print(f"Max rank per query: {max_rank_dedup_cutoff}\n")

# Test 3: Borda Voting Method (unweighted)
print("--- Test 3: Borda Voting Method (unweighted) ---")
print("\n[Step 1] Creating query rewrites...")
test_topic = topics_df[:1].copy()
original_qid = test_topic.iloc[0]["qid"]
original_query = test_topic.iloc[0]["query"]
print(f"Original Query ID: {original_qid}")
print(f"Original Query: {original_query}")
rewrite3 = QueryRewriteTransformer(n=5).transform(test_topic)
print(f"Number of rewrites created: {len(rewrite3)}")
print("\n[Step 2] Showing all rewrites with sub-query IDs:")
for idx, row in rewrite3.iterrows():
    print(f"  Sub-query ID: {row['qid']} | Query: {row['query']}")

print("\n[Step 3] Retrieving documents for each rewrite...")
bm25_results3 = (bm25 % RETRIEVAL_CUTOFF).transform(rewrite3)
print(f"Total documents retrieved: {len(bm25_results3)}")

print("\n[Step 4] Documents retrieved per rewrite:")
all_docs_by_rewrite = {}
for sub_qid in sorted(rewrite3["qid"].unique()):
    rewrite_docs = bm25_results3[bm25_results3["qid"] == sub_qid]
    all_docs_by_rewrite[sub_qid] = rewrite_docs
    print(f"  {sub_qid}: {len(rewrite_docs)} documents")
    # Show top 3 documents for each rewrite
    top_docs = (
        rewrite_docs.sort_values("score", ascending=False)
        .head(3)[["docno", "score"]]
        .values.tolist()
    )
    for i, (docno, score) in enumerate(top_docs, 1):
        print(f"    Rank {i}: docno={docno[:20]}... | score={score:.4f}")

# Analyze document overlap across rewrites
print("\n[Step 4.1] Analyzing document overlap across rewrites:")
all_unique_docs = set()
doc_rewrite_counts = {}
for sub_qid, rewrite_docs in all_docs_by_rewrite.items():
    docs_in_rewrite = set(rewrite_docs["docno"].unique())
    all_unique_docs.update(docs_in_rewrite)
    for docno in docs_in_rewrite:
        if docno not in doc_rewrite_counts:
            doc_rewrite_counts[docno] = []
        doc_rewrite_counts[docno].append(sub_qid)

print(f"  Total unique documents across all rewrites: " f"{len(all_unique_docs)}")
print("  Documents appearing in multiple rewrites:")
docs_in_multiple = {
    doc: rewrites for doc, rewrites in doc_rewrite_counts.items() if len(rewrites) > 1
}
print(f"    {len(docs_in_multiple)} documents appear in 2+ rewrites")
# Show top documents by frequency
doc_freq = sorted(
    [(doc, len(rewrites)) for doc, rewrites in doc_rewrite_counts.items()],
    key=lambda x: x[1],
    reverse=True,
)
print("  Top 5 most frequent documents (appearing in most rewrites):")
for docno, count in doc_freq[:5]:
    print(f"    {docno[:30]}... appears in {count}/{len(rewrite3)} rewrites")

print("\n[Step 5] Applying Borda Voting (unweighted)...")
borda_voting = BordaVotingTransformerV2(
    cutoff=RETRIEVAL_CUTOFF, weight=None, method="borda"
)
borda_results = borda_voting.transform(bm25_results3)
print(f"After Borda Voting: {len(borda_results)} documents")
print(f"Unique queries in result: {borda_results['qid'].nunique()}")
print(f"Unique documents in result: {borda_results['docno'].nunique()}")

print("\n[Step 6] Final Borda Voting ranking:")
borda_results_sorted = borda_results.sort_values(
    ["qid", "rank"], ascending=[True, True]
)
for idx, row in borda_results_sorted.head(10).iterrows():
    print(
        f"  Rank {row['rank']}: docno={row['docno'][:30]}... | "
        f"score={row['score']:.4f}"
    )
if len(borda_results_sorted) > 10:
    print(f"  ... and {len(borda_results_sorted) - 10} more documents")

max_rank_borda = borda_results.groupby("qid")["rank"].max().tolist()
print(f"\nMax rank per query: {max_rank_borda}\n")

# Test 4: Borda Voting Method (weighted)
# Use the same query as Test 3 for fair comparison
print("--- Test 4: Borda Voting Method (weighted) ---")
print("\n[Step 1] Creating query rewrites...")
print(f"Original Query ID: {original_qid}")
print(f"Original Query: {original_query}")
print("  (Using same query as Test 3 for comparison)")
rewrite4 = QueryRewriteTransformer(n=5).transform(test_topic)
print(f"Number of rewrites created: {len(rewrite4)}")

print("\n[Step 2] Retrieving documents for each rewrite...")
bm25_results4 = (bm25 % RETRIEVAL_CUTOFF).transform(rewrite4)
print(f"Total documents retrieved: {len(bm25_results4)}")

print("\n[Step 3] Documents retrieved per rewrite:")
for sub_qid in sorted(rewrite4["qid"].unique()):
    rewrite_docs = bm25_results4[bm25_results4["qid"] == sub_qid]
    print(f"  {sub_qid}: {len(rewrite_docs)} documents")

print("\n[Step 4] Applying Borda Voting with exponential weighting...")
print("  Weight function: exp(rank * 0.1)")
print("  This gives higher weight to documents ranked higher in each rewrite")
borda_voting_weighted = BordaVotingTransformerV2(
    cutoff=RETRIEVAL_CUTOFF,
    weight=lambda x: np.exp(x * 0.1),  # Exponential weighting
    method="borda",
)
borda_weighted_results = borda_voting_weighted.transform(bm25_results4)
print(f"After Borda Voting (weighted): " f"{len(borda_weighted_results)} documents")
unique_q_borda_w = borda_weighted_results["qid"].nunique()
print(f"Unique queries in result: {unique_q_borda_w}")
unique_docs_borda_w = borda_weighted_results["docno"].nunique()
print(f"Unique documents in result: {unique_docs_borda_w}")

print("\n[Step 5] Final Borda Voting ranking (weighted):")
borda_weighted_sorted = borda_weighted_results.sort_values(
    ["qid", "rank"], ascending=[True, True]
)
for idx, row in borda_weighted_sorted.head(10).iterrows():
    print(
        f"  Rank {row['rank']}: docno={row['docno'][:30]}... | "
        f"score={row['score']:.4f}"
    )
if len(borda_weighted_sorted) > 10:
    print(f"  ... and {len(borda_weighted_sorted) - 10} more documents")

max_rank_borda_w = borda_weighted_results.groupby("qid")["rank"].max().tolist()
print(f"\nMax rank per query: {max_rank_borda_w}")

# Comparison between weighted and unweighted
print("\n[Step 6] Comparing weighted vs unweighted Borda results:")
print(f"  Query ID: {original_qid}")
print(f"  Query: {original_query}")
print("  ✓ Both tests use the same query (comparison is valid)")

print("\n  Unweighted Borda top 5:")
for idx, row in borda_results_sorted.head(5).iterrows():
    print(f"    Rank {row['rank']}: {row['docno'][:30]}...")

print("\n  Weighted Borda top 5:")
for idx, row in borda_weighted_sorted.head(5).iterrows():
    print(f"    Rank {row['rank']}: {row['docno'][:30]}...")

# Check if rankings differ
unweighted_top5 = set(borda_results_sorted.head(5)["docno"])
weighted_top5 = set(borda_weighted_sorted.head(5)["docno"])
common_docs = unweighted_top5.intersection(weighted_top5)
print(f"\n  Common documents in top 5: {len(common_docs)}/{5}")
if len(common_docs) < 5:
    print("  -> Weighting affects the ranking!")
    print("  -> Documents unique to weighted top 5:")
    weighted_only = weighted_top5 - unweighted_top5
    for doc in list(weighted_only)[:3]:
        print(f"      {doc[:30]}...")
    if len(unweighted_top5 - weighted_top5) > 0:
        print("  -> Documents unique to unweighted top 5:")
        unweighted_only = unweighted_top5 - weighted_top5
        for doc in list(unweighted_only)[:3]:
            print(f"      {doc[:30]}...")
else:
    print("  -> Both methods agree on top 5")
print()

print("=" * 60)
print("TESTING COMPLETE")
print("=" * 60 + "\n")
