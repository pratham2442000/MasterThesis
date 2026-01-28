"""
Test file for Experiment 2b transformers (Rerank -> Merge) using a Simulated Reranker.
Tests pipeline mechanics without loading heavy T5 models.
"""
import os
import sys

import numpy as np
import pyterrier as pt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants
from code_base.experiment_utils import (
    QueryRewriteTransformer,
    MergeRewrites,
    preprocess_query,
)
from code_base.query_trasnformers import (
    Dedup,
    BordaVotingTransformerV2,
)
from code_base.wiki_loader_pyt import wiki_loader_pyt

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = constants.DATA_PATH
DATA_SPLIT = constants.DATA_SPLIT
INDEX_PATH = constants.INDEX_PATH
RETRIEVAL_CUTOFF = 20
N_REWRITES = 3


# -----------------------------
# Simulated Reranker
# -----------------------------
class SimulatedReranker(pt.Transformer):
    """
    A lightweight transformer that simulates a reranker.
    It deterministically modifies scores based on docno length, ensuring
    rankings change in a predictable way for testing.
    """

    def transform(self, input_df):
        df = input_df.copy()

        # Logic:
        # 1. Take current score
        # 2. Add a boost based on the last digit of the docno (if numeric) or length
        #    This ensures documents get re-ordered.

        def mock_score(row):
            # Extract a pseudo-random but deterministic feature from docno
            # e.g. using the length or hash
            discriminator = (hash(row['docno']) % 100) / 10.0
            return row['score'] + discriminator

        df["score"] = df.apply(mock_score, axis=1)
        return df


# -----------------------------
# Setup
# -----------------------------
print("Loading data...")
docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, DATA_SPLIT)

# Preprocess and limit to 1 topic
topics_df["query"] = topics_df["query"].apply(preprocess_query)
test_topic = topics_df.head(1).copy()
original_qid = test_topic.iloc[0]["qid"]
print(f"Testing with Query ID: {original_qid}")

# Initialize Components
index_ref = pt.IndexFactory.of(INDEX_PATH)
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")

# SWAPPED: Use SimulatedReranker instead of MonoT5
print("Initializing Simulated Reranker...")
simulated_reranker = SimulatedReranker()

print("\n" + "=" * 60)
print("TESTING EXPERIMENT 2B PIPELINES (Rerank -> Merge)")
print("=" * 60 + "\n")

# -----------------------------
# Common Steps (Rewrite -> BM25)
# -----------------------------
print("--- Phase 1: Common Setup (Rewrite -> BM25) ---")
rewrite_trans = QueryRewriteTransformer(n=N_REWRITES)
rewrites = rewrite_trans.transform(test_topic)
print(f"Generated {len(rewrites)} rewrites.")

bm25_res = (bm25 % RETRIEVAL_CUTOFF).transform(rewrites)
print(f"BM25 Retrieved {len(bm25_res)} total documents.")
print(f"BM25 Score range: {bm25_res['score'].min():.4f} to {bm25_res['score'].max():.4f}")

# -----------------------------
# Test 1: The Reranking Step
# -----------------------------
print("\n--- Phase 2: Reranking Individual Rewrites (Simulated) ---")
print("Applying Simulated Reranker to ALL retrieved documents...")

# Note: We skip 'get_text' here as SimulatedReranker doesn't need it,
# but in the real pipeline 'get_text' is just a pass-through for T5.
reranked = simulated_reranker.transform(bm25_res)

print(f"Reranked {len(reranked)} documents.")
print(f"New Score range: {reranked['score'].min():.4f} to {reranked['score'].max():.4f}")

# Verification: Check if rankings changed
first_rewrite_qid = rewrites.iloc[0]['qid']
print(f"\nChecking Rank Changes for Sub-Query {first_rewrite_qid}:")

bm25_top = bm25_res[bm25_res['qid'] == first_rewrite_qid].sort_values('score', ascending=False).head(5)[
    'docno'].tolist()
reranked_top = reranked[reranked['qid'] == first_rewrite_qid].sort_values('score', ascending=False).head(5)[
    'docno'].tolist()

print(f"  BM25 Top 5:     {bm25_top}")
print(f"  Reranked Top 5: {reranked_top}")

if bm25_top != reranked_top:
    print("  [SUCCESS] Ranking has changed after simulation.")
else:
    print("  [WARNING] Ranking is identical. The simulated boost might be too small relative to BM25 gaps.")

# -----------------------------
# Test 2: Pipeline Dedup (Rerank -> Merge -> Dedup)
# -----------------------------
print("\n--- Phase 3: Pipeline Dedup (Merge -> Dedup) ---")
print("Merging reranked results and deduplicating...")

merged = MergeRewrites().transform(reranked)
# Dedup checks 'qid' column, so MergeRewrites ensures they all map back to original_qid
deduped = Dedup().transform(merged)
print(f"After Dedup: {len(deduped)} unique documents.")

# Verify we kept the MAX score
print("Verifying Dedup logic (max score retention)...")
doc_counts = reranked.groupby('docno').size()
multi_docs = doc_counts[doc_counts > 1]

if not multi_docs.empty:
    test_doc = multi_docs.index[0]
    original_scores = reranked[reranked['docno'] == test_doc]['score'].tolist()
    final_score = deduped[deduped['docno'] == test_doc]['score'].values[0]

    print(f"  Doc {test_doc} scores: {original_scores}")
    print(f"  Final Deduped score: {final_score:.4f}")

    # Use small epsilon for float comparison
    if abs(final_score - max(original_scores)) < 0.0001:
        print("  [SUCCESS] Dedup kept the maximum reranked score.")
    else:
        print(f"  [FAIL] Dedup score {final_score} != max {max(original_scores)}")
else:
    print("  [INFO] No overlapping documents found.")

# -----------------------------
# Test 3: Pipeline Borda (Rerank -> Borda)
# -----------------------------
print("\n--- Phase 4: Pipeline Borda (Voting on Reranked Lists) ---")
# Borda works on the 'rank' within each query group.
# Ensure ranks are re-calculated based on new scores before Borda seeing them?
# Yes, typically PyTerrier components might assume sorted input or calc rank on the fly.
# BordaVotingTransformerV2 usually recalculates ranks internally or assumes input is just a set of results.
# Let's see if our simulated reranker output needs explicit sorting for Borda.

borda = BordaVotingTransformerV2(cutoff=RETRIEVAL_CUTOFF, method="borda")
borda_res = borda.transform(reranked)

print(f"Borda Result Count: {len(borda_res)}")
print(f"Top 3 Borda Results:")
print(borda_res.sort_values('score', ascending=False).head(3)[['docno', 'score', 'rank']])

# -----------------------------
# Test 4: Pipeline Weighted Borda
# -----------------------------
print("\n--- Phase 5: Pipeline Weighted Borda ---")
borda_weighted = BordaVotingTransformerV2(
    cutoff=RETRIEVAL_CUTOFF,
    weight=lambda x: np.exp(x * 0.1),
    method="borda"
)
borda_w_res = borda_weighted.transform(reranked)

print(f"Weighted Borda Result Count: {len(borda_w_res)}")
print(f"Top 3 Weighted Borda Results:")
print(borda_w_res.sort_values('score', ascending=False).head(3)[['docno', 'score']])

# Compare rankings
top_dedup = deduped.sort_values('score', ascending=False).head(5)['docno'].tolist()
top_borda = borda_res.sort_values('score', ascending=False).head(5)['docno'].tolist()

print("\n--- Comparison ---")
print(f"Dedup Top 5: {top_dedup}")
print(f"Borda Top 5: {top_borda}")

overlap = len(set(top_dedup).intersection(set(top_borda)))
print(f"Overlap: {overlap}/5 documents.")

print("\n" + "=" * 60)
print("TEST EXPERIMENT 2B COMPLETE")
print("=" * 60 + "\n")
