from collections import defaultdict

import numpy as np
import pandas as pd
import pyterrier as pt


def borda_count(rankings, k, all_docs, weight=None):
    """
    Borda count method.
    Returns a dictionary of {docno: borda_score} and the sorted top_k list.
    """
    if weight is None:
        def weight(x): return 1

    doc_scores = defaultdict(float)
    n_queries = len(rankings)

    for query_ranking in rankings:
        # list_len is the number of docs in this specific ranking
        list_len = len(query_ranking)

        for rank, doc in enumerate(query_ranking):
            # 1. Standard Borda: points = number of items below this one
            # raw_score = (list_len - rank)

            # 2. Normalized Borda:
            # Normalized between 0 and 1 to prevent longer lists dominating results
            # Top doc gets 1.0, bottom doc gets ~0.0
            normalized_rank = (list_len - rank) / list_len

            # Apply weight
            score = normalized_rank * weight(normalized_rank)

            doc_scores[doc] += score

    # Normalize by number of queries (average score)
    for doc in doc_scores:
        doc_scores[doc] /= n_queries

    # Select top k documents
    top_docs = sorted(all_docs, key=lambda x: doc_scores[x], reverse=True)[:k]

    return top_docs, doc_scores


def proportional_ranked_choice(rankings, k, method="borda", weight=None):
    # Flatten rankings and get all unique documents
    all_docs = set()
    for ranking in rankings:
        all_docs.update(ranking)
    all_docs = list(all_docs)

    if method == "borda":
        return borda_count(rankings, k, all_docs, weight=weight)
    else:
        raise ValueError("Method must be 'borda'.")


class BordaVotingTransformerV2(pt.Transformer):
    """
    Applied BEFORE MergeRewrites.
    Calculates Borda scores based on rankings from different rewrites.
    """

    def __init__(self, cutoff: int, weight=None, method="borda"):
        self.cutoff = cutoff
        self.weight = weight
        self.method = method

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        if res.empty:
            return res

        # 1. Extract 'orig_qid' safely
        # We assume format: "123" or "123_0", "123_1"
        res["orig_qid"] = res["qid"].astype(str).apply(lambda x: x.split("_")[0])

        # 2. Identify rewrite index (useful for debugging, strictly not needed for grouping if we just use unique qid)
        # But strictly speaking, we group by orig_qid, then gather lists from the sub-qids

        out_rows = []

        # Group by the ORIGINAL query ID (e.g., "101")
        for orig_qid, group in res.groupby("orig_qid"):

            # Collect rankings from the sub-queries (e.g., "101_0", "101_1")
            rankings = []
            # We group by 'qid' here because 'qid' currently holds the sub-query IDs
            for sub_qid, sub_group in group.groupby("qid"):
                # Sort by BM25 score to get the ranking for this specific rewrite
                ranking = sub_group.sort_values("score", ascending=False)["docno"].tolist()
                rankings.append(ranking)

            if not rankings:
                continue

            # Apply Borda
            top_docs, final_borda_scores = proportional_ranked_choice(
                rankings, k=self.cutoff, method=self.method, weight=self.weight
            )

            # Create result rows using the BORDA SCORE
            for rank, docno in enumerate(top_docs):
                out_rows.append(
                    {
                        "qid": orig_qid,  # Restore original QID here
                        "docno": docno,
                        "score": final_borda_scores[docno],  # Use the calculated voting score
                        "rank": rank + 1,
                    }
                )

        result_df = pd.DataFrame(out_rows)
        return result_df

class Dedup(pt.Transformer):
    # Remove duplicate (qid, docno) pairs, keeping only the highest scoring one
    def __init__(
            self,
    ):
        pass

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        # Keep max score per (qid, docno)
        res = res.sort_values(["qid", "docno", "score"], ascending=[True, True, False])
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        res = res.sort_values(["qid", "score"], ascending=[True, False])
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


class CombMAXTransformer(pt.Transformer):
    """
    CombMAX identifies the maximum ranking score assigned to each document across all retrieval results.
    This method ensures that relevant documents, even if highly ranked by only one retrieval system,
    are not overlooked.
    """

    def __init__(self):
        pass

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        if res.empty:
            return res

        # Sort by score descending so the first occurrence has the max score
        res = res.sort_values(["qid", "docno", "score"], ascending=[True, True, False])
        # Keep the one with the highest score
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        # Re-sort by qid and score for final ranking
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        # Re-assign ranks
        res["rank"] = res.groupby("qid").cumcount() + 1
        return res



# Example usage
if __name__ == "__main__":
    # Example rankings from 3 queries
    rankings = [
        ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],  # Query 1 results
        ["doc2", "doc3", "doc1", "doc6", "doc7"],  # Query 2 results
        ["doc3", "doc1", "doc4", "doc2", "doc8"],
    ]  # Query 3 results

    k = 3  # Number of top documents to select
    selected_docs = proportional_ranked_choice(rankings, k, method="borda")
    selected_docs_w = proportional_ranked_choice(
        rankings, k, method="borda", weight=lambda x: np.e ** (x * 0.1)
    )
    print(selected_docs, selected_docs_w)
