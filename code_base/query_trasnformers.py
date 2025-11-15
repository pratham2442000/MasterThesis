from collections import defaultdict

import numpy as np
import pandas as pd
import pyterrier as pt


def proportional_ranked_choice(rankings, k, method="borda", weight=None):
    """
    Select top k documents using proportional-ranked choice voting.
    """

    # Flatten rankings and get all unique documents
    all_docs = set()
    for ranking in rankings:
        all_docs.update(ranking)
    all_docs = list(all_docs)

    if method == "borda":
        return borda_count(rankings, k, all_docs, weight=weight)
    else:
        raise ValueError("Method must be 'borda'.")


def borda_count(rankings, k, all_docs, weight=None):
    """Borda count method - assign points based on rank positions."""
    if weight is None:
        def weight(x):
            return 1  # Uniform weight

    doc_scores = defaultdict(float)
    n_queries = len(rankings)

    for query_ranking in rankings:
        max_rank = len(query_ranking)
        for rank, doc in enumerate(query_ranking):
            # Borda score: higher ranks get more points
            score = (max_rank - rank) * weight(max_rank - rank)
            doc_scores[doc] += score

    # Normalize by number of queries
    for doc in doc_scores:
        doc_scores[doc] /= n_queries

    # Select top k documents
    top_docs = sorted(all_docs, key=lambda x: doc_scores[x], reverse=True)[:k]
    return top_docs


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


class BordaVotingTransformer(pt.Transformer):
    """
    Transformer that uses Borda count voting to combine document rankings
    from multiple query rewrites. Each rewrite's ranking is treated as a vote,
    and documents are scored based on their positions across all rewrites.
    """

    def __init__(self, cutoff: int, weight=None, method="borda"):
        """
        Args:
            cutoff: Number of top documents to return per query
            weight: Optional weight function for Borda scores (lambda function)
            method: Voting method (currently only 'borda' is supported)
        """
        self.cutoff = cutoff
        self.weight = weight
        self.method = method

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """
        Transform results by applying Borda voting per query.
        Expects results with columns: qid, docno, score
        """
        if res.empty:
            return res

        # Group by qid to process each query separately
        out_rows = []
        for qid, group in res.groupby("qid"):
            # Get rankings for each rewrite (identified by original qid structure)
            # Since MergeRewrites has already been applied, we need to reconstruct
            # the rankings from the sub-query IDs if they exist, or use score ordering

            # Sort by score to get ranking
            group_sorted = group.sort_values("score", ascending=False)

            # For Borda voting, we need multiple rankings. Since we've merged rewrites,
            # we'll use the score as a proxy for ranking position
            # However, we need to identify which documents came from which rewrite

            # If we have orig_qid column, we can group by that to get separate rankings
            if "orig_qid" in res.columns:
                rankings = []
                for orig_qid, rewrite_group in group.groupby("orig_qid"):
                    ranking = rewrite_group.sort_values("score", ascending=False)[
                        "docno"
                    ].tolist()
                    rankings.append(ranking)
            else:
                # Fallback: treat the entire group as one ranking
                # This won't work well for voting, but we'll use it as a fallback
                ranking = group_sorted["docno"].tolist()
                rankings = [ranking]

            # Apply Borda voting
            if len(rankings) > 0:
                top_docs = proportional_ranked_choice(
                    rankings, k=self.cutoff, method=self.method, weight=self.weight
                )

                # Create result rows with scores based on Borda ranking
                for rank, docno in enumerate(top_docs):
                    # Find the document in the original group to get its metadata
                    doc_row = group[group["docno"] == docno]
                    if not doc_row.empty:
                        # Use the max score from the document across all rewrites
                        max_score = doc_row["score"].max()
                        out_rows.append(
                            {
                                "qid": qid,
                                "docno": docno,
                                "score": max_score
                                         - rank * 0.001,  # Slight penalty for lower Borda rank
                                "rank": rank + 1,
                            }
                        )

        if not out_rows:
            return pd.DataFrame(columns=res.columns)

        result_df = pd.DataFrame(out_rows)
        return result_df


class BordaVotingTransformerV2(pt.Transformer):
    """
    Alternative implementation: Uses the sub-query structure to identify
    separate rankings from each rewrite before merging.
    This should be applied BEFORE MergeRewrites.
    """

    def __init__(self, cutoff: int, weight=None, method="borda"):
        """
        Args:
            cutoff: Number of top documents to return per query
            weight: Optional weight function for Borda scores (lambda function)
            method: Voting method (currently only 'borda' is supported)
        """
        self.cutoff = cutoff
        self.weight = weight
        self.method = method

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """
        Transform results by applying Borda voting per original query.
        Expects results with qid in format "original_qid_rewrite_index"
        """
        if res.empty:
            return res

        # Extract original qid and rewrite index
        res["orig_qid"] = res["qid"].apply(
            lambda x: x.split("_")[0] if "_" in str(x) else x
        )
        res["rewrite_idx"] = res["qid"].apply(
            lambda x: (
                int(x.split("_")[1]) if "_" in str(x) and len(x.split("_")) > 1 else 0
            )
        )

        out_rows = []
        for orig_qid, group in res.groupby("orig_qid"):
            # Get separate rankings for each rewrite
            rankings = []
            for rewrite_idx, rewrite_group in group.groupby("rewrite_idx"):
                ranking = rewrite_group.sort_values("score", ascending=False)[
                    "docno"
                ].tolist()
                rankings.append(ranking)

            # Apply Borda voting
            if len(rankings) > 0:
                top_docs = proportional_ranked_choice(
                    rankings, k=self.cutoff, method=self.method, weight=self.weight
                )

                # Create result rows
                for rank, docno in enumerate(top_docs):
                    # Get the best score for this document across all rewrites
                    doc_scores = group[group["docno"] == docno]["score"]
                    if not doc_scores.empty:
                        max_score = doc_scores.max()
                        out_rows.append(
                            {
                                "qid": orig_qid,
                                "docno": docno,
                                "score": max_score - rank * 0.001,
                                "rank": rank + 1,
                            }
                        )

        if not out_rows:
            return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

        result_df = pd.DataFrame(out_rows)
        return result_df


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
