"""
PyTerrier Transformers
======================
All custom PyTerrier Transformer classes for the experiment pipelines.

Includes:
- Query preprocessing and rewriting (QueryRewriteTransformer)
- Result merging (MergeRewrites, AttachOriginalQuery)
- Deduplication (Dedup, DedupAndCutoff)
- Fusion methods (BordaVotingTransformerV2, CombMAXTransformer, RRFTransformer)
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pyterrier as pt

import constants
from code_base.llm_q import rewrite_single_query


# ─────────────────────────────────────────────
# Query Preprocessing
# ─────────────────────────────────────────────

def preprocess_query(query: str) -> str:
    """Strip non-alphanumeric characters and lowercase a query string.

    Args:
        query: Raw query text.

    Returns:
        Cleaned, lowercased query.
    """
    query = "".join(e for e in query if e.isalnum() or e.isspace())
    return query.lower()


# ─────────────────────────────────────────────
# Query Rewriting Transformer
# ─────────────────────────────────────────────

class QueryRewriteTransformer(pt.Transformer):
    """Reads preprocessed LLM query rewrites and expands each topic into *n* sub-queries.

    Each original ``qid`` is expanded to ``{qid}_0, {qid}_1, …, {qid}_{n-1}``.
    The first sub-query (``_0``) is always the original query text.

    Args:
        n: Total number of queries per topic (1 original + n-1 rewrites).
        dataset_name: Dataset identifier used to locate the rewrite CSV.
            Defaults to ``constants.DATASET_NAME``.
    """

    def __init__(self, n: int, dataset_name: Optional[str] = None):
        self.n = n
        self.dataset_name = dataset_name or constants.DATASET_NAME

    def transform(self, topics_like: pd.DataFrame) -> pd.DataFrame:
        """Expand topics into rewritten sub-queries.

        Args:
            topics_like: DataFrame with at least ``qid`` and ``query`` columns.

        Returns:
            DataFrame with columns ``qid`` (sub-query id), ``orig_qid``,
            and ``query`` (preprocessed rewrite text).
        """
        if topics_like.empty:
            return pd.DataFrame(columns=["qid", "orig_qid", "query"])

        out_rows: list[dict] = []
        for _, row in topics_like.iterrows():
            qid = row["qid"]
            query = row["query"]
            try:
                rewrites = [query] + rewrite_single_query(
                    self.dataset_name, str(qid), self.n - 1
                )
            except (ValueError, FileNotFoundError, KeyError) as exc:
                print(
                    f"Warning: Could not load rewrites for qid {qid}: "
                    f"{exc}. Using original query."
                )
                rewrites = [query] * self.n

            for i, rw in enumerate(rewrites):
                out_rows.append({
                    "qid": f"{qid}_{i}",
                    "orig_qid": qid,
                    "query": preprocess_query(rw),
                })

        return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────
# Merge / Attach Helpers
# ─────────────────────────────────────────────

class MergeRewrites(pt.Transformer):
    """Collapse sub-query IDs back to original query IDs.

    Strips the ``_{index}`` suffix from ``qid`` so that downstream
    transformers see a single query pool per original topic.
    """

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Merge sub-query results by restoring original qid.

        Args:
            res: Result DataFrame with sub-query ``qid`` values
                (e.g. ``"101_0"``, ``"101_1"``).

        Returns:
            Same DataFrame with ``qid`` reset to the base ID.
        """
        res = res.copy()
        res["qid"] = res["qid"].astype(str).apply(lambda x: x.split("_")[0])
        return res


class AttachOriginalQuery(pt.Transformer):
    """Re-attach the original topic query text to result rows.

    MonoT5 and other cross-encoders need the *original* query, not the
    rewrite, in the ``query`` column.

    Args:
        query_lookup: Mapping ``{qid: query_text}`` built from topics.
    """

    def __init__(self, query_lookup: dict):
        self.query_lookup = {str(k): v for k, v in query_lookup.items()}

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Map qid → original query text.

        Falls back to ``orig_qid`` if ``qid`` is not found (pre-merge state).

        Args:
            res: Result DataFrame.

        Returns:
            DataFrame with ``query`` column replaced by original text.

        Raises:
            KeyError: If any qid cannot be resolved.
        """
        if res.empty:
            return res

        res = res.copy()
        mapped = res["qid"].astype(str).map(self.query_lookup)

        if mapped.isna().any() and "orig_qid" in res.columns:
            fallback = (
                res.loc[mapped.isna(), "orig_qid"]
                .astype(str)
                .map(self.query_lookup)
            )
            mapped.fillna(fallback, inplace=True)

        if mapped.isna().any():
            missing = res.loc[mapped.isna(), "qid"].unique()
            raise KeyError(
                f"Could not attach original queries for qids: {missing}. "
                "Ensure topics include these qids."
            )

        res["query"] = mapped
        return res


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

class Dedup(pt.Transformer):
    """Remove duplicate ``(qid, docno)`` pairs, keeping only the highest score."""

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate by keeping max-score per (qid, docno).

        Args:
            res: Result DataFrame.

        Returns:
            Deduplicated DataFrame sorted by ``(qid, score desc)``.
        """
        res = res.sort_values(
            ["qid", "docno", "score"], ascending=[True, True, False]
        )
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        return res


class DedupAndCutoff(pt.Transformer):
    """Deduplicate and keep only top-*cutoff* results per query.

    Args:
        cutoff: Maximum number of documents to retain per qid.
    """

    def __init__(self, cutoff: int):
        self.cutoff = cutoff

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate, then retain top-k documents per query.

        Args:
            res: Result DataFrame.

        Returns:
            Truncated DataFrame with ``rank`` column added.
        """
        res = res.sort_values(
            ["qid", "docno", "score"], ascending=[True, True, False]
        )
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res["rank"] = res.groupby("qid").cumcount() + 1
        return res[res["rank"] <= self.cutoff]


# ─────────────────────────────────────────────
# Borda Count Fusion
# ─────────────────────────────────────────────

def borda_count(
        rankings: List[List[str]],
        k: int,
        all_docs: List[str],
        weight: Optional[Callable[[float], float]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Compute normalized Borda-count scores across multiple rankings.

    Args:
        rankings: List of ranked document lists (one per sub-query).
        k: Number of top documents to return.
        all_docs: Universe of unique document IDs.
        weight: Optional weighting function applied to normalized rank.

    Returns:
        Tuple of (top-k document list, {docno: score} dict).
    """
    if weight is None:
        def weight(x):
            return 1

    doc_scores: Dict[str, float] = defaultdict(float)
    n_queries = len(rankings)

    for query_ranking in rankings:
        list_len = len(query_ranking)
        for rank, doc in enumerate(query_ranking):
            normalized_rank = (list_len - rank) / list_len
            score = normalized_rank * weight(normalized_rank)
            doc_scores[doc] += score

    for doc in doc_scores:
        doc_scores[doc] /= n_queries

    top_docs = sorted(all_docs, key=lambda x: doc_scores[x], reverse=True)[:k]
    return top_docs, doc_scores


def proportional_ranked_choice(
        rankings: List[List[str]],
        k: int,
        method: str = "borda",
        weight: Optional[Callable[[float], float]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Aggregate multiple rankings using a voting method.

    Args:
        rankings: List of ranked document lists.
        k: Number of top documents to select.
        method: Voting method (currently only ``"borda"``).
        weight: Optional weighting function for Borda.

    Returns:
        Tuple of (top-k doc list, score dict).

    Raises:
        ValueError: If method is not ``"borda"``.
    """
    all_docs = list({doc for ranking in rankings for doc in ranking})

    if method == "borda":
        return borda_count(rankings, k, all_docs, weight=weight)
    raise ValueError("Method must be 'borda'.")


class BordaVotingTransformerV2(pt.Transformer):
    """Fuse per-rewrite rankings using Borda voting *before* MergeRewrites.

    Groups results by original query ID (parsed from ``qid``), collects
    per-sub-query rankings, and produces a single fused ranking per query.

    Args:
        cutoff: Number of documents to keep per query after fusion.
        weight: Optional weighting function for Borda scores.
        method: Voting method (default ``"borda"``).
    """

    def __init__(
            self,
            cutoff: int,
            weight: Optional[Callable] = None,
            method: str = "borda",
    ):
        self.cutoff = cutoff
        self.weight = weight
        self.method = method

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Apply Borda voting across sub-query results.

        Args:
            res: Result DataFrame with sub-query ``qid`` values.

        Returns:
            Fused DataFrame with original ``qid``, ``docno``, ``score``, ``rank``.
        """
        if res.empty:
            return res

        res = res.copy()
        res["orig_qid"] = res["qid"].astype(str).apply(lambda x: x.split("_")[0])

        out_rows: list[dict] = []

        for orig_qid, group in res.groupby("orig_qid"):
            rankings = []
            for _, sub_group in group.groupby("qid"):
                ranking = (
                    sub_group.sort_values("score", ascending=False)["docno"]
                    .tolist()
                )
                rankings.append(ranking)

            if not rankings:
                continue

            top_docs, final_scores = proportional_ranked_choice(
                rankings, k=self.cutoff, method=self.method, weight=self.weight
            )

            for rank, docno in enumerate(top_docs):
                out_rows.append({
                    "qid": orig_qid,
                    "docno": docno,
                    "score": final_scores[docno],
                    "rank": rank + 1,
                })

        return pd.DataFrame(out_rows)


# ─────────────────────────────────────────────
# CombMAX Fusion
# ─────────────────────────────────────────────

class CombMAXTransformer(pt.Transformer):
    """Keep only the maximum score for each ``(qid, docno)`` pair.

    Ensures relevant documents highly ranked by *any* retrieval variant
    are not lost during fusion.
    """

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        """Take max score per (qid, docno) and re-rank.

        Args:
            res: Result DataFrame (post MergeRewrites).

        Returns:
            Deduplicated, re-ranked DataFrame.
        """
        if res.empty:
            return res

        res = res.sort_values(
            ["qid", "docno", "score"], ascending=[True, True, False]
        )
        res = res.drop_duplicates(subset=["qid", "docno"], keep="first")
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res["rank"] = res.groupby("qid").cumcount() + 1
        return res


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion (RRF)
# ─────────────────────────────────────────────

class RRFTransformer(pt.Transformer):
    def __init__(self, k: int = 60):
        self.k = k

    def transform(self, res: pd.DataFrame) -> pd.DataFrame:
        if res.empty:
            return res

        res = res.copy()

        # 1. Calculate ranks within each specific rewrite (qid)
        # We MUST do this before merging IDs.
        res = res.sort_values(["qid", "score"], ascending=[True, False])
        res["_rank"] = res.groupby("qid").cumcount() + 1

        # 2. Calculate the RRF weight
        res["_rrf_weight"] = 1.0 / (self.k + res["_rank"])

        # 3. Identify the grouping ID (prefer orig_qid if QueryRewriteTransformer was used)
        group_col = "orig_qid" if "orig_qid" in res.columns else "qid"

        # 4. Sum scores (Vectorized - much faster than iterrows)
        fused = res.groupby([group_col, "docno"], as_index=False).agg({
            "_rrf_weight": "sum",
            "query": "first"  # Keep the query text for downstream rerankers
        }).rename(columns={"_rrf_weight": "score", group_col: "qid"})

        # 5. Final Sort and Rank
        fused = fused.sort_values(["qid", "score"], ascending=[True, False])
        fused["rank"] = fused.groupby("qid").cumcount() + 1

        return fused[["qid", "docno", "score", "rank", "query"]]
