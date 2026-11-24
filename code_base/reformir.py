"""
ReformIR — Adaptive Retrieval via Query Reformulation
======================================================
PyTerrier Transformer implementing the ReformIR algorithm from:

    Anand et al., "ReformIR: Reformulation-based Adaptive Retrieval"
    (SIGIR 2024, DC-92)

Algorithm overview:
    1. Retrieve initial results with BM25 for original query
    2. Retrieve additional results with BM25 for each LLM reformulation
    3. Build hybrid candidate pool with weighted combination
    4. Iteratively:
       a. Prioritize candidates using learned weights
       b. Score top batch with cross-encoder (within budget)
       c. Generate RM3 pseudo-relevance feedback from scored docs
       d. Learn combination weights via constrained least-squares (lsq_linear)
    5. Emit final ranked list

Clean rewrite of original paper code. Preserves algorithm exactly,
fixes result-assembly bugs, removes dead code, adds documentation.
"""

from statistics import mean
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyterrier as pt
import scipy.optimize
from sklearn.preprocessing import minmax_scale

import ir_datasets

logger = ir_datasets.log.easy()


class ReformIR(pt.Transformer):
    """Adaptive re-ranking via query reformulation and weight learning.

    Consumes initial BM25 results, augments them with BM25 results from
    LLM-generated reformulations, then iteratively scores candidates with
    a cross-encoder while learning optimal combination weights.

    Args:
        scorer: PyTerrier transformer that scores ``(qid, query, docno) → score``.
            Typically ``get_text >> reranker``.
        num_results: Total documents to return per query (budget *c*).
        cross_enc_budget: Maximum number of cross-encoder batch iterations.
        top_s: Number of top-scored documents used for RM3 expansion.
        batch_size: Documents scored per cross-encoder iteration.
            Defaults to ``scorer.batch_size`` or 16.
        backfill: If True, pad results with unscored candidates when
            the scored pool is exhausted before reaching *num_results*.
        verbose: Show progress bar over queries.
        terrier_index: PyTerrier index reference (needed for RM3 expansion).
        bm25_retriever: BM25 retriever for reformulated queries and RM3.
        file_name: Path to CSV with reformulated queries.
            Expected columns: ``qid``, ``query`` (rewrites joined by ``--``).
    """

    def __init__(
        self,
        scorer: pt.Transformer,
        num_results: int = 100,
        cross_enc_budget: int = 7,
        top_s: int = 15,
        batch_size: Optional[int] = None,
        backfill: bool = False,
        verbose: bool = True,
        terrier_index=None,
        bm25_retriever=None,
        file_name: str = "reformulations.csv",
        # Legacy param (accepted for API compat, ignored)
        llm: str = "gpt-4o-mini",
    ):
        self.scorer = scorer
        self.num_results = num_results
        self.cross_enc_budget = cross_enc_budget
        self.top_s = top_s
        self.backfill = backfill
        self.verbose = verbose
        self.terrier_index = terrier_index
        self.bm25_retriever = bm25_retriever
        self.reformulation_file = file_name

        if batch_size is None:
            batch_size = getattr(scorer, "batch_size", 16)
        self.batch_size = batch_size

    # ─── Helpers ──────────────────────────────────────────

    @staticmethod
    def _minmax_scale_df(results: pd.DataFrame) -> pd.DataFrame:
        """Min-max scale scores within each query group."""
        results = results.copy()
        results["score"] = results.groupby("qid")["score"].transform(
            lambda x: minmax_scale(x) if len(x) > 1 else x
        )
        return results

    def _generate_rm3_query(
        self,
        qid: str,
        query: str,
        top_scored: List[Tuple[str, float]],
    ) -> Tuple[List[str], List[float]]:
        """Expand query via RM3 using top-scored documents.

        Args:
            qid: Query ID.
            query: Original query text.
            top_scored: List of ``(docno, score)`` tuples for RM3 feedback.

        Returns:
            ``(doc_ids, minmax_scaled_scores)`` from RM3-expanded retrieval.
        """
        batch = pd.DataFrame(
            [[docno, score] for docno, score in top_scored],
            columns=["docno", "score"],
        )
        batch["qid"] = qid
        batch["query"] = query

        expander = pt.rewrite.RM3(
            self.terrier_index, fb_terms=10, fb_docs=5, fb_lambda=0.3
        )
        expanded = expander.transform(batch)
        results = self.bm25_retriever.transform(expanded)
        results = results.sort_values("score", ascending=False)

        documents = results["docno"].tolist()[:128]
        scores = results["score"].tolist()[:128]
        scaled = list(minmax_scale(np.array(scores))) if scores else []
        return documents, scaled

    def _estimate_scores(
        self,
        candidates: List[str],
        hybrid_scores: Dict[str, float],
        alpha: float,
        delta: float,
        rm3_lookup: Optional[Dict[str, float]],
    ) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
        """Compute estimated scores for candidate prioritization.

        Args:
            candidates: Document IDs to score.
            hybrid_scores: Pre-computed weighted BM25 combination.
            alpha: Weight for hybrid BM25 scores.
            delta: Weight for RM3 scores.
            rm3_lookup: RM3 feature scores (None before first iteration).

        Returns:
            ``(estimated_scores, rm3_features_dict)``.
        """
        bm25_vals = np.array([hybrid_scores.get(d, 0.0) for d in candidates])

        if rm3_lookup:
            rm3_vals = np.array([rm3_lookup.get(d, 0.0) for d in candidates])
            rm3_features = dict(zip(candidates, rm3_vals))
            estimated = alpha * bm25_vals + delta * rm3_vals
            return estimated, rm3_features

        estimated = alpha * bm25_vals
        return estimated, None

    # ─── Main Transform ──────────────────────────────────

    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Run ReformIR adaptive re-ranking on initial BM25 results.

        Args:
            input_df: Initial BM25 results with columns
                ``qid``, ``query``, ``docno``, ``score``.

        Returns:
            Re-ranked DataFrame with columns
                ``qid``, ``query``, ``docno``, ``rank``, ``score``, ``iteration``.
        """
        reformulations_df = pd.read_csv(self.reformulation_file)
        input_df = self._minmax_scale_df(input_df)

        query_groups = dict(iter(input_df.groupby("qid")))
        qids = list(query_groups.keys())

        if self.verbose:
            qids = logger.pbar(qids, desc="ReformIR re-ranking", unit="query")

        all_result_rows: List[dict] = []

        for qid in qids:
            query_df = query_groups[qid]
            query = query_df["query"].iloc[0]

            # ── Parse reformulated queries from CSV ──
            reform_row = reformulations_df.loc[
                reformulations_df["qid"].astype(str) == str(qid)
            ]
            if reform_row.empty:
                reformulated_queries: List[str] = []
            else:
                reformulated_queries = str(reform_row["query"].values[0]).split("--")

            n_reforms = len(reformulated_queries)

            # ── Retrieve BM25 results for each reformulation ──
            reform_groups: Dict[str, pd.DataFrame] = {}
            if reformulated_queries:
                reform_topics = pd.DataFrame({
                    "qid": [f"{qid}{i}" for i in range(n_reforms)],
                    "query": reformulated_queries,
                })
                reform_results = self.bm25_retriever.transform(reform_topics)
                reform_results = self._minmax_scale_df(reform_results)
                reform_groups = dict(iter(reform_results.groupby("qid")))

            # ── Build per-source score dicts ──
            # Deterministic initial weights: equal share per source
            init_lambda = 1.0 / (n_reforms + 1) if n_reforms > 0 else 1.0

            # source_lambdas[i] = weight for reformulation i
            # source_lambdas[-1] = weight for original query (appended last)
            source_lambdas = [init_lambda] * n_reforms
            source_lambdas.append(init_lambda)  # original query

            # per_source_scores[i] = {docno: raw_score} for reformulation i
            # per_source_scores[qid] = {docno: raw_score} for original query
            per_source_scores: Dict = {}
            per_source_scores[qid] = dict(
                zip(query_df["docno"].values, query_df["score"].values)
            )

            # Hybrid scores: weighted combination across all sources
            hybrid_scores: Dict[str, float] = {
                docno: source_lambdas[-1] * score
                for docno, score in zip(
                    query_df["docno"].values, query_df["score"].values
                )
            }

            all_doc_ids = set(query_df["docno"].values)

            for reform_idx in range(n_reforms):
                reform_qid = f"{qid}{reform_idx}"
                per_source_scores[reform_idx] = {}

                if reform_qid in reform_groups:
                    rg = reform_groups[reform_qid]
                    for docno, score in zip(rg["docno"].values, rg["score"].values):
                        per_source_scores[reform_idx][docno] = score
                        all_doc_ids.add(docno)
                        if docno not in hybrid_scores:
                            hybrid_scores[docno] = source_lambdas[reform_idx] * score
                        else:
                            hybrid_scores[docno] += source_lambdas[reform_idx] * score

            # ── Iterative scoring loop ──
            candidates = list(all_doc_ids)
            # scored: {docno: (unscaled_ce_score, iteration)} — for final output
            scored: Dict[str, Tuple[float, int]] = {}
            # ce_scaled: {docno: (scaled_ce_score, iteration)} — for RM3 feedback
            ce_scaled: Dict[str, Tuple[float, int]] = {}
            alpha = 0.2
            delta = 0.2
            iteration = 0
            num_batch = 0
            rm3_features: Optional[Dict[str, float]] = None

            while len(scored) < self.num_results:
                remaining = [c for c in candidates if c not in scored]
                if not remaining:
                    break

                # ── RM3 expansion from top-scored documents ──
                rm3_lookup: Optional[Dict[str, float]] = None
                if num_batch > 0 and scored:
                    top_scored_pairs = sorted(
                        scored.items(), key=lambda x: x[1][0], reverse=True
                    )[:self.top_s]
                    feedback_pairs = [
                        (docno, score_tuple[0])
                        for docno, score_tuple in top_scored_pairs
                    ]

                    rm3_docs, rm3_scores_list = self._generate_rm3_query(
                        qid, query, feedback_pairs
                    )
                    # Aggregate RM3 scores (handles potential duplicates)
                    rm3_accum: Dict[str, List[float]] = defaultdict(list)
                    for doc, s in zip(rm3_docs, rm3_scores_list):
                        rm3_accum[doc].append(s)
                    rm3_lookup = {
                        doc: mean(scores_list)
                        for doc, scores_list in rm3_accum.items()
                    }

                # ── Estimate priority scores ──
                estimated, rm3_features = self._estimate_scores(
                    remaining, hybrid_scores, alpha, delta, rm3_lookup
                )

                # ── Select top batch candidates ──
                size = min(self.batch_size, self.num_results - len(scored))
                ranked_pairs = sorted(
                    zip(remaining, estimated), key=lambda x: x[1], reverse=True
                )
                batch_docnos = [doc for doc, _ in ranked_pairs[:size]]

                # ── Build per-source feature columns for regression ──
                # Columns: [reform_0, reform_1, ..., reform_n, original]
                reform_feature_cols: List[np.ndarray] = []
                for reform_idx in range(n_reforms):
                    feat = np.array([
                        per_source_scores[reform_idx].get(d, 0.0)
                        for d in batch_docnos
                    ]).reshape(-1, 1)
                    reform_feature_cols.append(feat)
                # Original query BM25 as last feature column
                orig_feat = np.array([
                    per_source_scores[qid].get(d, 0.0)
                    for d in batch_docnos
                ]).reshape(-1, 1)
                reform_feature_cols.append(orig_feat)

                # ── Score with cross-encoder (if within budget) ──
                if num_batch < self.cross_enc_budget:
                    batch_df = pd.DataFrame({
                        "docno": batch_docnos,
                        "qid": qid,
                        "query": query,
                    })
                    reranked = self.scorer(batch_df)
                    unscaled_scores = list(reranked["score"].values)
                    reranked = self._minmax_scale_df(reranked)
                    target_scores = list(reranked["score"].values)

                    # Hybrid BM25 feature (weighted combination)
                    hybrid_feat = np.array([
                        hybrid_scores.get(d, 0.0) for d in batch_docnos
                    ]).reshape(-1, 1)

                    # ── Weight learning via constrained least-squares ──
                    if num_batch > 0 and rm3_features is not None:
                        # Features: [hybrid, rm3, reform_0, ..., reform_n, orig]
                        rm3_feat = np.array([
                            rm3_features.get(d, 0.0) for d in batch_docnos
                        ]).reshape(-1, 1)
                        features = np.concatenate(
                            [hybrid_feat, rm3_feat] + reform_feature_cols, axis=1
                        )
                        params = scipy.optimize.lsq_linear(
                            features, target_scores,
                            lsq_solver="exact", bounds=(0, 1),
                        )
                        alpha = params.x[0]
                        delta = params.x[1]
                        for i in range(len(source_lambdas)):
                            if 2 + i < len(params.x):
                                source_lambdas[i] = params.x[2 + i]
                    else:
                        # Features: [hybrid, reform_0, ..., reform_n, orig]
                        features = np.concatenate(
                            [hybrid_feat] + reform_feature_cols, axis=1
                        )
                        params = scipy.optimize.lsq_linear(
                            features, target_scores,
                            lsq_solver="exact", bounds=(0, 1),
                        )
                        alpha = params.x[0]
                        for i in range(len(source_lambdas)):
                            if 1 + i < len(params.x):
                                source_lambdas[i] = params.x[1 + i]

                    # Record scores
                    ce_scaled.update({
                        d: (s, iteration)
                        for d, s in zip(reranked["docno"], reranked["score"])
                    })
                    scored.update({
                        d: (s, iteration)
                        for d, s in zip(reranked["docno"], unscaled_scores)
                    })
                else:
                    # Budget exhausted: fill remaining with estimated scores
                    fill_count = self.num_results - len(scored)
                    fill_pairs = ranked_pairs[:fill_count]
                    ce_scaled.update({d: (s, iteration) for d, s in fill_pairs})
                    scored.update({d: (s, iteration) for d, s in fill_pairs})

                num_batch += 1
                iteration += 1

            # ── Assemble output rows for this query ──
            sorted_results = sorted(
                scored.items(), key=lambda x: x[1][0], reverse=True
            )

            # Backfill unscored items if pool exhausted early
            if self.backfill and len(sorted_results) < self.num_results:
                last_score = sorted_results[-1][1][0] if sorted_results else 0.0
                unscored = [d for d in candidates if d not in scored]
                for i, docno in enumerate(unscored):
                    if len(sorted_results) >= self.num_results:
                        break
                    sorted_results.append((docno, (last_score - 1 - i, -1)))

            for rank, (docno, (score, iter_num)) in enumerate(sorted_results):
                all_result_rows.append({
                    "qid": qid,
                    "query": query,
                    "docno": docno,
                    "rank": rank,
                    "score": score,
                    "iteration": iter_num,
                })

        return pd.DataFrame(all_result_rows)
