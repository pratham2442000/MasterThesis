"""
Experiment 2 — Noise Analysis (Oracle & Dedup Oracle)
======================================================
Two oracle baselines for the rewrite-fusion experiments:

1. **Oracle**: For each query, picks the *single best rewrite* per metric.
   Upper-bound on what any selection strategy could achieve.

2. **Dedup Oracle**: Merges all sub-query results, deduplicates, and
   feeds the full (growing) document pool to the reranker — no cutoff.
   Shows what happens when the reranker sees *all* retrieved docs
   (document count grows with N rewrites).

Overlays pre-computed Experiment 2 fusion results for comparison.
"""

import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pyterrier as pt
import torch
from pyterrier.measures import R, nDCG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import (
    get_text_getter,
    save_experiment_results,
    get_reranker,
)
from code_base.transformers import (
    AttachOriginalQuery,
    Dedup,
    MergeRewrites,
    QueryRewriteTransformer,
)


# ─── Oracle Recall (best single rewrite) ─────────────────

def calculate_oracle_recall(
        res_df: pd.DataFrame,
        qrels_df: pd.DataFrame,
        cutoff: int = 10,
) -> pd.DataFrame:
    """Compute Oracle Recall@cutoff — best single rewrite per query.

    For each original query, selects the rewrite whose top-*cutoff*
    results achieve the highest recall, representing an upper bound.

    Args:
        res_df: Per-rewrite results with ``qid`` as ``"{base}_{idx}"``.
        qrels_df: Relevance judgments with base ``qid``.
        cutoff: Rank cutoff for recall computation.

    Returns:
        DataFrame with ``qid`` and ``oracle_recall`` columns.
    """
    res_df = res_df.copy()
    res_df["base_qid"] = (
        res_df["qid"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
    )

    qrels_map = (
        qrels_df[qrels_df["label"] > 0]
        .groupby("qid")["docno"]
        .apply(set)
        .to_dict()
    )
    qrels_map = {str(k): v for k, v in qrels_map.items()}

    rows = []
    for base_qid, group in res_df.groupby("base_qid"):
        relevant = qrels_map.get(base_qid, set())
        if not relevant:
            # No relevant docs for this topic — return NaN so pd.mean() skips
            # it, matching pt.Experiment behaviour (which also returns NaN).
            rows.append({"qid": base_qid, "oracle_recall": float("nan")})
            continue

        max_recall = 0.0
        for _, sub_group in group.groupby("qid"):
            top_k = sub_group.sort_values("rank").head(cutoff)
            hits = len(set(top_k["docno"]).intersection(relevant))
            max_recall = max(max_recall, hits / len(relevant))

        rows.append({"qid": base_qid, "oracle_recall": max_recall})

    return pd.DataFrame(rows)


# ─── Dedup Oracle (all sub-query docs merged → reranker) ──

def run_dedup_oracle(
        n,
        *,
        topics,
        qrels,
        index_ref,
        reranker,
        query_lookup,
        cutoff,
        eval_metrics,
):
    """Run the Dedup Oracle pipeline for *n* rewrites.

    Pipeline: rewrite → BM25 → MergeRewrites → AttachOriginalQuery
              → Dedup (no cutoff) → get_text → reranker

    The document pool grows with N since dedup keeps all unique docs
    across all sub-queries.

    Returns:
        Tuple of (metrics_dict, avg_docs_per_query).
        metrics_dict: ``{metric_name: value}`` averaged over queries.
        avg_docs_per_query: mean number of unique docs per query
            fed to the reranker after dedup.
    """
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
    attach = AttachOriginalQuery(query_lookup)
    get_text = get_text_getter(index_ref)

    # Run pre-reranker pipeline to count docs per query
    pre_reranker = rewrite >> bm25 >> MergeRewrites() >> attach >> Dedup()
    pre_reranker_df = pre_reranker.transform(topics)
    docs_per_query = pre_reranker_df.groupby("qid").size()
    avg_docs = float(docs_per_query.mean())
    print(f"    DedupOracle doc pool: min={docs_per_query.min()}, "
          f"avg={avg_docs:.1f}, max={docs_per_query.max()}")

    # Full pipeline with reranker
    dedup_oracle_pipeline = (
            rewrite >> bm25 >> MergeRewrites() >> attach
            >> Dedup() >> get_text >> reranker
    )

    result = pt.Experiment(
        [dedup_oracle_pipeline],
        topics,
        qrels,
        eval_metrics=eval_metrics,
        names=["DedupOracle"],
    )

    metrics = {}
    for col in result.columns:
        if col != "name":
            metrics[col] = float(result[col].iloc[0])
    return metrics, avg_docs


# ─── Find pre-computed exp2 results ───────────────────────

def _get_active_reranker_model() -> str:
    """Return the active reranker model name from constants."""
    if constants.RERANKER_TYPE == "cross_encoder":
        return constants.CROSS_ENCODER_MODEL_NAME
    elif constants.RERANKER_TYPE == "monot5":
        return constants.RERANKER_MODEL_NAME
    elif constants.RERANKER_TYPE == "qwen_reranker":
        return constants.QWEN_RERANKER_MODEL_NAME
    return ""


def find_latest_exp2_results() -> pd.DataFrame:
    """Find and load the latest Experiment 2 results CSV.

    Searches: RESULTS_PATH / DATASET_NAME / exp2 / run_* /

    Also validates that the loaded results used the **same reranker** as
    the current constants configuration.  Raises ``RuntimeError`` if there
    is a mismatch so results from different rerankers are never silently
    mixed together.

    Returns:
        DataFrame of experiment_2 results, or empty DataFrame if none found.
    """
    import json

    exp2_base = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME, "exp2"
    )
    pattern = os.path.join(exp2_base, "run_*", "experiment_2_results.csv")
    candidates = sorted(glob.glob(pattern), reverse=True)

    if not candidates:
        print("Warning: experiment_2 results CSV not found.")
        return pd.DataFrame()

    path = candidates[0]  # latest by timestamp
    run_dir = os.path.dirname(path)

    # ── Reranker consistency check ──────────────────────────
    meta_path = os.path.join(run_dir, "experiment_2_results_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        saved_type = meta.get("reranker_type", "")
        saved_model = meta.get("reranker_model_name", "")
        cur_type = constants.RERANKER_TYPE
        cur_model = _get_active_reranker_model()

        if saved_type != cur_type or saved_model != cur_model:
            raise RuntimeError(
                f"Reranker mismatch — exp2 results were produced with "
                f"'{saved_type}' / '{saved_model}' but current config is "
                f"'{cur_type}' / '{cur_model}'.\n"
                f"Either rerun Experiment 2 with the current reranker or "
                f"point to the correct results directory."
            )
        print(
            f"Reranker check passed: '{cur_type}' / '{cur_model}'"
        )
    else:
        print(
            f"Warning: no metadata file found at {meta_path}; "
            "skipping reranker consistency check."
        )

    print(f"Loaded experiment_2 results from {path}")
    return pd.read_csv(path)


# ─── Plotting ─────────────────────────────────────────────

def plot_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot Oracle + DedupOracle vs fusion method metrics.

    Args:
        df: Analysis results DataFrame.
        output_dir: Directory to save plots.
    """
    oracle_cols = [c for c in df.columns if c.startswith("Oracle_")]
    dedup_oracle_cols = [c for c in df.columns if c.startswith("DedupOracle_")]

    # Collect unique metric suffixes (e.g. "R@50", "R@10", "nDCG@10")
    metric_suffixes = set()
    for c in oracle_cols:
        metric_suffixes.add(c.split("_", 1)[1])
    for c in dedup_oracle_cols:
        metric_suffixes.add(c.split("_", 1)[1])

    for suffix in sorted(metric_suffixes):
        oracle_col = f"Oracle_{suffix}"
        dedup_col = f"DedupOracle_{suffix}"

        # Find matching fusion method columns
        matching_sys_cols = [
            c for c in df.columns
            if c.endswith(f"_{suffix}")
               and c != oracle_col
               and c != dedup_col
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Oracle line
        if oracle_col in df.columns:
            ax.plot(
                df["n"], df[oracle_col],
                label=f"Oracle ({suffix})",
                marker="*", linestyle="--", linewidth=2,
                color="black", markersize=8,
            )

        # DedupOracle line
        if dedup_col in df.columns:
            ax.plot(
                df["n"], df[dedup_col],
                label=f"Dedup Oracle ({suffix})",
                marker="X", linestyle="-.", linewidth=2,
                color="red", markersize=8,
            )
            # Annotate with avg doc count per query
            if "DedupOracle_avg_docs" in df.columns:
                for _, pt_row in df.iterrows():
                    ax.annotate(
                        f"{pt_row['DedupOracle_avg_docs']:.0f}d",
                        (pt_row["n"], pt_row[dedup_col]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=7, color="red", ha="center",
                    )

        # Rename BM25_MonoT5 → BM25_MiniLM to reflect actual reranker used
        _LABEL_MAP = {
            "BM25_MonoT5": "BM25_MiniLM",
        }
        markers = ["o", "s", "^", "D", "v", "p", "h", "<"]
        for i, col in enumerate(matching_sys_cols):
            label = col.rsplit(f"_{suffix}", 1)[0].replace("Pipeline_", "")
            label = _LABEL_MAP.get(label, label)
            ax.plot(
                df["n"], df[col],
                label=label, marker=markers[i % len(markers)], alpha=0.7,
            )

        ax.set_title(f"Noise Analysis: Oracle vs Fusion Methods ({suffix})")
        ax.set_xlabel("Number of Rewrites (N)")
        ax.set_ylabel(suffix)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        safe_suffix = suffix.replace("@", "at")
        filepath = os.path.join(output_dir, f"noise_analysis_{safe_suffix}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {filepath}")
        plt.close(fig)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run the noise (oracle) analysis."""

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())

    # Use same metrics as experiment_2
    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
    eval_metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff]

    # Build timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        constants.RESULTS_PATH,
        constants.DATASET_NAME,
        "exp2_noise",
        f"run_{timestamp}",
    )
    os.makedirs(results_path, exist_ok=True)

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
    reranker = get_reranker()
    get_text = get_text_getter(index_ref)

    # Load data via ExperimentRunner
    runner = ExperimentRunner(
        name="experiment_2_noise",
        results_subdir="exp2_noise",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=eval_metrics,
    )
    topics, qrels, query_lookup = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    # Load pre-computed experiment_2 results
    exp2_df = find_latest_exp2_results()

    # Determine recall cutoffs for oracle recall computation
    recall_cutoffs = []
    for m in eval_metrics:
        m_str = str(m)
        if m_str.startswith("R@"):
            val = int(m_str.split("@")[1])
            if val not in recall_cutoffs:
                recall_cutoffs.append(val)

    if not recall_cutoffs:
        recall_cutoffs = [50]  # fallback

    all_results = []

    print(f"Starting Noise Analysis for N={constants.EXPERIMENT_2_N_REWRITES}")
    print(f"Oracle recall cutoffs: {recall_cutoffs}")
    print(f"Eval metrics: {eval_metrics}")

    for n in constants.EXPERIMENT_2_N_REWRITES:
        print(f"\nProcessing N={n}...")

        # ── Oracle: best single rewrite per query ──
        rewriter = QueryRewriteTransformer(
            n, dataset_name=constants.DATASET_NAME
        )
        bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % cutoff
        attach = AttachOriginalQuery(query_lookup)
        # NOTE: attach original query BEFORE get_text+reranker so the reranker
        # scores with the original query text, not the rewrite text.
        oracle_pipeline = rewriter >> bm25 >> attach >> get_text >> reranker
        oracle_res = oracle_pipeline.transform(topics)

        row = {"n": n}

        # Oracle recall for each cutoff
        for rc in recall_cutoffs:
            oracle_metrics = calculate_oracle_recall(
                oracle_res, qrels, cutoff=rc
            )
            mean_oracle = oracle_metrics["oracle_recall"].mean()
            row[f"Oracle_R@{rc}"] = mean_oracle
            print(f"  Oracle R@{rc}: {mean_oracle:.4f}")

        # Oracle nDCG@10 — best single rewrite's nDCG
        oracle_res_copy = oracle_res.copy()
        oracle_res_copy["base_qid"] = (
            oracle_res_copy["qid"].astype(str).apply(lambda x: x.rsplit("_", 1)[0])
        )
        best_ndcg = 0.0
        n_queries_with_qrels = 0
        # Get unique sub-query ids grouped by base_qid
        for base_qid, group in oracle_res_copy.groupby("base_qid"):
            sub_qids = group["qid"].unique()
            max_ndcg_for_query = float("nan")
            for sq in sub_qids:
                sub_res = group[group["qid"] == sq].copy()
                sub_res["qid"] = base_qid  # restore original qid for eval
                try:
                    result = pt.Experiment(
                        [pt.Transformer.from_df(sub_res)],
                        topics[topics["qid"] == base_qid],
                        qrels,
                        eval_metrics=[nDCG @ 10],
                        names=["tmp"],
                    )
                    val = float(result["nDCG@10"].iloc[0])
                    if not pd.isna(val):
                        if pd.isna(max_ndcg_for_query):
                            max_ndcg_for_query = val
                        else:
                            max_ndcg_for_query = max(max_ndcg_for_query, val)
                except Exception:
                    pass
            if not pd.isna(max_ndcg_for_query):
                best_ndcg += max_ndcg_for_query
                n_queries_with_qrels += 1
        if n_queries_with_qrels > 0:
            row["Oracle_nDCG@10"] = best_ndcg / n_queries_with_qrels
            print(f"  Oracle nDCG@10: {row['Oracle_nDCG@10']:.4f}")

        # ── Dedup Oracle: all docs merged, deduped, no cutoff → reranker ──
        print(f"  Running DedupOracle N={n}...")
        dedup_metrics, avg_docs = run_dedup_oracle(
            n,
            topics=topics,
            qrels=qrels,
            index_ref=index_ref,
            reranker=reranker,
            query_lookup=query_lookup,
            cutoff=cutoff,
            eval_metrics=eval_metrics,
        )
        row["DedupOracle_avg_docs"] = avg_docs
        for metric_name, metric_val in dedup_metrics.items():
            row[f"DedupOracle_{metric_name}"] = metric_val
            print(f"  DedupOracle {metric_name}: {metric_val:.4f}")

        # ── Overlay experiment_2 fusion results ──
        if not exp2_df.empty:
            match = exp2_df[exp2_df["n"] == n]
            if not match.empty:
                r_cols = [
                    c for c in exp2_df.columns
                    if "_R@" in c or "_nDCG@" in c
                ]
                agg = match[r_cols].mean(numeric_only=True)
                for col in r_cols:
                    row[col] = float(agg[col])

        all_results.append(row)

    results_df = pd.DataFrame(all_results)
    print("\nAnalysis Complete.")
    print(results_df)

    save_experiment_results(
        results_df,
        experiment_name="experiment_2_noise_analysis",
        output_dir=results_path,
        filename="experiment_2_noise_analysis_results",
    )

    plot_comparison(results_df, results_path)


if __name__ == "__main__":
    main()
