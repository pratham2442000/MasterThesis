"""
Experiment 2 — Noise Analysis (Oracle Recall)
===============================================
Computes the "Oracle" recall for each N: the best single rewrite's
recall per query. Loads pre-computed Experiment 2 results and overlays
fusion method metrics for comparison.

Helps quantify how much recall the fusion methods lose to noise
introduced by poor rewrites.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_utils import get_text_getter, save_experiment_results, get_reranker
from code_base.transformers import QueryRewriteTransformer


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
        res_df["qid"].astype(str).apply(lambda x: x.split("_")[0])
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
            rows.append({"qid": base_qid, "oracle_recall": 0.0})
            continue

        max_recall = 0.0
        for _, sub_group in group.groupby("qid"):
            top_k = sub_group.sort_values("rank").head(cutoff)
            hits = len(set(top_k["docno"]).intersection(relevant))
            max_recall = max(max_recall, hits / len(relevant))

        rows.append({"qid": base_qid, "oracle_recall": max_recall})

    return pd.DataFrame(rows)


def find_latest_exp2_results() -> pd.DataFrame:
    """Find and load the latest Experiment 2 results CSV.

    Searches: RESULTS_PATH / DATASET_NAME / exp2 / run_* /

    Returns:
        DataFrame of experiment_2 results, or empty DataFrame.
    """
    exp2_base = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME, "exp2"
    )
    pattern = os.path.join(exp2_base, "run_*", "experiment_2_results.csv")
    candidates = sorted(glob.glob(pattern), reverse=True)

    if candidates:
        path = candidates[0]  # latest by timestamp
        print(f"Loaded experiment_2 results from {path}")
        return pd.read_csv(path)

    print("Warning: experiment_2 results CSV not found.")
    return pd.DataFrame()


def plot_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot Oracle vs fusion method recalls.

    Args:
        df: Analysis results DataFrame.
        output_dir: Directory to save the plot.
    """

    oracle_cols = [c for c in df.columns if "Oracle_" in c]

    for oracle_col in oracle_cols:
        # Extract cutoff from oracle col (e.g. "Oracle_R@50" -> "R@50")
        recall_suffix = oracle_col.split("_", 1)[1]  # e.g. "R@50"

        # Only plot sys cols with same recall cutoff
        matching_sys_cols = [
            c for c in df.columns
            if c.endswith(f"_{recall_suffix}") and "Oracle" not in c
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            df["n"], df[oracle_col],
            label=f"Oracle ({recall_suffix})",
            marker="*", linestyle="--", linewidth=2,
            color="black", markersize=8,
        )

        markers = ["o", "s", "^", "D", "v", "p"]
        for i, col in enumerate(matching_sys_cols):
            label = col.rsplit(f"_{recall_suffix}", 1)[0].replace("Pipeline_", "")
            ax.plot(
                df["n"], df[col],
                label=label, marker=markers[i % len(markers)], alpha=0.7,
            )

        ax.set_title(f"Fusion Noise Analysis: Oracle vs Fusion Methods ({oracle_col.split('_', 1)[1]})")
        ax.set_xlabel("Number of Rewrites (N)")
        ax.set_ylabel(f"{oracle_col.split('_', 1)[1]}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()

        filepath = os.path.join(output_dir, f"noise_analysis_plot_{oracle_col.split('_', 1)[1]}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {filepath}")
        plt.close(fig)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run the noise (oracle recall) analysis."""
    from datetime import datetime
    from code_base.experiment_runner import ExperimentRunner

    # Use same metrics as experiment_2
    cutoff = constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
    eval_metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ cutoff]

    # Build timestamped output dir matching ExperimentRunner pattern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        constants.RESULTS_PATH,
        constants.DATASET_NAME,
        "exp2_noise",
        f"run_{timestamp}",
    )
    os.makedirs(results_path, exist_ok=True)

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    reranker = get_reranker()
    get_text = get_text_getter(index_ref)

    # Load data via ExperimentRunner
    runner = ExperimentRunner(
        name="experiment_2_noise",
        results_subdir="exp2_noise",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=eval_metrics,
    )
    topics, qrels, _ = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    # Load pre-computed experiment_2 results (no recalculation)
    exp2_df = find_latest_exp2_results()

    # Determine recall cutoffs to compute oracle for
    # Match all R@X metrics from experiment_2
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

    for n in constants.EXPERIMENT_2_N_REWRITES:
        print(f"\nProcessing N={n}...")

        rewriter = QueryRewriteTransformer(
            n, dataset_name=constants.DATASET_NAME
        )
        bm25 = (
                pt.terrier.Retriever(index_ref, wmodel="BM25")
                % cutoff
        )
        oracle_pipeline = rewriter >> bm25 >> get_text >> reranker
        oracle_res = oracle_pipeline.transform(topics)

        row = {"n": n}

        # Compute oracle recall for each cutoff
        for rc in recall_cutoffs:
            oracle_metrics = calculate_oracle_recall(
                oracle_res, qrels, cutoff=rc
            )
            mean_oracle = oracle_metrics["oracle_recall"].mean()
            row[f"Oracle_R@{rc}"] = mean_oracle
            print(f"  Oracle R@{rc}: {mean_oracle:.4f}")

        # Overlay experiment_2 recall columns if available
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
