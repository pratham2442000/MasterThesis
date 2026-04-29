"""
Experiment 2 — Noise Analysis (Oracle Recall)
===============================================
Computes the "Oracle" Recall@10 for each N: the best single rewrite's
recall per query. Compares this upper bound against measured fusion
method results from Experiment 2.

Helps quantify how much recall the fusion methods lose to noise
introduced by poor rewrites.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import constants
from code_base.experiment_utils import get_text_getter, save_experiment_results
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


def load_comparison_results(results_path: str) -> pd.DataFrame:
    """Try to load Experiment 2 results CSV for overlay plotting.

    Args:
        results_path: Base results directory.

    Returns:
        DataFrame of experiment_2 results, or empty DataFrame.
    """
    candidates = [
        os.path.join(results_path, "experiment_2_results.csv"),
        os.path.join(results_path, "exp2", "experiment_2_results.csv"),
        os.path.join(constants.DATA_PATH, "experiment_2_results.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
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
    fig, ax = plt.subplots(figsize=(10, 6))

    oracle_col = next((c for c in df.columns if "Oracle_" in c), None)
    sys_cols = [c for c in df.columns if "_R@" in c and "Oracle" not in c]

    if oracle_col:
        ax.plot(
            df["n"], df[oracle_col],
            label="Oracle (Max Recall)",
            marker="*", linestyle="--", linewidth=2,
            color="black", markersize=8,
        )

    markers = ["o", "s", "^", "D", "v", "p"]
    for i, col in enumerate(sys_cols):
        label = col.split("_R@")[0].replace("Pipeline_", "")
        ax.plot(
            df["n"], df[col],
            label=label, marker=markers[i % len(markers)], alpha=0.7,
        )

    ax.set_title("Fusion Noise Analysis: Oracle vs Fusion Methods")
    ax.set_xlabel("Number of Rewrites (N)")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    filepath = os.path.join(output_dir, "noise_analysis_plot.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filepath}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run the noise (oracle recall) analysis."""
    # Reuse experiment_2 setup helpers

    results_path = os.path.join(constants.RESULTS_PATH, "exp2")
    os.makedirs(results_path, exist_ok=True)

    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)
    get_text = get_text_getter(index_ref)

    # Load data via experiment_2's runner-compatible path
    from code_base.experiment_runner import ExperimentRunner

    runner = ExperimentRunner(
        name="experiment_2_noise",
        results_subdir="exp2",
        topic_limit=constants.EXPERIMENT_2_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_2_EVAL_METRICS,
    )
    topics, qrels, _ = runner.load_data(
        filter_by_rewrites=True, use_tail=True,
    )

    exp2_df = load_comparison_results(results_path)

    all_results = []
    eval_cutoff = 10

    print(f"Starting Noise Analysis for N={constants.EXPERIMENT_2_N_REWRITES}")

    for n in constants.EXPERIMENT_2_N_REWRITES:
        print(f"\nProcessing N={n}...")

        rewriter = QueryRewriteTransformer(
            n, dataset_name=constants.DATASET_NAME
        )
        bm25 = (
                pt.terrier.Retriever(index_ref, wmodel="BM25")
                % constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
        )
        oracle_pipeline = rewriter >> bm25 >> get_text >> mono_t5
        oracle_res = oracle_pipeline.transform(topics)

        oracle_metrics = calculate_oracle_recall(
            oracle_res, qrels, cutoff=eval_cutoff
        )
        mean_oracle = oracle_metrics["oracle_recall"].mean()
        print(f"  Oracle R@{eval_cutoff}: {mean_oracle:.4f}")

        row = {"n": n, f"Oracle_R@{eval_cutoff}": mean_oracle}

        # Overlay experiment_2 results if available
        if not exp2_df.empty:
            n_col = next(
                (c for c in exp2_df.columns if c.lower() in ("n",)),
                None,
            )
            if n_col:
                match = exp2_df[exp2_df[n_col] == n]
                if not match.empty:
                    r_cols = [
                        c for c in exp2_df.columns
                        if f"R@{eval_cutoff}" in c
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
