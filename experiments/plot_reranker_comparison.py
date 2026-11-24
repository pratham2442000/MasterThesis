"""
Plot Reranker Comparison — All methods across cutoff values
============================================================
Combines results from exp1 (BM25 + Reranker), exp4 (ORE Adaptive),
and exp7 (ReformIR) into unified comparison plots.

Generates per-dataset:
  1. Recall@10 vs cutoff (c)
  2. nDCG@10 vs cutoff (c)

Note: R@50 is not available in these experiment results.
"""

import os
import sys
import glob

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_ROOT, "reranker_comparison_plots")

# Per-dataset mapping: experiment → (subdir_glob, column_map)
#   column_map: {display_name: csv_column_name}

# These are the systems we want to plot.
# BM25 baseline comes from exp1 (most complete sweep).
SYSTEM_STYLES = {
    "BM25":          {"color": "#6c757d", "marker": "o", "linestyle": "-",  "zorder": 1},
    "MiniLM-L6":     {"color": "#0d6efd", "marker": "s", "linestyle": "-",  "zorder": 2},
    "ORE Adaptive":  {"color": "#dc3545", "marker": "^", "linestyle": "-",  "zorder": 3},
    "ReformIR":      {"color": "#198754", "marker": "D", "linestyle": "-",  "zorder": 4},
}


def _find_latest_csv(dataset_dir, exp_subdir):
    """Find most recent results CSV in dataset/exp_subdir/run_*/."""
    pattern = os.path.join(dataset_dir, exp_subdir, "run_*", f"{exp_subdir.replace('/', '_')}*.csv")
    # Also try simpler pattern
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(dataset_dir, exp_subdir, "run_*", "*.csv")
        candidates = [f for f in glob.glob(pattern2) if "metadata" not in f]
    if not candidates:
        # Try non-run directory (old format)
        pattern3 = os.path.join(dataset_dir, exp_subdir, "*.csv")
        candidates = [f for f in glob.glob(pattern3) if "metadata" not in f]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _extract_series(df, c_col, metric_col):
    """Extract (c_values, metric_values) dropping NaNs."""
    mask = df[metric_col].notna()
    return df.loc[mask, c_col].values, df.loc[mask, metric_col].values


def load_dataset_data(dataset_dir):
    """Load and normalize data from exp1, exp4, exp7 for one dataset.

    Returns dict: {system_name: {metric: (c_array, values_array)}}
    """
    systems = {}

    # ── Experiment 1: BM25 + Reranker ──
    exp1_csv = _find_latest_csv(dataset_dir, "exp1")
    if exp1_csv:
        print(f"  exp1: {exp1_csv}")
        df1 = pd.read_csv(exp1_csv)

        # BM25 baseline
        if "BM25_R@10" in df1.columns:
            systems["BM25"] = {
                "R@10":    _extract_series(df1, "c", "BM25_R@10"),
                "nDCG@10": _extract_series(df1, "c", "BM25_nDCG@10"),
            }

        # Reranker — column could be "Pipeline_*" or "BM25_MiniLM-L6_*"
        for prefix in ["Pipeline", "BM25_MiniLM-L6", "BM25_MonoT5"]:
            r10_col = f"{prefix}_R@10"
            ndcg_col = f"{prefix}_nDCG@10"
            if r10_col in df1.columns:
                systems["MiniLM-L6"] = {
                    "R@10":    _extract_series(df1, "c", r10_col),
                    "nDCG@10": _extract_series(df1, "c", ndcg_col),
                }
                break
    else:
        print("  exp1: NOT FOUND")

    # ── Experiment 4: ORE Adaptive ──
    exp4_csv = _find_latest_csv(dataset_dir, "exp4")
    if exp4_csv:
        print(f"  exp4: {exp4_csv}")
        df4 = pd.read_csv(exp4_csv)
        if "ORE_Adaptive_R@10" in df4.columns:
            systems["ORE Adaptive"] = {
                "R@10":    _extract_series(df4, "c", "ORE_Adaptive_R@10"),
                "nDCG@10": _extract_series(df4, "c", "ORE_Adaptive_nDCG@10"),
            }
    else:
        print("  exp4: NOT FOUND")

    # ── Experiment 7: ReformIR ──
    exp7_csv = _find_latest_csv(dataset_dir, "exp7")
    if exp7_csv:
        print(f"  exp7: {exp7_csv}")
        df7 = pd.read_csv(exp7_csv)
        if "ReformIR_R@10" in df7.columns:
            systems["ReformIR"] = {
                "R@10":    _extract_series(df7, "c", "ReformIR_R@10"),
                "nDCG@10": _extract_series(df7, "c", "ReformIR_nDCG@10"),
            }
    else:
        print("  exp7: NOT FOUND")

    return systems


# ─── Pretty Plot Style ────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def make_comparison_plot(systems, metric, dataset_name, output_dir):
    """Create one comparison plot for a given metric."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sys_name, style in SYSTEM_STYLES.items():
        if sys_name not in systems:
            continue
        data = systems[sys_name]
        if metric not in data:
            continue
        c_vals, metric_vals = data[metric]
        if len(c_vals) == 0:
            continue

        ax.plot(
            c_vals, metric_vals,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=7,
            label=sys_name,
            alpha=0.85,
            zorder=style["zorder"],
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Collect all c values for tick marks
    all_c = set()
    for sys_name in systems:
        if metric in systems[sys_name]:
            all_c.update(systems[sys_name][metric][0])
    if all_c:
        ticks = sorted(all_c)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], rotation=45, fontsize=9)

    pretty_dataset = {
        "wiki_musique": "2WikiMultihopQA",
        "msmarco-passage": "MS MARCO Passage",
        "dpr-w100": "DPR-W100 (TriviaQA)",
    }.get(dataset_name, dataset_name)

    pretty_metric = {
        "R@10": "Recall@10",
        "nDCG@10": "nDCG@10",
    }.get(metric, metric)

    ax.set_title(f"{pretty_metric} vs. BM25 Cutoff — {pretty_dataset}", fontweight="bold")
    ax.set_xlabel("Initial BM25 Cutoff (c)")
    ax.set_ylabel(pretty_metric)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(framealpha=0.9, edgecolor="#ccc")
    fig.tight_layout()

    safe_metric = metric.replace("@", "_at_")
    fname = f"{dataset_name}_{safe_metric}_comparison.png"
    out_path = os.path.join(output_dir, fname)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    datasets = [d for d in os.listdir(RESULTS_ROOT)
                if os.path.isdir(os.path.join(RESULTS_ROOT, d))
                and d not in ("reranker_comparison_plots", "exp2_vs_2b")]

    metrics = ["R@10", "nDCG@10"]

    for ds in sorted(datasets):
        ds_dir = os.path.join(RESULTS_ROOT, ds)
        print(f"\n{'═' * 50}")
        print(f"Dataset: {ds}")
        print(f"{'═' * 50}")

        systems = load_dataset_data(ds_dir)
        if not systems:
            print("  No data found, skipping.")
            continue

        print(f"  Systems found: {list(systems.keys())}")

        for metric in metrics:
            make_comparison_plot(systems, metric, ds, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("\nNOTE: R@50 not available in experiment results.")
    print("      Only R@10 and nDCG@10 are computed.")


if __name__ == "__main__":
    main()
