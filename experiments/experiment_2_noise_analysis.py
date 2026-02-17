import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker

# Add parent directory to path to allow importing from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants

# Import experiment_2 to reuse setup and data loading functions
try:
    from experiments import experiment_2
except ImportError:
    import experiment_2

from code_base.experiment_utils import (
    QueryRewriteTransformer,
    save_experiment_results
)


def calculate_oracle_recall(res_df, qrels_df, cutoff=10):
    """
    Calculates the 'Oracle' Recall@cutoff for each original query.
    The Oracle selects the best performing rewrite for each query.
    
    Args:
        res_df: DataFrame with results from individual rewrites (qid must be 'baseqid_rewriteidx').
        qrels_df: DataFrame with qrels (qid is 'baseqid').
        cutoff: Rank cutoff for Recall calculation.
        
    Returns:
        DataFrame with ['qid', 'oracle_recall']
    """
    res_df = res_df.copy()
    # Parse base_qid from "{qid}_{idx}"
    res_df['base_qid'] = res_df['qid'].astype(str).apply(lambda x: x.split('_')[0])

    # Pre-calculate relevant docs per qid for fast lookup
    # key: str(qid), value: set of relevant docnos
    qrels_map = qrels_df[qrels_df['label'] > 0].groupby('qid')['docno'].apply(set).to_dict()
    qrels_map = {str(k): v for k, v in qrels_map.items()}

    results = []

    # Iterate over each base query
    for base_qid, group in res_df.groupby('base_qid'):
        if base_qid not in qrels_map:
            continue

        relevant_docs = qrels_map[base_qid]
        num_rel = len(relevant_docs)

        if num_rel == 0:
            results.append({'qid': base_qid, 'oracle_recall': 0.0})
            continue

        # Iterate over each rewrite for this query to find the max recall
        max_recall = 0.0

        for _, sub_group in group.groupby('qid'):
            # Get top K docs
            top_k = sub_group.sort_values("rank").head(cutoff)
            retrieved_docs = set(top_k['docno'])

            # Calculate Recall
            rel_ret = len(retrieved_docs.intersection(relevant_docs))
            recall = rel_ret / num_rel

            if recall > max_recall:
                max_recall = recall

        results.append({'qid': base_qid, 'oracle_recall': max_recall})

    return pd.DataFrame(results)


def load_comparison_results(results_path):
    """
    Attempts to load results from experiment_2 to compare against Oracle.
    """
    candidates = [
        os.path.join(results_path, 'experiment_2_results.csv'),
        os.path.join(results_path, 'exp2', 'experiment_2_results.csv'),
        os.path.join(constants.DATA_PATH, 'experiment_2_results.csv')
    ]

    for p in candidates:
        if os.path.exists(p):
            print(f"Loaded experiment_2 results from {p}")
            return pd.read_csv(p)

    print("Warning: experiment_2 results CSV not found. Fusion metrics will be skipped.")
    return pd.DataFrame()


def main():
    # 1. Setup
    if not pt.started():
        pt.init()

    results_path = experiment_2.setup_directories()
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)

    # Initialize Reranker
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)
    get_text = pt.text.get_text(index_ref, "text")

    # 2. Data
    topics, qrels, _ = experiment_2.load_data()

    # Load comparison data (Experiment 2 results) once
    exp2_df = load_comparison_results(results_path)

    # 3. Analysis Loop
    all_results = []
    n_rewrites_list = constants.EXPERIMENT_2_N_REWRITES

    # Determine evaluation cutoff from metric name (assuming R@10)
    # Defaulting to 10 if not explicitly parseable, but consistent with EXPERIMENT_2_EVAL_METRICS
    eval_cutoff = 10

    print(f"Starting Noise Analysis for N={n_rewrites_list}...")

    for n in n_rewrites_list:
        print(f"\nProcessing N={n}...")

        # --- A. Oracle Calculation (Max Individual Recall) ---
        print("  Calculating Oracle Baseline...")

        # Construct Oracle Pipeline: Rewrite -> BM25 -> GetText -> Rerank
        rewrite_trans = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
        bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

        oracle_pipeline = (rewrite_trans
                           >> bm25
                           >> get_text
                           >> mono_t5)

        # Run transform to get raw results
        oracle_res = oracle_pipeline.transform(topics)

        # Calculate Oracle Recall
        oracle_metrics = calculate_oracle_recall(oracle_res, qrels, cutoff=eval_cutoff)
        mean_oracle_recall = oracle_metrics['oracle_recall'].mean()
        print(f"  > Oracle R@{eval_cutoff}: {mean_oracle_recall:.4f}")

        # --- B. Match with Experiment 2 Results ---
        result_row = {'n': n, f'Oracle_R@{eval_cutoff}': mean_oracle_recall}

        if not exp2_df.empty:
            # Find column for 'n'
            n_col = next((c for c in exp2_df.columns if c.lower() in ('n', 'n_rewrites', 'nr')), None)

            if n_col:
                # Filter for current n
                matching = exp2_df[exp2_df[n_col] == n]

                if not matching.empty:
                    # Identify R@10 columns
                    metric_suffix = f"R@{eval_cutoff}"
                    metric_cols = [c for c in exp2_df.columns if metric_suffix in c]

                    # Average over runs if necessary and extract values
                    agg = matching[metric_cols].mean(numeric_only=True)

                    for col in metric_cols:
                        result_row[col] = float(agg[col])

        all_results.append(result_row)

    # 4. Save and Visualize
    results_df = pd.DataFrame(all_results)
    print("\nAnalysis Complete.")
    print(results_df)

    save_experiment_results(
        results_df,
        experiment_name="experiment_2_noise_analysis",
        output_dir=results_path,
        filename="experiment_2_noise_analysis_results"
    )

    # 5. Plotting
    plot_comparison(results_df, results_path)


def plot_comparison(df, output_dir):
    """
    Generates the comparison plot: N vs Recall for Oracle and Fusion methods.
    """
    plt.figure(figsize=(10, 6))

    # Identify Oracle column and System columns
    oracle_col = next((c for c in df.columns if "Oracle_" in c), None)
    sys_cols = [c for c in df.columns if "_R@" in c and "Oracle" not in c]

    if oracle_col:
        plt.plot(df['n'], df[oracle_col], label='Oracle (Max Recall)',
                 marker='*', linestyle='--', linewidth=2, color='black', markersize=8)

    # Plot Systems
    markers = ['o', 's', '^', 'D', 'v', 'p']
    for i, col in enumerate(sys_cols):
        label = col.split('_R@')[0]  # Remove metric suffix
        label = label.replace("Pipeline_", "")

        plt.plot(df['n'], df[col], label=label, marker=markers[i % len(markers)], alpha=0.7)

    plt.title("Fusion Noise Analysis: Oracle vs Fusion Methods")
    plt.xlabel("Number of Rewrites (N)")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    filename = os.path.join(output_dir, "noise_analysis_plot.png")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


if __name__ == "__main__":
    main()
