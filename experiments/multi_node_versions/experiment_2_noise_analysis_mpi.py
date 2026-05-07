import logging
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import get_text_getter, save_experiment_results, get_reranker
from code_base.transformers import (
    preprocess_query,
    QueryRewriteTransformer,
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_2_noise_analysis_mpi.log"


def setup_environment(rank: int):
    """Configures environment variables and PyTerrier for offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Initialize PyTerrier
    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    pt.init()
    if rank == 0:
        logging.info("PyTerrier initialized (Offline Mode).")


def setup_logging(rank: int, comm: MPI.Comm):
    """Sets up rank-aware logging."""
    file_mode = 'w' if rank == 0 else 'a'

    # Barrier to ensure rank 0 clears the file first
    if rank != 0:
        comm.Barrier()

    logging.basicConfig(
        filename=LOG_FILENAME,
        level=logging.INFO,
        format=f'%(asctime)s - Rank {rank} - %(message)s',
        filemode=file_mode
    )

    if rank == 0:
        comm.Barrier()





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
            logging.info(f"Loaded experiment_2 results from {p}")
            return pd.read_csv(p)

    logging.warning("experiment_2 results CSV not found. Fusion metrics will be skipped.")
    return pd.DataFrame()


def plot_comparison(df, output_dir):
    """
    Generates the comparison plot: N vs Recall for Oracle and Fusion methods.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
    logging.info(f"Plot saved to {filename}")
    plt.close()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    setup_logging(rank, comm)

    try:
        logging.info(f"Starting process. Rank {rank}/{size}")

        # 1. Environment & Data Setup
        setup_environment(rank)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        timestamp = comm.bcast(timestamp, root=0)
        RESULTS_PATH = os.path.join(
            constants.RESULTS_PATH, constants.DATASET_NAME, "exp2_noise", f"run_{timestamp}"
        )
        if rank == 0:
            os.makedirs(RESULTS_PATH, exist_ok=True)
            logging.info(f"Created results dir: {RESULTS_PATH}")

        # Load Dataset
        docs_df, topics_df, qrels_df = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
        topics_df["query"] = topics_df["query"].apply(preprocess_query)

        # Load rewritten queries
        filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
        filepath = os.path.join(constants.DATA_PATH, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rewrites file not found: {filepath}")

        rewritten_queries = pd.read_csv(filepath)

        # Filter topics to match rewrites and apply limit
        all_topics = topics_df[topics_df["qid"].isin(rewritten_queries["q_id"])].reset_index(drop=True)
        all_topics = all_topics.tail(constants.EXPERIMENT_2_TOPIC_LIMIT)

        # Distribute Work
        my_topics = np.array_split(all_topics, size)[rank]
        logging.info(f"Rank {rank}: Assigned {len(my_topics)} topics.")

        # 2. Initialize Models & Index
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
        reranker = get_reranker(models_base_dir=LOCAL_MODEL_BASE)
        get_text = get_text_getter(index_ref)

        # Evaluation cutoff (consistent with EXPERIMENT_2_EVAL_METRICS using R@10)
        eval_cutoff = 10

        # 3. Analysis Loop
        n_rewrites_list = constants.EXPERIMENT_2_N_REWRITES
        local_oracle_results = []  # List of dicts: {n, oracle_recalls: [floats]}

        if rank == 0:
            logging.info(f"Starting Noise Analysis loop for N_REWRITES: {n_rewrites_list}")

        for n in n_rewrites_list:
            logging.info(f"Rank {rank}: Processing N={n}...")

            # Construct Oracle Pipeline: Rewrite -> BM25 -> GetText -> Rerank
            rewrite_trans = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
            bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", threads=10) % constants.EXPERIMENT_2_RETRIEVAL_CUTOFF

            oracle_pipeline = (rewrite_trans
                               >> bm25
                               >> get_text
                               >> reranker)

            if len(my_topics) > 0:
                # Run transform to get raw results
                oracle_res = oracle_pipeline.transform(my_topics)

                # Calculate Oracle Recall for this rank's topics
                oracle_metrics = calculate_oracle_recall(oracle_res, qrels_df, cutoff=eval_cutoff)
                local_oracle_results.append({
                    'n': n,
                    'oracle_recalls': oracle_metrics['oracle_recall'].tolist()
                })
                logging.info(f"Rank {rank}: N={n}, local mean Oracle R@{eval_cutoff}: "
                             f"{oracle_metrics['oracle_recall'].mean():.4f}")
            else:
                local_oracle_results.append({'n': n, 'oracle_recalls': []})

        # 4. Gather & Aggregate
        logging.info(f"Rank {rank}: Finished loop. Waiting for gather.")
        all_oracle_results = comm.gather(local_oracle_results, root=0)

        if rank == 0:
            logging.info("Rank 0: Aggregating results...")

            # Load comparison data (Experiment 2 results)
            exp2_df = load_comparison_results(RESULTS_PATH)

            all_results = []

            for i, n in enumerate(n_rewrites_list):
                # Collect all oracle recalls for this N from all ranks
                all_recalls_for_n = []
                for rank_results in all_oracle_results:
                    if i < len(rank_results):
                        all_recalls_for_n.extend(rank_results[i]['oracle_recalls'])

                mean_oracle_recall = np.mean(all_recalls_for_n) if all_recalls_for_n else 0.0
                logging.info(f"N={n}: Global Oracle R@{eval_cutoff} = {mean_oracle_recall:.4f}")

                result_row = {'n': n, f'Oracle_R@{eval_cutoff}': mean_oracle_recall}

                # Match with Experiment 2 comparison data
                if not exp2_df.empty:
                    n_col = next((c for c in exp2_df.columns if c.lower() in ('n', 'n_rewrites', 'nr')), None)

                    if n_col:
                        matching = exp2_df[exp2_df[n_col] == n]

                        if not matching.empty:
                            metric_suffix = f"R@{eval_cutoff}"
                            metric_cols = [c for c in exp2_df.columns if metric_suffix in c]

                            agg = matching[metric_cols].mean(numeric_only=True)

                            for col in metric_cols:
                                result_row[col] = float(agg[col])

                all_results.append(result_row)

            # 5. Save and Visualize
            results_df = pd.DataFrame(all_results)
            logging.info("Analysis Complete.\n" + results_df.to_string())

            save_experiment_results(
                results_df,
                experiment_name="experiment_2_noise_analysis",
                output_dir=RESULTS_PATH,
                filename="experiment_2_noise_analysis_results"
            )

            # Plotting
            logging.info("Generating plots...")
            plot_comparison(results_df, RESULTS_PATH)

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        # Crucial for MPI: If one fails, kill all to prevent hanging
        comm.Abort(1)


if __name__ == "__main__":
    main()
