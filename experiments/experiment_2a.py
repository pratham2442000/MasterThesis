import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import R, nDCG
from pyterrier_t5 import MonoT5ReRanker
import pyterrier_dr as pt_dr

# Local imports
import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    save_experiment_results,
    preprocess_query,
    plot_with_regression,
    QueryRewriteTransformer,
    MergeRewrites,
    AttachOriginalQuery
)
from code_base.query_trasnformers import (
    Dedup,
    BordaVotingTransformerV2,
    CombMAXTransformer,
    RRFTransformer
)


# ---------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------
def setup_directories():
    """Ensure result directories exist."""
    results_path = os.path.join(constants.DATA_PATH, "experiment_results/experiment_2a")
    os.makedirs(results_path, exist_ok=True)
    print(f"Results directory ensured at: {results_path}")
    return results_path


def load_data():
    """Load and preprocess dataset, topics, and qrels."""
    print("Loading data...")
    docs_df, topics_df, qrels_df = dataset_factory.get_dataset(
        constants.DATASET_NAME, constants.DATA_SPLIT
    )

    # Preprocess queries
    topics_df["query"] = topics_df["query"].apply(preprocess_query)

    # Load rewritten queries to filter topics (aligned with experiment_2 logic)
    filename = f"{constants.DATASET_NAME}_rewritten_queries_dataset_{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
    filepath = os.path.join(constants.DATA_PATH, filename)

    if os.path.exists(filepath):
        rewritten_queries = pd.read_csv(filepath)
        topics = topics_df[topics_df["qid"].isin(rewritten_queries["q_id"])].reset_index(drop=True)
    else:
        print(f"Warning: Rewrites file not found at {filepath}. Using all topics.")
        topics = topics_df

    # Apply topic limit
    topics = topics.tail(constants.EXPERIMENT_2_TOPIC_LIMIT)

    print("Topics:")
    print(topics.info())
    print("\n\n")
    print("Qrels:")
    print(qrels_df.info())

    # Create lookup for AttachOriginalQuery
    query_lookup = {str(row["qid"]): row["query"] for _, row in topics.iterrows()}

    return topics, qrels_df, query_lookup


# ---------------------------------------------------------
# Pipeline Definitions
# ---------------------------------------------------------
def get_pipelines(n, index_ref, dense_retriever, dense_reranker, sparse_retriever, mono_t5, query_lookup, cutoff):
    """
    Constructs the dictionary of pipelines for a specific number of rewrites (N).
    """

    # Common Components
    rewrite = QueryRewriteTransformer(n, dataset_name=constants.DATASET_NAME)
    attach_query = AttachOriginalQuery(query_lookup)
    get_text = pt.text.get_text(index_ref, "text")

    # Baselines

    # 1. Dense Baseline
    dense_base = dense_retriever % cutoff

    # 2. Dense + MonoT5
    pipeline_dense_t5 = dense_base >> get_text >> mono_t5

    # 3. BM25 + Dense Rerank
    pipeline_bm25_dense = (
            sparse_retriever % cutoff
            >> dense_reranker
            >> pt.apply.generic(lambda df: df.sort_values(["qid", "score"], ascending=[True, False]))
    )

    # Pipeline 4: Dedup
    p_dedup = (rewrite
               >> dense_retriever % cutoff
               >> MergeRewrites()
               >> attach_query
               >> Dedup()
               >> get_text
               >> mono_t5)

    # Pipeline 5: Borda Voting
    borda_voting = BordaVotingTransformerV2(cutoff=cutoff, weight=None, method="borda")
    p_borda = (rewrite
               >> dense_retriever % cutoff
               >> borda_voting
               >> attach_query
               >> get_text
               >> mono_t5)

    # Pipeline 6: Weighted Borda
    borda_weighted = BordaVotingTransformerV2(
        cutoff=cutoff,
        weight=lambda x: np.exp(x * 0.1),
        method="borda"
    )
    p_borda_weighted = (rewrite
                        >> dense_retriever % cutoff
                        >> borda_weighted
                        >> attach_query
                        >> get_text
                        >> mono_t5)

    # Pipeline 7: CombMAX
    p_combmax = (rewrite
                 >> dense_retriever % cutoff
                 >> MergeRewrites()
                 >> attach_query
                 >> CombMAXTransformer()
                 >> get_text
                 >> mono_t5)

    # Pipeline 8: RRF (Reciprocal Rank Fusion)
    p_rrf = (rewrite
             >> dense_retriever % cutoff
             >> MergeRewrites()
             >> attach_query
             >> RRFTransformer()
             >> get_text
             >> mono_t5)

    return {
        "Dense": dense_base,
        "Dense_MonoT5": pipeline_dense_t5,
        "BM25_DenseRerank": pipeline_bm25_dense,
        "Pipeline_Dedup": p_dedup,
        "Pipeline_Borda": p_borda,
        "Pipeline_BordaWeighted": p_borda_weighted,
        "Pipeline_CombMAX": p_combmax,
        "Pipeline_RRF": p_rrf
    }


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    # 1. Setup
    results_path = setup_directories()

    print("Setting up dense retrieval...")
    # Initialize TCT-ColBERT model
    model = pt_dr.TctColBert(constants.DENSE_MODEL_NAME)
    index = pt_dr.FlexIndex(constants.DENSE_INDEX_PATH)
    print(f"Loaded dense index from {constants.DENSE_INDEX_PATH}.")

    # Create Dense Retriever and Reranker
    dense_retriever = model.query_encoder() >> index.retriever()
    dense_reranker = model.query_encoder() >> index.scorer()

    # Sparse Index and Retriever
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)
    sparse_retriever = pt.terrier.Retriever(index_ref, wmodel="BM25")

    # Initialize Reranker once
    mono_t5 = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    # 2. Data
    topics, qrels, query_lookup = load_data()

    # 3. Experiment Loop
    all_results = []
    eval_metrics = constants.EXPERIMENT_2_EVAL_METRICS + [R @ constants.EXPERIMENT_2_RETRIEVAL_CUTOFF]

    print(f"Starting experiment loop for N={constants.EXPERIMENT_2_N_REWRITES} topics... {len(topics)}")

    for n in constants.EXPERIMENT_2_N_REWRITES:
        print(f"Running pipeline for N={n}...")

        # Define systems
        systems_dict = get_pipelines(
            n, index_ref, dense_retriever, dense_reranker, sparse_retriever, mono_t5,
            query_lookup, constants.EXPERIMENT_2_RETRIEVAL_CUTOFF
        )

        # Run Experiment
        exp_df = pt.Experiment(
            list(systems_dict.values()),
            topics,
            qrels,
            eval_metrics=eval_metrics,
            names=list(systems_dict.keys()),
            verbose=True,
            batch_size=constants.EXPERIMENT_BATCH_SIZE,
            precompute_prefix=True
        )

        # Post-process: Add 'n' column and formatting
        exp_df['n'] = n
        print(exp_df)
        row_data = {"n": n}
        for _, row in exp_df.iterrows():
            system_name = row['name']
            for metric in eval_metrics:
                metric_str = str(metric)
                val = row[metric_str]

                # Standardize metric string for CSV column headers
                if metric_str == f"R@{constants.EXPERIMENT_2_RETRIEVAL_CUTOFF}":
                    key = f"{system_name}_R@c"  # Maps specific cutoff to generic key
                    metric_str = "R@10"  # For print display only
                else:
                    key = f"{system_name}_{metric_str}"

                row_data[key] = val

        print("\n")
        print(f"Experiment for N={n} complete. Results head:")
        print(row_data)
        print("\n")

        all_results.append(row_data)

    # 4. Save and Visualize
    results_df = pd.DataFrame(all_results)

    print("\nExperiment Complete. Results head:")
    print(results_df.head())

    save_experiment_results(
        results_df,
        experiment_name="experiment_2a",
        output_dir=results_path,
        filename="experiment_2a_results",
    )

    # Generate Plots
    generate_plots(results_df, results_path)


def generate_plots(df, output_dir):
    """Encapsulate plotting logic."""
    common_args = {
        "x_col": "n",
        "output_dir": output_dir,
        "use_log_scale": False
    }

    systems = [
        ("Dense", "Dense Baseline"),
        ("Dense_MonoT5", "Dense + MonoT5"),
        ("BM25_DenseRerank", "BM25 + Dense Rerank"),
        ("Pipeline_Dedup", "Dedup Method"),
        ("Pipeline_Borda", "Borda Voting"),
        ("Pipeline_BordaWeighted", "Borda Weighted"),
        ("Pipeline_CombMAX", "CombMAX"),
        ("Pipeline_RRF", "RRF")
    ]

    # 1. Recall Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_R@10" if "R@10" in df.columns else f"{sys}_R@c" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2a: Document Combination Methods Comparison (Recall)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_R_comparison.png",
        **common_args
    )

    # 2. NDCG Plot
    plot_metrics(
        df,
        metric_cols=[f"{sys}_nDCG@10" for sys, _ in systems],
        labels=[label for _, label in systems],
        title="Experiment 2a: Document Combination Methods Comparison (nDCG@10)",
        xlabel="Number of Rewrites",
        filename="/n_rewrites_vs_nDCG_comparison.png",
        **common_args
    )

    # 3. Regression Plot
    plot_with_regression(
        df,
        metric_cols=[f"{sys}_R@c" if f"{sys}_R@c" in df.columns else f"{sys}_R@10" for sys, _ in systems],
        title="Experiment 2a: R@10 vs Number of Rewrites (All Methods)",
        xlabel="Number of Rewrites",
        ylabel="R@10",
        filename="/n_rewrites_vs_R_regression_all_methods.png",
        **common_args
    )


if __name__ == "__main__":
    main()
