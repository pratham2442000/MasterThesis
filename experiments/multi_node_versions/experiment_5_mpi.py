import os
import sys

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI

# Add parent directory to path (repo root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import save_experiment_results, plot_metrics
from code_base.wiki_loader_pyt import get_gold_answers
from experiments.test_rag_pipeline import get_reader, build_retrieval_pipeline, build_rag_pipeline


def main():
    # 1. Enforce Offline Mode for Cluster
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Setup paths
    RESULTS_PATH = os.path.join(constants.RESULTS_PATH, "exp_rag_mpi")
    if rank == 0:
        os.makedirs(RESULTS_PATH, exist_ok=True)

    # Initialize PyTerrier (pin versions to avoid download)
    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    pt.init()

    # Load Data
    _, topics_df, _ = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
    all_topics = topics_df.head(constants.EXPERIMENT_5_TOPIC_LIMIT)
    my_topics = np.array_split(all_topics, size)[rank]
    gold_answers = get_gold_answers(my_topics)

    # Setup Index
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH)

    # Load Reranker (local path for offline cluster)
    from pyterrier_t5 import MonoT5ReRanker
    local_model_path = "/scratch/pjohari/MasterThesis/models"
    mono_t5_path = os.path.join(local_model_path, "castorini/monot5-base-msmarco")
    t5_base_path = os.path.join(local_model_path, "t5-base")

    if os.path.exists(mono_t5_path) and os.path.exists(t5_base_path):
        reranker = MonoT5ReRanker(
            model=mono_t5_path, tok_model=t5_base_path,
            verbose=False, batch_size=constants.T5_BATCH_SIZE,
        )
    elif os.path.exists(mono_t5_path):
        reranker = MonoT5ReRanker(
            model=mono_t5_path, verbose=False, batch_size=constants.T5_BATCH_SIZE,
        )
    else:
        reranker = MonoT5ReRanker(verbose=False, batch_size=constants.T5_BATCH_SIZE)

    # Load RAG Reader (get_reader handles OFFLINE_MODE/MODELS_CACHE_DIR)
    reader = get_reader(live=True)

    local_results = []
    C_VALUES = constants.EXPERIMENT_5_C_VALUES

    for c in C_VALUES:
        _, retrieval_pipe = build_retrieval_pipeline(index_ref, reranker, c)
        rag_pipe = build_rag_pipeline(retrieval_pipe, reader, constants.EXPERIMENT_5_TOP_K)

        if len(my_topics) > 0:
            exp = pt.Experiment(
                [rag_pipe],
                my_topics,
                gold_answers,
                eval_metrics=constants.EXPERIMENT_5_EVAL_METRICS,
                names=["Full_RAG"],
                verbose=(rank == 0),
                perquery=True,
            )
            exp['c'] = c
            local_results.append(exp)
        else:
            local_results.append(pd.DataFrame())

    # Aggregate results to Rank 0
    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        final_rows = []
        for i, c in enumerate(C_VALUES):
            dfs = [res[i] for res in all_results
                   if len(res) > i and not res[i].empty]
            if not dfs:
                continue
            combined = pd.concat(dfs, ignore_index=True)
            summary = {
                "c": c,
                "Full_RAG_EM": combined[combined['measure'] == 'EM']['value'].mean(),
                "Full_RAG_F1": combined[combined['measure'] == 'F1']['value'].mean(),
            }
            final_rows.append(summary)

        results_df = pd.DataFrame(final_rows)
        save_experiment_results(results_df, "experiment_rag_sweep", RESULTS_PATH)

        plot_metrics(
            results_df, x_col="c",
            metric_cols=["Full_RAG_EM", "Full_RAG_F1"],
            title="RAG Generation Quality Sweep (MPI)",
            output_dir=RESULTS_PATH, filename="rag_mpi_sweep.png",
        )


if __name__ == "__main__":
    main()
