import logging
import os
import sys
import traceback
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pyterrier as pt
from mpi4py import MPI

# Add parent directory to path (repo root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    save_experiment_results,
    get_text_getter,
    get_reader,
    get_reranker,
)
from code_base.dataset_factory import get_gold_answers
from pyterrier_rag.prompt import Concatenator



# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
LOCAL_MODEL_BASE = "/scratch/pjohari/MasterThesis/models"
LOG_FILENAME = "experiment_5_mpi.log"


def setup_environment(rank: int):
    """Configures environment variables and PyTerrier for offline mode."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    pt.terrier.set_version("5.11")
    pt.terrier.set_helper_version("0.0.8")
    pt.init()
    if rank == 0:
        logging.info("PyTerrier initialized (Offline Mode).")


def setup_logging(rank: int, comm: MPI.Comm):
    """Sets up rank-aware logging."""
    file_mode = 'w' if rank == 0 else 'a'

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





def build_rag_pipelines(c, *, index_ref, reranker, reader):
    """Full RAG pipeline for a given cutoff *c*."""
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", threads=1) % c
    retrieval_pipe = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )
    concat = Concatenator(
        max_elements=constants.EXPERIMENT_5_TOP_K,
        intermediate_format=lambda text="", **kw: f"- {text}",
    )
    rag_pipe = retrieval_pipe >> concat >> reader
    rag_pipe_without_reranker = bm25 >> get_text_getter(index_ref) >> concat >> reader

    return {
        "Full_RAG_with_reranker": rag_pipe,
        "Full_RAG_without_reranker": rag_pipe_without_reranker,
    }





def process_results(
        all_results_lists: List[List[pd.DataFrame]],
        c_values: List[int],
) -> pd.DataFrame:
    """Aggregates per-query results from all MPI ranks."""
    final_rows = []

    for i, c in enumerate(c_values):
        dfs = [
            rank_res[i] for rank_res in all_results_lists
            if i < len(rank_res) and not rank_res[i].empty
        ]
        if not dfs:
            logging.warning(f"No results found for c={c}")
            continue

        combined = pd.concat(dfs, ignore_index=True)

        row_data = {"c": c}
        systems = combined['name'].unique()

        for system in systems:
            sys_df = combined[combined['name'] == system]
            measures = sys_df['measure'].unique()
            for measure in measures:
                mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                row_data[f"{system}_{measure}"] = mean_val

        final_rows.append(row_data)

    return pd.DataFrame(final_rows)


def generate_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for experiment 5 results."""
    systems = [
        ("Full_RAG_with_reranker", f"Full RAG with {constants.RERANKER_NAME}"),
        ("Full_RAG_without_reranker", "Full RAG without reranker")
    ]

    # EM plot
    em_cols = [f"{s}_EM" for s, _ in systems if f"{s}_EM" in df.columns]
    em_labels = [lbl for s, lbl in systems if f"{s}_EM" in df.columns]
    if em_cols:
        plot_metrics(
            df, x_col="c",
            metric_cols=em_cols,
            labels=em_labels,
            title="Experiment 5: RAG Exact Match vs Retrieval Cutoff",
            xlabel="Retrieval Cutoff (c)",
            ylabel="EM",
            filename="exp5_cutoff_vs_EM.png",
            output_dir=output_dir,
            use_log_scale=True,
        )

    # F1 plot
    f1_cols = [f"{s}_F1" for s, _ in systems if f"{s}_F1" in df.columns]
    f1_labels = [lbl for s, lbl in systems if f"{s}_F1" in df.columns]
    if f1_cols:
        plot_metrics(
            df, x_col="c",
            metric_cols=f1_cols,
            labels=f1_labels,
            title="Experiment 5: RAG F1 vs Retrieval Cutoff",
            xlabel="Retrieval Cutoff (c)",
            ylabel="F1",
            filename="exp5_cutoff_vs_F1.png",
            output_dir=output_dir,
            use_log_scale=True,
        )

    llm_cols = [f"{s}_LLMasJudge" for s, _ in systems if f"{s}_LLMasJudge" in df.columns]
    llm_labels = [lbl for s, lbl in systems if f"{s}_LLMasJudge" in df.columns]
    if llm_cols:
        plot_metrics(
            df, x_col="c",
            metric_cols=llm_cols,
            labels=llm_labels,
            title="Experiment 5: RAG LLM-as-Judge vs Retrieval Cutoff",
            xlabel="Retrieval Cutoff (c)",
            ylabel="Score (1-10)",
            filename="exp5_cutoff_vs_LLMasJudge.png",
            output_dir=output_dir,
            use_log_scale=True,
        )


    # Combined EM + F1 plot
    combined_cols = em_cols + f1_cols
    combined_labels = [f"{lbl} (EM)" for lbl in em_labels] + [f"{lbl} (F1)" for lbl in f1_labels]
    if combined_cols:
        plot_metrics(
            df, x_col="c",
            metric_cols=combined_cols,
            labels=combined_labels,
            title="Experiment 5: RAG Generation Quality vs Retrieval Cutoff",
            xlabel="Retrieval Cutoff (c)",
            ylabel="Score",
            filename="exp5_cutoff_vs_EM_F1_combined.png",
            output_dir=output_dir,
            use_log_scale=True,
        )


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
            constants.RESULTS_PATH, constants.DATASET_NAME, "exp5", f"run_{timestamp}"
        )
        if rank == 0:
            os.makedirs(RESULTS_PATH, exist_ok=True)
            logging.info(f"Created results dir: {RESULTS_PATH}")

        # Load Data
        _, topics_df, _ = dataset_factory.get_dataset(constants.DATASET_NAME, constants.DATA_SPLIT)
        all_topics = topics_df.head(constants.EXPERIMENT_5_TOPIC_LIMIT)

        # Distribute Work
        my_topics = np.array_split(all_topics, size)[rank]
        gold_answers = get_gold_answers(my_topics)

        # LLMasJudge expects qrels with columns: query_id, relevance, text
        # PyTerrier internally renames qid -> query_id, so we only add text & relevance
        gold_answers["text"] = gold_answers["gold_answer"]
        gold_answers["relevance"] = 3  # default rel threshold in LLMasJudge is 3

        logging.info(f"Rank {rank}: Assigned {len(my_topics)} topics.")

        # 2. Initialize Models & Index
        index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
        reranker = get_reranker(models_base_dir=LOCAL_MODEL_BASE)
        reader, raw_backend = get_reader(live=True)
        logging.info(f"Rank {rank}: Models loaded.")

        # Share raw backend with LLMasJudge (avoid loading model twice → OOM)
        eval_metrics = constants.get_experiment_5_eval_metrics(
            shared_backend=raw_backend
        )

        # 3. Experiment Loop (varying cutoff c)
        local_results_list = []
        C_VALUES = constants.EXPERIMENT_5_C_VALUES

        if rank == 0:
            logging.info(f"Starting loop for C_VALUES: {C_VALUES}")

        for c in C_VALUES:
            systems_dict = build_rag_pipelines(c, index_ref=index_ref, reranker=reranker, reader=reader)

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    list(systems_dict.values()),
                    my_topics,
                    gold_answers,
                    eval_metrics=eval_metrics,
                    names=list(systems_dict.keys()),
                    verbose=(rank == 0),
                    perquery=True,
                    batch_size=constants.EXPERIMENT_BATCH_SIZE,
                    precompute_prefix=True,
                )
                exp_results['c'] = c
                local_results_list.append(exp_results)
            else:
                local_results_list.append(pd.DataFrame())

        # 4. Gather & Aggregate
        logging.info(f"Rank {rank}: Finished loop. Waiting for gather.")
        all_results_lists = comm.gather(local_results_list, root=0)

        if rank == 0:
            logging.info("Rank 0: Aggregating results...")
            results_df = process_results(all_results_lists, C_VALUES)

            logging.info("\n" + results_df.to_string())

            save_experiment_results(
                results_df,
                experiment_name="experiment_5",
                output_dir=RESULTS_PATH,
                filename="experiment_5_results",
            )

            # Generate Plots
            logging.info("Generating plots...")
            generate_plots(results_df, RESULTS_PATH)

    except Exception as e:
        logging.error(f"Rank {rank}: CRITICAL ERROR: {e}\n{traceback.format_exc()}")
        comm.Abort(1)


if __name__ == "__main__":
    main()
