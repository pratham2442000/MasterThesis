"""
Experiment 1 MPI — BM25 & Dense Cutoff Sweep (All Rerankers)
=============================================================
MPI-distributed version of experiment_1.py.

Systems:
  - BM25 (baseline)
  - BM25 + MonoT5
  - BM25 + MiniLM-L6 (CrossEncoder)
  - Dense (TCT-ColBERT)
  - Dense + MonoT5
  - Dense + MiniLM-L6

Sweeps ``constants.EXPERIMENT_1_C_VALUES``.
Topics split across MPI ranks. Results gathered on rank 0.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_dr as pt_dr
import torch
from mpi4py import MPI
from pyterrier.measures import R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    get_text_getter,
    plot_metrics,
    save_experiment_results,
)
from code_base.transformers import preprocess_query


# ─── Reranker Loaders ────────────────────────────────────

def _load_monot5(models_base_dir=None):
    from pyterrier_t5 import MonoT5ReRanker

    model_id = constants.RERANKER_MODEL_NAME
    tok_model = None
    cache_dir = models_base_dir or constants.MODELS_CACHE_DIR

    if getattr(constants, "OFFLINE_MODE", False) or models_base_dir:
        local_path = os.path.join(cache_dir, model_id.replace("/", os.sep))
        if os.path.exists(local_path):
            model_id = local_path
            t5_tok_path = os.path.join(cache_dir, "t5-base")
            if os.path.exists(t5_tok_path):
                tok_model = t5_tok_path

    kwargs = {
        "model": model_id,
        "verbose": False,
        "batch_size": constants.T5_BATCH_SIZE,
    }
    if tok_model is not None:
        kwargs["tok_model"] = tok_model

    return MonoT5ReRanker(**kwargs)


def _load_cross_encoder(models_base_dir=None):
    from code_base.cross_encoder_reranker import CrossEncoderReRanker

    model_id = constants.CROSS_ENCODER_MODEL_NAME
    cache_dir = models_base_dir or constants.MODELS_CACHE_DIR

    if getattr(constants, "OFFLINE_MODE", False) or models_base_dir:
        local_path = os.path.join(cache_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            model_id = local_path

    return CrossEncoderReRanker(
        model=model_id,
        batch_size=constants.CROSS_ENCODER_BATCH_SIZE,
        verbose=False,
    )


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, monot5, cross_encoder, dense_retriever):
    """Construct retrieval pipelines for a given cutoff *c*."""
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    text_getter = get_text_getter(index_ref)
    bm25_with_text = bm25 >> text_getter

    pipelines = {"BM25": bm25}

    if monot5 is not None:
        pipelines["BM25_MonoT5"] = bm25_with_text >> monot5

    if cross_encoder is not None:
        pipelines["BM25_MiniLM-L6"] = bm25_with_text >> cross_encoder

    if dense_retriever is not None:
        dense = dense_retriever % c
        dense_with_text = dense >> text_getter

        pipelines["Dense"] = dense
        if monot5 is not None:
            pipelines["Dense_MonoT5"] = dense_with_text >> monot5
        if cross_encoder is not None:
            pipelines["Dense_MiniLM-L6"] = dense_with_text >> cross_encoder

    return pipelines


# ─── Plot helpers ─────────────────────────────────────────

def _system_label(key):
    return {
        "BM25": "BM25 Baseline",
        "BM25_MonoT5": "BM25 + MonoT5",
        "BM25_MiniLM-L6": "BM25 + MiniLM-L6",
        "Dense": "Dense (TCT-ColBERT)",
        "Dense_MonoT5": "Dense + MonoT5",
        "Dense_MiniLM-L6": "Dense + MiniLM-L6",
    }.get(key, key)


def _build_plot_configs(system_keys):
    systems = [(k, _system_label(k)) for k in system_keys]
    return [
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on Recall@10",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "filename": "cutoff_vs_R10.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@50" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on Recall@50",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "Recall@50",
            "filename": "cutoff_vs_R50.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on nDCG@10",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "nDCG@10",
            "filename": "cutoff_vs_nDCG10.png",
            "use_log_scale": True,
        },
    ]


# ─── Main ─────────────────────────────────────────────────

def main():
    # Force offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(os.cpu_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Logging
    log_filename = "experiment_1_mpi.log"
    file_mode = 'w' if rank == 0 else 'a'
    if rank != 0:
        comm.Barrier()
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format=f'%(asctime)s - Rank {rank} - %(message)s',
        filemode=file_mode,
    )
    if rank == 0:
        comm.Barrier()

    try:
        logging.info(f"Starting. Rank {rank}/{size}")

        # Shared timestamp for results dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        timestamp = comm.bcast(timestamp, root=0)
        RESULTS_PATH = os.path.join(
            constants.RESULTS_PATH, constants.DATASET_NAME, "exp1", f"run_{timestamp}"
        )
        if rank == 0:
            os.makedirs(RESULTS_PATH, exist_ok=True)

        C_VALUES = constants.EXPERIMENT_1_C_VALUES
        EVAL_METRICS = constants.EXPERIMENT_1_EVAL_METRICS
        TOPIC_LIMIT = constants.EXPERIMENT_1_TOPIC_LIMIT

        # Load data
        _, topics_df, qrels_df = dataset_factory.get_dataset(
            constants.DATASET_NAME, constants.DATA_SPLIT
        )
        topics_df["query"] = topics_df["query"].apply(preprocess_query)
        all_topics = topics_df.head(TOPIC_LIMIT)

        # Distribute topics across ranks
        my_topics = np.array_split(all_topics, size)[rank]
        logging.info(f"Rank {rank}: {len(my_topics)} topics")

        # PyTerrier init
        pt.terrier.set_version("5.11")
        pt.terrier.set_helper_version("0.0.8")
        pt.init()

        index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)

        # Load rerankers (skip failures)
        LOCAL_MODEL_BASE = os.environ.get("LOCAL_MODEL_BASE", os.path.abspath(constants.MODELS_CACHE_DIR))
        monot5, cross_encoder = None, None

        try:
            monot5 = _load_monot5(models_base_dir=LOCAL_MODEL_BASE)
            logging.info(f"Rank {rank}: MonoT5 loaded")
        except Exception as e:
            logging.warning(f"Rank {rank}: MonoT5 not loaded — {e}")

        try:
            cross_encoder = _load_cross_encoder(models_base_dir=LOCAL_MODEL_BASE)
            logging.info(f"Rank {rank}: CrossEncoder loaded")
        except Exception as e:
            logging.warning(f"Rank {rank}: CrossEncoder not loaded — {e}")

        # Load dense retriever
        dense_retriever = None
        dense_index_path = constants.DENSE_INDEX_PATH

        if dense_index_path and os.path.exists(
                os.path.join(dense_index_path, "pt_meta.json")
        ):
            try:
                model = pt_dr.TctColBert(
                    constants.DENSE_MODEL_NAME, batch_size=64, device=device,
                )
                index = pt_dr.FlexIndex(dense_index_path)
                dense_retriever = model.query_encoder() >> index.retriever()
                logging.info(f"Rank {rank}: Dense retriever loaded from {dense_index_path}")
            except Exception as e:
                logging.warning(f"Rank {rank}: Dense retriever not loaded — {e}")
        else:
            logging.warning(f"Rank {rank}: Dense index not found at '{dense_index_path}'")

        # Run sweep
        local_results_list = []

        for c in C_VALUES:
            systems_dict = build_pipelines(
                c,
                index_ref=index_ref,
                monot5=monot5,
                cross_encoder=cross_encoder,
                dense_retriever=dense_retriever,
            )

            current_metrics = list(EVAL_METRICS) + [R @ c]

            if len(my_topics) > 0:
                exp_results = pt.Experiment(
                    list(systems_dict.values()),
                    my_topics,
                    qrels_df,
                    eval_metrics=current_metrics,
                    names=list(systems_dict.keys()),
                    verbose=(rank == 0),
                    batch_size=constants.EXPERIMENT_BATCH_SIZE,
                    perquery=True,
                    precompute_prefix=True,
                )
                exp_results['c'] = c
                local_results_list.append(exp_results)
            else:
                local_results_list.append(pd.DataFrame())

        # Gather to rank 0
        all_results_lists = comm.gather(local_results_list, root=0)

        if rank == 0:
            logging.info("Aggregating results...")
            final_summary_list = []

            for i, c in enumerate(C_VALUES):
                dfs_for_c = [
                    rank_res[i] for rank_res in all_results_lists
                    if not rank_res[i].empty
                ]
                if not dfs_for_c:
                    logging.warning(f"No results for c={c}")
                    continue

                combined_df = pd.concat(dfs_for_c, ignore_index=True)
                summary_row = {"c": c}
                systems = combined_df['name'].unique()

                for system in systems:
                    sys_df = combined_df[combined_df['name'] == system]
                    measures = sys_df['measure'].unique()
                    for measure in measures:
                        mean_val = sys_df[sys_df['measure'] == measure]['value'].mean()
                        if measure == f"R@{c}":
                            key = f"{system}_R@c"
                        else:
                            key = f"{system}_{measure}"
                        summary_row[key] = mean_val

                final_summary_list.append(summary_row)
                logging.info(f"c={c}: {summary_row}")

            results_df = pd.DataFrame(final_summary_list)

            save_experiment_results(
                results_df,
                experiment_name="experiment_1",
                output_dir=RESULTS_PATH,
                filename="experiment_1_results",
            )

            logging.info("Final Results:")
            logging.info(results_df.to_string())

            # Plots
            system_keys = [
                col.replace("_R@10", "") for col in results_df.columns
                if col.endswith("_R@10")
            ]
            plot_configs = _build_plot_configs(system_keys)
            for cfg in plot_configs:
                cfg = dict(cfg)
                cfg.pop("type", None)
                cfg["df"] = results_df
                cfg["output_dir"] = RESULTS_PATH
                plot_metrics(**cfg)

            logging.info("Done.")

    except Exception as e:
        logging.error(f"Rank {rank}: Error — {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()