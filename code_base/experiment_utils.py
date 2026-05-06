"""
Experiment Utilities
====================
Reusable utilities for experiments:

- **Plotting**: ``plot_metrics``, ``plot_with_regression``
- **Metadata**: ``collect_experiment_metadata``, ``save_experiment_results``,
  ``load_experiment_metadata``, ``find_experiment_files``

All PyTerrier Transformer classes have been moved to
``code_base.transformers``.
"""

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import constants


def get_text_getter(index_ref=None):
    """Returns a PyTerrier transformer that adds the 'text' column.
    
    For msmarco-passage, the pre-built index does not contain the 'text' 
    attribute in its metadata. This function automatically fetches the 
    text directly from the ir_datasets corpus instead. For other datasets, 
    it falls back to reading from the index metaindex.
    """
    import pyterrier as pt
    if constants.DATASET_NAME == "msmarco-passage":
        dataset = pt.get_dataset("irds:msmarco-passage")
        return pt.text.get_text(dataset, "text")
    elif constants.DATASET_NAME == "dpr-w100":
        dataset = pt.get_dataset("irds:dpr-w100")
        return pt.text.get_text(dataset, "text")
    else:
        return pt.text.get_text(index_ref, "text")


def get_reranker():
    """Returns a reranker based on ``constants.RERANKER_TYPE``.

    Supports ``"monot5"`` (MonoT5ReRanker) and ``"cross_encoder"``
    (CrossEncoderReRanker). Respects offline mode for MonoT5.

    Returns:
        A PyTerrier Transformer that reranks documents.
    """
    rtype = getattr(constants, "RERANKER_TYPE", "monot5")

    if rtype == "cross_encoder":
        from code_base.cross_encoder_reranker import CrossEncoderReRanker

        model_id = constants.CROSS_ENCODER_MODEL_NAME
        if getattr(constants, "OFFLINE_MODE", False):
            local_path = os.path.join(
                constants.MODELS_CACHE_DIR, model_id.split("/")[-1]
            )
            if os.path.exists(local_path):
                model_id = local_path
                print(f"Loading offline CrossEncoder: {model_id}")

        print(f"Using CrossEncoderReRanker ({model_id})")
        return CrossEncoderReRanker(
            model=model_id,
            batch_size=constants.CROSS_ENCODER_BATCH_SIZE,
            verbose=False,
        )

    else:  # monot5
        from pyterrier_t5 import MonoT5ReRanker

        model_id = constants.RERANKER_MODEL_NAME
        if getattr(constants, "OFFLINE_MODE", False):
            local_path = os.path.join(
                constants.MODELS_CACHE_DIR, model_id.split("/")[-1]
            )
            if os.path.exists(local_path):
                model_id = local_path
                print(f"Loading offline MonoT5: {model_id}")

        print(f"Using MonoT5ReRanker ({model_id})")
        return MonoT5ReRanker(
            model=model_id,
            verbose=False,
            batch_size=constants.T5_BATCH_SIZE,
        )

# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def _setup_plot(
        df: pd.DataFrame,
        x_col: str,
        metric_cols: List[str],
        labels: Optional[List[str]],
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        use_log_scale: Optional[bool],
        figsize: tuple,
):
    """Shared plot initialization used by public plotting functions.

    Args:
        df: DataFrame with results.
        x_col: Column name for x-axis.
        metric_cols: Columns to plot.
        labels: Legend labels (defaults to column names).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        use_log_scale: Whether to log-scale x-axis. ``None`` = auto-detect.
        figsize: Figure dimensions.

    Returns:
        Tuple of ``(fig, ax, labels, title, xlabel, ylabel, use_log_scale)``.
    """
    if labels is None or len(labels) != len(metric_cols):
        labels = list(metric_cols)
    if title is None:
        title = f"Metrics vs. {x_col}"
    if xlabel is None:
        xlabel = x_col.replace("_", " ").title()
    if ylabel is None:
        ylabel = metric_cols[0].replace("_", " ")

    if use_log_scale is None:
        if len(df[x_col]) > 2 and (df[x_col].max() / df[x_col].min() > 5):
            use_log_scale = True
        else:
            use_log_scale = False

    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, labels, title, xlabel, ylabel, use_log_scale


def _apply_log_ticks(ax, df, x_col, use_log_scale):
    """Apply log-scale formatting to x-axis if requested."""
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_xticks(df[x_col])
        ax.set_xticklabels([str(v) for v in df[x_col]])


def _save_or_show(fig, filename, output_dir):
    """Save figure to file, or show interactively."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Chart saved as {filepath}")
    else:
        print("Displaying chart...")
        plt.show()
    plt.close(fig)


def plot_metrics(
        df: pd.DataFrame,
        x_col: str,
        metric_cols: List[str],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        filename: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_log_scale: Optional[bool] = None,
        figsize: tuple = (10, 6),
) -> None:
    """Generate a line plot of metrics over a variable.

    Args:
        df: DataFrame with results.
        x_col: Column name for x-axis.
        metric_cols: List of column names to plot.
        labels: Legend labels (one per metric_col).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        filename: Output filename (without directory).
        output_dir: Directory to save plot. ``None`` → display interactively.
        use_log_scale: Log-scale x-axis. ``None`` → auto-detect.
        figsize: Figure size ``(width, height)``.
    """
    fig, ax, labels, title, xlabel, ylabel, use_log_scale = _setup_plot(
        df, x_col, metric_cols, labels, title, xlabel, ylabel,
        use_log_scale, figsize,
    )

    markers = ["o", "x", "s", "^", "D", "v", "*", "+"]
    for i, (col, label) in enumerate(zip(metric_cols, labels)):
        marker = markers[i % len(markers)]
        ax.plot(
            df[x_col], np.round(df[col], 3),
            marker=marker, label=label, alpha=0.7,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    _apply_log_ticks(ax, df, x_col, use_log_scale)
    fig.tight_layout()

    if filename is None:
        safe = metric_cols[0].replace("@", "_at_").replace("/", "_")
        filename = f"{x_col}_vs_{safe}.png"
    filename = filename.lstrip("/")

    _save_or_show(fig, filename, output_dir)


def plot_with_regression(
        df: pd.DataFrame,
        x_col: str,
        metric_cols: List[str],
        labels: Optional[List[str]] = None,
        use_log_scale: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        filename: Optional[str] = None,
        output_dir: Optional[str] = None,
        figsize: tuple = (10, 6),
        print_stats: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Plot metrics with linear regression lines.

    Uses ``log(x)`` for regression when ``use_log_scale`` is True.

    Args:
        df: DataFrame with results.
        x_col: Column name for x-axis.
        metric_cols: Columns to plot.
        labels: Legend labels.
        use_log_scale: Use ``log(x)`` for regression.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        filename: Output filename.
        output_dir: Output directory. Defaults to ``constants.DATA_PATH``.
        figsize: Figure size.
        print_stats: Print regression statistics to stdout.

    Returns:
        Dict mapping metric column → regression statistics.
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    if labels is None or len(labels) != len(metric_cols):
        labels = list(metric_cols)

    # Regression x values
    if use_log_scale:
        x_reg = np.log(df[x_col])
        x_reg_label = f"log({x_col})"
    else:
        x_reg = df[x_col].values.astype(float)
        x_reg_label = x_col

    # Compute regressions
    regression_stats: Dict[str, Dict[str, float]] = {}
    for col, label in zip(metric_cols, labels):
        y = df[col]
        slope, intercept, r_value, p_value, std_err = linregress(x_reg, y)
        stats = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "std_err": std_err,
        }
        regression_stats[col] = stats

        if print_stats:
            print(f"\n{label}:")
            print(f"  Equation: {col} = {slope:.4f} * {x_reg_label} + "
                  f"{intercept:.4f}")
            print(f"  R²: {stats['r_squared']:.4f}")
            print(f"  p-value: {p_value:.4g}")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "x", "s", "^", "D", "v", "*", "+"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_cols)))

    for i, (col, label) in enumerate(zip(metric_cols, labels)):
        marker = markers[i % len(markers)]
        ax.plot(
            df[x_col], np.round(df[col], 3),
            marker=marker, linestyle="", color=colors[i],
            label=f"{label} (Measured)",
        )

    # Regression lines
    x_min, x_max = df[x_col].min(), df[x_col].max()
    if use_log_scale:
        x_plot = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        x_reg_plot = np.log(x_plot)
    else:
        x_plot = np.linspace(x_min, x_max, 50)
        x_reg_plot = x_plot

    for i, (col, label) in enumerate(zip(metric_cols, labels)):
        s = regression_stats[col]
        y_line = s["intercept"] + s["slope"] * x_reg_plot
        ax.plot(
            x_plot, y_line,
            color=colors[i], linestyle="--", alpha=0.7,
            label=f"{label} Regression",
        )

    if title is None:
        log_text = "Log " if use_log_scale else ""
        title = f"Metrics vs. {x_col} with {log_text}Linear Regression"
    if xlabel is None:
        scale_type = "log scale" if use_log_scale else "linear scale"
        xlabel = f"{x_col} ({scale_type})"
    if ylabel is None:
        ylabel = metric_cols[0].replace("_", " ")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    _apply_log_ticks(ax, df, x_col, use_log_scale)
    fig.tight_layout()

    if filename is None:
        safe = metric_cols[0].replace("@", "_at_").replace("/", "_")
        log_suffix = "_log" if use_log_scale else ""
        filename = f"{x_col}_vs_{safe}{log_suffix}_regression.png"
    filename = filename.lstrip("/")

    _save_or_show(fig, filename, output_dir)
    return regression_stats


# ─────────────────────────────────────────────
# RAG Pipeline Builders
# ─────────────────────────────────────────────

def build_retrieval_pipeline(index_ref, reranker, cutoff):
    """BM25 + T5 reranker (retrieval-only, no generation).

    Args:
        index_ref: PyTerrier index reference.
        reranker: Reranker transformer (e.g. MonoT5ReRanker).
        cutoff: BM25 retrieval cutoff.

    Returns:
        Tuple of ``(bm25_baseline, retrieval_pipeline)``.
    """
    import pyterrier as pt

    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")
    bm25_retrieved = bm25 % cutoff

    pipeline = (
            bm25_retrieved
            >> get_text_getter(index_ref)
            >> reranker
    )
    return bm25_retrieved, pipeline


def build_rag_pipeline(retrieval_pipeline, reader, top_k):
    """Full RAG: Retrieve + Rerank → Concatenate top-k → Generate answer.

    Args:
        retrieval_pipeline: A PyTerrier retrieval pipeline.
        reader: A RAG reader transformer (e.g. pyterrier_rag Reader).
        top_k: Number of passages to concatenate for the generator.

    Returns:
        Complete RAG pipeline.
    """
    from pyterrier_rag.prompt import Concatenator

    concat = Concatenator(max_elements=top_k)
    rag_pipeline = retrieval_pipeline >> concat >> reader
    return rag_pipeline


def get_reader(live: bool):
    """Returns the RAG reader.

    Args:
        live: If False, returns a DummyReader (no model).
              If True, returns a pyterrier_rag Reader with HuggingFaceBackend.

    Returns:
        A PyTerrier Transformer that produces ``qanswer`` column.
    """
    if not live:
        import pyterrier as pt

        class DummyReader(pt.Transformer):
            def transform(self, inp):
                result = inp.copy()
                if "qcontext" in result.columns:
                    result["qanswer"] = result["qcontext"].apply(
                        lambda ctx: f"Dummy answer based on: {str(ctx)[:50]}"
                    )
                else:
                    result["qanswer"] = "no context available"
                return result

        print("Using DummyReader (test mode — no model loaded)")
        return DummyReader()

    model_id = constants.RAG_MODEL_NAME
    if getattr(constants, "OFFLINE_MODE", False):
        local_path = os.path.join(constants.MODELS_CACHE_DIR, model_id.split('/')[-1])
        if os.path.exists(local_path):
            model_id = local_path
            print(f"Loading offline HuggingFace model from local cache: {model_id}")
        else:
            print(f"Loading HF model {model_id} (offline mode env set, relying on HF cache)")
    else:
        print(f"Loading HuggingFace model: {model_id}")

    from pyterrier_rag import HuggingFaceBackend
    from pyterrier_rag.readers import Reader

    backend = HuggingFaceBackend(
        model_id,
        max_new_tokens=constants.RAG_MODEL_MAX_NEW_TOKENS,
        generation_args={"temperature": 0.1},
    )
    reader = Reader(backend, prompt=constants.RAG_PROMPT)
    return reader


# ─────────────────────────────────────────────
# Metadata & Result I/O
# ─────────────────────────────────────────────

def collect_experiment_metadata(
        experiment_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect metadata from constants for an experiment.

    Args:
        experiment_name: Experiment identifier.
        additional_metadata: Extra key-value pairs to include.

    Returns:
        Metadata dictionary.
    """
    metadata: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "data_path": constants.DATA_PATH,
        "data_split": constants.DATA_SPLIT,
        "dataset_name": constants.DATASET_NAME,
        "index_path": constants.INDEX_PATH,
        "reranker_type": constants.RERANKER_TYPE,
        "reranker_model_name": (
            constants.CROSS_ENCODER_MODEL_NAME
            if constants.RERANKER_TYPE == "cross_encoder"
            else constants.RERANKER_MODEL_NAME
        ),
    }

    if experiment_name == "experiment_1":
        metadata.update({
            "topic_limit": constants.EXPERIMENT_1_TOPIC_LIMIT,
            "c_values": constants.EXPERIMENT_1_C_VALUES,
            "t5_batch_size": constants.T5_BATCH_SIZE,
            "experiment_batch_size": constants.EXPERIMENT_BATCH_SIZE,
            "eval_metrics": [str(m) for m in constants.EXPERIMENT_1_EVAL_METRICS],
        })
    elif experiment_name == "experiment_2":
        metadata.update({
            "topic_limit": constants.EXPERIMENT_2_TOPIC_LIMIT,
            "n_rewrites": constants.EXPERIMENT_2_N_REWRITES,
            "retrieval_cutoff": constants.EXPERIMENT_2_RETRIEVAL_CUTOFF,
            "t5_batch_size": constants.T5_BATCH_SIZE,
            "experiment_batch_size": constants.EXPERIMENT_BATCH_SIZE,
            "eval_metrics": [str(m) for m in constants.EXPERIMENT_2_EVAL_METRICS],
            "llm_rewrites_per_query": constants.LLM_REWRITES_PER_QUERY,
            "llm_temperature": constants.LLM_TEMPERATURE,
            "llm_model_name": constants.LLM_MODEL_NAME,
        })
    elif experiment_name == "experiment_5":
        metadata.update({
            "topic_limit": constants.EXPERIMENT_5_TOPIC_LIMIT,
            "c_values": constants.EXPERIMENT_5_C_VALUES,
            "retrieval_metrics": [str(m) for m in constants.EXPERIMENT_5_RETRIEVAL_METRICS],
            "eval_metrics": [str(m) for m in constants.EXPERIMENT_5_EVAL_METRICS],
            "top_k": constants.EXPERIMENT_5_TOP_K,
            "offline_mode": constants.OFFLINE_MODE,
            "rag_model_name": constants.RAG_MODEL_NAME,
            "reranker_type": constants.RERANKER_TYPE,
            "reranker_model_name": constants.RERANKER_MODEL_NAME,
        })

    elif experiment_name == "experiment_6":
        metadata.update({
            "topic_limit": constants.EXPERIMENT_6_TOPIC_LIMIT,
            "retrieval_cutoff": constants.EXPERIMENT_6_RETRIEVAL_CUTOFF,
            "n_rewrites": constants.EXPERIMENT_6_N_REWRITES,
            "eval_metrics": [str(m) for m in constants.EXPERIMENT_6_EVAL_METRICS],
            "top_k": constants.EXPERIMENT_6_TOP_K,
            "offline_mode": constants.OFFLINE_MODE,
            "rag_model_name": constants.RAG_MODEL_NAME,
            "reranker_type": constants.RERANKER_TYPE,
            "reranker_model_name": constants.RERANKER_MODEL_NAME,
        })

    if additional_metadata:
        metadata.update(additional_metadata)

    return metadata


def save_experiment_results(
        results_df: Any,
        experiment_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        include_timestamp: bool = True,
) -> str:
    """Save experiment results to CSV with a companion metadata JSON.

    Args:
        results_df: DataFrame or PyTerrier result object.
        experiment_name: Experiment identifier.
        metadata: Metadata dict. If ``None``, auto-collected.
        output_dir: Output directory. Defaults to ``constants.DATA_PATH``.
        filename: Base filename (no extension).
        include_timestamp: Append timestamp to auto-generated filename.

    Returns:
        Path to the saved CSV file.
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH
    if metadata is None:
        metadata = collect_experiment_metadata(experiment_name)

    # Coerce to DataFrame
    if not isinstance(results_df, pd.DataFrame):
        if hasattr(results_df, "data"):
            results_df = results_df.data
        elif hasattr(results_df, "to_csv"):
            results_df = pd.DataFrame(results_df)
        else:
            results_df = pd.DataFrame(results_df)

    if filename is None:
        filename = f"{experiment_name}_results"
        if include_timestamp:
            filename = f"{filename}_{datetime.now():%Y%m%d_%H%M%S}"

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{filename}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    json_path = os.path.join(output_dir, f"{filename}_metadata.json")
    with open(json_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"Metadata saved to: {json_path}")

    return csv_path


def load_experiment_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load experiment metadata from a JSON file.

    Args:
        metadata_path: Path to the ``*_metadata.json`` file.

    Returns:
        Metadata dictionary.
    """
    with open(metadata_path, "r") as fh:
        return json.load(fh)


def find_experiment_files(
        experiment_name: str,
        output_dir: Optional[str] = None,
        latest: bool = True,
) -> Dict[str, Any]:
    """Locate CSV and metadata files for an experiment.

    Args:
        experiment_name: Experiment identifier.
        output_dir: Directory to search. Defaults to ``constants.DATA_PATH``.
        latest: Return only the most-recently-modified files.

    Returns:
        Dict with ``"csv"`` and ``"metadata"`` keys pointing to file paths.

    Raises:
        FileNotFoundError: If no CSV files are found.
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    csv_pattern = os.path.join(output_dir, f"{experiment_name}_results.csv")
    csv_files = glob.glob(csv_pattern)

    meta_pattern = os.path.join(
        output_dir, f"{experiment_name}_results_metadata.json"
    )
    meta_files = glob.glob(meta_pattern)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found matching {csv_pattern}"
        )

    if latest:
        csv_files.sort(key=os.path.getmtime, reverse=True)
        meta_files.sort(key=os.path.getmtime, reverse=True)
        return {
            "csv": csv_files[0] if csv_files else None,
            "metadata": meta_files[0] if meta_files else None,
        }

    return {"csv": csv_files, "metadata": meta_files}
