"""
Experiment Utilities Module
===========================
Reusable utilities for experiments including:
- Plotting functions for visualizing results
- Metadata storage and retrieval for experiment results
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

import constants


def preprocess_query(query: str) -> str:
    query = "".join(e for e in query if e.isalnum() or e.isspace())
    return query.lower()


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
):
    """
    Generates and saves a plot for metrics over a variable.

    Args:
        df: DataFrame with results
        x_col: Column name for x-axis (e.g., 'c', 'cutoff')
        metric_cols: List of column names to plot
        labels: Optional list of labels for each metric column.
            If None, uses column names.
        title: Plot title. If None, generates from metric names.
        xlabel: X-axis label. If None, uses x_col.
        ylabel: Y-axis label. If None, uses first metric name.
        filename: Output filename. If None, generates from metric names.
        output_dir: Directory to save plot. If None, uses constants.DATA_PATH.
        use_log_scale: Whether to use log scale for x-axis. If None,
            auto-detects.
        figsize: Figure size tuple (width, height).
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    if labels is None:
        labels = metric_cols
    elif len(labels) != len(metric_cols):
        labels = metric_cols  # Fallback if length mismatch

    if title is None:
        title = f"Metrics vs. {x_col}"
    if xlabel is None:
        xlabel = x_col.replace("_", " ").title()
    if ylabel is None:
        ylabel = metric_cols[0].replace("_", " ")

    plt.figure(figsize=figsize)

    markers = ["o", "x", "s", "^", "D", "v", "*", "+"]
    for i, (metric_col, label) in enumerate(zip(metric_cols, labels)):
        marker = markers[i % len(markers)]
        plt.plot(df[x_col], np.round(df[metric_col], 3), marker=marker, label=label, alpha=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Auto-detect log scale if not specified
    if use_log_scale is None:
        if len(df[x_col]) > 2 and (df[x_col].max() / df[x_col].min() > 5):
            use_log_scale = True
        else:
            use_log_scale = False

    if use_log_scale:
        plt.xscale("log")
        plt.xticks(df[x_col], labels=[str(v) for v in df[x_col]])

    plt.tight_layout()

    if filename is None:
        safe_metric = metric_cols[0].replace("@", "_at_").replace("/", "_")
        filename = f"/{x_col}_vs_{safe_metric}.png"

    if not filename.startswith("/"):
        filename = "/" + filename

    filepath = output_dir + filename
    plt.savefig(filepath)
    print(f"Chart saved as {filepath}")
    plt.close()


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
    """
    Plots metrics with linear regression lines (using log(x) if specified).

    Args:
        df: DataFrame with results
        x_col: Column name for x-axis
        metric_cols: List of column names to plot
        labels: Optional list of labels for each metric column
        use_log_scale: Whether to use log(x) for regression
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
        output_dir: Directory to save plot
        figsize: Figure size tuple
        print_stats: Whether to print regression statistics

    Returns:
        Dictionary with regression statistics for each metric
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    if labels is None:
        labels = metric_cols
    elif len(labels) != len(metric_cols):
        labels = metric_cols

    # Prepare x values for regression
    if use_log_scale:
        x_regression = np.log(df[x_col])
        x_regression_label = f"log({x_col})"
    else:
        x_regression = df[x_col]
        x_regression_label = x_col

    # Perform regression for each metric
    regression_stats = {}
    for metric_col, label in zip(metric_cols, labels):
        y = df[metric_col]
        slope, intercept, r_value, p_value, std_err = linregress(x_regression, y)
        r_squared = r_value ** 2

        regression_stats[metric_col] = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "p_value": p_value,
            "std_err": std_err,
        }

        if print_stats:
            print(f"\n{label}:")
            eq_str = (
                f"  Equation: {metric_col} = {slope:.4f} * "
                f"{x_regression_label} + {intercept:.4f}"
            )
            print(eq_str)
            print(f"  R-squared: {r_squared:.4f}")
            print(f"  P-value: {p_value:.4g}")

    # Create plot
    plt.figure(figsize=figsize)

    markers = ["o", "x", "s", "^", "D", "v", "*", "+"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_cols)))

    # Plot measured data points
    for i, (metric_col, label) in enumerate(zip(metric_cols, labels)):
        marker = markers[i % len(markers)]
        plt.plot(
            df[x_col],
            np.round(df[metric_col], 3),
            marker=marker,
            linestyle="",
            label=f"{label} (Measured)",
            color=colors[i],
        )

    # Plot regression lines
    x_min, x_max = df[x_col].min(), df[x_col].max()
    if use_log_scale:
        x_values_for_plot = np.logspace(np.log10(x_min), np.log10(x_max), 50)
        x_regression_for_plot = np.log(x_values_for_plot)
    else:
        x_values_for_plot = np.linspace(x_min, x_max, 50)
        x_regression_for_plot = x_values_for_plot

    for i, (metric_col, label) in enumerate(zip(metric_cols, labels)):
        stats = regression_stats[metric_col]
        y_regression = stats["intercept"] + stats["slope"] * x_regression_for_plot
        plt.plot(
            x_values_for_plot,
            y_regression,
            color=colors[i],
            linestyle="--",
            alpha=0.7,
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

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    if use_log_scale:
        plt.xscale("log")
        plt.xticks(df[x_col], labels=[str(v) for v in df[x_col]])
    plt.tight_layout()

    if filename is None:
        safe_metric = metric_cols[0].replace("@", "_at_").replace("/", "_")
        log_suffix = "_log" if use_log_scale else ""
        filename = f"/{x_col}_vs_{safe_metric}{log_suffix}_regression.png"

    if not filename.startswith("/"):
        filename = "/" + filename

    filepath = output_dir + filename
    plt.savefig(filepath)
    print(f"Chart saved as {filepath}")
    plt.close()

    return regression_stats


def collect_experiment_metadata(
        experiment_name: str,
        additional_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collects metadata from constants and additional sources for an experiment.

    Args:
        experiment_name: Name of the experiment
            (e.g., 'experiment_1', 'experiment_2')
        additional_metadata: Optional dictionary with additional metadata
            to include

    Returns:
        Dictionary with all metadata
    """
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "data_path": constants.DATA_PATH,
        "data_split": constants.DATA_SPLIT,
        "dataset_name": constants.DATASET_NAME,
        "index_path": constants.INDEX_PATH,
    }

    # Add experiment-specific constants
    if experiment_name == "experiment_1":
        metadata.update(
            {
                "topic_limit": constants.EXPERIMENT_1_TOPIC_LIMIT,
                "c_values": constants.EXPERIMENT_1_C_VALUES,
                "t5_batch_size": constants.T5_BATCH_SIZE,
                "experiment_batch_size": constants.EXPERIMENT_BATCH_SIZE,
                "eval_metrics": constants.EXPERIMENT_1_EVAL_METRICS,
            }
        )
    elif experiment_name == "experiment_2":
        metadata.update(
            {
                "topic_limit": constants.EXPERIMENT_2_TOPIC_LIMIT,
                "n_rewrites": constants.EXPERIMENT_2_N_REWRITES,
                "retrieval_cutoff": constants.EXPERIMENT_2_RETRIEVAL_CUTOFF,
                "recall_c": constants.EXPERIMENT_1_C_VALUES,
                "t5_batch_size": constants.T5_BATCH_SIZE,
                "experiment_batch_size": constants.EXPERIMENT_BATCH_SIZE,
                "eval_metrics": constants.EXPERIMENT_2_EVAL_METRICS,
                "llm_rewrites_per_query": constants.LLM_REWRITES_PER_QUERY,
                "llm_temperature": constants.LLM_TEMPERATURE,
                "llm_model_name": constants.LLM_MODEL_NAME,
            }
        )

    # Add any additional metadata
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
    """
    Saves experiment results with metadata to CSV and JSON files.

    Args:
        results_df: DataFrame or PyTerrier Experiment results with
            experiment results
        experiment_name: Name of the experiment
        metadata: Optional metadata dictionary. If None, collects from
            constants.
        output_dir: Directory to save files. If None, uses
            constants.DATA_PATH.
        filename: Base filename (without extension). If None, uses
            experiment_name.
        include_timestamp: Whether to include timestamp in filename.

    Returns:
        Path to the saved CSV file
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    if metadata is None:
        metadata = collect_experiment_metadata(experiment_name)

    # Convert PyTerrier results to DataFrame if needed
    if not isinstance(results_df, pd.DataFrame):
        if hasattr(results_df, "to_csv"):
            # It's a Styler or similar, get the underlying DataFrame
            if hasattr(results_df, "data"):
                results_df = results_df.data
            else:
                results_df = pd.DataFrame(results_df)
        else:
            results_df = pd.DataFrame(results_df)

    if filename is None:
        filename = f"{experiment_name}_results"
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

    # Save results CSV
    csv_path = os.path.join(output_dir, f"{filename}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Save metadata JSON
    json_path = os.path.join(output_dir, f"{filename}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {json_path}")

    return csv_path


def load_experiment_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Loads experiment metadata from a JSON file.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        Dictionary with metadata
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def find_experiment_files(
        experiment_name: str,
        output_dir: Optional[str] = None,
        latest: bool = True, ):
    """
    Finds CSV and metadata files for an experiment.

    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to search. If None, uses constants.DATA_PATH.
        latest: If True, returns only the latest files. If False, returns
            all matches.

    Returns:
        Dictionary with keys 'csv' and 'metadata' containing file paths,
        or None if not found.
    """
    if output_dir is None:
        output_dir = constants.DATA_PATH

    import glob

    # Find CSV files
    csv_pattern = os.path.join(output_dir, f"{experiment_name}_results*.csv")
    csv_files = glob.glob(csv_pattern)

    # Find metadata files
    metadata_pattern = os.path.join(
        output_dir, f"{experiment_name}_results*_metadata.json"
    )
    metadata_files = glob.glob(metadata_pattern)

    if not csv_files:
        raise FileNotFoundError("No CSV files found for the experiment.")

    if latest:
        # Sort by modification time and return latest
        csv_files.sort(key=os.path.getmtime, reverse=True)
        metadata_files.sort(key=os.path.getmtime, reverse=True)
        return {
            "csv": csv_files[0] if csv_files else None,
            "metadata": metadata_files[0] if metadata_files else None,
        }
    else:
        return {
            "csv": csv_files,
            "metadata": metadata_files,
        }
