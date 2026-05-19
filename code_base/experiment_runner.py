"""
Experiment Runner
=================
Unified harness for running PyTerrier IR experiments.

Eliminates boilerplate repeated across experiment scripts:
directory creation, data loading, sweep loops, result collection,
saving, and plotting.

Usage
-----
::

    runner = ExperimentRunner(
        name="experiment_1",
        results_subdir="experiment_results/experiment_1",
        topic_limit=constants.EXPERIMENT_1_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_1_EVAL_METRICS,
    )
    results_dir = runner.setup()
    topics, qrels, query_lookup = runner.load_data(filter_by_rewrites=False)
    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_1_C_VALUES,
        sweep_param="c",
        pipeline_factory=my_pipeline_factory,
        topics=topics,
        qrels=qrels,
    )
    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, plot_configs)
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pyterrier as pt

import constants
from code_base import dataset_factory
from code_base.experiment_utils import (
    plot_metrics,
    plot_with_regression,
    save_experiment_results,
)
from code_base.transformers import preprocess_query


class ExperimentRunner:
    """Unified experiment execution harness.

    Args:
        name: Experiment identifier (e.g. ``"experiment_1"``).
        results_subdir: Sub-path under ``constants.RESULTS_PATH`` for
            this experiment's output.
        topic_limit: Maximum number of topics to use.
        eval_metrics: List of PyTerrier measure objects (e.g. ``[R@10, nDCG@10]``).
        extra_metrics_fn: Optional callable ``(sweep_value) -> [Measure]``
            to append dynamic metrics per sweep step.
    """

    def __init__(
            self,
            name: str,
            results_subdir: str,
            topic_limit: int,
            eval_metrics: list,
            extra_metrics_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.results_subdir = results_subdir
        self.topic_limit = topic_limit
        self.eval_metrics = list(eval_metrics)
        self.extra_metrics_fn = extra_metrics_fn
        self._start_time = datetime.now()

    # ── Setup ────────────────────────────────────────────────

    def setup(self) -> str:
        """Create results directory and return its path.

        Path structure: ``RESULTS_PATH / DATASET_NAME / results_subdir / run_YYYYMMDD_HHMMSS``

        Each run gets unique timestamped directory.
        Results are grouped by dataset automatically.

        Returns:
            Absolute path to the results directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            constants.RESULTS_PATH,
            constants.DATASET_NAME,
            self.results_subdir,
            f"run_{timestamp}",
        )
        os.makedirs(results_path, exist_ok=True)
        print(f"Results directory ensured at: {results_path}")
        return results_path

    # ── Data Loading ─────────────────────────────────────────

    def load_data(
            self,
            filter_by_rewrites: bool = False,
            use_tail: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Load and preprocess dataset.

        Args:
            filter_by_rewrites: If True, keep only topics that have
                precomputed LLM rewrites available.
            use_tail: If True, use ``tail(topic_limit)`` instead of
                ``head(topic_limit)`` for topic selection.

        Returns:
            Tuple of ``(topics_df, qrels_df, query_lookup)``.
        """
        print("Loading data...")
        _, topics_df, qrels_df = dataset_factory.get_dataset(
            constants.DATASET_NAME, constants.DATA_SPLIT
        )

        topics_df["query"] = topics_df["query"].apply(preprocess_query)

        if filter_by_rewrites:
            topics_df = self._filter_by_rewrites(topics_df)

        if use_tail:
            topics = topics_df.tail(self.topic_limit)
        else:
            topics = topics_df.head(self.topic_limit)

        query_lookup = {
            str(row["qid"]): row["query"] for _, row in topics.iterrows()
        }

        print(f"Topics loaded: {len(topics)}")
        return topics, qrels_df, query_lookup

    # ── Sweep Loop ───────────────────────────────────────────

    def run_sweep(
            self,
            sweep_values: list,
            sweep_param: str,
            pipeline_factory: Callable[..., Dict[str, Any]],
            topics: pd.DataFrame,
            qrels: pd.DataFrame,
            factory_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Run a parameter sweep over ``sweep_values``.

        For each value, calls ``pipeline_factory(value, **factory_kwargs)``
        which must return ``{system_name: pt.Transformer}``.  Then runs
        ``pt.Experiment`` and collects results.

        Args:
            sweep_values: Values to iterate over.
            sweep_param: Name of the swept parameter (used as column name).
            pipeline_factory: Callable returning ``{name: pipeline}`` dict.
            topics: Topic DataFrame.
            qrels: Qrels DataFrame.
            factory_kwargs: Extra keyword arguments passed to factory.

        Returns:
            DataFrame with one row per sweep value, columns for each
            ``{system}_{metric}``.
        """
        if factory_kwargs is None:
            factory_kwargs = {}

        all_rows: list[dict] = []

        print(f"\nStarting sweep over {sweep_param}={sweep_values}, "
              f"topics={len(topics)}")

        for val in sweep_values:
            print(f"\n{'─' * 40}")
            print(f"Running {sweep_param}={val} ...")

            systems_dict = pipeline_factory(val, **factory_kwargs)

            current_metrics = list(self.eval_metrics)
            if self.extra_metrics_fn is not None:
                current_metrics += self.extra_metrics_fn(val)

            exp_df = pt.Experiment(
                list(systems_dict.values()),
                topics,
                qrels,
                eval_metrics=current_metrics,
                names=list(systems_dict.keys()),
                verbose=True,
                batch_size=constants.EXPERIMENT_BATCH_SIZE,
                precompute_prefix=True,
            )

            row = self._collect_row(
                exp_df, current_metrics, sweep_param, val
            )
            all_rows.append(row)
            self._print_summary(row)

        return pd.DataFrame(all_rows)

    # ── Save ─────────────────────────────────────────────────

    def save(
            self,
            results_df: pd.DataFrame,
            output_dir: str,
            filename: Optional[str] = None,
    ) -> str:
        """Save results CSV and metadata JSON.

        Args:
            results_df: Aggregated results DataFrame.
            output_dir: Directory to save files.
            filename: Base filename (no extension). Defaults to
                ``"{name}_results"``.

        Returns:
            Path to saved CSV file.
        """
        if filename is None:
            filename = f"{self.name}_results"

        print("\nFinal Results DataFrame:")
        print(results_df)

        self._end_time = datetime.now()
        execution_time = self._end_time - self._start_time

        return save_experiment_results(
            results_df,
            experiment_name=self.name,
            output_dir=output_dir,
            filename=filename,
            execution_time=execution_time,
        )

    # ── Plotting ─────────────────────────────────────────────

    def plot(
            self,
            results_df: pd.DataFrame,
            output_dir: str,
            plot_configs: List[dict],
    ) -> None:
        """Generate plots from a list of configuration dicts.

        Each config dict is passed to ``plot_metrics`` or
        ``plot_with_regression`` depending on the ``"type"`` key:

        - ``"type": "metrics"`` → ``plot_metrics(**config)``
        - ``"type": "regression"`` → ``plot_with_regression(**config)``

        The DataFrame and ``output_dir`` are injected automatically.

        Args:
            results_df: Results DataFrame.
            output_dir: Directory for plot output.
            plot_configs: List of plot configuration dicts.
        """
        for cfg in plot_configs:
            cfg = dict(cfg)  # copy
            plot_type = cfg.pop("type", "metrics")
            cfg["df"] = results_df
            cfg["output_dir"] = output_dir

            if plot_type == "regression":
                plot_with_regression(**cfg)
            else:
                plot_metrics(**cfg)

    # ── Private Helpers ──────────────────────────────────────

    def _filter_by_rewrites(self, topics_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only topics with precomputed LLM rewrites."""
        filename = (
            f"{constants.DATASET_NAME}_rewritten_queries_dataset_"
            f"{constants.LLM_REWRITES_PER_QUERY}_{constants.DATA_SPLIT}.csv"
        )
        filepath = os.path.join(constants.DATA_PATH, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Rewrites file not found: {filepath}")

        rewritten = pd.read_csv(filepath)
        filtered = topics_df[
            topics_df["qid"].astype(str).isin(rewritten["q_id"].astype(str))
        ].reset_index(drop=True)
        return filtered

    @staticmethod
    def _collect_row(
            exp_df: pd.DataFrame,
            eval_metrics: list,
            sweep_param: str,
            sweep_value: Any,
    ) -> dict:
        """Extract a single result row from ``pt.Experiment`` output.

        Maps metric columns like ``R@{c}`` to a generic ``_R@c`` key to
        keep CSV columns consistent across sweep values.

        Args:
            exp_df: Output of ``pt.Experiment``.
            eval_metrics: Metric objects used in the experiment.
            sweep_param: Name of the swept parameter.
            sweep_value: Current sweep value.

        Returns:
            Dict with ``{sweep_param: value, system_metric: score, …}``.
        """
        row: dict = {sweep_param: sweep_value}
        for _, exp_row in exp_df.iterrows():
            system_name = exp_row["name"]
            for metric in eval_metrics:
                metric_str = str(metric)
                val = exp_row[metric_str]
                key = f"{system_name}_{metric_str}"
                row[key] = val

        return row

    @staticmethod
    def _print_summary(row: dict) -> None:
        """Print a compact summary of one sweep row."""
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in row.items()]
        print("  " + ", ".join(parts))
