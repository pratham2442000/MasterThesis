"""
Experiment 7 — ReformIR Cutoff Sweep
======================================
Measures how BM25 retrieval cutoff (c) affects Recall and nDCG when
using ReformIR adaptive re-ranking (with LLM query reformulations and
RM3 pseudo-relevance feedback) vs baselines.

Systems compared:
  1. BM25 baseline
  2. BM25 + Reranker (MonoT5 / MiniLM-L6)
  3. BM25 + ReformIR  (full adaptive reformulation + RM3 + lambda fitting)

Two ce_modes are tested:
  - "linear": ce scales with c (full CE coverage, paper's design)
  - "fixed":  ce=7 always (tests lsq_linear approximation at large c)

Sweeps ``constants.EXPERIMENT_7_C_VALUES``.
"""

import ast
import os
import sys
import time

import pandas as pd
import pyterrier as pt
from pyterrier.measures import R

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add ReformIR code directory to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ReformIR-DC92", "code"))
)

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import get_text_getter, get_reranker
from code_base.reformir import ReformIR



# ─── Rewrite Format Adapter ─────────────────────────────────

def build_reformulation_csv(n_rewrites: int = 5) -> str:
    """Convert the user's rewrite CSV to ReformIR's expected format.

    Reads ``{DATASET_NAME}_rewritten_queries_dataset_{LLM_REWRITES_PER_QUERY}_{DATA_SPLIT}.csv``
    (columns: ``q_id``, ``rewritten_queries`` as Python list repr)
    and writes a temp CSV with columns ``qid``, ``query`` where
    rewrites are joined by ``--``.

    Args:
        n_rewrites: Number of rewrites to select per query.

    Returns:
        Path to the temporary reformulation CSV.
    """
    n_total = constants.LLM_REWRITES_PER_QUERY
    split = constants.DATA_SPLIT
    src_filename = (
        f"{constants.DATASET_NAME}_rewritten_queries_dataset_{n_total}_{split}.csv"
    )
    src_path = os.path.join(constants.DATA_PATH, src_filename)

    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Rewrite file not found: {src_path}\n"
            "Run the LLM rewrite pipeline first."
        )

    src_df = pd.read_csv(src_path)
    rows = []
    for _, row in src_df.iterrows():
        qid = str(row["q_id"])
        rewrites = ast.literal_eval(row["rewritten_queries"])
        selected = rewrites[:n_rewrites]
        # Strip colons — Terrier interprets ":" as field operators and
        # crashes with "Index must have 1 or more fields" if the index
        # was built without fields.
        selected = [q.replace(":", " ") for q in selected]
        rows.append({"qid": qid, "query": "--".join(selected)})

    out_df = pd.DataFrame(rows)

    # Write to a persistent temp file inside the workspace
    out_dir = os.path.join(
        constants.RESULTS_PATH, constants.DATASET_NAME, "exp7_cache"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reformir_rewrites.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Reformulation CSV written: {out_path}  ({len(out_df)} queries, {n_rewrites} rewrites each)")
    return out_path


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(
        c,
        *,
        index_ref,
        reranker,
        reformulation_path,
        ce_mode="linear",
):
    """Construct retrieval pipelines for a given cutoff *c*.

    ReformIR parameters are scaled per-c via ``constants.reformir_params_for_budget(c)``.
    batch_size is always REFORMIR_BATCH_SIZE (16, paper's value), NOT the
    scorer's inference batch_size (T5_BATCH_SIZE=64).

    Args:
        c: BM25 retrieval cutoff / ReformIR budget.
        index_ref: PyTerrier index reference (loaded into memory).
        reranker: Reranker instance.
        reformulation_path: Path to the reformulation CSV.
        ce_mode: "linear" (full coverage) or "fixed" (ce=7, partial).
    """
    bs = constants.REFORMIR_BATCH_SIZE  # 16 — paper's loop batch size
    params = constants.reformir_params_for_budget(c, batch_size=bs, ce_mode=ce_mode)
    real_scored = params["cross_enc_budget"] * bs
    print(
        f"  [c={c}, ce_mode={ce_mode}] ReformIR params: ce={params['cross_enc_budget']}, "
        f"batch_size={bs}, real_scored={real_scored}, "
        f"top_s={params['top_s']}, rm3_pool={params['reform_bm25_num_results']}"
    )

    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c
    scorer = get_text_getter(index_ref) >> reranker
    pipeline_reranker = bm25 >> scorer

    # Build reform_bm25 with num_results scaled to c so the RM3 expansion pool
    # is large enough. The original code hardcoded num_results=100 which becomes
    # a bottleneck at large c.
    class _SanitizeQuery(pt.Transformer):
        def transform(self, df):
            df = df.copy()
            df["query"] = df["query"].str.replace(":", " ", regex=False)
            return df

    _raw_bm25 = pt.terrier.Retriever(
        index_ref, wmodel="BM25",
        num_results=params["reform_bm25_num_results"]
    )
    reform_bm25 = _SanitizeQuery() >> _raw_bm25

    pipeline_reformir = (
            bm25
            >> ReformIR(
        scorer,
        num_results=c,
        cross_enc_budget=params["cross_enc_budget"],
        top_s=params["top_s"],
        batch_size=bs,  # ← paper's 16, NOT T5_BATCH_SIZE (64)
        terrier_index=index_ref,
        bm25_retriever=reform_bm25,
        verbose=False,
        file_name=reformulation_path,
    )
    )

    return {
        "BM25": bm25,
        f"BM25_{constants.RERANKER_NAME}": pipeline_reranker,
        "ReformIR": pipeline_reformir,
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    (f"BM25_{constants.RERANKER_NAME}", f"BM25 + {constants.RERANKER_NAME}"),
    ("ReformIR", "ReformIR"),
]


def _make_plot_configs(ce_mode: str) -> list:
    """Generate plot configs for a given ce_mode."""
    suffix = f"_{ce_mode}"
    return [
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
            "labels": [lbl for _, lbl in SYSTEMS],
            "title": f"Effect of Initial Cutoff (c) on Recall (R@10) — ReformIR [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "filename": f"cutoff_vs_R_reformir{suffix}.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
            "labels": [lbl for _, lbl in SYSTEMS],
            "title": f"Effect of Initial Cutoff (c) on NDCG@10 — ReformIR [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "NDCG@10",
            "filename": f"cutoff_vs_NDCG_at_10_reformir{suffix}.png",
            "use_log_scale": True,
        },
        {
            "type": "regression",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
            "labels": [lbl for _, lbl in SYSTEMS],
            "use_log_scale": True,
            "title": f"R@10 vs. Initial Cutoff (c) Regression — ReformIR [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) (log scale)",
            "ylabel": "Recall (R@10)",
            "filename": f"cutoff_R_at_10_log_regression_reformir{suffix}.png",
        },
    ]


def _make_timing_plot_config(ce_mode: str) -> dict:
    """Plot config for wall-clock timing per sweep step."""
    return {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["wall_clock_s"],
        "labels": ["Wall-clock (s)"],
        "title": f"Wall-clock Time per Cutoff (c) — ReformIR [{ce_mode}]",
        "xlabel": "Initial Cutoff (c)",
        "ylabel": "Seconds",
        "filename": f"cutoff_vs_time_reformir_{ce_mode}.png",
        "use_log_scale": True,
    }


# ─── Timed Sweep ──────────────────────────────────────────

def run_timed_sweep(runner, sweep_values, topics, qrels, factory_kwargs, ce_mode):
    """Wrap runner.run_sweep with per-step wall-clock timing.

    Returns results_df with an extra 'wall_clock_s' column.
    """
    all_rows = []

    print(f"\n{'═' * 50}")
    print(f"  ce_mode = {ce_mode}")
    print(f"{'═' * 50}")

    for val in sweep_values:
        t0 = time.time()

        # Build pipelines for this (c, ce_mode)
        kwargs = dict(factory_kwargs, ce_mode=ce_mode)
        systems_dict = build_pipelines(val, **kwargs)

        current_metrics = list(runner.eval_metrics)
        if runner.extra_metrics_fn is not None:
            current_metrics += runner.extra_metrics_fn(val)

        print(f"\n{'─' * 40}")
        print(f"Running c={val}, ce_mode={ce_mode} ...")

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

        elapsed = time.time() - t0

        row = runner._collect_row(exp_df, current_metrics, "c", val)
        row["wall_clock_s"] = round(elapsed, 2)
        row["ce_mode"] = ce_mode
        all_rows.append(row)
        runner._print_summary(row)

    return pd.DataFrame(all_rows)


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 7."""
    print("Starting Experiment 7 (ReformIR)...")

    runner = ExperimentRunner(
        name="experiment_7",
        results_subdir="exp7",
        topic_limit=constants.EXPERIMENT_7_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_7_EVAL_METRICS,
        extra_metrics_fn=lambda c: [R @ c],
    )

    results_dir = runner.setup()
    topics, qrels, _ = runner.load_data(filter_by_rewrites=True)

    # ── Build reformulation CSV from user's rewrites ──
    reformulation_path = build_reformulation_csv(
        n_rewrites=constants.EXPERIMENT_7_N_REWRITES
    )

    # ── Load index + reranker ──
    print("Loading PyTerrier index...")
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
    reranker = get_reranker()

    factory_kwargs = {
        "index_ref": index_ref,
        "reranker": reranker,
        "reformulation_path": reformulation_path,
    }

    # ── Run both ce_modes ──
    all_results = []
    for ce_mode in ("linear", "fixed"):
        results_df = run_timed_sweep(
            runner,
            sweep_values=constants.EXPERIMENT_7_C_VALUES,
            topics=topics,
            qrels=qrels,
            factory_kwargs=factory_kwargs,
            ce_mode=ce_mode,
        )
        all_results.append(results_df)

        # Save and plot per-mode
        mode_dir = os.path.join(results_dir, ce_mode)
        os.makedirs(mode_dir, exist_ok=True)
        runner.save(results_df, mode_dir, filename=f"experiment_7_results_{ce_mode}")
        runner.plot(results_df, mode_dir, _make_plot_configs(ce_mode))
        runner.plot(results_df, mode_dir, [_make_timing_plot_config(ce_mode)])

    # ── Combined results ──
    combined_df = pd.concat(all_results, ignore_index=True)
    runner.save(combined_df, results_dir, filename="experiment_7_results_combined")

    print("\nExperiment 7 complete. Both ce_modes saved.")


if __name__ == "__main__":
    main()
