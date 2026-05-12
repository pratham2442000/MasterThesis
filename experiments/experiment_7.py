"""
Experiment 7 — ReformIR Cutoff Sweep
======================================
Measures how BM25 retrieval cutoff (c) affects Recall and nDCG when
using ReformIR adaptive re-ranking (with LLM query reformulations and
RM3 pseudo-relevance feedback) vs baselines.

Systems compared:
  1. BM25 baseline
  2. BM25 + Reranker (MonoT5 / MiniLM-L6)
  3. BM25 + ReformRank  (query-expansion-only baseline)
  4. BM25 + ReformIR    (full adaptive reformulation + RM3 + lambda fitting)

Sweeps ``constants.EXPERIMENT_7_C_VALUES``.
"""

import ast
import os
import sys

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

from reformir import ReformIR
from baselines.genqr_ensemble_baseline import ReformRank


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
        reform_bm25,
):
    """Construct retrieval pipelines for a given cutoff *c*.

    Args:
        c: BM25 retrieval cutoff / ReformIR budget.
        index_ref: PyTerrier index reference (loaded into memory).
        reranker: Reranker instance.
        reformulation_path: Path to the reformulation CSV.
        reform_bm25: BM25 retriever used internally by ReformIR/ReformRank.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c

    scorer = get_text_getter(index_ref) >> reranker

    pipeline_reranker = bm25 >> scorer

    # pipeline_reformrank = (
    #     bm25
    #     >> ReformRank(
    #         scorer=scorer,
    #         bm25_retriever=reform_bm25,
    #         budget=c,
    #         batch_size=constants.T5_BATCH_SIZE,
    #         file_name=reformulation_path,
    #     )
    # )

    pipeline_reformir = (
            bm25
            >> ReformIR(
        scorer,
        num_results=c,
        cross_enc_budget=constants.EXPERIMENT_7_CROSS_ENC_BUDGET,
        top_s=constants.EXPERIMENT_7_TOP_S,
        batch_size=constants.T5_BATCH_SIZE,
        terrier_index=index_ref,
        bm25_retriever=reform_bm25,
        verbose=False,
        file_name=reformulation_path,
    )
    )

    return {
        "BM25": bm25,
        f"BM25_{constants.RERANKER_NAME}": pipeline_reranker,
        # "ReformRank": pipeline_reformrank,
        "ReformIR": pipeline_reformir,
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    (f"BM25_{constants.RERANKER_NAME}", f"BM25 + {constants.RERANKER_NAME}"),
    ("ReformIR", "ReformIR"),
]

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Effect of Initial Cutoff (c) on Recall (R@10) — ReformIR",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "filename": "cutoff_vs_R_reformir.png",
        "use_log_scale": True,
    },
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Effect of Initial Cutoff (c) on NDCG@10 — ReformIR",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "NDCG@10",
        "filename": "cutoff_vs_NDCG_at_10_reformir.png",
        "use_log_scale": True,
    },
    {
        "type": "regression",
        "x_col": "c",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "use_log_scale": True,
        "title": "R@10 vs. Initial Cutoff (c) Regression — ReformIR",
        "xlabel": "Initial Cutoff (c) (log scale)",
        "ylabel": "Recall (R@10)",
        "filename": "cutoff_R_at_10_log_regression_reformir.png",
    },
]


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

    # BM25 retriever used internally by ReformIR/ReformRank for reformulated queries.
    # Wrap with query sanitiser: colons in LLM rewrites crash Terrier's
    # field-based matching on indexes built without fields.
    class _SanitizeQuery(pt.Transformer):
        def transform(self, df):
            df = df.copy()
            df["query"] = df["query"].str.replace(":", " ", regex=False)
            return df

    _raw_bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25", num_results=100)
    reform_bm25 = _SanitizeQuery() >> _raw_bm25

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_7_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "reranker": reranker,
            "reformulation_path": reformulation_path,
            "reform_bm25": reform_bm25,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
