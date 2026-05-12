"""
Experiment 4 — ORE Adaptive Cutoff Sweep
=========================================
Measures how BM25 retrieval cutoff (c) affects Recall and nDCG
when using ORE Adaptive re-ranking vs BM25 baseline and MonoT5.

Sweeps ``constants.EXPERIMENT_4_C_VALUES`` (same as Experiment 1).
"""

import os
import sys

import pyterrier as pt
import pyterrier_alpha as pta
import torch
from pyterrier.measures import R
from pyterrier_adaptive import CorpusGraph, Laff
from pyterrier_dr import FlexIndex, TasB

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add ORE directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ORE")))

import constants
from code_base.experiment_runner import ExperimentRunner
from code_base.experiment_utils import get_text_getter, get_reranker
from code_base import dataset_factory
from ore_adaptive import OREAdaptive


# ─── Artifact Loader Helpers ────────────────────────────────
def _is_local_path(path: str) -> bool:
    """True when *path* refers to a local file/directory, not an HF artifact."""
    return path.startswith("./") or path.startswith("/") or os.path.exists(path)


def _load_flex_index(path: str) -> FlexIndex:
    if _is_local_path(path):
        return FlexIndex(path)
    return pta.Artifact.from_hf(path)


def _load_corpus_graph(path: str) -> CorpusGraph:
    if _is_local_path(path):
        return CorpusGraph.load(path)
    return pta.Artifact.from_hf(path)


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, reranker, dual_encoder,
                    corpus_index, graph, laff_graph, text_dict):
    """Construct retrieval pipelines for a given cutoff *c*.

    Args:
        c: BM25 retrieval cutoff / ORE budget.
        index_ref: PyTerrier index reference.
        reranker: Reranker instance.
        dual_encoder: TasB dual encoder model.
        corpus_index: FlexIndex for dense vectors.
        graph: Corpus graph for ORE.
        laff_graph: LAFF graph for ORE.
        text_dict: Mapping docno → text for BM25 scoring inside ORE.

    Returns:
        Dict mapping system names to pipelines.
    """
    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c

    pipeline_t5 = (
            bm25
            >> get_text_getter(index_ref)
            >> reranker
    )

    scorer = (
            get_text_getter(index_ref)
            >> reranker
    )

    pipeline_ore = (
            bm25
            >> OREAdaptive(
        dual_encoder, scorer, corpus_index, graph, laff_graph,
        budget=c, cross_enc_budget=constants.EXPERIMENT_4_CROSS_ENC_BUDGET,
        param_bounds=(0.25, 0.95), num_bm25_calls=0,
        verbose=False, top_s=constants.EXPERIMENT_4_TOP_S1,
        top_s2=constants.EXPERIMENT_4_TOP_S2,
        batch_size=constants.T5_BATCH_SIZE,
        text_dict=text_dict,
        bm25_index=index_ref,
    )
    )

    return {
        "BM25": bm25,
        "BM25_MonoT5": pipeline_t5,
        "ORE_Adaptive": pipeline_ore,
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    ("BM25_MonoT5", f"BM25 + {constants.RERANKER_NAME}"),
    ("ORE_Adaptive", "BM25 + ORE Adaptive"),
]

PLOT_CONFIGS = [
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Effect of Initial Cutoff (c) on Recall (R@10) — ORE",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "filename": "cutoff_vs_R_ore.png",
        "use_log_scale": True,
    },
    {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "title": "Effect of Initial Cutoff (c) on NDCG@10 — ORE",
        "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
        "ylabel": "NDCG@10",
        "filename": "cutoff_vs_NDCG_at_10_ore.png",
        "use_log_scale": True,
    },
    {
        "type": "regression",
        "x_col": "c",
        "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
        "labels": [lbl for _, lbl in SYSTEMS],
        "use_log_scale": True,
        "title": "R@10 vs. Initial Cutoff (c) Regression — ORE",
        "xlabel": "Initial Cutoff (c) (log scale)",
        "ylabel": "Recall (R@10)",
        "filename": "cutoff_R_at_10_log_regression_ore.png",
    },
]


# ─── Main ─────────────────────────────────────────────────

def main():
    print("Starting Experiment 4...")
    """Run Experiment 4."""

    runner = ExperimentRunner(
        name="experiment_4",
        results_subdir="exp4",
        topic_limit=constants.EXPERIMENT_4_TOPIC_LIMIT,
        eval_metrics=constants.EXPERIMENT_4_EVAL_METRICS,
        extra_metrics_fn=lambda c: [R @ c],
    )

    print("Setting up experiment...")
    results_dir = runner.setup()
    topics, qrels, _ = runner.load_data()

    # ── Build text_dict from corpus (needed by OREAdaptive) ──
    print("Loading corpus for text_dict...")
    docs_df, _, _ = dataset_factory.get_dataset(
        constants.DATASET_NAME, constants.DATA_SPLIT
    )
    if docs_df is not None:
        text_dict = dict(zip(docs_df["docno"], docs_df["text"]))
        del docs_df  # free memory
        print(f"  text_dict built: {len(text_dict)} entries")
    else:
        text_dict = {}
        print("  WARNING: docs_df is None — OREAdaptive BM25 scoring may fail")

    print("Loading PyTerrier index...")
    index_ref = pt.IndexFactory.of(constants.INDEX_PATH, memory=True)
    reranker = get_reranker()

    # ── ORE dependencies ──
    print("Loading ORE dependencies...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_encoder = TasB.dot(batch_size=1, device=device)

    print(f"  FlexIndex: {constants.EXPERIMENT_4_FLEX_INDEX_PATH}")
    corpus_index = _load_flex_index(constants.EXPERIMENT_4_FLEX_INDEX_PATH)

    print(f"  CorpusGraph: {constants.EXPERIMENT_4_CORPUS_GRAPH}")
    graph = _load_corpus_graph(constants.EXPERIMENT_4_CORPUS_GRAPH)

    # On-the-fly LAFF: precomputed graph was incomplete, compute affinity at query time
    print(f"  LAFF base graph: {constants.EXPERIMENT_4_LAFF_GRAPH}")
    base_laff_graph = _load_corpus_graph(
        constants.EXPERIMENT_4_LAFF_GRAPH
    ).to_limit_k(constants.EXPERIMENT_4_LAFF_K)
    text_loader = pt.apply.generic(lambda df: df.assign(text=df['docno'].map(text_dict).fillna('')))
    laff = Laff(device=device)

    class CachedGraph:
        def __init__(self, base_graph):
            self.base_graph = base_graph
            self.cache = {}

        def neighbours(self, docid, weights=False):
            key = (docid, weights)
            if key not in self.cache:
                self.cache[key] = self.base_graph.neighbours(docid, weights=weights)
            return self.cache[key]

    laff_graph = CachedGraph(laff.wrap_graph(base_laff_graph, text_loader))

    results_df = runner.run_sweep(
        sweep_values=constants.EXPERIMENT_4_C_VALUES,
        sweep_param="c",
        pipeline_factory=build_pipelines,
        topics=topics,
        qrels=qrels,
        factory_kwargs={
            "index_ref": index_ref,
            "reranker": reranker,
            "dual_encoder": dual_encoder,
            "corpus_index": corpus_index,
            "graph": graph,
            "laff_graph": laff_graph,
            "text_dict": text_dict,
        },
    )

    runner.save(results_df, results_dir)
    runner.plot(results_df, results_dir, PLOT_CONFIGS)


if __name__ == "__main__":
    main()
