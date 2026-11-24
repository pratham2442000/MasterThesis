"""
Experiment 4 — ORE Adaptive Cutoff Sweep
==========================================
Measures how BM25 retrieval cutoff (c) affects Recall and nDCG
when using ORE Adaptive re-ranking vs baselines.

Systems:
  - BM25 (baseline)
  - BM25 + Reranker
  - ORE Adaptive (TasB dual encoder + corpus graph + LAFF)

Two ce_modes are tested:
  - "linear": ce scales with c (full CE coverage, isolates graph navigation)
  - "fixed":  ce=7 always (tests λ-regression approximation at large c)

Sweeps ``constants.EXPERIMENT_4_C_VALUES``.

ORE hyperparameters are scaled per-c via ``constants.ore_params_for_budget(c)``:
  - cross_enc_budget: scales with ce_mode (see ORE/ORE_PARAMS.md)
  - top_s:  scales as max(25, c // 4)  — cheap LAFF lookup, improves navigation
  - top_s2: scales as max(15, c // 6)  — cheap heap sort, improves frontier expansion

Corpus graph is loaded from the dataset-specific path configured in
  constants._DATASET_CONFIGS (local path or HuggingFace artifact).
"""

import os
import sys
import time

import pandas as pd
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
from code_base.ore_adaptive import OREAdaptive


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


def _try_load_precomputed_laff(laff_graph_path: str):
    """Try loading precomputed LAFF graph. Returns CorpusGraph or None."""
    try:
        graph = _load_corpus_graph(laff_graph_path)
        print(f"  Loaded precomputed LAFF graph from {laff_graph_path}")
        return graph
    except Exception as e:
        print(f"  Precomputed LAFF not available ({e}), will compute on-the-fly")
        return None


def _load_base_corpus_graph() -> CorpusGraph:
    """Load the base corpus graph from the dataset-specific path.

    ``EXPERIMENT_4_CORPUS_GRAPH_LOCAL`` is derived from the active dataset's
    ``CORPUS_GRAPH_PATH`` in constants.  It may be:
      - A local file path (wiki_musique, dpr-w100 when pre-built)
      - A HuggingFace artifact ID (msmarco-passage, dpr-w100 fallback)

    The function detects which case applies and loads accordingly.
    """
    path = constants.EXPERIMENT_4_CORPUS_GRAPH_LOCAL
    if not path:
        raise ValueError(
            f"No corpus graph configured for dataset '{constants.DATASET_NAME}'. "
            "Set CORPUS_GRAPH_PATH in _DATASET_CONFIGS."
        )

    if _is_local_path(path):
        # Resolve relative paths
        resolved = path if os.path.isabs(path) else os.path.join(".", path.lstrip("./"))
        if os.path.exists(resolved):
            print(f"  CorpusGraph: loading local {path}")
            return CorpusGraph.load(resolved)
        # Local path configured but not yet built — error clearly
        raise FileNotFoundError(
            f"Local corpus graph not found at '{resolved}'. "
            "Run the graph build script first, or set an HF artifact path "
            "in _DATASET_CONFIGS."
        )

    # HF artifact ID (e.g. macavaney/msmarco-passage.corpusgraph.bm25.16)
    print(f"  CorpusGraph: loading from HF → {path}")
    graph = pta.Artifact.from_hf(path)
    print("  CorpusGraph: HF artifact loaded")
    return graph


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, reranker, dual_encoder,
                    corpus_index, graph, laff_graph, text_dict,
                    ce_mode="linear"):
    """Construct retrieval pipelines for a given cutoff *c*.

    ORE parameters are scaled per-c via ``constants.ore_params_for_budget(c)``.

    Args:
        c: BM25 retrieval cutoff / ORE budget.
        index_ref: PyTerrier index reference.
        reranker: Reranker instance.
        dual_encoder: TasB dual encoder model.
        corpus_index: FlexIndex for dense vectors.
        graph: Corpus graph for ORE Adaptive.
        laff_graph: LAFF graph for ORE (precomputed or wrapped).
        text_dict: Mapping docno → text for BM25 scoring inside ORE.
        ce_mode: "linear" (full coverage) or "fixed" (ce=7, partial).

    Returns:
        Dict mapping system names to pipelines.
    """
    bs = constants.ORE_BATCH_SIZE  # 16 — paper's loop batch size
    ore_params = constants.ore_params_for_budget(c, batch_size=bs, ce_mode=ce_mode)
    real_scored = ore_params["cross_enc_budget"] * bs
    print(
        f" \n\n [c={c}, ce_mode={ce_mode}] ORE params: ce={ore_params['cross_enc_budget']}, "
        f"batch_size={bs}, real_scored={real_scored}, "
        f"top_s={ore_params['top_s']}, top_s2={ore_params['top_s2']}"
    )

    bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25") % c

    scorer = (
            get_text_getter(index_ref)
            >> reranker
    )

    pipeline_reranker = bm25 >> scorer

    pipeline_ore = (
            bm25
            >> OREAdaptive(
        dual_encoder, scorer, corpus_index, graph, laff_graph,
        budget=c,
        cross_enc_budget=ore_params["cross_enc_budget"],
        top_s=ore_params["top_s"],
        top_s2=ore_params["top_s2"],
        param_bounds=(0.25, 0.95),
        num_bm25_calls=0,
        verbose=False,
        batch_size=bs,
        text_dict=text_dict,
        bm25_index=index_ref,
    )
    )

    return {
        "BM25": bm25,
        f"BM25_{constants.RERANKER_NAME}": pipeline_reranker,
        "ORE_Adaptive": pipeline_ore,
    }


# ─── Plot Configuration ──────────────────────────────────

SYSTEMS = [
    ("BM25", "BM25 Baseline"),
    (f"BM25_{constants.RERANKER_NAME}", f"BM25 + {constants.RERANKER_NAME}"),
    ("ORE_Adaptive", "ORE Adaptive"),
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
            "title": f"Effect of Initial Cutoff (c) on Recall (R@10) — ORE [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "filename": f"cutoff_vs_R_ore{suffix}.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in SYSTEMS],
            "labels": [lbl for _, lbl in SYSTEMS],
            "title": f"Effect of Initial Cutoff (c) on NDCG@10 — ORE [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "NDCG@10",
            "filename": f"cutoff_vs_NDCG_at_10_ore{suffix}.png",
            "use_log_scale": True,
        },
        {
            "type": "regression",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in SYSTEMS],
            "labels": [lbl for _, lbl in SYSTEMS],
            "use_log_scale": True,
            "title": f"R@10 vs. Initial Cutoff (c) Regression — ORE [{ce_mode}]",
            "xlabel": "Initial Cutoff (c) (log scale)",
            "ylabel": "Recall (R@10)",
            "filename": f"cutoff_R_at_10_log_regression_ore{suffix}.png",
        },
    ]


def _make_timing_plot_config(ce_mode: str) -> dict:
    """Plot config for wall-clock timing per sweep step."""
    return {
        "type": "metrics",
        "x_col": "c",
        "metric_cols": ["wall_clock_s"],
        "labels": ["Wall-clock (s)"],
        "title": f"Wall-clock Time per Cutoff (c) — ORE [{ce_mode}]",
        "xlabel": "Initial Cutoff (c)",
        "ylabel": "Seconds",
        "filename": f"cutoff_vs_time_ore_{ce_mode}.png",
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
    """Run Experiment 4 — ORE Adaptive cutoff sweep."""
    print("Starting Experiment 4 (ORE Adaptive)...")

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

    # ── ORE Adaptive dependencies ──
    print("Loading ORE Adaptive dependencies...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_encoder = TasB.dot(batch_size=1, device=device)

    print(f"  FlexIndex: {constants.EXPERIMENT_4_FLEX_INDEX_PATH}")
    corpus_index = _load_flex_index(constants.EXPERIMENT_4_FLEX_INDEX_PATH)

    graph = _load_base_corpus_graph()

    # ── LAFF graph: try precomputed first, fall back to on-the-fly ──
    print("Loading LAFF graph...")
    laff_graph = _try_load_precomputed_laff(constants.LAFF_GRAPH_PATH)

    if laff_graph is not None:
        # Precomputed LAFF loaded — limit k and use directly
        laff_graph = laff_graph.to_limit_k(constants.EXPERIMENT_4_LAFF_K)
    else:
        # Fall back to on-the-fly LAFF computation
        print(f"  Building on-the-fly LAFF from base: {constants.EXPERIMENT_4_LAFF_GRAPH}")
        base_laff_graph = _load_corpus_graph(
            constants.EXPERIMENT_4_LAFF_GRAPH
        ).to_limit_k(constants.EXPERIMENT_4_LAFF_K)
        text_loader = pt.apply.generic(
            lambda df: df.assign(text=df['docno'].map(text_dict).fillna(''))
        )
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

            def to_limit_k(self, k):
                """Pass-through — on-the-fly graph is already limited."""
                return self

        laff_graph = CachedGraph(laff.wrap_graph(base_laff_graph, text_loader))

    factory_kwargs = {
        "index_ref": index_ref,
        "reranker": reranker,
        "dual_encoder": dual_encoder,
        "corpus_index": corpus_index,
        "graph": graph,
        "laff_graph": laff_graph,
        "text_dict": text_dict,
    }

    # ── Run both ce_modes ──
    all_results = []
    for ce_mode in ("linear", "fixed"):
        results_df = run_timed_sweep(
            runner,
            sweep_values=constants.EXPERIMENT_4_C_VALUES,
            topics=topics,
            qrels=qrels,
            factory_kwargs=factory_kwargs,
            ce_mode=ce_mode,
        )
        all_results.append(results_df)

        # Save and plot per-mode
        mode_dir = os.path.join(results_dir, ce_mode)
        os.makedirs(mode_dir, exist_ok=True)
        runner.save(results_df, mode_dir, filename=f"experiment_4_results_{ce_mode}")
        runner.plot(results_df, mode_dir, _make_plot_configs(ce_mode))
        runner.plot(results_df, mode_dir, [_make_timing_plot_config(ce_mode)])

    # ── Combined results ──
    combined_df = pd.concat(all_results, ignore_index=True)
    runner.save(combined_df, results_dir, filename="experiment_4_results_combined")

    print("\nExperiment 4 complete. Both ce_modes saved.")


if __name__ == "__main__":
    main()
