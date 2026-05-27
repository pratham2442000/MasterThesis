"""
Experiment 4 — ORE Adaptive Cutoff Sweep
==========================================
Measures how BM25 retrieval cutoff (c) affects Recall and nDCG
when using ORE Adaptive re-ranking vs baselines.

Systems:
  - BM25 (baseline)
  - BM25 + Reranker
  - ORE Adaptive (TasB dual encoder + corpus graph + LAFF)

Sweeps ``constants.EXPERIMENT_4_C_VALUES``.

ORE hyperparameters are scaled per-c via ``constants.ore_params_for_budget(c)``:
  - cross_enc_budget: FIXED at 7 (preserves ORE efficiency — see ORE/ORE_PARAMS.md)
  - top_s:  scales as max(25, c // 4)  — cheap LAFF lookup, improves navigation
  - top_s2: scales as max(15, c // 6)  — cheap heap sort, improves frontier expansion

Corpus graph for dpr-w100:
  Loaded from HuggingFace (astappiev/ragwiki-corpusgraph) if local index not built.
  Built for rag:nq_wiki (Wikipedia NQ passages = same corpus as dpr-w100), k=16 BM25.
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
# pyrefly: ignore [missing-import]
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
    """Load the base corpus graph, preferring local over HF.

    For dpr-w100 the local graph may not be built yet; in that case we fall
    back to astappiev/ragwiki-corpusgraph on HuggingFace, which was built for
    the same Wikipedia NQ passage corpus (rag:nq_wiki) with BM25 k=16.
    """
    local = constants.EXPERIMENT_4_CORPUS_GRAPH_LOCAL
    hf = constants.EXPERIMENT_4_CORPUS_GRAPH_HF

    local_exists = os.path.exists(local) if (local and not local.startswith("./")) else \
        os.path.exists(os.path.join(".", local.lstrip("./"))) if local else False

    if local and local_exists:
        print(f"  CorpusGraph: loading local {local}")
        return CorpusGraph.load(local)

    print(f"  CorpusGraph: local not found, loading from HF → {hf}")
    graph = pta.Artifact.from_hf(hf)
    print("  CorpusGraph: HF artifact loaded")
    return graph


# ─── Pipeline Factory ────────────────────────────────────

def build_pipelines(c, *, index_ref, reranker, dual_encoder,
                    corpus_index, graph, laff_graph, text_dict):
    """Construct retrieval pipelines for a given cutoff *c*.

    ORE parameters (top_s, top_s2) are scaled with c while cross_enc_budget
    stays fixed.  See constants.ore_params_for_budget and ORE/ORE_PARAMS.md.

    Args:
        c: BM25 retrieval cutoff / ORE budget.
        index_ref: PyTerrier index reference.
        reranker: Reranker instance.
        dual_encoder: TasB dual encoder model.
        corpus_index: FlexIndex for dense vectors.
        graph: Corpus graph for ORE Adaptive.
        laff_graph: LAFF graph for ORE (precomputed or wrapped).
        text_dict: Mapping docno → text for BM25 scoring inside ORE.

    Returns:
        Dict mapping system names to pipelines.
    """
    ore_params = constants.ore_params_for_budget(c)
    print(
        f"  [c={c}] ORE params: ce={ore_params['cross_enc_budget']}, "
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
        batch_size=constants.T5_BATCH_SIZE,
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

def _system_label(key):
    return {
        "BM25": "BM25 Baseline",
        f"BM25_{constants.RERANKER_NAME}": f"BM25 + {constants.RERANKER_NAME}",
        "ORE_Adaptive": "ORE Adaptive",
    }.get(key, key)


def _build_plot_configs(system_keys):
    """Build PLOT_CONFIGS list from active system keys."""
    systems = [(k, _system_label(k)) for k in system_keys]

    return [
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on Recall (R@10) — ORE Adaptive",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "filename": "cutoff_vs_R_ore.png",
            "use_log_scale": True,
        },
        {
            "type": "metrics",
            "x_col": "c",
            "metric_cols": [f"{s}_nDCG@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "title": "Effect of Initial Cutoff (c) on NDCG@10 — ORE Adaptive",
            "xlabel": "Initial Cutoff (c) for BM25 Retrieval",
            "ylabel": "NDCG@10",
            "filename": "cutoff_vs_NDCG_at_10_ore.png",
            "use_log_scale": True,
        },
        {
            "type": "regression",
            "x_col": "c",
            "metric_cols": [f"{s}_R@10" for s, _ in systems],
            "labels": [lbl for _, lbl in systems],
            "use_log_scale": True,
            "title": "R@10 vs. Initial Cutoff (c) Regression — ORE Adaptive",
            "xlabel": "Initial Cutoff (c) (log scale)",
            "ylabel": "Recall (R@10)",
            "filename": "cutoff_R_at_10_log_regression_ore.png",
        },
    ]


# ─── Main ─────────────────────────────────────────────────

def main():
    """Run Experiment 4 — ORE Adaptive cutoff sweep."""
    print("Starting Experiment 4...")

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

    # Determine which systems actually ran
    system_keys = [col.replace("_R@10", "") for col in results_df.columns
                   if col.endswith("_R@10")]
    plot_configs = _build_plot_configs(system_keys)
    runner.plot(results_df, results_dir, plot_configs)


if __name__ == "__main__":
    main()
