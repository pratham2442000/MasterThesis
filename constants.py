"""
Constants Configuration
========================
Central configuration for all experiments.

Change ``DATASET_NAME`` to switch all experiments to a different dataset.
All dataset-specific paths are automatically derived from the config
dictionaries below.
"""

import numpy as np
import pyterrier_rag.measures as rag_measures
from pyterrier.measures import R, nDCG

from code_base.llm_judge import LLMasJudge

# ==================== Dataset Selection ====================
# Change this ONE value to switch all experiments to a different dataset.
DATASET_NAME = "wiki_musique"  # Options: "wiki_musique", "msmarco-passage", "dpr-w100"

# ==================== Reranker Configuration ====================
# Options: "monot5", "cross_encoder"
RERANKER_TYPE = "cross_encoder"
_RERANKER_NAME_MAP = {
    "cross_encoder": "MiniLM-L6",
    "monot5": "MonoT5",
}
RERANKER_NAME = _RERANKER_NAME_MAP.get(RERANKER_TYPE, RERANKER_TYPE)
RERANKER_MODEL_NAME = "castorini/monot5-base-msmarco"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_BATCH_SIZE = 64
QWEN_RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
QWEN_RERANKER_BATCH_SIZE = 32
QWEN_RERANKER_GPU_MEMORY_UTILIZATION = 0.65
T5_BATCH_SIZE = 64
EXPERIMENT_BATCH_SIZE = 10

# ==================== Dataset-Specific Configuration ====================
_DATASET_CONFIGS = {
    "wiki_musique": {
        "DATA_PATH": (
            #   "/scratch/pjohari/MasterThesis/data/2wikimultihopQA"
            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
        ),
        "DATA_SPLIT": "dev",
        "INDEX_PATH": "./index/wiki_musique_index/data.properties",
        "TCT_INDEX_PATH": "./indices/wiki_musique_tct_colbert",
        "TASB_INDEX_PATH": "./indices/wiki_musique.tasb.flex",
        "CORPUS_GRAPH_PATH": "./indices/wiki_musique.corpusgraph.bm25.16",
        "LAFF_BASE_GRAPH_PATH": "./indices/wiki_musique.corpusgraph.bm25.128",
        "LAFF_GRAPH_PATH": "./indices/wiki_musique.corpusgraph.bm25.128.laff",
        "TYPE_DISTRIBUTION": {
            "compositional": 0.5,
            "comparison": 0.3,
            "bridge_comparison": 0.1,
            "inference": 0.1,
        },
    },
    "msmarco-passage": {
        "DATA_PATH": (
            #   "/scratch/pjohari/MasterThesis/data/msmarco-passage"
            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/msmarco-passage"
        ),
        "DATA_SPLIT": "19",  # -> irds:msmarco-passage/trec-dl-2019/judged
        "INDEX_PATH": "./index/msmarco_passage_index/data.properties",
        "TCT_INDEX_PATH": "./indices/msmarco_passage_tct_colbert",
        "TASB_INDEX_PATH": "./indices/msmarco_passage_dedup.tasb.flex",
        "CORPUS_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.16",
        "LAFF_BASE_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.128",
        "LAFF_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.128.laff",
        "TYPE_DISTRIBUTION": None,  # MSMARCO has no query types
    },
    "dpr-w100": {
        "DATA_PATH": (
            #   "/scratch/pjohari/MasterThesis/data/dpr-w100"
            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/dpr-w100"
        ),
        "DATA_SPLIT": "dev",  # -> irds:dpr-w100/trivia-qa/dev
        "INDEX_PATH": "./index/dpr_w100_index/data.properties",
        "TCT_INDEX_PATH": "./indices/dpr_w100_tct_colbert",
        "TASB_INDEX_PATH": "./indices/dpr_w100.tasb.flex",
        "CORPUS_GRAPH_PATH": "./indices/dpr_w100.corpusgraph.bm25.16",
        "LAFF_BASE_GRAPH_PATH": "./indices/dpr_w100.corpusgraph.bm25.128",
        "LAFF_GRAPH_PATH": "./indices/dpr_w100.corpusgraph.bm25.128.laff",
        "TYPE_DISTRIBUTION": None,  # TriviaQA has no query types
    },
}

# ==================== Derived Data Configuration ====================
DATA_PATH = _DATASET_CONFIGS[DATASET_NAME]["DATA_PATH"]
DATA_SPLIT = _DATASET_CONFIGS[DATASET_NAME]["DATA_SPLIT"]
INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["INDEX_PATH"]
DENSE_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["TCT_INDEX_PATH"]
EXPERIMENT_2_TYPE_DISTRIBUTION = _DATASET_CONFIGS[DATASET_NAME]["TYPE_DISTRIBUTION"]
TCT_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME].get("TCT_INDEX_PATH", "")
TASB_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME].get("TASB_INDEX_PATH", "")
CORPUS_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("CORPUS_GRAPH_PATH", "")
LAFF_BASE_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("LAFF_BASE_GRAPH_PATH", "")
LAFF_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("LAFF_GRAPH_PATH", "")

RESULTS_PATH = (
    #   "/scratch/pjohari/MasterThesis/results/"
    "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/results/"
)

# TCT-ColBERT v2-HNP — same model used by build_all.py to build TCT_INDEX_PATH
DENSE_MODEL_NAME = "castorini/tct_colbert-v2-hnp-msmarco"
# Point DENSE_INDEX_PATH at the same index built by build_all.py
DENSE_INDEX_PATH = TCT_INDEX_PATH
MODELS_CACHE_DIR = "./models"  # Path to local offline models

# ==================== Experiment 1 Configuration ====================
EXPERIMENT_1_TOPIC_LIMIT = 100
EXPERIMENT_1_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_1_EVAL_METRICS = [R @ 10, R @ 50, nDCG @ 10]

# ==================== Experiment 2 Configuration ====================
EXPERIMENT_2_TOPIC_LIMIT = 100
EXPERIMENT_2_N_REWRITES = [int(i) for i in np.linspace(1, 30, 10, dtype=int)]
EXPERIMENT_2_RETRIEVAL_CUTOFF = 50
EXPERIMENT_2_EVAL_METRICS = [R @ 50, R @ 10, nDCG @ 10]

# ==================== Experiment 3 Configuration ====================
EXPERIMENT_3_TOPIC_LIMIT = 100
EXPERIMENT_3_BEST_N_REWRITES = 10
EXPERIMENT_3_C_VALUES = [int(i) for i in np.geomspace(10, 100, num=10, dtype=int)]
EXPERIMENT_3_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 4 Configuration (ORE) ==============
EXPERIMENT_4_TOPIC_LIMIT = 100
EXPERIMENT_4_C_VALUES = [int(i) for i in np.geomspace(10, 500, num=10, dtype=int)]
EXPERIMENT_4_EVAL_METRICS = [R @ 10, R @ 50, nDCG @ 10]
EXPERIMENT_4_FLEX_INDEX_PATH = TASB_INDEX_PATH  # derived from dataset
# Corpus graph: local path if built, else fall back to HuggingFace artifact.
# astappiev/ragwiki-corpusgraph is built for rag:nq_wiki (Wikipedia NQ passages,
# same corpus as dpr-w100) with BM25 k=16 neighbours.
EXPERIMENT_4_CORPUS_GRAPH_LOCAL = CORPUS_GRAPH_PATH
EXPERIMENT_4_CORPUS_GRAPH_HF = "astappiev/ragwiki-corpusgraph"
EXPERIMENT_4_LAFF_GRAPH = LAFF_BASE_GRAPH_PATH  # derived from dataset
EXPERIMENT_4_LAFF_K = 16
# ORE base hyperparameters (tuned for c=100 in the paper).
# Do NOT change these — use ore_params_for_budget(c) to get scaled values.
EXPERIMENT_4_CROSS_ENC_BUDGET = 7  # fixed; see ORE/ORE_PARAMS.md
EXPERIMENT_4_TOP_S1 = 25  # baseline for c=100
EXPERIMENT_4_TOP_S2 = 15  # baseline for c=100

LAFF_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("LAFF_GRAPH_PATH", "")


def ore_params_for_budget(c: int) -> dict:
    """Return scaled ORE hyperparameters for a given BM25 retrieval budget c.

    ce (cross_enc_budget) is intentionally kept fixed at 7 to preserve ORE's
    efficiency advantage: at large c the cross-encoder only scores ~112 docs
    regardless of budget.  top_s and top_s2 are cheap array operations and
    scale so the cluster navigation and frontier expansion stay proportional to
    the candidate pool size.  See ORE/ORE_PARAMS.md for full derivation.

    Args:
        c: BM25 retrieval cutoff (budget) for the current sweep step.

    Returns:
        Dict with keys ``cross_enc_budget``, ``top_s``, ``top_s2``.
    """
    return {
        "cross_enc_budget": EXPERIMENT_4_CROSS_ENC_BUDGET,  # fixed — never scale
        "top_s": max(EXPERIMENT_4_TOP_S1, c // 4),  # cheap: LAFF array lookup
        "top_s2": max(EXPERIMENT_4_TOP_S2, c // 6),  # cheap: heapq on scores
    }


# ==================== Experiment 5 Configuration (Full RAG) =====
EXPERIMENT_5_TOPIC_LIMIT = 100
EXPERIMENT_5_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_5_RETRIEVAL_METRICS = [R @ 10, R @ 100]
EXPERIMENT_5_TOP_K = 5  # Passages fed to generator


def get_experiment_5_eval_metrics(shared_backend=None):
    """Build eval metrics for Experiment 5, optionally sharing model backend.

    Args:
        shared_backend: Existing HuggingFace backend from RAG reader.
            Pass this to avoid loading Llama twice (OOM on single GPU).
    """
    return [rag_measures.EM, rag_measures.F1,
            LLMasJudge(f"{MODELS_CACHE_DIR}/Llama-3.2-3B-Instruct",
                       shared_backend=shared_backend)]


# ==================== Experiment 6 Configuration (RAG + Rewrites) =====
EXPERIMENT_6_TOPIC_LIMIT = 100
EXPERIMENT_6_RETRIEVAL_CUTOFF = 20
EXPERIMENT_6_N_REWRITES = [int(i) for i in np.linspace(1, 30, 10, dtype=int)]
EXPERIMENT_6_TOP_K = 5  # Passages fed to generator


def get_experiment_6_eval_metrics(shared_backend=None):
    """Build eval metrics for Experiment 6, optionally sharing model backend."""
    return [rag_measures.EM, rag_measures.F1,
            LLMasJudge(f"{MODELS_CACHE_DIR}/Llama-3.2-3B-Instruct",
                       shared_backend=shared_backend)]


# ==================== Experiment 7 Configuration (ReformIR) =====
EXPERIMENT_7_TOPIC_LIMIT = 100
EXPERIMENT_7_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_7_EVAL_METRICS = [R @ 10, R @ 50, nDCG @ 10]
# Base hyperparameters — validated by paper at c=50 (ce=4) and c=100 (ce=7).
# Do NOT change directly — use reformir_params_for_budget(c) in experiment code.
# See ReformIR-DC92/REFORMIR_PARAMS.md for full derivation.
EXPERIMENT_7_CROSS_ENC_BUDGET = 7  # baseline for c=100; scales with c
EXPERIMENT_7_TOP_S = 15  # fixed — RM3 quality plateaus quickly
EXPERIMENT_7_N_REWRITES = 5  # number of LLM rewrites fed to ReformIR
# Paper's active-learning loop batch size (NOT the scorer's inference batch).
# Each iteration of the while-loop in reformir.py scores exactly this many docs.
# The paper used 16; T5_BATCH_SIZE (64) is the scorer's inference batch size.
REFORMIR_BATCH_SIZE = 16


def reformir_params_for_budget(
        c: int,
        batch_size: int = 16,
        ce_mode: str = "linear",
) -> dict:
    """Return scaled ReformIR hyperparameters for a given BM25 budget c.

    Two ce modes control how the cross-encoder budget scales with c:

    - ``"linear"``: ce = ceil(c / batch_size).  Every budget doc gets a real
      CE score.  The lsq_linear regression fits weights each iteration but
      never estimates unscored docs (there are none).  This isolates the
      *reformulation* contribution: does ReformIR find better candidates?
      Matches the paper's operating points at c=50 (ce=4) and c=100 (ce=7).

    - ``"fixed"``: ce = 7 (the paper's baseline), capped at
      ceil(c / batch_size) for small c where 7 would overshoot.  At large c
      the regression must *predict* scores for unscored tail docs.  This
      tests whether the lsq_linear approximation holds beyond c=100.

    top_s is fixed at 15: RM3 term selection quality plateaus quickly.

    reform_bm25_num_results scales with c to fix the hardcoded-100 bug.

    Args:
        c: BM25 retrieval cutoff (budget) for the current sweep step.
        batch_size: Active-learning loop batch size (default 16, paper value).
        ce_mode: ``"linear"`` (full coverage) or ``"fixed"`` (ce=7, partial).

    Returns:
        Dict with keys ``cross_enc_budget``, ``top_s``, ``reform_bm25_num_results``.
    """
    import math
    max_batches = math.ceil(c / batch_size)  # total iterations to exhaust budget

    if ce_mode == "fixed":
        ce = min(EXPERIMENT_7_CROSS_ENC_BUDGET, max_batches)
    else:  # "linear"
        ce = max_batches  # full coverage

    return {
        "cross_enc_budget": ce,
        "top_s": EXPERIMENT_7_TOP_S,
        "reform_bm25_num_results": max(128, c),
    }



# ==================== LLM Query Rewriting Configuration ====================
LLM_REWRITES_PER_QUERY = 30
LLM_TEMPERATURE = 0.7
LLM_MODEL_NAME = "gpt-4o-mini"

# ==================== API Permission Configuration ====================
REQUIRE_API_PERMISSION = True

# ==================== RAG Generation Configuration ====================
OFFLINE_MODE = True

RAG_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# DENSE_MODEL_NAME = "castorini/tct_colbert-msmarco"

RAG_PROMPT = (
    "You are given passages and a question. "
    "Answer the question using ONLY information from the passages below. "
    "Give a short answer (1-5 words). Do not explain. Do not add extra text.\n\n"
    "Passages:\n{{ qcontext }}\n\n"
    "Question: {{ query }}\n"
    "Answer:"
)

RAG_MODEL_MAX_NEW_TOKENS = 512
RAG_CONCATENATOR_TOP_K = 5

# TODO have documents as the x axis not the number of queries (i.e find the average number of documents considered after merge(i.e before reranker) for each query rewrite) #still need to figure out how
# fed-forward index
