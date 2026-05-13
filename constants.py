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

# ==================== Dataset-Specific Configuration ====================
_DATASET_CONFIGS = {
    "wiki_musique": {
        "DATA_PATH": (
              "/scratch/pjohari/MasterThesis/data/2wikimultihopQA"
#            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
        ),
        "DATA_SPLIT": "dev",
        "INDEX_PATH": "./index/wiki_musique_index/data.properties",
        "DENSE_INDEX_PATH": "./index/wiki_musique_dense_index",
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
              "/scratch/pjohari/MasterThesis/data/msmarco-passage"
#            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/msmarco-passage"
        ),
        "DATA_SPLIT": "19",  # -> irds:msmarco-passage/trec-dl-2019/judged
        "INDEX_PATH": "./index/msmarco_passage_index/data.properties",
        "DENSE_INDEX_PATH": "./index/msmarco_passage_dense_index",
        "TASB_INDEX_PATH": "macavaney/msmarco-passage-v2-dedup.tcthnp.flex",
        "CORPUS_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.16",
        "LAFF_BASE_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.128.laff",
        "LAFF_GRAPH_PATH": "macavaney/msmarco-passage.corpusgraph.bm25.128.laff",
        "TYPE_DISTRIBUTION": None,  # MSMARCO has no query types
    },
    "dpr-w100": {
        "DATA_PATH": (
              "/scratch/pjohari/MasterThesis/data/dpr-w100"
#            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/dpr-w100"
        ),
        "DATA_SPLIT": "dev",  # -> irds:dpr-w100/trivia-qa/dev
        "INDEX_PATH": "./index/dpr_w100_index/data.properties",
        "DENSE_INDEX_PATH": "./index/dpr_w100_dense_index",
        "TYPE_DISTRIBUTION": None,  # TriviaQA has no query types
    },
}

# ==================== Derived Data Configuration ====================
DATA_PATH = _DATASET_CONFIGS[DATASET_NAME]["DATA_PATH"]
DATA_SPLIT = _DATASET_CONFIGS[DATASET_NAME]["DATA_SPLIT"]
INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["INDEX_PATH"]
DENSE_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["DENSE_INDEX_PATH"]
EXPERIMENT_2_TYPE_DISTRIBUTION = _DATASET_CONFIGS[DATASET_NAME]["TYPE_DISTRIBUTION"]
TCT_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME].get("TCT_INDEX_PATH", "")
TASB_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME].get("TASB_INDEX_PATH", "")
CORPUS_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("CORPUS_GRAPH_PATH", "")
LAFF_BASE_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("LAFF_BASE_GRAPH_PATH", "")
LAFF_GRAPH_PATH = _DATASET_CONFIGS[DATASET_NAME].get("LAFF_GRAPH_PATH", "")

RESULTS_PATH = (
      "/scratch/pjohari/MasterThesis/results/"
#    "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/results/"
)

DENSE_MODEL_NAME = "castorini/tct_colbert-msmarco"
MODELS_CACHE_DIR = "./models"  # Path to local offline models

# ==================== Experiment 1 Configuration ====================
EXPERIMENT_1_TOPIC_LIMIT = 100
EXPERIMENT_1_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_1_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 2 Configuration ====================
EXPERIMENT_2_TOPIC_LIMIT = 100
EXPERIMENT_2_N_REWRITES = [int(i) for i in np.linspace(1, 30, 10, dtype=int)]
EXPERIMENT_2_RETRIEVAL_CUTOFF = 50
EXPERIMENT_2_EVAL_METRICS = [R @ 50, R @ 10, nDCG @ 10]

# ==================== Experiment 2c Configuration ====================
EXPERIMENT_2C_REWRITING_COUNT = 15
EXPERIMENT_2C_C_VALUES = EXPERIMENT_1_C_VALUES

# ==================== Experiment 3 Configuration ====================
EXPERIMENT_3_TOPIC_LIMIT = 100
EXPERIMENT_3_BEST_N_REWRITES = 10
EXPERIMENT_3_C_VALUES = [int(i) for i in np.geomspace(10, 100, num=10, dtype=int)]
EXPERIMENT_3_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 4 Configuration (ORE) ==============
EXPERIMENT_4_TOPIC_LIMIT = 100
EXPERIMENT_4_C_VALUES = [int(i) for i in np.geomspace(10, 500, num=10, dtype=int)]
EXPERIMENT_4_EVAL_METRICS = [R @ 10, nDCG @ 10]
EXPERIMENT_4_FLEX_INDEX_PATH = TASB_INDEX_PATH  # derived from dataset
EXPERIMENT_4_CORPUS_GRAPH = CORPUS_GRAPH_PATH  # derived from dataset
EXPERIMENT_4_LAFF_GRAPH = LAFF_BASE_GRAPH_PATH  # derived from dataset
EXPERIMENT_4_LAFF_K = 16
EXPERIMENT_4_CROSS_ENC_BUDGET = 7
EXPERIMENT_4_TOP_S1 = 25
EXPERIMENT_4_TOP_S2 = 15

# ==================== Experiment 5 Configuration (Full RAG) =====
EXPERIMENT_5_TOPIC_LIMIT = 100
EXPERIMENT_5_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_5_RETRIEVAL_METRICS = [R @ 10, R @ 100]
EXPERIMENT_5_EVAL_METRICS = [rag_measures.EM, rag_measures.F1,
                             LLMasJudge("causal", f"{MODELS_CACHE_DIR}/Llama-3.2-3B-Instruct")]
EXPERIMENT_5_TOP_K = 5  # Passages fed to generator

# ==================== Experiment 6 Configuration (RAG + Rewrites) =====
EXPERIMENT_6_TOPIC_LIMIT = 100
EXPERIMENT_6_RETRIEVAL_CUTOFF = 20
EXPERIMENT_6_N_REWRITES = [int(i) for i in np.linspace(1, 30, 10, dtype=int)]
EXPERIMENT_6_EVAL_METRICS = [rag_measures.EM, rag_measures.F1, ]
EXPERIMENT_6_TOP_K = 5  # Passages fed to generator

# ==================== Experiment 7 Configuration (ReformIR) =====
EXPERIMENT_7_TOPIC_LIMIT = 100
EXPERIMENT_7_C_VALUES = [int(i) for i in np.geomspace(10, 1000, num=10, dtype=int)]
EXPERIMENT_7_EVAL_METRICS = [R @ 10, nDCG @ 10]
EXPERIMENT_7_CROSS_ENC_BUDGET = 7
EXPERIMENT_7_TOP_S = 15
EXPERIMENT_7_N_REWRITES = 5  # number of LLM rewrites used by ReformIR

# ==================== Reranker Configuration ====================
# Options: "monot5", "cross_encoder", "qwen_reranker"
RERANKER_TYPE = "cross_encoder"
_RERANKER_NAME_MAP = {
    "cross_encoder": "MiniLM-L6",
    "monot5": "MonoT5",
    "qwen_reranker": "Qwen3-Reranker",
}
RERANKER_NAME = _RERANKER_NAME_MAP.get(RERANKER_TYPE, RERANKER_TYPE)
RERANKER_MODEL_NAME = "castorini/monot5-base-msmarco"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_BATCH_SIZE = 64
QWEN_RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
QWEN_RERANKER_BATCH_SIZE = 32
QWEN_RERANKER_GPU_MEMORY_UTILIZATION = 0.65
T5_BATCH_SIZE = 16
EXPERIMENT_BATCH_SIZE = 10

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

# TODO use mini-lm 0.6 reranker (info in sunar paper) ✅
# TODO use qwen 0.6 (repo: https://github.com/Mandeep-Rhttps://github.com/Mandeep-Rathee/gar_reproathee/gar_repro)✅
# TODO LLM as judge 
# TODO ORE laff index is still not made for wiki✅
# TODO have documents as the x axis not the number of queries (i.e find the average number of documents considered after merge(i.e before reranker) for each query rewrite) #still need to figure out how
# fed-forward index
