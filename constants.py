"""
Constants Configuration
========================
Central configuration for all experiments.

Change ``DATASET_NAME`` to switch all experiments to a different dataset.
All dataset-specific paths are automatically derived from the config
dictionaries below.
"""

import numpy as np
from pyterrier.measures import R, nDCG

# ==================== Dataset Selection ====================
# Change this ONE value to switch all experiments to a different dataset.
DATASET_NAME = "wiki_musique"  # Options: "wiki_musique", "msmarco-passage"

# ==================== Dataset-Specific Configuration ====================
_DATASET_CONFIGS = {
    "wiki_musique": {
        "DATA_PATH": (
            # "/scratch/pjohari/MasterThesis/data/2wikimultihopQA"
            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA"
        ),
        "DATA_SPLIT": "test",
        "INDEX_PATH": "./index/wiki_musique_index/data.properties",
        "DENSE_INDEX_PATH": "./index/wiki_musique_dense_index",
        "TYPE_DISTRIBUTION": {
            "compositional": 0.5,
            "comparison": 0.3,
            "bridge_comparison": 0.1,
            "inference": 0.1,
        },
    },
    "msmarco-passage": {
        "DATA_PATH": (
            # "/scratch/pjohari/MasterThesis/data/msmarco-passage"
            "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/data/msmarco-passage"
        ),
        "DATA_SPLIT": "19",  # -> irds:msmarco-passage/trec-dl-2019/judged
        "INDEX_PATH": "./index/msmarco_passage_index/data.properties",
        "DENSE_INDEX_PATH": "./index/msmarco_passage_dense_index",
        "TYPE_DISTRIBUTION": None,  # MSMARCO has no query types
    },
}

# ==================== Derived Data Configuration ====================
DATA_PATH = _DATASET_CONFIGS[DATASET_NAME]["DATA_PATH"]
DATA_SPLIT = _DATASET_CONFIGS[DATASET_NAME]["DATA_SPLIT"]
INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["INDEX_PATH"]
DENSE_INDEX_PATH = _DATASET_CONFIGS[DATASET_NAME]["DENSE_INDEX_PATH"]
EXPERIMENT_2_TYPE_DISTRIBUTION = _DATASET_CONFIGS[DATASET_NAME]["TYPE_DISTRIBUTION"]

RESULTS_PATH = (
    # "/scratch/pjohari/MasterThesis/results/"
    "/mnt/c/Users/prats/Documents/Uni(Tudelft)/MSc/MasterThesis/results/"
)

DENSE_MODEL_NAME = "castorini/tct_colbert-msmarco"

# ==================== Experiment 1 Configuration ====================
EXPERIMENT_1_TOPIC_LIMIT = 500
EXPERIMENT_1_C_VALUES = [int(i) for i in np.geomspace(10, 200, num=5, dtype=int)]
EXPERIMENT_1_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 2 Configuration ====================
EXPERIMENT_2_TOPIC_LIMIT = 500
EXPERIMENT_2_N_REWRITES = [int(i) for i in np.linspace(1, 30, 10, dtype=int)]
EXPERIMENT_2_RETRIEVAL_CUTOFF = 20
EXPERIMENT_2_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 2c Configuration ====================
EXPERIMENT_2C_REWRITING_COUNT = 15
EXPERIMENT_2C_C_VALUES = EXPERIMENT_1_C_VALUES

# ==================== Experiment 3 Configuration ====================
EXPERIMENT_3_TOPIC_LIMIT = 500
EXPERIMENT_3_BEST_N_REWRITES = 10
EXPERIMENT_3_C_VALUES = [int(i) for i in np.geomspace(10, 100, num=10, dtype=int)]
EXPERIMENT_3_EVAL_METRICS = [R @ 10, nDCG @ 10]

# ==================== Experiment 4 Configuration (ORE) ==============
EXPERIMENT_4_TOPIC_LIMIT = 500
EXPERIMENT_4_C_VALUES = [int(i) for i in np.geomspace(10, 500, num=10, dtype=int)]
EXPERIMENT_4_EVAL_METRICS = [R @ 10, nDCG @ 10]
EXPERIMENT_4_FLEX_INDEX_PATH = "macavaney/msmarco-passage-v2-dedup.tcthnp.flex"
EXPERIMENT_4_CORPUS_GRAPH = "macavaney/msmarco-passage.corpusgraph.bm25.16"
EXPERIMENT_4_LAFF_GRAPH = "macavaney/msmarco-passage.corpusgraph.bm25.128.laff"
EXPERIMENT_4_LAFF_K = 16
EXPERIMENT_4_CROSS_ENC_BUDGET = 7
EXPERIMENT_4_TOP_S1 = 25
EXPERIMENT_4_TOP_S2 = 15

# ==================== Reranker Configuration ====================
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
MODELS_CACHE_DIR = "./models"  # Path to local offline models
RAG_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
RERANKER_MODEL_NAME = "castorini/monot5-base-msmarco"
DENSE_MODEL_NAME = "castorini/tct_colbert-msmarco"

RAG_MODEL_MAX_NEW_TOKENS = 128
RAG_CONCATENATOR_TOP_K = 5
