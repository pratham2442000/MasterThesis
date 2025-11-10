"""
Constants configuration file for experiments.
Modify values here to reconfigure experiments without changing code.
"""

import numpy as np

# ==================== Data Configuration ====================
DATA_PATH = (
    "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/" "MasterThesis/data/2wikimultihopQA"
)
DATA_SPLIT = "test"
DATASET_NAME = "wiki_musique"  # Options: "wiki_musique", "msmarco-passage"

# ==================== Index Configuration ====================
INDEX_PATH = "./index/wiki_musique_index/data.properties"

# ==================== Experiment 1 Configuration ====================
# Limits the number of topics (queries) used for experiment 1
# default 1000
EXPERIMENT_1_TOPIC_LIMIT = 100

# Initial Cutoff Values (c) for BM25 Retrieval in experiment 1
# Defines the range of 'c' to experiment with
# List or single int
EXPERIMENT_1_C_VALUES = [int(i) for i in np.linspace(10, 100, 4, dtype=int)]

# ==================== Experiment 2 Configuration ====================
# Limits the number of topics (queries) used for experiment 2
# default 1000
EXPERIMENT_2_TOPIC_LIMIT = 20

# Number of rewritten queries per original query
EXPERIMENT_2_N_REWRITES = [int(i) for i in np.linspace(1, 10, 2, dtype=int)]

# Docs per query (or per expanded query)
EXPERIMENT_2_RETRIEVAL_CUTOFF = 200

# Defines the range of 'c' to experiment with
# List or single int
EXPERIMENT_2_C_VALUES = [int(i) for i in np.linspace(10, 100, 2, dtype=int)]

# ==================== Reranker Configuration ====================
T5_BATCH_SIZE = 16
EXPERIMENT_BATCH_SIZE = 10

# ==================== LLM Query Rewriting Configuration ====================
# Number of (max) rewritten queries to generate per original query
LLM_REWRITES_PER_QUERY = 30

# LLM Model Configuration
LLM_TEMPERATURE = 0.7
LLM_MODEL_NAME = "gpt-4o-mini"  # Can be overridden by OPENAI_MODEL env var

# ==================== API Permission Configuration ====================
# Set to True to require explicit permission before making API calls
# Set to False to automatically allow API calls (for automated scripts)
REQUIRE_API_PERMISSION = True

# ==================== Evaluation Metrics ====================
# Experiment 1 metrics
EXPERIMENT_1_EVAL_METRICS = [R @ 10, nDCG @ 10]

# Experiment 2 metrics
EXPERIMENT_2_EVAL_METRICS = [R @ 10, nDCG @ 10]
