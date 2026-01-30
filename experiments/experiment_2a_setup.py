"""
Experiment 2a Setup Script
==========================
This script creates the dense index for experiment 2a using TCT-ColBERT.
This must be run before experiment_2a.py.
"""

import os
import sys

# Add parent directory to path to allow importing constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pyterrier as pt

if not pt.started():
    pt.init()
import pyterrier_dr as pt_dr

import constants
from code_base import dataset_factory


def main():
    print("=" * 60)
    print("Experiment 2a Setup: Creating Dense Index")
    print("=" * 60)

    # Configuration
    model_name = constants.DENSE_MODEL_NAME
    index_path = constants.DENSE_INDEX_PATH
    dataset_name = constants.DATASET_NAME
    data_split = constants.DATA_SPLIT

    print(f"Model: {model_name}")
    print(f"Index Path: {index_path}")
    print(f"Dataset: {dataset_name} ({data_split})")
    print("-" * 60)

    # Load dataset
    print("Loading dataset...")
    try:
        docs_df, _, _ = dataset_factory.get_dataset(dataset_name, data_split)
        print(f"Loaded {len(docs_df)} documents.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize TCT-ColBERT model
    print("Initializing TCT-ColBERT model...")
    # As per user instruction: model = pyterrier_dr.TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
    # But using constant value
    model = pt_dr.TctColBert(model_name, verbose=True)

    # Initialize Faiss Index
    print(f"Initializing FlexIndex at {index_path}...")
    index = pt_dr.FlexIndex(index_path)

    if os.path.exists(os.path.join(index_path, "index.dpr")):
        print(f"Warning: Index already exists at {index_path}.")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Skipping indexing.")
            return

    # Index documents
    print("Indexing documents... This may take a while.")
    # Ensure docs_df has 'docno' and 'text'
    if "docno" not in docs_df.columns or "text" not in docs_df.columns:
        raise ValueError("docs_df must contain 'docno' and 'text' columns")

    pipeline = model.doc_encoder() >> index.indexer(mode="overwrite")
    pipeline.index(docs_df.to_dict(orient='records'))
    print("Indexing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
