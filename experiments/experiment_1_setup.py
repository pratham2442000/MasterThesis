"""
Experiment 1 Setup Script
==========================
Creates a BM25 index from the dataset corpus.
This must be run before experiment_1.py.

For wiki_musique: indexes the custom corpus from JSON files.
For msmarco-passage: uses PyTerrier's built-in MSMARCO index.
"""
import os
import sys

print("trying now to load pyterrier")

import pyterrier as pt
print("pyterrier loaded")

## Add parent directory to path to import constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import constants
from code_base.dataset_factory import get_dataset


def main():
    if not pt.started():
        pt.init()

    # Constants
    DATASET_NAME = constants.DATASET_NAME
    SPLIT = constants.DATA_SPLIT
    # constants.INDEX_PATH is usually .../data.properties, we need the directory
    INDEX_DIR = os.path.dirname(constants.INDEX_PATH)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Index Directory: {INDEX_DIR}")
    print(f"Split: {SPLIT}")

    # Load dataset via factory (returns normalized schema)
    print("Loading data...")
    docs_df, topics_df, qrels_df = get_dataset(DATASET_NAME, SPLIT)

    if docs_df is None:
        # For datasets like MSMARCO that use pre-built indexes
        print(f"\nDataset '{DATASET_NAME}' does not provide a custom corpus DataFrame.")
        print("Using PyTerrier's built-in index support instead.")

        # For MSMARCO, we can build an index from ir_datasets
        print(f"Building index from ir_datasets for {DATASET_NAME}...")
        dataset = pt.get_dataset(f'irds:msmarco-passage')
        indexer = pt.IterDictIndexer(INDEX_DIR, overwrite=True, meta={'docno': 256, 'text': 4096})
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=['text'])

        print(f"Index created successfully: {index_ref}")
        print("Setup complete.")
        return

    # Clean documents (as per notebook)
    print("Cleaning documents...")
    docs_df = docs_df.dropna(subset=['docno', 'text'])
    docs_df['docno'] = docs_df['docno'].astype(str)
    docs_df = docs_df[docs_df['text'].str.strip() != '']

    print(f"Documents loaded: {len(docs_df)}")

    # Create index
    print(f"Creating index at {INDEX_DIR}...")
    indexer = pt.IterDictIndexer(INDEX_DIR, overwrite=True, meta={'docno': 256, 'text': 4096})
    index_ref = indexer.index(docs_df.to_dict(orient='records'))

    print(f"Index created successfully: {index_ref}")
    print("Setup complete.")


if __name__ == "__main__":
    main()
