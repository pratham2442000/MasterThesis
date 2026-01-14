import os
import sys

print("trying now to load pyterrier")

import pyterrier as pt
print("pyterrier loaded")

## Add parent directory to path to import constants and code_base
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from code_base.wiki_loader_pyt import wiki_loader_pyt
import constants


def main():
    if not pt.started():
        pt.init()

    # Constants
    DATA_PATH = constants.DATA_PATH
    SPLIT = constants.DATA_SPLIT
    # constants.INDEX_PATH is usually .../data.properties, we need the directory
    INDEX_DIR = os.path.dirname(constants.INDEX_PATH)

    print(f"Index Directory: {INDEX_DIR}")
    print(f"Data Path: {DATA_PATH}")
    print(f"Split: {SPLIT}")

    # Load dataframes
    print("Loading data...")
    docs_df, topics_df, qrels_df = wiki_loader_pyt(DATA_PATH, SPLIT)

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
