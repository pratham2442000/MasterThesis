import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
import tqdm


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Helper to load a JSON file with error handling."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading {filepath.name}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_corpus(corpus_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Loads and cleans the corpus, returning a DataFrame and a Title->DocNo mapping.
    """
    corpus_data = load_json_file(corpus_path)

    # Convert dictionary to list of dicts for DataFrame creation
    docs_list = [
        {"docno": str(doc_id), "title": snippet['title'], "text": snippet['text']}
        for doc_id, snippet in tqdm.tqdm(corpus_data.items(), desc="Processing Docs")
    ]
    
    docs_df = pd.DataFrame(docs_list)

    # 
    # Clean duplicates based on text content
    initial_count = len(docs_df)
    docs_df = docs_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"Removed {initial_count - len(docs_df)} duplicate documents.")

    # Data cleanup
    docs_df['docno'] = docs_df['docno'].astype(str)
    docs_df = docs_df.dropna(subset=['docno', 'text'])
    docs_df = docs_df[docs_df['text'].str.strip() != '']

    # Create Title -> DocNo(s) mapping
    # A title might map to multiple docnos if texts were different but titles same
    title_to_docnos = docs_df.groupby('title')['docno'].apply(list).to_dict()

    return docs_df, title_to_docnos


def process_queries(dataset_path: Path, title_to_docnos: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses queries and generates relevance judgments (qrels) based on context titles.
    """
    query_data = load_json_file(dataset_path)
    
    topics_list = []
    qrels_list = []

    for data in tqdm.tqdm(query_data, desc="Processing Queries"):
        query_id = data["_id"]

        # Add Topic
        topics_list.append({
            "qid": query_id,
            "query": data["question"],
            "type": data.get("type", "unknown")
        })

        # Add Qrels (Relevance Judgments)
        # The dataset provides titles in 'context'; we map these titles back to docnos
        for fact in data.get("context", []):
            doc_title = fact[0]
            if doc_title in title_to_docnos:
                for docno in title_to_docnos[doc_title]:
                    qrels_list.append({
                        "qid": query_id,
                        "docno": docno,
                        "label": 1
                    })

    topics_df = pd.DataFrame(topics_list)
    qrels_df = pd.DataFrame(qrels_list)

    return topics_df, qrels_df


def wiki_loader_pyt(data_path: str, split: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main loader function for 2WikiMultihopQA data compatible with PyTerrier.
    """
    # 
    base_path = Path(data_path)
    dataset_path = base_path / f"{split}.json"
    corpus_path = base_path / "wiki_musique_corpus.json"

    # 1. Process Corpus
    docs_df, title_to_docnos = process_corpus(corpus_path)
    print(f"Final Corpus: {len(docs_df)} documents.")

    # 2. Process Topics & Qrels
    topics_df, qrels_df = process_queries(dataset_path, title_to_docnos)

    print(f"\nFinal Statistics:")
    print(f"Topics: {len(topics_df)}")
    print(f"Qrels: {len(qrels_df)}")
    print("-" * 30)

    return docs_df, topics_df, qrels_df

if __name__ == "__main__":
    # Configuration
    BASE_DIR = Path("/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data/2wikimultihopQA")
    SPLIT = 'dev'

    try:
        docs, topics, qrels = wiki_loader_pyt(str(BASE_DIR), SPLIT)

        print("\n--- Corpus Head ---")
        print(docs.head())
        print(docs.info())

        print("\n--- Topics Head ---")
        print(topics.head())

        print("\n--- Qrels Head ---")
        print(qrels.head())

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
