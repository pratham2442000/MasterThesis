#Load AmbigQA into pyterrier data format

#use the download_data_ambigqa.py script to download the data
#convert the files into pyterrier data format

import pandas as pd
import json
import os
import tqdm
import re


def preprocess_query(query):
    """
    Preprocess query to avoid TerrierQL parser errors by:
    - Removing or escaping special characters
    - Ensuring no unbalanced quotes or parentheses
    """
    # Replace parentheses and other special characters that cause parser errors
    query = re.sub(r'[\(\)]', ' ', query)  # Remove parentheses
    query = re.sub(r'[?!,;:]', ' ', query)  # Replace punctuation with spaces
    query = re.sub(r'\s+', ' ', query).strip()  # Normalize whitespace
    return query


def ambiq_loader_pyt(data_path: str, split: str, corpus_name: str = "wikipedia_split") -> tuple:
    """
    Load AmbigQA data into PyTerrier format

    Args:
        data_path: Path to the data directory
        split: Dataset split (train, dev, test)
        corpus_name: Name of the corpus file

    Returns:
        Tuple of (docs_df, topics_df, qrels_df)
    """
    # File paths
    if split == "train":
        dataset_path = os.path.join(data_path, "ambigqa", "train_light.json")
    elif split == "dev":
        dataset_path = os.path.join(data_path, "ambigqa", "dev_light.json")
    else:
        dataset_path = os.path.join(data_path, f"data.nqopen.{split}", f"nqopen-{split}.json")

    corpus_path = os.path.join(data_path, corpus_name, "psgs_w100.tsv")

    # Load corpus
    print(f"Processing corpus from {corpus_path}...")
    docs_list = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                doc_id, text, title = parts[0], parts[1], parts[2]
                docs_list.append({
                    "docno": doc_id,
                    "text": text,
                    "title": title
                })
            break

    docs_df = pd.DataFrame(docs_list)
    print(f"Created docs_df with {len(docs_df)} documents.")

    # Load questions and answers
    print(f"\nProcessing {split} set for topics and qrels...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    topics_list = []
    qrels_list = []

    for item in tqdm.tqdm(data):
        query_id = item.get("id", item.get("_id"))
        query = item.get("question")

        # Preprocess query to prevent TerrierQL parser errors
        clean_query = preprocess_query(query)

        topics_list.append({
            "qid": query_id,
            "query": clean_query,
            "original_query": query  # Keep original query for reference
        })

        # Process relevance judgments
        # AmbigQA provides annotations differently based on format
        if "annotations" in item:
            # For AmbigQA format
            for annotation in item["annotations"]:
                if "positive_ctxs" in annotation:
                    for ctx in annotation["positive_ctxs"]:
                        qrels_list.append({
                            "qid": query_id,
                            "docno": ctx["passage_id"],
                            "label": 1
                        })
        elif "context" in item:
            # For simplified format
            for ctx in item["context"]:
                if isinstance(ctx, list) and len(ctx) > 0:
                    doc_id = ctx[0]  # Assuming first element is document ID
                    qrels_list.append({
                        "qid": query_id,
                        "docno": doc_id,
                        "label": 1
                    })

    topics_df = pd.DataFrame(topics_list)
    qrels_df = pd.DataFrame(qrels_list)

    # Clean up and validate
    docs_df = docs_df.dropna(subset=['docno', 'text'])
    docs_df['docno'] = docs_df['docno'].astype(str)

    print(f"Created topics_df with {len(topics_df)} queries.")
    print(f"Created qrels_df with {len(qrels_df)} relevance judgments.")

    return docs_df, topics_df, qrels_df


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "/mnt/c/Users/prath/Documents/Uni(Tudelft)/MSc/MasterThesis/data"

    # Load development set
    docs_df, topics_df, qrels_df = ambiq_loader_pyt(DATA_PATH, split="dev")

    print("\nCorpus DataFrame Sample:")
    print(docs_df.head())
    print(docs_df.info())

    print("\nTopics DataFrame Sample:")
    print(topics_df.head())
    print(topics_df.info())

    print("\nQrels DataFrame Sample:")
    print(qrels_df.head())
    print(qrels_df.info())