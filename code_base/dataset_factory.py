"""
Dataset Factory
================
Provides a unified interface for loading datasets.
All datasets return the same schema: (docs_df, topics_df, qrels_df)

topics_df columns: qid, query, type
qrels_df columns:  qid, docno, label
docs_df columns:   docno, text  (may be None for datasets with pre-built indexes)
"""
import os

import pyterrier as pt

import constants
from code_base.wiki_loader_pyt import wiki_loader_pyt


def get_dataset(name: str, SPLIT: str):
    """
    Load a dataset by name and split.

    Args:
        name: Dataset identifier ("wiki_musique" or "msmarco-passage")
        SPLIT: Data split. For wiki_musique this is "test"/"dev"/"train".
               For msmarco-passage this is "19" or "20" (TREC-DL year).

    Returns:
        (docs_df, topics_df, qrels_df) with normalized column names.
        docs_df may be None for datasets that use pre-built indexes.
    """
    if name == "wiki_musique":
        DATA_PATH = constants.DATA_PATH
        return wiki_loader_pyt(DATA_PATH, SPLIT)

    elif name == "msmarco-passage":
        dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{SPLIT}/judged')

        # Get components
        topics_df = dataset.get_topics()
        qrels_df = dataset.get_qrels()

        # Normalize topic column names
        # irds returns: qid, query  (already correct for trec-dl)
        if "query_id" in topics_df.columns:
            topics_df = topics_df.rename(columns={"query_id": "qid"})
        # Some irds versions use 'title' instead of 'query'
        if "title" in topics_df.columns and "query" not in topics_df.columns:
            topics_df = topics_df.rename(columns={"title": "query"})

        # Normalize qrels column names
        if "query_id" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"query_id": "qid"})
        if "doc_id" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"doc_id": "docno"})
        if "relevance" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"relevance": "label"})

        # Ensure string types for IDs
        topics_df["qid"] = topics_df["qid"].astype(str)
        qrels_df["qid"] = qrels_df["qid"].astype(str)
        qrels_df["docno"] = qrels_df["docno"].astype(str)

        # Add 'type' column for compatibility with wiki_musique experiments
        if "type" not in topics_df.columns:
            topics_df["type"] = "unknown"

        # Ensure DATA_PATH directory exists for storing rewrites etc.
        os.makedirs(constants.DATA_PATH, exist_ok=True)

        # MSMARCO corpus is ~8.8M passages; we don't load it into a DataFrame.
        # Instead, experiments should use a pre-built index (from dataset_setup).
        # Return None for corpus — setup scripts handle indexing differently.
        corpus_df = None

        print(f"MSMARCO-passage TREC-DL 20{SPLIT} loaded:")
        print(f"  Topics: {len(topics_df)}")
        print(f"  Qrels: {len(qrels_df)}")

        return corpus_df, topics_df, qrels_df

    else:
        raise ValueError(f"Dataset {name} not recognized.")
