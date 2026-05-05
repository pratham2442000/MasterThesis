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

import pandas as pd
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

    elif name == "dpr-w100":
        dataset = pt.get_dataset(f'irds:dpr-w100/trivia-qa/{SPLIT}')

        # Get components
        topics_df = dataset.get_topics()
        qrels_df = dataset.get_qrels()

        # Normalize topic column names
        # irds returns: query_id, text, answers
        if "query_id" in topics_df.columns:
            topics_df = topics_df.rename(columns={"query_id": "qid"})
        if "text" in topics_df.columns and "query" not in topics_df.columns:
            topics_df = topics_df.rename(columns={"text": "query"})

        # Store answers tuple as list in 'answer' column (multi-answer)
        if "answers" in topics_df.columns:
            topics_df["answer"] = topics_df["answers"].apply(
                lambda x: list(x) if isinstance(x, (list, tuple)) else [str(x)]
            )
            topics_df = topics_df.drop(columns=["answers"])

        # Normalize qrels column names
        if "query_id" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"query_id": "qid"})
        if "doc_id" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"doc_id": "docno"})
        if "relevance" in qrels_df.columns:
            qrels_df = qrels_df.rename(columns={"relevance": "label"})

        # Filter qrels: only label > 0 (1 = BM25 match, 2 = human annotated)
        qrels_before = len(qrels_df)
        qrels_df = qrels_df[qrels_df["label"] > 0].reset_index(drop=True)

        # Drop 'iteration' column if present (not needed)
        if "iteration" in qrels_df.columns:
            qrels_df = qrels_df.drop(columns=["iteration"])

        # Ensure string types for IDs
        topics_df["qid"] = topics_df["qid"].astype(str)
        qrels_df["qid"] = qrels_df["qid"].astype(str)
        qrels_df["docno"] = qrels_df["docno"].astype(str)

        # Add 'type' column for compatibility
        if "type" not in topics_df.columns:
            topics_df["type"] = "trivia"

        # Ensure DATA_PATH directory exists
        os.makedirs(constants.DATA_PATH, exist_ok=True)

        # DPR-W100 corpus is ~21M passages; don't load into DataFrame.
        corpus_df = None

        print(f"DPR-W100 TriviaQA {SPLIT} loaded:")
        print(f"  Topics: {len(topics_df)}")
        print(f"  Qrels: {len(qrels_df)} (filtered from {qrels_before}, kept label > 0)")

        return corpus_df, topics_df, qrels_df

    else:
        raise ValueError(f"Dataset {name} not recognized.")


def get_gold_answers_triviaqa(topics_df):
    """Expand multi-answer TriviaQA topics into one row per (qid, gold_answer).

    pyterrier_rag EM/F1 measures take max() across rows per qid,
    so multiple rows = multi-answer support.

    Args:
        topics_df: DataFrame with 'qid' and 'answer' (list of strings).

    Returns:
        DataFrame with columns: qid, gold_answer.
    """
    rows = []
    for _, row in topics_df.iterrows():
        answers = row.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        for ans in answers:
            rows.append({"qid": str(row["qid"]), "gold_answer": ans})
    return pd.DataFrame(rows)


def get_gold_answers(topics_df):
    """Unified gold-answers dispatcher.

    Routes to appropriate gold answer extraction based on DATASET_NAME.

    Args:
        topics_df: DataFrame with 'qid' and 'answer' column.

    Returns:
        DataFrame with columns: qid, gold_answer.
    """
    if constants.DATASET_NAME == "dpr-w100":
        return get_gold_answers_triviaqa(topics_df)
    else:
        # wiki_musique / msmarco: single answer per topic
        ga = topics_df[["qid"]].copy()
        ga["gold_answer"] = topics_df["answer"].fillna("") if "answer" in topics_df.columns else ""
        return ga
