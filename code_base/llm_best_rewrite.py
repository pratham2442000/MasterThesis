"""
LLM Best-Rewrite Selector
===========================
For each query and each N in EXPERIMENT_2_N_REWRITES, asks an LLM to pick
the single best rewrite from the first N rewrites.  The winning rewrite is
stored in a CSV with one column per N value.

CSV layout::

    q_id | best_of_1 | best_of_4 | best_of_7 | … | best_of_30
    -----+-----------+-----------+-----------+---+-----------
    101  | <orig>    | <pick>    | <pick>    | … | <pick>

``best_of_1`` is always the original query (no API call needed).

Usage
-----
Called from ``scripts/dataset_setup.py --steps best_rewrite``, or
directly::

    python -m code_base.llm_best_rewrite
"""

import ast
import json
import os
import sys
from typing import List, Optional

# Add parent directory to path to import constants
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import pandas as pd
from tqdm import tqdm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

import constants

try:
    from langchain_openai import ChatOpenAI

    _MODEL_BACKEND = "openai"
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore

        _MODEL_BACKEND = "openai"
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LangChain OpenAI backend not available. Install "
            "'langchain-openai' or set up a compatible chat model."
        ) from e


# ─────────────────────────────────────────────
# Permission helper (mirrors llm_q.py)
# ─────────────────────────────────────────────

def _request_api_permission() -> bool:
    """Request permission before making API calls."""
    if not constants.REQUIRE_API_PERMISSION:
        return True

    print("\n" + "=" * 60)
    print("API CALL PERMISSION REQUEST")
    print("=" * 60)
    print("This will make API calls to OpenAI to select best rewrites (costs may apply).")
    print("Do you want to proceed? (yes/no): ", end="")

    response = input().strip().lower()
    return response in ("yes", "y")


def _build_model():
    """Build the OpenAI model with temperature=0 for deterministic selection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
    model_name = os.environ.get("OPENAI_MODEL", constants.LLM_MODEL_NAME)
    return ChatOpenAI(model=model_name, temperature=0)


# ─────────────────────────────────────────────
# Core selection logic
# ─────────────────────────────────────────────

def _select_best_rewrite(
    question_text: str,
    rewrites: List[str],
    live: bool = False,
    request_permission: bool = True,
) -> int:
    """Ask an LLM which of the candidate rewrites will retrieve the best results.

    Args:
        question_text: The original query.
        rewrites: List of candidate rewrites to choose from.
        live: If False, returns index 0 without API call (test mode).
        request_permission: If True and REQUIRE_API_PERMISSION is True, asks user.

    Returns:
        Index (0-based) of the best rewrite in *rewrites*.
    """
    if not live:
        return 0

    if request_permission and not _request_api_permission():
        print("API call cancelled. Returning index 0.")
        return 0

    if len(rewrites) <= 1:
        return 0

    model = _build_model()

    # Build numbered candidate list
    candidates_str = "\n".join(
        f"{i}: {rw}" for i, rw in enumerate(rewrites)
    )

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert in information retrieval.\n\n"
                    "Given an original query and a numbered list of candidate "
                    "query rewrites, select the ONE rewrite that is most likely "
                    "to retrieve the most relevant documents when used with "
                    "BM25 and dense retrieval systems.\n\n"
                    "Consider:\n"
                    "- Keyword coverage and specificity\n"
                    "- Entity surfacing and attribute extraction\n"
                    "- Retrieval diversity and recall potential\n"
                    "- Conciseness vs. informativeness tradeoff\n\n"
                    "Return strictly JSON: {{\"best_index\": <int>}}\n"
                    "where <int> is the 0-based index of the best rewrite.\n"
                    "Do not include any explanation."
                ),
            ),
            (
                "user",
                "Original query: {question}\n\nCandidate rewrites:\n{candidates}",
            ),
        ]
    )

    parser = JsonOutputParser()
    chain = template | model | parser

    try:
        output = chain.invoke({"question": question_text, "candidates": candidates_str})

        if isinstance(output, dict) and "best_index" in output:
            idx = int(output["best_index"])
        elif isinstance(output, int):
            idx = output
        else:
            print(f"Unexpected output format: {output}. Defaulting to 0.")
            idx = 0

        # Clamp to valid range
        idx = max(0, min(idx, len(rewrites) - 1))

    except Exception as e:
        print(f"LLM selection failed: {e}. Defaulting to index 0.")
        idx = 0

    return idx


def select_best_rewrite(
    question_text: str,
    rewrites: List[str],
    live: bool = False,
) -> int:
    """Public entry point — selects best rewrite index (no permission prompt).

    Args:
        question_text: Original query.
        rewrites: Candidate rewrites.
        live: If False, test mode.

    Returns:
        Index of best rewrite.
    """
    return _select_best_rewrite(
        question_text, rewrites, live=live, request_permission=False
    )


# ─────────────────────────────────────────────
# Batch processing
# ─────────────────────────────────────────────

def preprep_best_rewrites(
    name: str,
    split: Optional[str] = None,
    path_dt: Optional[str] = None,
    live: bool = False,
    n_values: Optional[List[int]] = None,
) -> pd.DataFrame:
    """For each query, select the best rewrite for each N value.

    Reads the existing rewrites CSV produced by ``llm_q.preprep_dataset``,
    then for each N in *n_values* asks the LLM to pick the best from the
    first N rewrites.

    Output CSV: ``{name}_best_rewrite_{split}.csv``

    Columns: ``q_id, original_query, best_of_1, best_of_4, …, best_of_30``

    Args:
        name: Dataset name.
        split: Data split (defaults to constants.DATA_SPLIT).
        path_dt: Data directory (defaults to constants.DATA_PATH).
        live: If False, test mode (always picks index 0).
        n_values: List of N values to evaluate. Defaults to
            constants.EXPERIMENT_2_N_REWRITES.

    Returns:
        DataFrame with best rewrites.
    """
    if split is None:
        split = constants.DATA_SPLIT
    if path_dt is None:
        path_dt = constants.DATA_PATH
    if n_values is None:
        n_values = constants.EXPERIMENT_2_N_REWRITES

    # Load existing rewrites CSV
    n_rewrites = constants.LLM_REWRITES_PER_QUERY
    rewrites_filename = f"{name}_rewritten_queries_dataset_{n_rewrites}_{split}.csv"
    rewrites_filepath = os.path.join(path_dt, rewrites_filename)

    if not os.path.exists(rewrites_filepath):
        raise FileNotFoundError(
            f"Rewrites CSV not found: {rewrites_filepath}. "
            "Run 'dataset_setup.py --steps rewrites' first."
        )

    rewrites_df = pd.read_csv(rewrites_filepath)
    rewrites_df["q_id"] = rewrites_df["q_id"].astype(str)

    # Need original queries — load topics
    from code_base.dataset_factory import get_dataset

    _, topics_df, _ = get_dataset(name, split)
    query_lookup = dict(zip(topics_df["qid"].astype(str), topics_df["query"]))

    # Output file
    output_filename = f"{name}_best_rewrite_{split}.csv"
    output_filepath = os.path.join(path_dt, output_filename)

    # Build column names
    col_names = [f"best_of_{n}" for n in n_values]

    # Check for existing partial results (incremental mode)
    existing_q_ids = set()
    existing_rows = []
    if os.path.exists(output_filepath):
        existing_df = pd.read_csv(output_filepath)
        existing_df["q_id"] = existing_df["q_id"].astype(str)
        existing_q_ids = set(existing_df["q_id"].values)
        existing_rows = existing_df.to_dict("records")
        print(
            f"Found existing best-rewrite CSV with {len(existing_q_ids)} queries. "
            "Skipping already-processed q_ids."
        )

    # Filter to unprocessed queries
    pending_df = rewrites_df[~rewrites_df["q_id"].isin(existing_q_ids)]

    if pending_df.empty:
        print("All queries already processed. Returning existing dataset.")
        return pd.read_csv(output_filepath)

    # Request permission once
    if live and not _request_api_permission():
        print("Best-rewrite selection cancelled by user.")
        if existing_rows:
            return pd.DataFrame(existing_rows)
        return pd.DataFrame(columns=["q_id", "original_query"] + col_names)

    new_rows = []
    for _, row in tqdm(
        pending_df.iterrows(),
        total=len(pending_df),
        desc="Selecting best rewrites",
    ):
        q_id = str(row["q_id"])
        rewrites = ast.literal_eval(row["rewritten_queries"])
        original_query = query_lookup.get(q_id, rewrites[0] if rewrites else "")

        result_row = {"q_id": q_id, "original_query": original_query}

        for n in n_values:
            col = f"best_of_{n}"

            if n <= 1:
                # best_of_1 is always original query
                result_row[col] = original_query
                continue

            # Take first n rewrites (they are 0-indexed from LLM output)
            candidates = rewrites[:n]

            # Pad if not enough rewrites
            if len(candidates) < n:
                candidates = candidates + [candidates[-1]] * (n - len(candidates))

            best_idx = _select_best_rewrite(
                original_query,
                candidates,
                live=live,
                request_permission=False,
            )
            result_row[col] = candidates[best_idx]

        new_rows.append(result_row)

    # Combine with existing
    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)

    # Save
    print(f"Saving best rewrites to: {output_filepath}")
    df.to_csv(output_filepath, index=False)
    return df


def get_best_rewrite(
    name: str,
    q_id: str,
    n: int,
    split: Optional[str] = None,
    path_dt: Optional[str] = None,
) -> str:
    """Look up the best rewrite for a given query and N value.

    Args:
        name: Dataset name.
        q_id: Query ID.
        n: N value (from EXPERIMENT_2_N_REWRITES).
        split: Data split.
        path_dt: Data directory.

    Returns:
        The best rewrite string.
    """
    if split is None:
        split = constants.DATA_SPLIT
    if path_dt is None:
        path_dt = constants.DATA_PATH

    filename = f"{name}_best_rewrite_{split}.csv"
    filepath = os.path.join(path_dt, filename)

    df = pd.read_csv(filepath)
    df["q_id"] = df["q_id"].astype(str)

    col = f"best_of_{n}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found. Available: {list(df.columns)}")

    match = df[df["q_id"] == str(q_id)]
    if match.empty:
        raise ValueError(f"q_id {q_id} not found in best-rewrite dataset.")

    return match[col].values[0]


if __name__ == "__main__":
    # Quick test
    example_q = "What are the health benefits of regular exercise?"
    example_rewrites = [
        "health benefits exercise",
        "why is exercise good for you",
        "advantages of regular physical activity on health",
        "exercise impact on human body wellness",
        "medical benefits of working out regularly",
    ]
    idx = select_best_rewrite(example_q, example_rewrites, live=False)
    print(f"Test mode — best index: {idx}, rewrite: {example_rewrites[idx]}")
