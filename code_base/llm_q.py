import ast
import json
import os
import sys
from typing import List

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
    # Fallback to community ChatOpenAI import path (older LangChain versions)
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore

        _MODEL_BACKEND = "openai"
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LangChain OpenAI backend not available. Install "
            "'langchain-openai' or set up a compatible chat model."
        ) from e


def _request_api_permission() -> bool:
    """
    Request permission before making API calls.
    Returns True if permission is granted, False otherwise.
    """
    if not constants.REQUIRE_API_PERMISSION:
        return True

    print("\n" + "=" * 60)
    print("API CALL PERMISSION REQUEST")
    print("=" * 60)
    print("This will make an API call to OpenAI (costs may apply).")
    print("Do you want to proceed? (yes/no): ", end="")

    response = input().strip().lower()
    return response in ("yes", "y")


def _build_model(temperature: float = 0.7):
    """Build the OpenAI model. Uses constants for default values."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
    # Model name via env; default from constants
    model_name = os.environ.get("OPENAI_MODEL", constants.LLM_MODEL_NAME)
    # Use provided temperature or default from constants
    if temperature == 0.7:  # Default value
        temp = constants.LLM_TEMPERATURE
    else:
        temp = temperature
    return ChatOpenAI(model=model_name, temperature=temp)


def _normalize_to_k(items: List[str], k: int) -> List[str]:
    cleaned = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    if len(cleaned) >= k:
        return cleaned[:k]
    return cleaned + [cleaned[-1] if cleaned else ""] * (k - len(cleaned))


def _query_llm(
    question_text: str, k: int, live: bool = False, request_permission: bool = True
) -> List[str]:
    """
    Return exactly k rewritten queries via LangChain with robust parsing.

    Args:
        question_text: The original question to rewrite
        k: Number of rewritten queries to generate
        live: If False, returns dummy paraphrases without API call (test mode)
        request_permission: If True and REQUIRE_API_PERMISSION is True, asks for permission

    Returns:
        List of k rewritten queries
    """
    if not live:
        # Test mode: return dummy paraphrases without making API call
        return [f"{question_text} (paraphrase {i+1})" for i in range(k)]

    # Request permission before making API call
    if request_permission and not _request_api_permission():
        print("API call cancelled by user. Returning original query repeated k times.")
        return [question_text] * k

    model = _build_model()

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You rewrite search queries. Return strictly JSON: a list of "
                    "exactly {k} unique, concise paraphrases. Do not include any "
                    "explanation or keys, only the JSON array."
                ),
            ),
            ("user", "Original question: {question}"),
        ]
    )

    parser = JsonOutputParser()
    chain = template | model | parser
    try:
        output = chain.invoke({"k": k, "question": question_text})
        print(output)
        # parser returns Python object already
        if not isinstance(output, list):
            # In rare cases models wrap with an object – try to coerce
            if isinstance(output, dict):
                # attempt common keys
                for key in ("queries", "paraphrases", "results", "items"):
                    if key in output and isinstance(output[key], list):
                        output = output[key]
                        break
        if not isinstance(output, list):
            raise ValueError("Parsed output is not a list")
        result = _normalize_to_k([str(x) for x in output], k)
    except Exception as e:
        # Last resort: try to load as JSON text or fallback to identity
        print(f"Parsing failed: {e}")
        try:
            loaded = json.loads(str(e))
            if isinstance(loaded, list):
                result = _normalize_to_k([str(x) for x in loaded], k)
            else:
                result = _normalize_to_k([question_text], k)
        except Exception as pain:
            print(f"Parsing failed: {pain}")
            result = _normalize_to_k([question_text], k)

    return result


def query_llm(question_text: str, k: int, live: bool = False) -> List[str]:
    """
    Public function to query LLM for rewritten queries.
    This is the main entry point for query rewriting.

    Args:
        question_text: The original question to rewrite
        k: Number of rewritten queries to generate
        live: If False, returns dummy paraphrases without API call (test mode)

    Returns:
        List of k rewritten queries
    """
    # For batch processing, only request permission once at the start
    # This is handled by the caller (preprep_dataset)
    return _query_llm(question_text, k, live=live, request_permission=False)


def preprep_dataset(name: str, questions, split=None, path_dt=None, live: bool = False):
    """
    Preprocess a list of questions to get rewritten queries each.
    Uses constants for default configuration.

    Args:
        name: Dataset name
        questions: List of (question, q_id) tuples
        split: Data split (defaults to constants.DATA_SPLIT)
        path_dt: Path to data directory (defaults to constants.DATA_PATH)
        live: If False, uses test mode without API calls
    """
    if split is None:
        split = constants.DATA_SPLIT
    if path_dt is None:
        path_dt = constants.DATA_PATH

    # Request permission once at the start if live mode
    if live and not _request_api_permission():
        print("Dataset preprocessing cancelled by user.")
        return pd.DataFrame(columns=["q_id", "rewritten_queries"])

    all_rewritten = []
    n_rewrites = constants.LLM_REWRITES_PER_QUERY

    filename = f"{name}_rewritten_queries_dataset_{n_rewrites}_{split}.csv"

    if os.path.exists(os.path.join(path_dt, filename)):
        print(
            f"Rewritten queries dataset already exists at {os.path.join(path_dt, filename)}. "
            "Loading existing dataset and skipping q_ids that are already present."
        )

        existing_df = pd.read_csv(os.path.join(path_dt, filename))
        existing_q_ids = set(existing_df["q_id"].values)
        questions = [
            (q, q_id) for q, q_id in questions if q_id not in existing_q_ids
        ]
        all_rewritten.extend(
            zip(
                existing_df["q_id"].values,
                existing_df["rewritten_queries"].apply(lambda x: ast.literal_eval(x)),
            )
        )

    if not questions:
        print("No new questions to process. Returning existing dataset.")
        return pd.DataFrame(all_rewritten, columns=["q_id", "rewritten_queries"])

    for question, q_id in tqdm(questions, desc="Rewriting queries"):
        rewritten = _query_llm(
            question, n_rewrites, live=live, request_permission=False
        )
        all_rewritten.append((q_id, rewritten))

    # Save to dataframe
    columns = ["q_id", "rewritten_queries"]
    df = pd.DataFrame(all_rewritten, columns=columns)

    filepath = os.path.join(path_dt, filename)
    print(f"Saving rewritten queries to CSV: {filepath}")
    df.to_csv(filepath, index=False)
    return df


def rewrite_single_query(
        name, q_id: str, k: int, split=None, path_dt=None
) -> List[str]:
    """
    Rewrite a single query to get k rewritten queries from preprocessed dataset.
    Uses constants for default configuration.

    Args:
        name: Dataset name
        q_id: Query ID to retrieve
        k: Number of rewritten queries to return
        split: Data split (defaults to constants.DATA_SPLIT)
        path_dt: Path to data directory (defaults to constants.DATA_PATH)

    Returns:
        List of k rewritten queries
    """
    if split is None:
        split = constants.DATA_SPLIT
    if path_dt is None:
        path_dt = constants.DATA_PATH

    n_rewrites = constants.LLM_REWRITES_PER_QUERY
    filename = f"{name}_rewritten_queries_dataset_{n_rewrites}_{split}.csv"
    filepath = os.path.join(path_dt, filename)

    df = pd.read_csv(filepath)
    if q_id not in df["q_id"].values:
        raise ValueError(f"q_id {q_id} not found in the dataset.")

    rewritten = df[df["q_id"] == q_id]["rewritten_queries"].apply(lambda x: ast.literal_eval(x)).values[0]
    rewritten = _normalize_to_k(rewritten, k)
    return rewritten


if __name__ == "__main__":
    example_q = "What are the health benefits of regular exercise?"
    k = 5
    print(query_llm(example_q, k, live=True))
