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

import random

random.seed(42)


try:
    from langchain_openai import ChatOpenAI

    _MODEL_BACKEND = "openai"
except Exception:
    # Fallback to community ChatOpenAI import path (older LangChain versions)
    try:
        # noinspection PyProtectedMember
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
                    "You are an expert retrieval engineer.\n\n"
                    "Your task is to generate high-diversity, retrieval-oriented rewrites of a user query.\n\n"
                    "Follow these rules strictly:\n\n"
                    "1. Diversity Types (you MUST cover all):\n\n"
                    "A. Lexical Diversity\n"
                    "Use different surface forms, synonyms, and industry-specific vocabulary.\n"
                    "Avoid trivial paraphrases (e.g., swapping “nation” with “country”).\n"
                    "Make rewrites differ by at least 30–40% token overlap.\n\n"
                    "B. Structural Diversity\n"
                    "Produce different grammatical forms:\n"
                    "question → declarative\n"
                    "“X of Y” → “Y’s X”\n"
                    "long → short\n"
                    "inverted order\n"
                    "keyword-style queries\n"
                    "multi-hop versions\n\n"
                    "C. Semantic Diversity\n"
                    "Introduce meaningfully different retrieval intents:\n"
                    "entity identification\n"
                    "attribute extraction\n"
                    "contextual expansion\n"
                    "source-specific phrasing (“IMDb”, “Wikipedia”, “Wikidata”)\n"
                    "catalog / KB schema terms (“citizenship”, “birthplace”, “founded”, “parent organization”)\n"
                    "related concepts that commonly co-occur in documents\n\n"
                    "D. Multi-Hop Coverage\n"
                    "If the original query has latent steps (e.g., film → director → nationality), generate rewrites that address each hop separately.\n\n"
                    "E. Document-Style Diversity\n"
                    "Match the styles of documents from:\n"
                    "Wikipedia\n"
                    "Wikidata\n"
                    "StackExchange\n"
                    "IMDb / Rotten Tomatoes\n"
                    "News\n"
                    "Scholarly articles\n"
                    "Product listings\n"
                    "APIs and schemas\n\n"
                    "F. Domain Adaptation\n"
                    "If the domain is:\n"
                    "movies → use film metadata terms\n"
                    "people → biography, occupation, awards\n"
                    "geography → location hierarchy\n"
                    "products → specs, compatibility\n"
                    "science → definitions, formulas, variations\n"
                    "legal → statutes, regulations\n"
                    "finance → metrics, ratios, filings\n"
                    "web search → keywords + entities\n"
                    "Always produce rewrites that reflect real-world search logs, not artificial paraphrases.\n\n"
                    "2. Rewrites That Are Forbidden\n"
                    "Never produce rewrites that are:\n"
                    "trivial paraphrases\n"
                    "differ only by one swapped noun (“nation/country/origin”)\n"
                    "only reorder words with no semantic change\n"
                    "too vague or generic\n"
                    "contradict the original question\n"
                    "introduce irrelevant information\n"
                    "hallucinate specifics\n\n"
                    "3. Rewrites That Are Required\n"
                    "For every user query, ALWAYS produce the following categories:\n"
                    "Entity-Surfacing Rewrite (“Who/what is X?”, “Identify the entity Y”, “Metadata about Z”)\n"
                    "Attribute-Focused Rewrite (“citizenship”, “height”, “release date”, “definition”, “formula”, “causes”, etc.)\n"
                    "Source-Targeted Rewrite (“Wikipedia page for X”, “IMDb listing for Y”, “scholarly definition of Z”)\n"
                    "Keyword / Telegraphed Query (short, high-precision retrieval queries)\n"
                    "Decomposed / Multi-Hop Version\n"
                    "Break the query into constituent retrieval pieces.\n"
                    "Schema / KB-Style Rewrite\n"
                    "Use vocabulary like:\n"
                    "“instance of”, “country of citizenship”, “located in”, “parent company”, “date founded”,\n"
                    "“chemical composition”, “mathematical definition”.\n"
                    "Context Expansion Rewrite\n"
                    "Add relevant context typically present in documents that aid recall (e.g., “movie”, “biography”, “technical specification”, “historical background”).\n\n"
                    "4. Output Format\n"
                    "Produce {k} rewrites.\n"
                    "Each rewrite must be meaningfully distinct.\n"
                    "Order does not matter.\n"
                    "No commentary, only rewrites.\n"
                    "No numbering if asked for “raw list”.\n"
                    "Return strictly JSON: a list of exactly {k} unique, concise paraphrases. Do not include any explanation or keys, only the JSON array.\n\n"
                    "5. Optimization Behavior\n"
                    "Maximize recall, not precision.\n"
                    "Assume both BM25 and dense retrievers will be used.\n"
                    "Favor rewrites that expose latent entities, metadata, or structured attributes.\n"
                    "For complex or ambiguous queries, generate rewrites that cover different plausible interpretations.\n"
                    "Balance specificity and generality to increase coverage."
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


def preprep_dataset(
        name: str,
        questions,
        split=None,
        path_dt=None,
        live: bool = False,
        type_distribution: dict[str, float] = None,
        sample_limit: int = None,
):
    """
    Preprocess a list of questions to get rewritten queries each.
    Uses constants for default configuration.

    Args:
        name: Dataset name
        questions: List of (question, q_id) or (question, q_id, type) tuples
        split: Data split (defaults to constants.DATA_SPLIT)
        path_dt: Path to data directory (defaults to constants.DATA_PATH)
        live: If False, uses test mode without API calls
        type_distribution: Dict mapping query type to fraction (e.g. {'comparison': 0.8, 'bridge': 0.2})
                           If provided, questions must contain type information.
        sample_limit: Total number of queries to sample if type_distribution is used. 
                      If None, uses len(questions).
    """
    if split is None:
        split = constants.DATA_SPLIT
    if path_dt is None:
        path_dt = constants.DATA_PATH

    # Request permission once at the start if live mode
    if live and not _request_api_permission():
        print("Dataset preprocessing cancelled by user.")
        return pd.DataFrame(columns=["q_id", "rewritten_queries"])

    # Handle distribution sampling if requested
    if type_distribution:

        print(f"Applying type distribution: {type_distribution}")

        # Verify questions have type info
        if not questions or len(questions[0]) < 3:
            print(
                "Warning: Type distribution requested but questions tuples do not contain type info. Skipping distribution sampling.")
        else:
            # Group by type
            by_type = {}
            for q in questions:
                q_type = q[2]
                if q_type not in by_type:
                    by_type[q_type] = []
                by_type[q_type].append(q)

            # Determine total sample size
            total_count = sample_limit if sample_limit is not None else len(questions)

            sampled_questions = []
            for q_type, fraction in type_distribution.items():
                if q_type not in by_type:
                    print(f"Warning: Type '{q_type}' not found in questions.")
                    continue

                target_count = int(total_count * fraction)
                available = by_type[q_type]

                if len(available) < target_count:
                    print(
                        f"Warning: Not enough questions of type '{q_type}' (have {len(available)}, want {target_count}). Taking all.")
                    sampled_questions.extend(available)
                else:
                    sampled_questions.extend(random.sample(available, target_count))

            # Update questions list
            questions = sampled_questions
            # Shuffle to mix types
            random.shuffle(questions)
            print(f"Sampled {len(questions)} questions based on distribution.")


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
            q for q in questions if q[1] not in existing_q_ids
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

    # sample limit if provided
    if sample_limit is not None:
        questions = random.sample(questions, sample_limit)

    for item in tqdm(questions, desc="Rewriting queries"):
        question = item[0]
        q_id = item[1]
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
    df["q_id"] = df["q_id"].astype(str)
    
    if q_id not in df["q_id"].values:
        raise ValueError(f"q_id {q_id} not found in the dataset.")

    rewritten = df[df["q_id"] == q_id]["rewritten_queries"].apply(lambda x: ast.literal_eval(x)).values[0]
    rewritten = _normalize_to_k(rewritten, k)
    return rewritten


if __name__ == "__main__":
    example_q = "What are the health benefits of regular exercise?"
    k = 5
    print(query_llm(example_q, k, live=True))
