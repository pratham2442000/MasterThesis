"""
LLM-as-Judge Measure
=====================
Custom ir_measures measure that uses an LLM to rate the quality of
generated answers on a 1–10 scale (pointwise evaluation).

Mirrors the pyterrier_rag ``LLMasJudge`` interface but fixes the
BackendOutput handling and works fully offline.

Usage::

    from code_base.llm_judge import LLMasJudge

    judge = LLMasJudge("causal", "./models/Llama-3.2-3B-Instruct")
    # Use in pt.Experiment as eval_metrics=[judge]
"""

import re

import ir_measures

# ─── Prompt Templates ────────────────────────────────────

SYSTEM_MESSAGE = "You are a helpful assistant."

JUDGE_PROMPT = (
    "You are an expert evaluator assessing a question-answering system.\n\n"
    "[User Query]\n"
    "{query}\n\n"
    "[Gold Reference Answer]\n"
    "{reference}\n\n"
    "[System Generated Answer]\n"
    "{prediction}\n\n"
    "Based on the Gold Reference Answer, evaluate the factual accuracy and semantic correctness "
    "of the System Generated Answer to the User Query.\n"
    "Does the generated answer correctly convey the same information as the gold reference?\n"
    "Provide a brief explanation, then rate the response on a scale of 1 to 10 by "
    'strictly following this format: "[[rating]]", for example: "Rating: [[10]]" or "Rating: [[1]]".'
)

# ─── Module-level cache ──────────────────────────────────

_backend_obj = None
_backend_key = None  # (backend_type, model_id) to detect config changes


def _get_backend(backend_type, model_id):
    """Lazy-init and cache the backend singleton."""
    global _backend_obj, _backend_key
    key = (backend_type, model_id)
    if _backend_obj is None or _backend_key != key:
        from pyterrier_rag.backend import get_backend
        _backend_obj = get_backend(backend_type, model_id)
        _backend_key = key
    return _backend_obj


def _parse_rating(text):
    """Extract an integer rating from LLM output.

    Handles formats like:
        - ``Rating: [[5]]``
        - ``[[7]]``
        - ``5``
        - Free-text followed by a number
    """
    # Try [[N]] pattern first (official format)
    match = re.search(r"\[\[(\d+)]]", text)
    if match:
        return int(match.group(1))

    # Fallback: first standalone integer
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))

    return 0


def _llmjudge_fn(qrels, res, backend_type, model_id, rel=3, agg="max"):
    """Evaluate system output using an LLM judge.

    Keeps the same interface as pyterrier_rag's ``llmjudge_fn``:

    - ``qrels`` must have columns ``query_id``, ``relevance``, ``text``.
    - ``res`` must have columns ``query_id``, ``qanswer``.

    Args:
        qrels: Reference DataFrame (gold answers).
        res: System output DataFrame.
        backend_type: Backend identifier ('causal', 'openai', etc.).
        model_id: Model name or local path.
        rel: Minimum relevance threshold for gold answers.
        agg: Aggregation over multiple gold references
             ('max', 'min', 'avg', 'sum', 'none').

    Returns:
        Aggregated score (int/float) or list if ``agg='none'``.
    """
    assert len(res), "Empty res df provided"
    assert len(qrels), "Empty qrels df provided"

    # Filter by relevance threshold
    qrels = qrels[qrels.relevance >= rel]
    assert len(qrels), f"No qrels with relevance >= {rel}"

    backend = _get_backend(backend_type, model_id)

    predictions = res["qanswer"].to_list()
    assert len(predictions) == 1, "Expected exactly 1 prediction per query"

    queries = res["query"].to_list()
    query = queries[0] if len(queries) > 0 else ""

    references = qrels["text"].to_list()

    # One prompt per (prediction, reference) pair
    prediction = predictions[0]
    prompts = [
        JUDGE_PROMPT.format(query=query, reference=r, prediction=prediction)
        for r in references
    ]

    outputs = backend.generate(prompts)

    parsed_ints = []
    for output in outputs:
        # BackendOutput has a .text attribute
        text = output.text if hasattr(output, "text") else str(output)
        score = _parse_rating(text)
        # Clamp to 1-10 range
        score = max(1, min(10, score)) if score > 0 else 0
        parsed_ints.append(score)

    # Aggregate scores across references
    if agg == "max":
        return max(parsed_ints)
    elif agg == "avg":
        return sum(parsed_ints) / len(parsed_ints)
    elif agg == "sum":
        return sum(parsed_ints)
    elif agg == "min":
        return min(parsed_ints)
    elif agg == "none":
        return parsed_ints
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")


def LLMasJudge(backend_type, model_id):
    """Create an ir_measures by-query measure using an LLM as judge.

    Drop-in replacement for ``pyterrier_rag.measures.LLMasJudge``.

    Args:
        backend_type: Backend identifier ('causal', 'openai', etc.).
        model_id: Model name or local path.

    Returns:
        ir_measures Measure usable in ``pt.Experiment(eval_metrics=[...])``.
    """
    return ir_measures.define_byquery(
        lambda qrels, res: _llmjudge_fn(
            qrels, res, backend_type=backend_type, model_id=model_id
        ),
        name="LLMasJudge",
        support_cutoff=False,
    )


__all__ = ["LLMasJudge", "JUDGE_PROMPT"]
