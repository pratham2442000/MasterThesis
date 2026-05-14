"""
Robust LLM-as-Judge implementation for PyTerrier RAG
===================================================


Expected input columns
----------------------

Run dataframe (`res`)
~~~~~~~~~~~~~~~~~~~~~
Required:
    - qanswer

Optional:
    - query
    - qid / query_id

Qrels dataframe (`qrels`)
~~~~~~~~~~~~~~~~~~~~~~~~~
Required:
    - text
    - relevance

Optional:
    - qid / query_id

Example
-------

    import pyterrier as pt
    from code_base.llm_judge import LLMasJudge

    judge = LLMasJudge(
        backend_type="causal",
        model_id="./models/Llama-3.2-3B-Instruct",
        aggregation="max",
    )

    pt.Experiment(
        [pipeline],
        topics,
        qrels,
        eval_metrics=[judge],
    )
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, List

import ir_measures

# -------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------

SYSTEM_MESSAGE = "You are a precise and rigorous evaluator."

JUDGE_PROMPT = """
You are an expert evaluator assessing a question-answering system.

[User Query]
{query}

[Gold Reference Answer]
{reference}

[System Generated Answer]
{prediction}

Evaluate whether the generated answer is factually accurate and semantically
consistent with the gold reference answer.

Scoring rubric:
1-2   : Completely incorrect
3-4   : Mostly incorrect
5-6   : Partially correct
7-8   : Mostly correct
9-10  : Fully correct

Provide:
1. A short explanation
2. A rating in EXACTLY this format:

Rating: [[N]]

Where N is an integer from 1 to 10.
""".strip()


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def _validate_columns(df, required, df_name):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{df_name} missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )


def _safe_query(res):
    """
    Query text is often dropped by pt.Experiment().
    Try multiple fallback strategies.
    """

    if "query" in res.columns and len(res["query"]):
        q = res["query"].iloc[0]
        if isinstance(q, str):
            return q

    if "question" in res.columns and len(res["question"]):
        q = res["question"].iloc[0]
        if isinstance(q, str):
            return q

    warnings.warn(
        "No query text found in run dataframe. "
        "LLM judge prompt quality may degrade."
    )

    return ""


def _parse_rating(text: str) -> int:
    """
    Robust parsing for multiple output styles.
    """

    if not text:
        return 0

    patterns = [
        r"\[\[(\d+)\]\]",  # [[7]]
        r"rating\s*:\s*(\d+)",  # Rating: 7
        r"score\s*:\s*(\d+)",  # Score: 7
        r"\b(10|[1-9])\b",  # standalone integer
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                return max(1, min(10, val))
            except Exception:
                pass

    return 0


def _normalize_outputs(outputs: Any) -> List[str]:
    """
    Normalize all backend output formats into List[str].

    Handles:
        - str
        - BackendOutput
        - List[str]
        - List[BackendOutput]
        - generators
        - nested structures
    """

    if outputs is None:
        return []

    if isinstance(outputs, str):
        return [outputs]

    if hasattr(outputs, "text"):
        return [str(outputs.text)]

    if not isinstance(outputs, Iterable):
        return [str(outputs)]

    normalized = []

    for item in outputs:

        if item is None:
            continue

        if isinstance(item, str):
            normalized.append(item)

        elif hasattr(item, "text"):
            normalized.append(str(item.text))

        else:
            normalized.append(str(item))

    return normalized


# -------------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------------


@dataclass
class LLMJudgeEvaluator:
    backend_type: str
    model_id: str
    aggregation: str = "max"
    rel: int = 3

    def __post_init__(self):

        from pyterrier_rag.backend import get_backend

        self.backend = get_backend(
            self.backend_type,
            self.model_id,
        )

    # -----------------------------------------------------------------

    def _generate(self, prompts: List[str]) -> List[str]:
        """
        Handle both batched and non-batched backends.
        """

        try:
            outputs = self.backend.generate(prompts)
            normalized = _normalize_outputs(outputs)

            if len(normalized) == len(prompts):
                return normalized

        except Exception:
            pass

        # Fallback: sequential generation
        outputs = []

        for prompt in prompts:

            out = self.backend.generate(prompt)
            normalized = _normalize_outputs(out)

            if normalized:
                outputs.append(normalized[0])
            else:
                outputs.append("")

        return outputs

    # -----------------------------------------------------------------

    def score(self, query, prediction, references):

        if not references:
            raise ValueError("No reference answers available")

        prompts = [
            JUDGE_PROMPT.format(
                query=query,
                reference=reference,
                prediction=prediction,
            )
            for reference in references
        ]

        raw_outputs = self._generate(prompts)

        scores = []

        for output in raw_outputs:

            score = _parse_rating(output)

            if score <= 0:
                warnings.warn(
                    f"Could not parse rating from output:\n{output}"
                )
                continue

            scores.append(score)

        if not scores:
            return 0.0

        return self._aggregate(scores)

    # -----------------------------------------------------------------

    def _aggregate(self, scores):

        if self.aggregation == "max":
            return float(max(scores))

        if self.aggregation == "min":
            return float(min(scores))

        if self.aggregation == "avg":
            return float(sum(scores) / len(scores))

        if self.aggregation == "sum":
            return float(sum(scores))

        raise ValueError(
            f"Unknown aggregation method: {self.aggregation}"
        )


# -------------------------------------------------------------------------
# ir_measures Wrapper
# -------------------------------------------------------------------------


def LLMasJudge(
        backend_type: str,
        model_id: str,
        aggregation: str = "max",
        rel: int = 3,
):
    """
    Create a robust ir_measures-compatible LLM judge metric.
    """

    evaluator = LLMJudgeEvaluator(
        backend_type=backend_type,
        model_id=model_id,
        aggregation=aggregation,
        rel=rel,
    )

    metric_name = (
        f"LLMasJudge("
        f"{backend_type},"
        f"{model_id},"
        f"agg={aggregation}"
        f")"
    )

    # -----------------------------------------------------------------

    def _measure_fn(qrels, res):

        if len(qrels) == 0:
            return 0.0

        if len(res) == 0:
            return 0.0

        _validate_columns(
            qrels,
            required=["text", "relevance"],
            df_name="qrels",
        )

        _validate_columns(
            res,
            required=["qanswer"],
            df_name="res",
        )

        # Filter relevance
        qrels_filtered = qrels[qrels["relevance"] >= rel]

        if len(qrels_filtered) == 0:
            return 0.0

        prediction = str(res["qanswer"].iloc[0])

        query = _safe_query(res)

        references = [
            str(x)
            for x in qrels_filtered["text"].tolist()
            if x is not None
        ]

        if not references:
            return 0.0

        return evaluator.score(
            query=query,
            prediction=prediction,
            references=references,
        )

    # -----------------------------------------------------------------

    return ir_measures.define_byquery(
        _measure_fn,
        name=metric_name,
        support_cutoff=False,
        run_inputs=["qanswer"],
        qrel_inputs=["text", "relevance"],
    )


__all__ = [
    "LLMasJudge",
    "LLMJudgeEvaluator",
    "JUDGE_PROMPT",
]
