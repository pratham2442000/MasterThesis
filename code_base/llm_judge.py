"""
Robust LLM-as-Judge implementation for PyTerrier RAG
===================================================

Uses LangChain with a local HuggingFace model (Llama) for offline
evaluation. No external API calls required.

Expected input columns
----------------------

Run dataframe (``res``)
~~~~~~~~~~~~~~~~~~~~~
Required:
    - qanswer

Optional:
    - query
    - qid / query_id

Qrels dataframe (``qrels``)
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
from dataclasses import dataclass, field
from typing import Any

import ir_measures
import pandas as pd

# -------------------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------------------

SYSTEM_MESSAGE = "You are a precise and rigorous evaluator."

# Max tokens for judge generation — only need a single integer
JUDGE_MAX_NEW_TOKENS = 2000

# Llama-3.2-Instruct chat-template wrapped prompts
JUDGE_PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a strict factual evaluator. Compare the system answer to the gold reference. "
    "Reply with ONLY a single integer from 1 to 10.\n\n"
    "Scoring rules (strict):\n"
    "- 1-2: The answer is completely wrong (e.g., gives a year when a name is expected, "
    "answers 'None' or 'No information', or provides an irrelevant fact).\n"
    "- 3-4: The answer is of the correct type (e.g., a person name) but factually incorrect.\n"
    "- 5-6: The answer is partially correct (e.g., contains a correct detail but misses the main answer).\n"
    "- 7-8: The answer is mostly correct but has minor inaccuracies or missing context.\n"
    "- 9-10: The answer exactly matches the gold reference in all essential aspects.\n\n"
    "Examples:\n"
    "Query: which film has the director died first A or B\n"
    "Gold: Film A\n"
    "System: 1979\n"
    "Score: 1\n\n"
    "Query: why did person's wife die\n"
    "Gold: stroke\n"
    "System: 1951\n"
    "Score: 1\n\n"
    "Query: where was the director of film X born\n"
    "Gold: New York\n"
    "System: None of the above\n"
    "Score: 1\n\n"
    "Query: who is person's father\n"
    "Gold: John Smith\n"
    "System: John Smith Jr.\n"
    "Score: 5\n\n"
    "Query: what nationality is the director of film Y\n"
    "Gold: United States\n"
    "System: American\n"
    "Score: 9\n"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Query: {query}\n"
    "Gold answer: {reference}\n"
    "System answer: {prediction}\n"
    "Score (1-10):<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


JUDGE_PROMPT_NO_REF = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "You are a strict evaluator. Assess whether the system answer is reasonable and self-contained.\n"
    "Rate on a scale of 1 (nonsensical/refusal) to 10 (excellent).\n"
    "- If the answer says 'None of the above', 'No information', or refuses to answer, assign 1.\n"
    "- If it contains only a year or a single word without context, assign 3.\n"
    "- Give high scores only for complete, informative answers.\n"
    "- Reply with ONLY the integer.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "Query: {query}\n"
    "System answer: {prediction}\n"
    "Score (1-10):<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


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


def _safe_query(res, topics_lookup=None):
    """
    Query text is often dropped by pt.Experiment().
    Try: res columns → topics_lookup by qid → empty string.
    """

    if "query" in res.columns and len(res["query"]):
        q = res["query"].iloc[0]
        if isinstance(q, str) and q.strip():
            return q

    if "question" in res.columns and len(res["question"]):
        q = res["question"].iloc[0]
        if isinstance(q, str) and q.strip():
            return q

    # Fallback: look up from topics via qid
    if topics_lookup is not None:
        qid = None
        for col in ("qid", "query_id"):
            if col in res.columns and len(res[col]):
                qid = str(res[col].iloc[0])
                break
        if qid and qid in topics_lookup:
            return topics_lookup[qid]

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
        r"rating\s*:\s*\[*(\d+)\]*",  # Rating: 7 or Rating: [7]
        r"score\s*:\s*(\d+)",  # Score: 7
        r"\*\*rating\*\*\s*:\s*(\d+)",  # **Rating**: 7
        r"rate[d]?\s*(\d+)\s*(?:out)?\s*(?:of)?\s*10",  # rated 7 out of 10
        r"(\d+)\s*/\s*10",  # 7/10
        r"^\s*(\d+)\s*$",  # Just a number
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                return max(1, min(10, val))
            except Exception:
                pass

    # Last resort: first standalone integer in entire text
    m = re.search(r"\b(10|[1-9])\b", text)
    if m:
        try:
            return max(1, min(10, int(m.group(1))))
        except Exception:
            pass

    return 0


def clean_prediction(text: str) -> str:
    """Strip trailing '###' chat-template artifacts from generated answer."""
    return re.sub(r"\n*###.*$", "", text, flags=re.DOTALL).strip()


# -------------------------------------------------------------------------
# LangChain Model Builder
# -------------------------------------------------------------------------


def _build_langchain_model(model_id: str, shared_backend: Any = None):
    """
    Build a LangChain-compatible LLM from a local HuggingFace model.

    Auto-detects shared_backend type:
    - HuggingFacePipeline (langchain) → reuse directly
    - pyterrier_rag HuggingFaceBackend → extract model/tokenizer, wrap
    - transformers Pipeline → wrap in HuggingFacePipeline
    - None → load fresh from model_id

    Returns a LangChain LLM (HuggingFacePipeline).
    """
    from langchain_huggingface import HuggingFacePipeline

    # Case 1: Already a LangChain HuggingFacePipeline
    if shared_backend is not None and isinstance(shared_backend, HuggingFacePipeline):
        print("LLM Judge: reusing shared LangChain HuggingFacePipeline")
        return shared_backend

    # Case 2: Raw transformers pipeline
    if shared_backend is not None and hasattr(shared_backend, "task"):
        # Looks like a transformers.Pipeline object
        print("LLM Judge: wrapping shared transformers pipeline")
        return HuggingFacePipeline(pipeline=shared_backend)

    # Case 3: pyterrier_rag HuggingFaceBackend — extract model + tokenizer
    if shared_backend is not None:
        model_obj = None
        tokenizer_obj = None

        # Try common attribute names on pyterrier_rag backends
        for attr in ("model", "_model", "llm"):
            if hasattr(shared_backend, attr):
                model_obj = getattr(shared_backend, attr)
                break

        for attr in ("tokenizer", "_tokenizer", "tok"):
            if hasattr(shared_backend, attr):
                tokenizer_obj = getattr(shared_backend, attr)
                break

        if model_obj is not None and tokenizer_obj is not None:
            import transformers
            print("LLM Judge: extracting model/tokenizer from shared backend")
            tokenizer_obj.padding_side = "left"
            pipe = transformers.pipeline(
                "text-generation",
                model=model_obj,
                tokenizer=tokenizer_obj,
                max_new_tokens=JUDGE_MAX_NEW_TOKENS,
                do_sample=False,
                return_full_text=False,
            )
            return HuggingFacePipeline(pipeline=pipe)

        # If extraction failed, warn and fall through to fresh load
        warnings.warn(
            f"Could not extract model from shared_backend "
            f"(type={type(shared_backend).__name__}). "
            f"Loading fresh model from {model_id}."
        )

    # Case 4: Load fresh model
    import transformers
    print(f"LLM Judge: loading model from {model_id}")
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        do_sample=False,
        device_map="auto",
        torch_dtype="auto",
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


# -------------------------------------------------------------------------
# Evaluator
# -------------------------------------------------------------------------


@dataclass
class LLMJudgeEvaluator:
    model_id: str
    aggregation: str = "max"
    rel: int = 3
    shared_backend: Any = None
    _llm: Any = field(default=None, init=False, repr=False)

    @property
    def llm(self):
        """Lazy-load LangChain LLM on first use."""
        if self._llm is None:
            self._llm = _build_langchain_model(
                self.model_id,
                shared_backend=self.shared_backend,
            )
        return self._llm

    # -----------------------------------------------------------------

    def _generate(self, prompt_text: str) -> str:
        """
        Generate text from a single prompt using LangChain.
        Returns raw string output.
        """
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        template = PromptTemplate.from_template("{prompt}")
        chain = template | self.llm | StrOutputParser()

        try:
            output = chain.invoke({"prompt": prompt_text})
            return str(output).strip()
        except Exception as e:
            warnings.warn(f"LLM generation failed: {e}")
            return ""

    # -----------------------------------------------------------------

    def score(self, query, prediction, references=None):

        if not references:
            # No reference text — use reference-free prompt
            prompt = JUDGE_PROMPT_NO_REF.format(
                query=query,
                prediction=prediction,
            )
            raw_output = self._generate(prompt)
            score_val = _parse_rating(raw_output)

            print(f"\n[LLM Judge] query={query!r}")
            print(f"[LLM Judge] prediction={prediction!r}")
            print(f"[LLM Judge] raw_output={raw_output!r}")
            print(f"[LLM Judge] parsed_score={score_val}")

            if score_val <= 0:
                warnings.warn(
                    f"Could not parse rating from output:\n{raw_output}"
                )
                return 0.0

            return float(score_val)

        scores = []

        for i, reference in enumerate(references):
            prompt = JUDGE_PROMPT.format(
                query=query,
                reference=reference,
                prediction=prediction,
            )
            raw_output = self._generate(prompt)
            score_val = _parse_rating(raw_output)

            print(f"\n[LLM Judge] query={query!r}")
            print(f"[LLM Judge] reference[{i}]={reference!r}")
            print(f"[LLM Judge] prediction={prediction!r}")
            print(f"[LLM Judge] raw_output={raw_output!r}")
            print(f"[LLM Judge] parsed_score={score_val}")

            if score_val <= 0:
                warnings.warn(
                    f"Could not parse rating from output:\n{raw_output}"
                )
                continue

            scores.append(score_val)

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
        model_id: str,
        aggregation: str = "max",
        rel: int = 3,
        shared_backend=None,
        topics_df=None,
):
    """
    Create an ir_measures-compatible LLM judge metric.

    Uses LangChain with a local HuggingFace model (e.g. Llama)
    for fully offline evaluation.

    Args:
        model_id: Path to local HuggingFace model directory
            (e.g. "./models/Llama-3.2-3B-Instruct").
        aggregation: How to aggregate scores across references.
            One of "max", "min", "avg", "sum".
        rel: Minimum relevance threshold for filtering qrels.
        shared_backend: An existing backend instance to reuse.
            Accepts: LangChain HuggingFacePipeline, transformers
            Pipeline, or pyterrier_rag HuggingFaceBackend.
            Avoids loading the model twice (OOM on single GPU).
        topics_df: DataFrame with 'qid' and 'query' columns.
            Used to look up query text when pipeline drops it.
    """

    evaluator = LLMJudgeEvaluator(
        model_id=model_id,
        aggregation=aggregation,
        rel=rel,
        shared_backend=shared_backend,
    )

    # Build qid → query lookup
    topics_lookup = None
    if topics_df is not None:
        qid_col = "qid" if "qid" in topics_df.columns else "query_id"
        q_col = "query" if "query" in topics_df.columns else "question"
        if qid_col in topics_df.columns and q_col in topics_df.columns:
            topics_lookup = dict(zip(
                topics_df[qid_col].astype(str),
                topics_df[q_col].astype(str),
            ))

    metric_name = (
        f"LLMasJudge("
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
            required=["relevance"],
            df_name="qrels",
        )

        _validate_columns(
            res,
            required=["qanswer"],
            df_name="res",
        )

        # Filter relevance
        if isinstance(qrels, pd.Series):
            qrels = pd.DataFrame([qrels])
        if isinstance(qrels, pd.DataFrame):
            qrels = qrels.reset_index(drop=True)
            # Deduplicate columns if ir_measures passes frames with duplicate labels
            qrels = qrels.loc[:, ~qrels.columns.duplicated()]

        # Filter relevance — use .loc to avoid reindex issues
        mask = qrels["relevance"] >= rel
        qrels_filtered = qrels.loc[mask].reset_index(drop=True)

        if len(qrels_filtered) == 0:
            return 0.0

        prediction = clean_prediction(str(res["qanswer"].iloc[0]))

        query = _safe_query(res, topics_lookup=topics_lookup)

        # text column optional
        references = None
        if "text" in qrels_filtered.columns:
            references = [
                str(x)
                for x in qrels_filtered["text"].tolist()
                if x is not None and str(x).strip()
            ]
            if not references:
                references = None

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
        qrel_inputs=["relevance", "text"],
    )


__all__ = [
    "LLMasJudge",
    "LLMJudgeEvaluator",
    "JUDGE_PROMPT",
    "JUDGE_PROMPT_NO_REF",
]
