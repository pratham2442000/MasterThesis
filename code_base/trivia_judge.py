"""
TriviaQA LLM-as-Judge Measure
==============================
Custom ir_measures measure for evaluating TriviaQA answers.

Checks if the generated answer semantically matches ANY of the gold
answers using an LLM judge. Returns 1 (correct) or 0 (incorrect).

Usage::

    from code_base.trivia_judge import TriviaQAJudge

    judge = TriviaQAJudge(backend_type="openai", model_id="gpt-4o-mini")
    # Use in pt.Experiment as eval_metrics=[judge]
"""

import ir_measures
import pandas as pd

JUDGE_PROMPT = (
    "You are evaluating a question-answering system.\n\n"
    "Question: {query}\n"
    "Generated Answer: {prediction}\n"
    "Accepted Answers: {gold_answers}\n\n"
    "Does the generated answer correctly answer the question "
    "and semantically match ANY of the accepted answers?\n"
    "Consider synonyms, paraphrases, and equivalent meanings.\n"
    "Reply ONLY with '1' (correct) or '0' (incorrect)."
)


def _trivia_judge_fn(qrels, res, backend_type, model_id):
    """Evaluate if generated answer matches any gold answer.

    Args:
        qrels: DataFrame with 'gold_answer' column (one row per answer).
        res: DataFrame with 'qanswer' and 'query' columns.
        backend_type: LLM backend type (e.g. 'openai', 'hf').
        model_id: Model name or path.

    Returns:
        1 if correct, 0 if incorrect.
    """
    assert len(res), "Empty res df provided"
    assert len(qrels), "Empty qrels df provided"

    if isinstance(qrels, pd.Series):
        qrels = pd.DataFrame([qrels])
    if isinstance(qrels, pd.DataFrame):
        qrels = qrels.reset_index(drop=True)

    prediction = res.iloc[0]["qanswer"]
    query = res.iloc[0].get("query", "")
    gold_answers = qrels["gold_answer"].tolist()

    # Format gold answers as readable list
    gold_str = ", ".join(f'"{a}"' for a in gold_answers)

    prompt_text = JUDGE_PROMPT.format(
        query=query,
        prediction=prediction,
        gold_answers=gold_str,
    )

    try:
        from pyterrier_rag.backend import get_backend

        backend = get_backend(backend_type, model_id)
        outputs = backend.generate([prompt_text])
        result = outputs[0].strip()

        # Parse: extract first digit (1 or 0)
        for char in result:
            if char in ("0", "1"):
                return int(char)

        # Fallback: if no clear digit, treat as 0
        return 0

    except Exception as e:
        print(f"  !! TriviaQAJudge error: {e}")
        return 0


def TriviaQAJudge(backend_type="openai", model_id="gpt-4o-mini"):
    """Create an ir_measures by-query measure using LLM as judge.

    Evaluates if the generated answer semantically matches any gold
    answer from the TriviaQA dataset.

    Args:
        backend_type: LLM backend identifier ('openai', 'hf', etc.).
        model_id: Model name or path.

    Returns:
        ir_measures Measure object usable in pt.Experiment.
    """
    return ir_measures.define_byquery(
        lambda qrels, res: _trivia_judge_fn(
            qrels, res, backend_type=backend_type, model_id=model_id
        ),
        name="TriviaQAJudge",
        support_cutoff=False,
        run_inputs=["qanswer"],
        qrel_inputs=["gold_answer"],
    )


__all__ = ["TriviaQAJudge", "JUDGE_PROMPT"]
""" Description: Custom LLM-as-Judge for TriviaQA multi-answer evaluation. Checks semantic match against any gold answer."""
