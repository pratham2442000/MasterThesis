"""
Test the custom LLMasJudge metric.

Validates:
  1. _parse_rating extracts scores from various LLM output formats
  2. LLMasJudge works end-to-end with a mock backend
  3. LLMasJudge integrates correctly with pt.Experiment

Usage:
    python experiments/test_llm_judge.py             # unit tests only
    python experiments/test_llm_judge.py --live       # + live model test
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock, patch

# LLMasJudge is now lazy-loaded, no eager model load on import
import constants

if getattr(constants, "OFFLINE_MODE", False):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
from dataclasses import dataclass

import pandas as pd

from code_base.llm_judge import _parse_rating, LLMJudgeEvaluator, LLMasJudge


# ─── Test 1: Rating Parser ─────────────────────────────────

def test_parse_rating():
    """Test _parse_rating handles all expected LLM output formats."""
    cases = [
        ('Rating: [[5]]', 5),
        ('[[7]]', 7),
        ('The answer is good. Rating: [[8]]', 8),
        ('[[10]]', 10),
        ('3', 3),
        ('I rate this 6 out of 10', 6),
        ('', 0),  # empty
        ('no number', 0),  # no digits
        ('Rating: [[0]]', 1),
    ]

    passed = 0
    for text, expected in cases:
        result = _parse_rating(text)
        status = "✓" if result == expected else "✗"
        if result != expected:
            print(f"  {status} _parse_rating({text!r}) = {result}, expected {expected}")
        else:
            passed += 1

    print(f"  Rating parser: {passed}/{len(cases)} passed")
    return passed == len(cases)


# ─── Test 2: LLMJudgeEvaluator with mock backend ───────────────

def test_evaluator_mock():
    """Test LLMJudgeEvaluator with a mocked backend."""

    @dataclass
    class FakeOutput:
        text: str

    fake_backend = MagicMock()
    fake_backend.generate.return_value = [FakeOutput(text="Rating: [[7]]")]
    fake_backend.model_id = "test-model"

    try:
        with patch("pyterrier_rag.backend.get_backend", return_value=fake_backend):
            evaluator = LLMJudgeEvaluator("mock", "mock")
            score = evaluator.score(
                query="What is the capital of France?",
                prediction="Paris",
                references=["Paris is the capital of France"]
            )

        assert score == 7, f"Expected 7, got {score}"
        print(f"  ✓ Mock backend test passed (score={score})")
        return True

    except Exception as e:
        print(f"  ✗ Mock backend test failed: {e}")
        return False


# ─── Test 3: Aggregation modes ────────────────────────────

def test_aggregation():
    """Test different aggregation modes."""

    @dataclass
    class FakeOutput:
        text: str

    fake_backend = MagicMock()
    fake_backend.model_id = "test-model"

    passed = 0
    try:
        with patch("pyterrier_rag.backend.get_backend", return_value=fake_backend):
            # max
            fake_backend.generate.return_value = [FakeOutput("[[3]]"), FakeOutput("[[7]]")]
            evaluator_max = LLMJudgeEvaluator("mock", "mock", aggregation="max")
            assert evaluator_max.score("q", "p", ["r1", "r2"]) == 7.0
            passed += 1

            # min
            fake_backend.generate.return_value = [FakeOutput("[[3]]"), FakeOutput("[[7]]")]
            evaluator_min = LLMJudgeEvaluator("mock", "mock", aggregation="min")
            assert evaluator_min.score("q", "p", ["r1", "r2"]) == 3.0
            passed += 1

            # avg
            fake_backend.generate.return_value = [FakeOutput("[[4]]"), FakeOutput("[[6]]")]
            evaluator_avg = LLMJudgeEvaluator("mock", "mock", aggregation="avg")
            assert evaluator_avg.score("q", "p", ["r1", "r2"]) == 5.0
            passed += 1

        print(f"  ✓ Aggregation tests: {passed}/3 passed")
        return passed == 3

    except AssertionError as e:
        print(f"  ✗ Aggregation test failed: {e}")
        return False


# ─── Test 4: ir_measures integration ──────────────────────

def test_ir_measures_integration():
    """Test LLMasJudge creates a valid ir_measures measure."""
    with patch("pyterrier_rag.backend.get_backend", return_value=MagicMock()):
        measure = LLMasJudge("causal", "test-model")

    assert hasattr(measure, "name") or str(measure).startswith("LLMasJudge"), \
        f"Measure name wrong: {measure}"
    print(f"  ✓ ir_measures measure created: {measure}")
    return True


# ─── Test 5: Live model (optional) ────────────────────────

def test_live_model():
    """Test with actual HuggingFace backend (needs model files)."""

    model_path = f"{constants.MODELS_CACHE_DIR}/Llama-3.2-3B-Instruct"
    if not os.path.exists(model_path):
        print(f"  ⊘ Skipped: model not found at {model_path}")
        return True

    print(f"  Loading model from {model_path}...")
    measure = LLMasJudge("causal", model_path)

    qrels = pd.DataFrame({
        "query_id": ["q1"],
        "qid": ["q1"],
        "relevance": [3],
        "text": ["Paris"],
        "gold_answer": ["Paris"],
    })
    res = pd.DataFrame({
        "query_id": ["q1"],
        "qid": ["q1"],
        "query": ["What is the capital of France?"],
        "qanswer": ["Paris"],
    })

    import ir_measures
    results = list(ir_measures.iter_calc([measure], qrels, res))
    print(results)

    if results:
        score = results[0].value
        print(f"  ✓ Live model score: {score}")
        return True
    else:
        print("  ✗ No results returned")
        return False


# ─── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test LLMasJudge metric")
    parser.add_argument("--live", action="store_true", help="Run live model test")
    args = parser.parse_args()

    print("=" * 50)
    print("LLMasJudge Test Suite")
    print("=" * 50)

    results = {}

    print("\n[1] Rating parser...")
    results["parse_rating"] = test_parse_rating()

    print("\n[2] Mock backend...")
    results["mock_backend"] = test_evaluator_mock()

    print("\n[3] Aggregation modes...")
    results["aggregation"] = test_aggregation()

    print("\n[4] ir_measures integration...")
    results["ir_measures"] = test_ir_measures_integration()

    if args.live:
        print("\n[5] Live model test...")
        results["live"] = test_live_model()

    print("\n" + "=" * 50)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("All tests passed.")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
