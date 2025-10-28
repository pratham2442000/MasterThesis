import json
import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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


def _build_model(temperature: float = 0.2):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in environment.")
    # Model name via env; default to gpt-4o-mini for cost/speed
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=temperature)


def _normalize_to_k(items: List[str], k: int) -> List[str]:
    cleaned = [s.strip() for s in items if isinstance(s, str) and s.strip()]
    if len(cleaned) >= k:
        return cleaned[:k]
    return cleaned + [cleaned[-1] if cleaned else ""] * (k - len(cleaned))


def query_llm(question_text: str, k: int) -> List[str]:
    """Return exactly k rewritten queries via LangChain with robust parsing."""
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
        try:
            loaded = json.loads(str(e))
            if isinstance(loaded, list):
                result = _normalize_to_k([str(x) for x in loaded], k)
            else:
                result = _normalize_to_k([question_text], k)
        except Exception:
            result = _normalize_to_k([question_text], k)

    return result


if __name__ == "__main__":
    example_q = "What are the health benefits of regular exercise?"
    k = 5
    print(query_llm(example_q, k))
