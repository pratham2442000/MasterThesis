from dexter.llms.llama_engine import LlamaEngine
import torch
import json
import re

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
model_name = "meta-llama/Llama-2-7b-chat-hf"

llm_instance = LlamaEngine(data="", model_name=model_name, temperature=0.3, top_n=1)



def query_llm(question_text: str, k: int) -> list[str]:

    system_prompt = (
        f"Generate {k} paraphrased queries for the following question. "
        f"Respond strictly in JSON as a list of {k} strings. "
        f"Do not include any text before or after the JSON."
    )
    user_prompt = f"Question: {question_text}\nAfree -hnswer:"

    chain_answer = llm_instance.get_llama_completion(system_prompt, user_prompt)
    if not chain_answer:
        return None

    try:
        queries = json.loads(chain_answer)
        if not isinstance(queries, list):
            raise ValueError("Output is not a list")
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    except (json.JSONDecodeError, ValueError):
        print("Warning: Model output not valid JSON. Falling back to line parsing.")
        lines = [q.strip() for q in chain_answer.split('\n') if q.strip()]
        queries = [re.sub(r'^\d+[\).\s-]*', '', q).strip() for q in lines]

    if len(queries) != k:
        print(f"Warning: Expected {k} queries, got {len(queries)}. Adjusting...")
        if len(queries) > k:
            queries = queries[:k]
        else:
            queries += [""] * (k - len(queries))

    return queries


if __name__ == "__main__":
    question = "What are the health benefits of regular exercise?"
    k = 5
    paraphrased_queries = query_llm(question, k)
    print(paraphrased_queries)