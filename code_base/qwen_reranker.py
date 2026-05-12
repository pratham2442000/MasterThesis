"""
Qwen3-Reranker for PyTerrier (vLLM backend)
=============================================
A pt.Transformer wrapping ``Qwen/Qwen3-Reranker-4B`` via vLLM.

Qwen3-Reranker uses a generative yes/no relevance judgement:
it produces logprobs for "yes"/"no" tokens and converts them
to a calibrated relevance score via softmax.

Requires ``vllm>=0.8.5``.

Usage::

    from code_base.qwen_reranker import QwenRerankerVLLM

    reranker = QwenRerankerVLLM()                          # default 4B
    pipeline = bm25 >> get_text >> reranker
"""

import math

import pandas as pd
import pyterrier as pt
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.inputs.data import TokensPrompt


class QwenRerankerVLLM(pt.Transformer):
    """PyTerrier reranker using Qwen3-Reranker with vLLM.

    Scores each (query, document) pair by computing the softmax-
    normalised probability that the model outputs "yes" (relevant)
    vs "no" (not relevant).

    Args:
        model: HuggingFace model name or local path.
        instruction: Task instruction prepended to every pair.
        batch_size: Number of pairs processed per vLLM generate call.
        max_length: Maximum input token length (truncated).
        tensor_parallel_size: GPUs for tensor parallelism (auto if None).
        max_model_len: vLLM max model length.
        enable_prefix_caching: Enable vLLM prefix caching.
        gpu_memory_utilization: Fraction of GPU memory for vLLM.
        text_field: Column containing document text.
        verbose: Print progress information.
        device: Pin to a specific GPU (e.g. ``'cuda:1'``).
    """

    def __init__(
            self,
            model: str = "Qwen/Qwen3-Reranker-4B",
            instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
            batch_size: int = 32,
            max_length: int = 8192,
            tensor_parallel_size: int = None,
            max_model_len: int = 10000,
            enable_prefix_caching: bool = True,
            gpu_memory_utilization: float = 0.8,
            text_field: str = "text",
            verbose: bool = False,
            device: str = None,
    ):
        self.model_name = model
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_length = max_length
        self.text_field = text_field
        self.verbose = verbose
        self.device = device

        # ── GPU / tensor-parallel setup ──────────────────────────
        import os

        gpu_id = None
        original_cuda_visible = None
        physical_gpu_id = None

        if device is not None and "cuda" in str(device):
            gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0

            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if original_cuda_visible is not None:
                available_gpus = [int(g.strip()) for g in original_cuda_visible.split(",")]
                physical_gpu_id = (
                    available_gpus[gpu_id] if gpu_id < len(available_gpus) else available_gpus[0]
                )
            else:
                physical_gpu_id = gpu_id

            if verbose:
                print(f"Assigning QwenRerankerVLLM to logical GPU {gpu_id} (physical GPU {physical_gpu_id})")
                print(f"Original CUDA_VISIBLE_DEVICES: {original_cuda_visible}")

            self.tensor_parallel_size = 1
        else:
            self.tensor_parallel_size = (
                tensor_parallel_size if tensor_parallel_size is not None else torch.cuda.device_count()
            )

        # ── Tokenizer ────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── vLLM model ───────────────────────────────────────────
        if physical_gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
            if verbose:
                print(f"Temporarily setting CUDA_VISIBLE_DEVICES={physical_gpu_id}")

        try:
            self.model = LLM(
                model=model,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        finally:
            if physical_gpu_id is not None:
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                if verbose:
                    print(f"Restored CUDA_VISIBLE_DEVICES to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")

        # ── Suffix / token setup ─────────────────────────────────
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self.true_token, self.false_token],
        )

    # ── Internal helpers ─────────────────────────────────────────

    def _format_instruction(self, query: str, doc: str) -> list:
        """Format a query-document pair with the instruction as chat messages."""
        return [
            {
                "role": "system",
                "content": (
                    'Judge whether the Document meets the requirements based on the '
                    'Query and the Instruct provided. Note that the answer can only '
                    'be "yes" or "no".'
                ),
            },
            {
                "role": "user",
                "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
            },
        ]

    def _process_inputs(self, pairs: list) -> list:
        """Tokenize and prepare inputs for vLLM."""
        messages = [self._format_instruction(q, d) for q, d in pairs]
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False,
        )
        max_len = self.max_length - len(self.suffix_tokens)
        messages = [ele[:max_len] + self.suffix_tokens for ele in messages]
        return [TokensPrompt(prompt_token_ids=ele) for ele in messages]

    def _compute_scores(self, messages: list) -> list:
        """Compute relevance scores for the inputs using vLLM."""
        outputs = self.model.generate(messages, self.sampling_params, use_tqdm=False)
        scores = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]

            true_logit = final_logits[self.true_token].logprob if self.true_token in final_logits else -10
            false_logit = final_logits[self.false_token].logprob if self.false_token in final_logits else -10

            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            scores.append(true_score / (true_score + false_score))

        return scores

    # ── PyTerrier interface ──────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rerank documents in the input DataFrame.

        Expects columns ``query`` and *text_field* (default ``text``).
        Returns the DataFrame with an updated ``score`` column,
        sorted descending per ``qid``.
        """
        if len(df) == 0:
            return df

        if "query" not in df.columns:
            raise ValueError("DataFrame must have a 'query' column")
        if self.text_field not in df.columns:
            raise KeyError(f"Input dataframe missing text column: {self.text_field}")

        pairs = list(zip(df["query"].tolist(), df[self.text_field].tolist()))

        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i: i + self.batch_size]
            if self.verbose:
                print(f"Processing batch {i // self.batch_size + 1}/{(len(pairs) - 1) // self.batch_size + 1}")
            inputs = self._process_inputs(batch)
            all_scores.extend(self._compute_scores(inputs))

        result = df.copy()
        result["score"] = all_scores

        if "qid" in result.columns:
            result = result.sort_values(["qid", "score"], ascending=[True, False])
            result["rank"] = result.groupby("qid").cumcount()
        else:
            result = result.sort_values("score", ascending=False)

        return result.reset_index(drop=True)

    def __del__(self):
        """Cleanup vLLM resources."""
        try:
            destroy_model_parallel()
        except Exception:
            pass

    def __str__(self):
        return f"QwenReranker({self.model_name})"

    def __repr__(self):
        return (
            f"QwenRerankerVLLM(model={self.model_name!r}, "
            f"batch_size={self.batch_size}, tensor_parallel={self.tensor_parallel_size})"
        )
