"""
Cross-Encoder Reranker for PyTerrier
=====================================
A pt.Transformer wrapping HuggingFace cross-encoder models
(e.g. ``cross-encoder/ms-marco-MiniLM-L6-v2``).

Cross-encoders use ``AutoModelForSequenceClassification`` —
they take (query, document) pairs and output a single relevance
logit. Much faster than T5-based rerankers.

Usage::

    from code_base.cross_encoder_reranker import CrossEncoderReRanker

    reranker = CrossEncoderReRanker()                          # default MiniLM-L6
    reranker = CrossEncoderReRanker("cross-encoder/ms-marco-MiniLM-L12-v2")  # bigger

    pipeline = bm25 >> get_text >> reranker
"""

import pyterrier as pt
import torch
from pyterrier.model import add_ranks
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CrossEncoderReRanker(pt.Transformer):
    """PyTerrier reranker using a HuggingFace cross-encoder model.

    Supports any ``AutoModelForSequenceClassification`` cross-encoder
    trained on MS MARCO (or similar). Default model:
    ``cross-encoder/ms-marco-MiniLM-L6-v2``.

    Args:
        model: HuggingFace model name or local path.
        batch_size: Batch size for inference.
        text_field: Column name containing document text.
        max_length: Max token length for tokenizer truncation.
        verbose: Show progress bar.
    """

    def __init__(
            self,
            model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
            batch_size: int = 64,
            text_field: str = "text",
            max_length: int = 512,
            verbose: bool = True,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.text_field = text_field
        self.max_length = max_length
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()

    def __str__(self):
        return f"CrossEncoder({self.model_name})"

    def __repr__(self):
        return (
            f"CrossEncoderReRanker(model={self.model_name!r}, "
            f"batch_size={self.batch_size}, device={self.device})"
        )

    def transform(self, run):
        # 1. Validation check
        if self.text_field not in run.columns:
            raise KeyError(f"Input dataframe missing text column: {self.text_field}")

        scores = []
        queries = run["query"].tolist()
        texts = run[self.text_field].tolist()

        it = range(0, len(queries), self.batch_size)
        if self.verbose:
            it = pt.tqdm(it, desc=str(self), unit="batches")

        # 2. Use autocast for potential speed boost on GPU
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            for start_idx in it:
                end_idx = start_idx + self.batch_size
                batch_queries = queries[start_idx:end_idx]
                batch_texts = texts[start_idx:end_idx]

                enc = self.tokenizer(
                    batch_queries,
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**enc).logits

                # 3. Flexible logit handling
                if logits.shape[-1] > 1:
                    # Take the 'relevant' class logit (usually index 1)
                    batch_scores = logits[:, 1].cpu().tolist()
                else:
                    batch_scores = logits.squeeze(-1).cpu().tolist()

                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

        # 4. Standard PyTerrier cleanup
        res = run.drop(columns=["score", "rank"], errors="ignore").assign(score=scores)
        return add_ranks(res)
