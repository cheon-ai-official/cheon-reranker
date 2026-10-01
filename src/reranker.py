"""Cheon Reranker: score every candidate for a query in one call.

The order the candidates are passed in can change their scores, so pass them
in your first stage's order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "cheonai/cheon-reranker-0.6b-v1"

# How the query and each candidate are written into the sequence.
QUERY_TEMPLATE = "query: {}\n"
CANDIDATE_TEMPLATE = "candidate: {}"


@dataclass(frozen=True)
class Ranked:
    """One candidate after reranking."""

    index: int
    """Position of the candidate in the list that was passed in."""
    score: float
    """Relevance score, higher is better. A relative signal within one call, not a probability."""
    document: str
    """The candidate text as it was passed in."""


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CheonReranker:
    """A Cheon Reranker model loaded from the Hugging Face Hub.

    Args:
        model: Hub repository id or local directory of the model.
        device: Where to run it ("cuda", "mps", "cpu", ...). Picked automatically when omitted.
        dtype: Weight precision. The released weights are float32.
        revision: Hub branch, tag or commit to pin the model to.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        revision: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = torch.device(device or default_device())
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision, trust_remote_code=True, dtype=dtype)
        self.model.to(self.device).eval()
        self.max_tokens: int = self.model.config.max_sequence_length
        self.max_candidates: int = self.model.config.max_candidates_per_context

    def _token_ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _check(self, query: str, documents: Sequence[str]) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(documents, str) or len(documents) == 0:
            raise ValueError("documents must be a non-empty list of strings")
        if len(documents) > self.max_candidates:
            raise ValueError(
                f"at most {self.max_candidates} candidates per call, got {len(documents)}; "
                "rerank the top of your first-stage results"
            )
        for position, text in enumerate(documents):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"document {position} has no text")

    def encode(self, query: str, documents: Sequence[str]) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids for one forward pass and the (start, end) token span of every candidate.

        The query line comes first, then the candidates in order. Everything has
        to fit in `max_tokens`, so each candidate gets an equal share of what the
        query leaves, and a longer candidate is cut at the end.
        """
        self._check(query, documents)
        ids = self._token_ids(QUERY_TEMPLATE.format(query.strip()))
        budget = (self.max_tokens - len(ids)) // len(documents)
        if budget < 1:
            raise ValueError(
                f"the query takes {len(ids)} of {self.max_tokens} tokens, "
                f"which leaves no room for {len(documents)} candidates"
            )
        spans = []
        for text in documents:
            piece = self._token_ids(CANDIDATE_TEMPLATE.format(text.strip()))[:budget]
            spans.append((len(ids), len(ids) + len(piece)))
            ids.extend(piece)
        return ids, spans

    @torch.inference_mode()
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per candidate, in the order they were passed in."""
        ids, spans = self.encode(query, documents)
        input_ids = torch.tensor([ids], device=self.device)
        scores = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), doc_spans=spans)
        return scores.float().cpu().tolist()

    def rerank(self, query: str, documents: Sequence[str], top_n: int | None = None) -> list[Ranked]:
        """The candidates from best to worst, optionally only the best `top_n`."""
        if top_n is not None and top_n < 1:
            raise ValueError("top_n must be at least 1")
        scores = self.score(query, documents)
        order = sorted(range(len(documents)), key=lambda index: -scores[index])
        return [Ranked(index, scores[index], documents[index]) for index in order[:top_n]]
