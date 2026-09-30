"""BM25 over Kiwi morphemes, the first stage of the two-stage recipe.

Particles and endings attach to Korean words, so splitting on spaces keeps
'예비인가를' and '예비인가' apart. Kiwi splits text into morphemes first, and
BM25 (rank_bm25 defaults, k1=1.5 and b=0.75) runs on those.

Needs the `bm25` extra: pip install "cheon-reranker[bm25]".
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi


@lru_cache(maxsize=1)
def _kiwi() -> Kiwi:
    return Kiwi()


def morphemes(text: str) -> list[str]:
    """The surface forms of the morphemes in `text`."""
    return [token.form for token in _kiwi().tokenize(text)]


class KiwiBM25:
    """A BM25 index over a fixed list of texts."""

    def __init__(self, texts: Sequence[str]) -> None:
        self._bm25 = BM25Okapi([morphemes(text) for text in texts])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """The `k` best texts as (index, score), best first; ties keep the corpus order."""
        scores = self._bm25.get_scores(morphemes(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(index), float(scores[index])) for index in order]
