"""Hybrid search over one collection: keyword (BM25) + meaning (embeddings), merged by rank.

Why both: keywords are exact ("ささみ", "Greek yogurt") but miss other wordings; embeddings find
"something for my lats without a bar" but blur near-identical foods (raw / boiled, skin / no skin).

How they're merged (Reciprocal Rank Fusion): each search produces a ranking, and a document earns
1 / (60 + rank) from each. Ranks are used because BM25 scores and cosines are on different scales.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .bm25 import BM25
from .embed import EmbeddingUnavailable
from .text import is_japanese, shared_stretch, tokenize

RRF_K = 60
POOL = 50          # how many results each search contributes
MODES = ("hybrid", "keyword", "vector")


@dataclass
class Doc:
    id: str                         # "mext:12004", "usda:171287", "ex:Barbell_Squat"
    kind: str                       # "food" | "exercise"
    title: str
    text: str                       # what is searched (title + everyday names + details)
    data: dict[str, Any] = field(default_factory=dict)      # the facts returned to the model
    vector: list[float] | None = None


@dataclass
class Hit:
    doc: Doc
    score: float
    coverage: float                 # share of the query's words found in the document (0..1)
    cosine: float | None            # similarity of meaning (None in keyword mode)


class Index:
    def __init__(self, docs: list[Doc], embed_query: Callable[[str], list[float]] | None = None):
        self.docs = docs
        self.embed_query = embed_query
        # A food's length is its official name only. Counting the added everyday words would push the
        # most common foods DOWN the ranking for having synonyms.
        self.bm25 = BM25([tokenize(d.text) for d in docs],
                         [len(tokenize(d.title if d.kind == "food" else d.text)) for d in docs])
        self.last_mode = "keyword"          # what the last search actually used (embeddings can be down)

    def search(self, query: str, *, limit: int = 5, keep: Callable[[Doc], bool] = lambda d: True,
               mode: str = "hybrid") -> list[Hit]:
        """- keep: filter applied BEFORE ranking, so filters never leave fewer results than exist
        - mode: "hybrid" | "keyword" | "vector" (the last two are for comparing in evals)
        - embeddings unavailable in hybrid mode -> keyword only, and last_mode says so
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        allowed = {i for i, d in enumerate(self.docs) if keep(d)}
        words = tokenize(query)
        rankings: list[list[int]] = []
        cosines: dict[int, float] = {}

        if mode != "keyword":
            try:
                cosines = self._cosines(query, allowed)
            except EmbeddingUnavailable:
                if mode == "vector":
                    raise
        if cosines:
            rankings.append(_top(cosines))
        self.last_mode = mode if cosines or mode == "keyword" else "keyword"

        if mode != "vector":
            scores = self.bm25.scores(words)
            rankings.append(_top({i: s for i, s in scores.items() if i in allowed}))

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
        def coverage(i: int) -> float:
            return (shared_stretch(query, self.docs[i].text) if is_japanese(query)
                    else self.bm25.coverage(words, i))
        return [Hit(self.docs[i], fused[i], coverage(i), cosines.get(i)) for i in _top(fused)[:limit]]

    def _cosines(self, query: str, allowed: set[int]) -> dict[int, float]:
        if self.embed_query is None:
            raise EmbeddingUnavailable("no embedding function")
        with_vectors = [i for i in allowed if self.docs[i].vector is not None]
        if not with_vectors:
            raise EmbeddingUnavailable("no vectors in the index")
        q = self.embed_query(query)
        return {i: math.sumprod(q, self.docs[i].vector) for i in with_vectors}


def _top(scores: dict[int, float]) -> list[int]:
    """Best first; ties broken by position so results are repeatable."""
    return sorted(scores, key=lambda i: (-scores[i], i))[:POOL]
