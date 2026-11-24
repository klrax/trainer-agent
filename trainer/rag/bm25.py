"""Keyword search (BM25), written out so every step is visible.

score(doc) = sum over query words of   idf(word) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len / avg_len))
- idf: rare words count more ("納豆" says more than "生")
- tf: more mentions count more, but with diminishing returns (K1)
- long documents are penalised, since they contain more words by chance (B)
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

K1, B = 1.2, 0.75


class BM25:
    def __init__(self, docs: list[list[str]], lengths: list[int] | None = None):
        """lengths: the length to use per document, when it shouldn't count every token (see Index)."""
        self.lengths = lengths if lengths is not None else [len(d) for d in docs]
        self.avg_len = (sum(self.lengths) / len(docs) if docs else 0.0) or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)   # word -> [(doc number, count)]
        for i, tokens in enumerate(docs):
            for word, count in Counter(tokens).items():
                self.postings[word].append((i, count))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)      # what a word nobody uses would weigh

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for word in set(query):
            for i, tf in self.postings.get(word, ()):
                norm = K1 * (1 - B + B * self.lengths[i] / self.avg_len)
                out[i] += self.idf[word] * tf * (K1 + 1) / (tf + norm)
        return out

    def coverage(self, query: list[str], doc: int) -> float:
        """0..1: how much of the query (weighted by rarity) this document contains."""
        words = set(query)
        total = sum(self.idf.get(w, self.max_idf) for w in words)
        hit = sum(self.idf[w] for w in words if w in self.idf and any(i == doc for i, _ in self.postings[w]))
        return hit / total if total else 0.0
