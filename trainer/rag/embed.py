"""Embeddings from a local Ollama model: text in, a list of numbers that captures its meaning out.

EmbeddingGemma expects a short label in front of the text, different for questions and documents.
Its vectors can be cut to 256 numbers (Matryoshka) and re-scaled to length 1: smaller and faster to compare.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

MODEL = "embeddinggemma:300m"
DIMS = 256
URL = "http://localhost:11434/api/embed"


class EmbeddingUnavailable(Exception):
    """Ollama isn't running or the model isn't installed. Search falls back to keywords."""


def query_text(q: str) -> str:
    return f"task: search result | query: {q}"


def document_text(title: str, text: str) -> str:
    return f"title: {title} | text: {text}"


def unit(vec: list[float]) -> list[float]:
    """Cut to DIMS and scale to length 1, so cosine similarity is just a dot product."""
    vec = vec[:DIMS]
    norm = math.sqrt(math.sumprod(vec, vec)) or 1.0
    return [x / norm for x in vec]


def embed(texts: list[str], *, timeout: float = 300, url: str = URL) -> list[list[float]]:
    body = json.dumps({"model": MODEL, "input": texts, "keep_alive": "30m"}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            vectors = json.load(r)["embeddings"]
    except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError) as e:
        raise EmbeddingUnavailable(str(e)) from e
    return [unit(v) for v in vectors]
