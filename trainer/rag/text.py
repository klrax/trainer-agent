"""Turning Japanese and English text into search tokens.

- English words: lowercased, with a simple plural rule ("eggs" -> "egg").
- Japanese has no spaces, so each run of kana/kanji becomes overlapping 2-character tokens:
  "鶏むね" -> ["鶏む", "むね"]. A 1-character run stays as it is ("卵" -> ["卵"]).
"""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[぀-ヿ㐀-鿿々ー]+")
_STOP = {"a", "an", "the", "of", "and", "or", "with", "in", "for", "to", "my", "some", "i", "had", "ate",
         "one", "two", "three", "four", "five", "six", "half"}      # amounts don't identify a food


def normalize(text: str) -> str:
    """Full-width letters to half-width, lowercase: "ＬＧ２１" -> "lg21"."""
    return unicodedata.normalize("NFKC", text).lower()


def is_japanese(text: str) -> bool:
    return bool(re.search(r"[぀-ヿ㐀-鿿]", text))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for run in _WORD.findall(normalize(text)):
        if is_japanese(run):
            tokens += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
        elif run not in _STOP:
            tokens.append(_singular(run))
    return tokens


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("oes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def shared_stretch(query: str, text: str) -> float:
    """0..1: how much of each query word appears in the text as ONE unbroken stretch (weighted by length).

    Used for Japanese, where character pairs overlap by accident: スターバックス shares ター, バッ, ック and
    クス with a tempura batter mix, but its longest unbroken stretch there is only ックス (3 of 7).
    """
    words, text = _WORD.findall(normalize(query)), normalize(text)
    total = sum(len(w) for w in words)
    return sum(_longest_common(w, text) for w in words) / total if total else 0.0


def _longest_common(word: str, text: str) -> int:
    for size in range(len(word), 0, -1):
        if any(word[i:i + size] in text for i in range(len(word) - size + 1)):
            return size
    return 0
