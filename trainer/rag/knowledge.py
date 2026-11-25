"""What the agent's tools call: look up foods and exercises, and do the nutrition arithmetic.

The split: retrieval FINDS the right entry, code does the MATH, the model talks to the user.
The model never multiplies grams by calories itself.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import embed as emb
from .index import Doc, Hit, Index
from .ingest import DB, PFC, load_docs
from .text import is_japanese, normalize, tokenize

MAX_GRAMS = 3000
MAX_LIMIT = 10
# Below BOTH limits a result is marked "weak": few of the words AND little of the meaning match.
# The meaning limit differs: food names are short and all alike ("unicorn steak" scored 0.51, real foods
# 0.47-0.61), so for foods meaning alone must be very close. Set on about 25 queries: provisional.
WEAK_COVERAGE = 0.7
WEAK_COSINE = {"food": 0.6, "exercise": 0.45}
TRAINING_CATEGORIES = {"strength", "powerlifting", "olympic weightlifting"}
# What search_exercises can ask for. "training" (the default) is lifting; the rest are free-exercise-db categories.
CATEGORIES = {"training": TRAINING_CATEGORIES, "cardio": {"cardio"}, "stretching": {"stretching"},
              "plyometrics": {"plyometrics"}, "strongman": {"strongman"}}
BODY_PARTS = {("back", "背中"): ["lats", "middle back", "lower back", "traps"],
              ("leg", "脚", "下半身"): ["quadriceps", "hamstrings", "glutes", "calves"],
              ("arm", "腕"): ["biceps", "triceps", "forearms"],
              ("chest", "胸"): ["chest"], ("shoulder", "肩"): ["shoulders"],
              ("ab", "core", "腹筋"): ["abdominals"], ("glute", "お尻"): ["glutes"]}
LEVELS = ["beginner", "intermediate", "expert"]


class KnowledgeBase:
    def __init__(self, foods: list[Doc], exercises: list[Doc], embed_query=None):
        self.foods = Index(foods, embed_query)
        self.exercises = Index(exercises, embed_query)
        self.food_by_id = {d.id: d for d in foods}

    @classmethod
    def load(cls, path: Path = DB) -> "KnowledgeBase":
        if not path.exists():
            raise ValueError("food and exercise data is not installed (run: uv run python -m trainer.rag.ingest). "
                             "Tell the user, and say that any numbers you give are your own estimates")
        con = sqlite3.connect(path)
        try:
            return cls(load_docs(con, "food"), load_docs(con, "exercise"), embed_query=_embed_query)
        finally:
            con.close()

    def search_foods(self, query: str, limit: int = 5, mode: str = "hybrid") -> dict[str, Any]:
        """{"results": [{"food_id", "name", "source", "match", "per_100g": {...}, "portions": [...]}], "search": mode}

        - a Japanese query searches only the Japanese table; otherwise both tables
        - "match" is "good" or "weak"; weak means "this may not be the food they meant: ask"
        """
        _check_query(query, limit)
        japanese = is_japanese(query)
        hits = self.foods.search(query, limit=limit, mode=mode,
                                 keep=lambda d: not japanese or d.data["source"] == "MEXT")
        results = []
        for h in hits:
            d = h.doc.data
            row = {"food_id": h.doc.id, "name": d["name"], "source": d["source"], "match": _match(h),
                   "per_100g": d["per_100g"]}
            if d["portions"]:
                row["portions"] = [_portion(d["per_100g"], p) for p in d["portions"][:3]]
            if d["estimated"]:
                row["note"] = "the table marks some of these values as estimates"
            results.append(row)
        return {"results": results, "search": self.foods.last_mode}

    def nutrition(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """kcal and PFC for given amounts, and their sum.

            kb.nutrition([{"food_id": "mext:...", "name": "鶏卵 全卵 生", "grams": 100}])
            -> {"items": [{"food_id", "name", "grams", "kcal", "protein_g", "fat_g", "carbs_g"}],
                "total": {"kcal", "protein_g", "fat_g", "carbs_g"}, "sources": ["MEXT"]}

        - food_id and name must belong to the same entry, or ValueError. Why: a small model searched for
          chicken, then passed an egg's id and reported egg numbers as chicken. Asking for both makes a
          wrong id visible to the code, and the error tells the model which food it actually picked.
        - an unknown food_id, or grams <= 0 or > 3000, raises ValueError
        - a value the table didn't measure is None for that item; the total leaves it out and
          "incomplete" names it, so an unknown is never silently counted as zero
        """
        if not items:
            raise ValueError("items is empty: pass at least one {food_id, grams}")
        rows, total, incomplete = [], dict.fromkeys(PFC, 0.0), []
        for item in items:
            doc, grams = self.food_by_id.get(item["food_id"]), item["grams"]
            if doc is None:
                raise ValueError(f"unknown food_id {item['food_id']}: use an id returned by search_foods")
            if not _same_name(item["name"], doc.data["name"]):
                raise ValueError(f"{doc.id} is {doc.data['name']!r}, not {item['name']!r}: copy food_id and name "
                                 "from the same search_foods result")
            if isinstance(grams, bool) or not isinstance(grams, (int, float)) or not 0 < grams <= MAX_GRAMS:
                raise ValueError(f"grams must be a number above 0 and up to {MAX_GRAMS}, got {grams!r}")
            row: dict[str, Any] = {"food_id": doc.id, "name": doc.data["name"], "grams": grams}
            for key in PFC:
                per_100g = doc.data["per_100g"][key]
                if per_100g is None:
                    row[key] = None
                    incomplete.append(f"{key} of {doc.data['name']}")
                    continue
                exact = per_100g * grams / 100
                total[key] += exact                       # add exact values, round only what is shown
                row[key] = _round(key, exact)
            rows.append(row)
        out = {"items": rows, "total": {k: _round(k, v) for k, v in total.items()},
               "sources": sorted({self.food_by_id[r["food_id"]].data["source"] for r in rows})}
        if incomplete:
            out["incomplete"] = incomplete
        return out

    def search_exercises(self, query: str, muscles: list[str] | None = None, equipment: list[str] | None = None,
                         level: str | None = None, category: str = "training", limit: int = 5,
                         mode: str = "hybrid") -> dict[str, Any]:
        """Filters are exact and applied before ranking.

        - muscles: the exercise's PRIMARY muscle is one of these (back = lats, middle back, lower back, traps).
          Left out, but the query names a body part ("back exercises") -> filled in by code, because models
          often skip it and meaning search alone returns shoulder work for "back"
        - equipment: any of these (so a user without a barbell never sees barbell work)
        - level: this level or easier
        - category: "training" (lifting, the default), "cardio", "stretching", "plyometrics" or "strongman".
          It used to be include_stretching=True, which hid cardio behind a stretching flag: the model never
          found cardio (cardio eval 0/3)
        """
        _check_query(query, limit)
        muscles = muscles or body_part_muscles(query)
        if level is not None and level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {sorted(CATEGORIES)}")

        def keep(doc: Doc) -> bool:
            d = doc.data
            if d["category"] not in CATEGORIES[category]:
                return False
            if muscles and not set(muscles) & set(d["primary_muscles"]):
                return False
            if equipment and d["equipment"] not in equipment:
                return False
            return level is None or d["level"] in LEVELS[:LEVELS.index(level) + 1]

        hits = self.exercises.search(query, limit=limit, keep=keep, mode=mode)
        return {"results": [{"exercise_id": h.doc.id, "name": h.doc.data["name"], "match": _match(h),
                             "primary_muscles": h.doc.data["primary_muscles"],
                             "secondary_muscles": h.doc.data["secondary_muscles"],
                             "equipment": h.doc.data["equipment"], "level": h.doc.data["level"],
                             "how_to": " ".join(h.doc.data["instructions"])[:300],
                             "source": h.doc.data["source"]} for h in hits],
                "muscles_filter": muscles or None, "search": self.exercises.last_mode}


@lru_cache(maxsize=1)
def default_kb() -> KnowledgeBase:
    """Loaded on first use and kept (about 11,000 documents)."""
    return KnowledgeBase.load()


def body_part_muscles(query: str) -> list[str] | None:
    """"dumbbell back exercises" -> the back muscles. None if the query names no body part, or a lift
    whose NAME contains one ("back squat", "leg press", "one arm row")."""
    words, text = set(tokenize(query)), normalize(query)
    if words & {"squat", "press", "curl", "extension", "raise", "row"} or ("one" in text and "arm" in words):
        return None
    found = [m for names, muscles in BODY_PARTS.items()
             if any((n in text) if is_japanese(n) else (n in words) for n in names) for m in muscles]
    return list(dict.fromkeys(found)) or None


def _portion(per_100g: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """A household portion with its numbers worked out, so the model has nothing to multiply."""
    return {**p, **{k: None if per_100g[k] is None else _round(k, per_100g[k] * p["grams"] / 100)
                    for k in ("kcal", "protein_g")}}


def _same_name(given: Any, actual: str) -> bool:
    if not isinstance(given, str):
        return False
    a, b = ("".join(normalize(x).split()) for x in (given, actual))
    return len(a) >= 2 and (a in b or b in a)


def _embed_query(query: str) -> list[float]:
    # short timeout: a slow embedding server should cost seconds, then fall back to keywords
    return emb.embed([emb.query_text(query)], timeout=20)[0]


def _match(h: Hit) -> str:
    weak = h.coverage < WEAK_COVERAGE and (h.cosine is None or h.cosine < WEAK_COSINE[h.doc.kind])
    return "weak" if weak else "good"


def _check_query(query: str, limit: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is empty")
    if not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be 1-{MAX_LIMIT}")


def _round(key: str, value: float) -> float:
    return round(value) if key == "kcal" else round(value, 1)
