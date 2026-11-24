"""Does search find the right entry? Retrieval is measured on its own, without the LLM: free and repeatable.

    uv run python -m evals.retrieval_eval                          # compare keyword / vector / hybrid
    uv run python -m evals.retrieval_eval --misses                 # show what went wrong
    uv run python -m evals.retrieval_eval --find foods "ゆで卵"     # candidates to choose labels from

Cases: evals/retrieval_cases.jsonl, one JSON object per line.
    {"tool": "foods", "query": "ゆで卵 2個", "want": ["mext:12005"]}
    {"tool": "exercises", "query": "back exercise without a bar", "filters": {"equipment": ["dumbbell"]}, "want": ["ex:..."]}
    {"tool": "foods", "query": "moon cheese supreme", "want": []}        # nothing fits: should be marked weak
- want: every id you would accept. [] means "no entry fits".
- "example": true lines only show the format and are not scored.

Numbers (cases with a non-empty want):
- hit@1  share whose FIRST result is wanted        (what the model usually uses)
- hit@5  share with a wanted result in the top 5   (what the model gets to see)
- MRR    average of 1/rank of the first wanted result (0 if not in the top 5)
Weak@none: share of "nothing fits" cases whose results are ALL marked weak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.rag.embed import EmbeddingUnavailable
from trainer.rag.index import MODES
from trainer.rag.knowledge import KnowledgeBase

CASES = Path(__file__).with_name("retrieval_cases.jsonl")


def load_cases(path: Path = CASES) -> list[dict]:
    cases = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            assert case["tool"] in ("foods", "exercises") and isinstance(case["want"], list) and case["query"].strip()
        except (ValueError, KeyError, AssertionError) as e:
            raise SystemExit(f"{path.name} line {n}: bad case ({type(e).__name__} {e})")
        if not case.get("example"):
            cases.append(case)
    return cases


def run_case(kb: KnowledgeBase, case: dict, mode: str, limit: int = 5) -> list[dict]:
    search = kb.search_foods if case["tool"] == "foods" else kb.search_exercises
    return search(case["query"], limit=limit, mode=mode, **case.get("filters", {}))["results"]


def result_id(row: dict) -> str:
    return row.get("food_id") or row["exercise_id"]


def score(kb: KnowledgeBase, cases: list[dict], mode: str) -> tuple[dict, list[str]]:
    hit1 = hit5 = mrr = weak = 0.0
    misses = []
    wanted = [c for c in cases if c["want"]]
    none = [c for c in cases if not c["want"]]
    for c in wanted:
        rows = run_case(kb, c, mode)
        ranks = [i for i, r in enumerate(rows, start=1) if result_id(r) in c["want"]]
        hit1 += ranks[:1] == [1]
        hit5 += bool(ranks)
        mrr += 1 / ranks[0] if ranks else 0
        if ranks[:1] != [1]:
            got = f"{result_id(rows[0])} {rows[0]['name']}" if rows else "nothing"
            misses.append(f"  {c['query']!r}: wanted {c['want']}, first was {got}"
                          + (f" (wanted is #{ranks[0]})" if ranks else " (wanted not in top 5)"))
    for c in none:
        rows = run_case(kb, c, mode)
        ok = all(r["match"] == "weak" for r in rows)
        weak += ok
        if not ok:
            misses.append(f"  {c['query']!r}: nothing fits, but {rows[0]['name']!r} was marked good")
    n, m = len(wanted), len(none)
    return {"hit@1": hit1 / n if n else None, "hit@5": hit5 / n if n else None, "MRR": mrr / n if n else None,
            "weak@none": weak / m if m else None, "cases": f"{n}+{m}"}, misses


def find(kb: KnowledgeBase, tool: str, query: str) -> None:
    """Pool candidates from keyword AND vector search, so labels aren't limited to what hybrid already finds."""
    seen = {}
    for mode in ("keyword", "vector"):
        for rank, row in enumerate(run_case(kb, {"tool": tool, "query": query}, mode, limit=10), start=1):
            seen.setdefault(result_id(row), [row, {}])[1][mode] = rank
    for id_, (row, ranks) in sorted(seen.items(), key=lambda x: min(x[1][1].values())):
        facts = row.get("per_100g") or row["primary_muscles"]
        print(f"{id_:32} {row['name'][:60]:60} {facts} {ranks}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true", help="list every case whose first result was wrong")
    ap.add_argument("--find", nargs=2, metavar=("foods|exercises", "QUERY"))
    opts = ap.parse_args()
    kb = KnowledgeBase.load()
    if opts.find:
        return find(kb, *opts.find)
    cases = load_cases()
    if not cases:
        raise SystemExit(f"no scored cases yet: add yours to {CASES.name} (the example lines don't count)")
    print(f"{'mode':8} {'hit@1':>6} {'hit@5':>6} {'MRR':>6} {'weak@none':>10}  cases (wanted + none)")
    for mode in MODES:
        try:
            r, misses = score(kb, cases, mode)
        except EmbeddingUnavailable as e:
            print(f"{mode:8} skipped: embeddings unavailable ({e})")
            continue
        used = kb.foods.last_mode if any(c["tool"] == "foods" for c in cases) else kb.exercises.last_mode
        cells = [f"{r[k]:.2f}" if r[k] is not None else "-" for k in ("hit@1", "hit@5", "MRR", "weak@none")]
        note = "" if used == mode else f"  (fell back to {used}!)"
        print(f"{mode:8} {cells[0]:>6} {cells[1]:>6} {cells[2]:>6} {cells[3]:>10}  {r['cases']}{note}")
        if opts.misses:
            print("\n".join(misses))


if __name__ == "__main__":
    main()
