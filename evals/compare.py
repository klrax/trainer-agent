"""Compare models on the same scenarios.

    uv run python -m evals.run_evals --provider ollama --model qwen3:0.6b -n 3 --out data/compare.jsonl     (per model)
    uv run python -m evals.compare data/compare.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import median


def table(rows: list[dict]) -> str:
    configs, scenarios = list(dict.fromkeys(r["config"] for r in rows)), list(dict.fromkeys(r["scenario"] for r in rows))
    by = defaultdict(list)
    for r in rows:
        by[r["scenario"], r["config"]].append(r)

    def passed(rs):
        return f"{sum(not r['problems'] for r in rs)}/{len(rs)}" if rs else "-"
    lines = [["", *configs]]
    lines += [[s, *(passed(by[s, c]) for c in configs)] for s in scenarios]
    of = {c: [r for r in rows if r["config"] == c] for c in configs}
    tools = {c: [t for r in of[c] for t in r["tools"]] for c in configs}
    lines += [["ALL PASSED", *(passed(of[c]) for c in configs)],
              ["median seconds", *(f"{median(r['seconds'] for r in of[c]):.0f}" for c in configs)],
              ["slowest seconds", *(f"{max(r['seconds'] for r in of[c]):.0f}" for c in configs)],
              ["median output tokens", *(f"{median(r['output_tokens'] for r in of[c]):.0f}" for c in configs)],
              ["tool calls (errors)", *(f"{len(tools[c])} ({sum(t[2] for t in tools[c])})" for c in configs)]]
    width = [max(len(row[i]) for row in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, width)) for row in lines)


if __name__ == "__main__":
    print(table([json.loads(line) for path in sys.argv[1:] for line in open(path)]))
