"""Evals: proof the agent behaves, not just a demo that worked once.

Unit tests check YOUR code with a fake model. Evals check the WHOLE agent with the REAL model:
each scenario seeds a fresh store, sends one user message, then checks what the agent DID
(which tools, what got saved) and what it SAID.

The model is non-deterministic, so each scenario runs several times and you report a pass rate.
"Scenario X passed 5/5 after changing Y" is the kind of result this produces.

Run:  uv run python -m evals.run_evals            (costs real tokens: a few model calls per run)
      uv run python -m evals.run_evals -k meal -n 1
"""
from __future__ import annotations

import argparse
import re
import datetime as dt
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from trainer.agent import AgentResult, run_turn
from trainer.cli import build_call_model
from trainer.llm import UsageLog
from trainer.prompts import SYSTEM
from trainer.store import Store
from trainer.tools import TOOLS, run_tool

TODAY = dt.date(2026, 11, 23)  # a Monday, so "today" is the same in every run

KAT = {"name": "Kat", "sex": "female", "age": 30, "height_cm": 165, "weight_kg": 60,
       "activity": "moderate", "goal": "maintain", "train_days": ["Mon", "Wed", "Fri"],
       "experience": "1 year of lifting"}


@dataclass
class Scenario:
    name: str
    seed: Callable[[Store], None]                       # put data in the store before the chat
    user: str                                           # the one user message
    check: Callable[[AgentResult, Store], list[str]]    # returns a list of problems; [] = pass


def called(result: AgentResult, tool: str) -> bool:
    return any(name == tool for name, _ in result.tool_calls)


JUDGE = {"provider": "claude", "model": None}   # set from the command line in main()


def judge(question: str, answer: str) -> bool:
    """LLM-as-judge for things code can't check ("did it tell them to stop?"). Yes/no only.

    Uses the same provider as the agent unless you pass --judge. (A judge should be at least as
    strong as the model it grades; a small local model judging itself is weak evidence. Say so.)
    """
    call_model = build_call_model(JUDGE["provider"], "You grade a fitness coach's replies.", [],
                                  effort="low", model=JUDGE["model"])
    resp = call_model([{"role": "user", "content":
                        f"Here is a fitness coach's reply:\n<reply>\n{answer}\n</reply>\n\n{question}\n"
                        "Answer with exactly one word: YES or NO."}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip().upper()
    return text.startswith("YES")


# ── scenarios ──────────────────────────────────────────────────────────────────────────
def check_todays_workout(r: AgentResult, s: Store) -> list[str]:
    """Outcome, not path: right session named, and no weights invented (the seeded store has no history)."""
    problems = []
    if not (called(r, "get_context") or called(r, "get_session_for")):
        problems.append("didn't look up today's session (get_context / get_session_for)")
    if "full body a" not in r.text.lower():
        problems.append("didn't name today's session (Full body A)")
    if re.search(r"\d+(\.\d+)?\s*kg", r.text, re.IGNORECASE):
        problems.append("stated a weight in kg although there is no workout history (invented)")
    return problems


def check_logs_breakfast(r: AgentResult, s: Store) -> list[str]:
    meals = s.meals_on(TODAY.isoformat())
    if len(meals) != 1:
        return [f"expected 1 meal logged today, found {len(meals)}"]
    if not 200 <= meals[0]["kcal"] <= 600:
        return [f"2 eggs + toast estimated at {meals[0]['kcal']} kcal (expected 200-600)"]
    return []


SCENARIOS = [
    Scenario("todays_workout", lambda s: s.update_profile(**KAT),
             "What's my workout today?", check_todays_workout),
    Scenario("logs_breakfast", lambda s: s.update_profile(**KAT),
             "I just had 2 scrambled eggs and a slice of toast for breakfast.", check_logs_breakfast),
    # TODO (yours): write at least 3 more. Ideas:
    # - knee_pain: "My knee hurts sharply when I squat. What weight should I squat today?"
    #     check: judge("Does the coach tell them to stop or skip squats and suggest a professional?", r.text)
    # - no_profile: empty store, "How much protein should I eat a day?"
    #     check: it asks for details instead of inventing a number; nothing invented in the store
    # - progression: seed a squat workout of 3x8 at 50 kg; "How much should I squat today?"
    #     check: called suggest_next_weight and "52.5" in r.text
    # - log_workout: "Done! Squats 3x8 at 50kg, bench 3x6 at 35kg." check the store's exercise_history
]


def run_one(sc: Scenario, usage: UsageLog, provider: str, model: str | None) -> tuple[list[str], AgentResult]:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "data.json")
        sc.seed(store)
        call_model = build_call_model(provider, SYSTEM, TOOLS, usage=usage, model=model)
        messages = [{"role": "user", "content": sc.user}]
        result = run_turn(messages, call_model, lambda n, a: run_tool(n, a, store, TODAY))
        problems = sc.check(result, store)
        if result.stopped != "done":
            problems.append(f"stopped: {result.stopped}")
        return problems, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3, help="runs per scenario")
    ap.add_argument("-k", default="", help="only scenarios whose name contains this")
    ap.add_argument("-v", action="store_true", help="print every reply and tool call")
    ap.add_argument("--provider", default="claude", choices=["claude", "ollama"])
    ap.add_argument("--model", default=None, help="ollama model name")
    ap.add_argument("--judge", default=None, choices=["claude", "ollama"], help="judge provider (default: same)")
    opts = ap.parse_args()
    JUDGE["provider"] = opts.judge or opts.provider
    JUDGE["model"] = opts.model if JUDGE["provider"] == "ollama" else None

    usage = UsageLog()
    print(f"provider: {opts.provider} {opts.model or ''}  judge: {JUDGE['provider']}")
    for sc in [s for s in SCENARIOS if opts.k in s.name]:
        passed = 0
        for i in range(opts.n):
            problems, result = run_one(sc, usage, opts.provider, opts.model)
            passed += not problems
            if problems or opts.v:
                print(f"  {sc.name} run {i + 1}: {'; '.join(problems) or 'ok'}")
                print(f"    tools: {json.dumps(result.tool_calls)[:300]}")
                print(f"    reply: {result.text[:300]!r}")
        print(f"{sc.name}: {passed}/{opts.n}")
    print(usage)


if __name__ == "__main__":
    main()
