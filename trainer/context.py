"""What the get_context tool returns: everything the coach needs at the start of a conversation.

Why one tool: small models (qwen3:1.7b) often skipped get_profile and asked for details that were
already saved. One obvious "call this first" tool is easier for a weak model to pick than two.
To give the coach more starting information later (this week's workouts, today's nutrition,
active injuries...), add fields here; the tool, the prompt and the loop don't change.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from .coaching import CARDIO, SESSION_EXERCISES, WEEKDAYS, next_weight, session_for
from .store import Store


def build_context(store: Store, today: dt.date) -> dict[str, Any]:
    """Everything the coach needs to start, in one tool result:

        {"date": "2026-11-23", "weekday": "Mon", "profile": {...} or None,
         "today": {"session": "Full body A" or None (rest day / unknown),
                   "exercises": [{"name": "Back squat", "sets": 3, "reps": "5-8",
                                  "suggested_weight_kg": 52.5 or None (no history)}, ...]}}
    """
    profile = store.get_profile()
    context = {
        "date": today.isoformat(),
        "weekday": WEEKDAYS[today.weekday()],
        "profile": profile,
        "today": _todays_plan(store, today, profile),
        "cardio": CARDIO,
    }
    if not profile:   # seen: "no training days yet, let me know when..." and no question. Tell it what to do next.
        context["next_step"] = ("No profile yet. Before planning anything, ask for: sex, age, height, weight, "
                                "activity level, goal, which weekdays they can train, and experience.")
    return context


def _todays_plan(store: Store, today: dt.date, profile: dict[str, Any] | None) -> dict[str, Any]:
    train_days = (profile or {}).get("train_days") or []
    try:
        session = session_for(today.isoformat(), train_days)
    except ValueError:                       # no profile yet, or an unsupported number of training days
        return {"session": None, "exercises": [], "note": "no valid training days in the profile yet"}
    if session is None:
        return {"session": None, "exercises": [], "note": "rest day"}
    exercises = [
        {"name": name, "sets": sets, "reps": reps, "suggested_weight_kg": _suggested_weight(store, name, reps)}
        for name, sets, reps in SESSION_EXERCISES[session]
    ]
    return {"session": session, "exercises": exercises}


def _suggested_weight(store: Store, exercise: str, reps: str) -> float | None:
    """next_weight from the last 2 logged sessions. None if there's no history, or for timed
    exercises like "30-60s" that have no weight."""
    low, _, high = reps.partition("-")
    if not (low.isdigit() and high.isdigit()):
        return None
    history = [h["sets"] for h in store.exercise_history(exercise, limit=2)]
    return next_weight(history, int(low), int(high))
