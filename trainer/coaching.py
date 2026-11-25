"""The trainer's rules, as plain Python.

Why this matters for the agent: LLMs are bad at arithmetic and inconsistent with rules.
So the MATH lives here (calories, macros, the weight to lift next) and is unit-tested.
The model's job is the conversation: asking questions, explaining, adapting.
Principle: deterministic logic in tools, judgment in the model.

Run:  uv run pytest -k s2
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

# Activity multipliers (standard TDEE factors) and calorie change per goal.
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
GOAL_ADJUST = {"cut": -500, "maintain": 0, "bulk": 300}

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]   # index 0 = Monday, like date.weekday()

# training days per week -> the session for each training day, in order
SPLITS = {
    2: ["Full body A", "Full body B"],
    3: ["Full body A", "Full body B", "Full body A"],
    4: ["Upper", "Lower", "Upper", "Lower"],
    5: ["Push", "Pull", "Legs", "Upper", "Lower"],
    6: ["Push", "Pull", "Legs", "Push", "Pull", "Legs"],
}

# What each session contains: (exercise, sets, rep range). Data, not graded: edit it to your taste.
# The model reads this through the system prompt (see prompts.py).
SESSION_EXERCISES = {
    "Full body A": [("Back squat", 3, "5-8"), ("Bench press", 3, "6-10"), ("Barbell row", 3, "8-12"), ("Plank", 3, "30-60s")],
    "Full body B": [("Romanian deadlift", 3, "6-10"), ("Overhead press", 3, "6-10"), ("Lat pulldown", 3, "8-12"), ("Walking lunge", 2, "10-12")],
    "Upper": [("Bench press", 3, "6-10"), ("Barbell row", 3, "8-12"), ("Overhead press", 3, "8-10"), ("Lat pulldown", 3, "10-12")],
    "Lower": [("Back squat", 3, "5-8"), ("Romanian deadlift", 3, "6-10"), ("Leg press", 3, "10-12"), ("Calf raise", 3, "12-15")],
    "Push": [("Bench press", 4, "6-10"), ("Overhead press", 3, "8-10"), ("Incline DB press", 3, "10-12"), ("Triceps pushdown", 3, "12-15")],
    "Pull": [("Deadlift", 3, "3-5"), ("Pull-up", 3, "5-10"), ("Barbell row", 3, "8-12"), ("Biceps curl", 3, "12-15")],
    "Legs": [("Back squat", 4, "5-8"), ("Romanian deadlift", 3, "8-10"), ("Leg press", 3, "10-12"), ("Calf raise", 3, "12-15")],
}


def daily_targets(profile: dict[str, Any]) -> dict[str, int]:
    """Daily calories and macros. Uses profile keys: sex, age, height_cm, weight_kg, activity, goal.

    Steps, in this order (use round() at each step exactly as written):
    1. bmr  = 10*weight_kg + 6.25*height_cm - 5*age + (5 if sex == "male" else -161)     (Mifflin-St Jeor)
    2. kcal = round(bmr * ACTIVITY[activity] + GOAL_ADJUST[goal])
    3. protein_g = round(1.8 * weight_kg)
    4. fat_g     = round(kcal * 0.25 / 9)                          (25% of calories; fat has 9 kcal/g)
    5. carbs_g   = round((kcal - protein_g*4 - fat_g*9) / 4)       (the rest; protein & carbs have 4 kcal/g)

    What to return: {"kcal": ..., "protein_g": ..., "fat_g": ..., "carbs_g": ...}

    Errors:
    - activity not in ACTIVITY raises ValueError
    - goal not in GOAL_ADJUST raises ValueError

    Example: female, 30, 165 cm, 60 kg, moderate, maintain
        bmr = 1320.25;  kcal = round(2046.39) = 2046
        -> {"kcal": 2046, "protein_g": 108, "fat_g": 57, "carbs_g": 275}
    """
    if profile["activity"] not in ACTIVITY:
        raise ValueError(f"unknown activity {profile['activity']!r}; use one of {list(ACTIVITY)}")
    if profile["goal"] not in GOAL_ADJUST:
        raise ValueError(f"unknown goal {profile['goal']!r}; use one of {list(GOAL_ADJUST)}")
    weight_kg, height_cm, age = profile["weight_kg"], profile["height_cm"], profile["age"]
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if profile["sex"] == "male" else -161)
    kcal = round(bmr * ACTIVITY[profile["activity"]] + GOAL_ADJUST[profile["goal"]])
    protein_g = round(1.8 * weight_kg)
    fat_g = round(kcal * 0.25 / 9)
    carbs_g = round((kcal - protein_g * 4 - fat_g * 9) / 4)
    return {"kcal": kcal, "protein_g": protein_g, "fat_g": fat_g, "carbs_g": carbs_g}


def day_totals(meals: list[dict[str, Any]]) -> dict[str, float]:
    """- the sum of each of kcal, protein_g, carbs_g, fat_g over the meals
    - day_totals([]) -> {"kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    """
    return {key: sum(m[key] for m in meals) for key in ("kcal", "protein_g", "carbs_g", "fat_g")}


def remaining(targets: dict[str, float], meals: list[dict[str, Any]]) -> dict[str, float]:
    """- for each of kcal, protein_g, carbs_g, fat_g: targets[key] - day_totals(meals)[key]
    - can be negative (ate more than the target): that's fine, return it as is
    """
    totals = day_totals(meals)
    return {key: targets[key] - totals[key] for key in ("kcal", "protein_g", "carbs_g", "fat_g")}


def weekly_split(days_per_week: int) -> list[str]:
    """- weekly_split(3) -> ["Full body A", "Full body B", "Full body A"]   (a COPY of SPLITS[3])
    - days_per_week not in SPLITS (so not 2-6) raises ValueError
    """
    if days_per_week not in SPLITS:
        raise ValueError(f"days_per_week must be 2-6, got {days_per_week}")
    return list(SPLITS[days_per_week])


def session_for(date: str, train_days: list[str]) -> str | None:
    """Which session is planned on a date.

    Input:
    - date is "YYYY-MM-DD".  dt.date.fromisoformat(date).weekday() gives 0 for Monday ... 6 for Sunday
    - train_days are weekday names from WEEKDAYS, in ANY order, e.g. ["Sat", "Mon", "Wed"]

    Steps:
    1. sort train_days Monday-first:  sorted(train_days, key=WEEKDAYS.index) -> ["Mon", "Wed", "Sat"]
    2. split = weekly_split(len(train_days))    (raises ValueError for 1 or 7 days: let it)
    3. the date's weekday is a training day -> the split entry at the same position
    4. not a training day -> None (a rest day)

    Example with train_days ["Sat", "Mon", "Wed"]:
    - Monday 2026-11-23    -> "Full body A"   (1st training day)
    - Wednesday 2026-11-25 -> "Full body B"   (2nd)
    - Saturday 2026-11-28  -> "Full body A"   (3rd)
    - Tuesday 2026-11-24   -> None
    """
    ordered = sorted(train_days, key=WEEKDAYS.index)
    split = weekly_split(len(ordered))
    weekday = WEEKDAYS[dt.date.fromisoformat(date).weekday()]
    if weekday not in ordered:
        return None
    return split[ordered.index(weekday)]


# Offered when the user asks for cardio. In get_context because the small model never called search_exercises for
# cardio (0/3 with a prompt rule), but it reliably reads get_context.
CARDIO = {"options": ["brisk walk (treadmill or outside)", "stationary bike", "rowing machine", "elliptical"],
          "how": "20-30 min at an easy pace you could talk at, after lifting"}


def next_weight(history: list[list[dict[str, Any]]], rep_min: int, rep_max: int,
                step: float = 2.5) -> float | None:
    """Progressive overload ("double progression"): the weight to use next time for one exercise.

    Input:
    - history: past sessions of this exercise, OLDEST first. Each session is a list of sets
      {"reps": int, "weight_kg": float}.
    - rep_min, rep_max: the target rep range, e.g. 6 and 8

    Words:
    - w = the heaviest weight_kg in the LAST session
    - a session is "missed" if any of its sets has reps < rep_min

    Rules, checked in this order:
    1. history is empty -> None   (no data: the model should ask, or start light)
    2. EVERY set of the last session has reps >= rep_max -> w + step
    3. the last TWO sessions are both "missed" -> deload: math.floor(w * 0.9 / step) * step
    4. otherwise -> w   (same weight, try for more reps)

    Examples (rep range 6-8):
    - [[8, 8, 8 reps at 40]]                  -> 42.5
    - [[8, 8, 7 reps at 40]]                  -> 40
    - [[6, 5, 5 at 40]]                       -> 40     (one missed session: keep trying)
    - [[5, 5 at 40], [6, 5 at 40]]            -> 35.0   (two missed in a row: floor(36 / 2.5) * 2.5)
    """
    if not history:
        return None
    last = history[-1]
    w = max(s["weight_kg"] for s in last)

    def missed(session):
        return any(s["reps"] < rep_min for s in session)

    if all(s["reps"] >= rep_max for s in last):
        return w + step
    if len(history) >= 2 and missed(history[-1]) and missed(history[-2]):
        return math.floor(w * 0.9 / step) * step
    return w
