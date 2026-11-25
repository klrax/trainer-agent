"""Tools: the agent's hands.

Two halves:
- TOOLS: what the MODEL sees. A name, a description, and a JSON schema for the input.
  The model reads the description to decide WHEN to call a tool, so write it for a reader:
  say what it returns and when to use it.
- run_tool(): what YOUR CODE does when the model asks for a tool.

Run:  uv run pytest -k s3
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from . import coaching
from .context import build_context
from .rag import meals
from .rag.knowledge import CATEGORIES, LEVELS, KnowledgeBase, default_kb
from .store import PROFILE_FIELDS, Store

MUSCLES = ["abdominals", "abductors", "adductors", "biceps", "calves", "chest", "forearms", "glutes", "hamstrings",
           "lats", "lower back", "middle back", "neck", "quadriceps", "shoulders", "traps", "triceps"]
EQUIPMENT = ["bands", "barbell", "body only", "cable", "dumbbell", "e-z curl bar", "exercise ball", "foam roll",
             "kettlebells", "machine", "medicine ball", "other"]
DATE = {"type": "string", "description": "A date as YYYY-MM-DD."}
DAYS_AGO = {"type": "integer", "description": "0 = today (default), 1 = yesterday."}
FOOD_ITEMS = {
    "type": "array",
    "description": "One entry per food.",
    "items": {
        "type": "object",
        "properties": {
            "food": {"type": "string", "description": "One food in plain words, with how it was cooked, e.g. 'scrambled eggs'."},
            "quantity": {"type": "number", "description": "How many units, e.g. 2."},
            "unit": {"type": "string", "enum": meals.UNITS, "description": "Use 'g' when a weight is given."},
        },
        "required": ["food", "quantity", "unit"],
    },
}

# ── Part A: the schemas ────────────────────────────────────────────────────────────────
# Three are done as examples. Write the other 7. Required per tool (the test checks these):
#   update_profile:       required []      properties: every key in store.PROFILE_FIELDS
#                         (hint: "enum" for sex ["female","male"], activity list(coaching.ACTIVITY),
#                          goal list(coaching.GOAL_ADJUST); train_days is an array of WEEKDAYS strings)
#   get_daily_targets:    required []
#   get_day_nutrition:    required []                     optional "days_ago" (0 = today)
#   get_session_for:      required ["date"]
#   get_exercise_history: required ["exercise"]            optional "limit" (integer)
#   suggest_next_weight:  required ["exercise", "rep_min", "rep_max"]
#   log_workout:          required ["date", "exercises"]   exercises: array of {name, sets: [{reps, weight_kg}]}
# Every tool needs: "name", a non-empty "description", and "input_schema" with
#   "type": "object", "properties": {...}, "required": [...]   (every required name must be in properties)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_context",
        "description": "Call this FIRST in every conversation, before answering anything. Returns today's date, "
                       "the weekday, the user's saved profile (null if none yet), and today's plan: the session "
                       "(null on a rest day) and each exercise with sets, reps and suggested_weight_kg. "
                       "Never ask for details already in the profile, never guess the date, and use "
                       "suggested_weight_kg exactly; when it is null, do NOT make up a weight.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "log_meal",
        "description": "Records one meal the user says they ALREADY ate. List each food with its amount; calories and "
                       "protein are calculated for you and returned with what is left today. One meal = one call. "
                       "Never call it for a suggestion or a question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": FOOD_ITEMS,
                "days_ago": DAYS_AGO,
            },
            "required": ["items"],
        },
    },
    {
        "name": "update_profile",
        "description": "Saves or changes LASTING profile facts. Send only the fields that changed; the others are "
                       "kept. Call it when the user tells you a lasting fact about themselves (new weight, new goal, "
                       "an injury, which days they can train). Never for a passing state (a cold, tiredness, a bad "
                       "night's sleep, normal soreness): don't save those, just adapt today's advice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "sex": {"type": "string", "enum": ["female", "male"], "description": "Used for the calorie formula."},
                "age": {"type": "integer"},
                "height_cm": {"type": "number"},
                "weight_kg": {"type": "number"},
                "activity": {"type": "string", "enum": list(coaching.ACTIVITY),
                             "description": "Daily activity OUTSIDE the gym sessions."},
                "goal": {"type": "string", "enum": list(coaching.GOAL_ADJUST),
                         "description": "cut = lose fat, maintain, bulk = gain muscle."},
                "train_days": {"type": "array", "items": {"type": "string", "enum": coaching.WEEKDAYS},
                               "description": "Weekdays they can train, 2 to 6 of them, e.g. [\"Mon\", \"Wed\", \"Fri\"]."},
                "experience": {"type": "string", "description": "e.g. 'new to lifting', '2 years'."},
                "injuries": {"type": "string", "description": "Lasting injuries (e.g. 'left knee meniscus'); not "
                                                         "illnesses like a cold. '' if none."},
            },
            "required": [],
        },
    },
    {
        "name": "get_daily_targets",
        "description": "Returns today's calorie and macro targets ({kcal, protein_g, fat_g, carbs_g}), calculated "
                       "from the profile. Never calculate these yourself; use this tool.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_day_nutrition",
        "description": "Returns the meals logged on a day, their totals, and what is left of each target "
                       "(negative = over the target). Use it for 'how am I doing today?'.",
        "input_schema": {"type": "object", "properties": {"days_ago": DAYS_AGO}, "required": []},
    },
    {
        "name": "get_session_for",
        "description": "Returns which training session is planned on a date, e.g. {\"session\": \"Full body A\"}, "
                       "or {\"session\": null} for a rest day. The exercises of each session are in your instructions.",
        "input_schema": {"type": "object", "properties": {"date": DATE}, "required": ["date"]},
    },
    {
        "name": "get_exercise_history",
        "description": "Returns the user's recent logged sets for one exercise, oldest first: "
                       "[{date, sets: [{reps, weight_kg}]}]. Use it to talk about their progress.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exercise": {"type": "string", "description": "Exercise name as in the program, e.g. 'Back squat'."},
                "limit": {"type": "integer", "description": "How many past sessions to return (default 5)."},
            },
            "required": ["exercise"],
        },
    },
    {
        "name": "suggest_next_weight",
        "description": "Returns the weight to use next for one exercise, from the user's logged history "
                       "(progressive overload). Call this before recommending ANY working weight; never guess one. "
                       "weight_kg is null when there is no history for the exercise.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exercise": {"type": "string", "description": "Exercise name as in the program, e.g. 'Back squat'."},
                "rep_min": {"type": "integer", "description": "Bottom of the rep range, e.g. 5 for '5-8'."},
                "rep_max": {"type": "integer", "description": "Top of the rep range, e.g. 8 for '5-8'."},
            },
            "required": ["exercise", "rep_min", "rep_max"],
        },
    },
    {
        "name": "log_workout",
        "description": "Records a workout the user says they FINISHED, one entry per exercise: '3 sets of 8 at 40 kg' "
                       "= {name, sets: 3, reps: 8, weight_kg: 40}. Use the program's exercise names so history and "
                       "weight suggestions find them. The date is worked out from days_ago. Never call it for a "
                       "question or a plan. Confirm with the user what you logged.",
        "input_schema": {
            "type": "object",
            "properties": {
                "exercises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "e.g. 'Back squat'."},
                            "sets": {"type": "integer", "description": "How many sets, e.g. 3."},
                            "reps": {"type": "integer", "description": "Reps in each set."},
                            "weight_kg": {"type": "number", "description": "0 for bodyweight."},
                        },
                        "required": ["name", "sets", "reps", "weight_kg"],
                    },
                },
                "days_ago": DAYS_AGO,
            },
            "required": ["exercises"],
        },
    },
    {
        "name": "search_foods",
        "description": "Find which entries the Japanese (MEXT) and US (USDA) food tables have for one food, e.g. when "
                       "another tool says a food was not found, or to compare foods. Search one food at a time, in the "
                       "user's own words. To get numbers for an amount, use get_food_nutrition, not this.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "One food, e.g. '鶏むね肉 皮なし' or 'boiled egg'."},
                "limit": {"type": "integer", "description": "1-10, default 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_food_nutrition",
        "description": "Calculates kcal and protein, fat, carbs for foods the user asks about, and the total. Saves "
                       "nothing. Use it for 'how much protein is in ...?'. Never do this arithmetic yourself.",
        "input_schema": {"type": "object", "properties": {"items": FOOD_ITEMS}, "required": ["items"]},
    },
    {
        "name": "suggest_meal",
        "description": "Call this whenever the user wants an idea for what to eat (high-protein lunch, dinner, snack). "
                       "Returns meals with grams, calories and protein already worked out from today's remaining "
                       "targets. Saves nothing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal": {"type": "string", "enum": list(meals.MEALS_LEFT)},
                "protein_g": {"type": "number", "description": "Minimum. Only if the user named a number."},
                "max_kcal": {"type": "number", "description": "Only if the user named a number."},
                "avoid": {"type": "array", "items": {"type": "string", "enum": meals.AVOID},
                          "description": "Allergies and dislikes the user mentioned."},
            },
            "required": ["meal"],
        },
    },
    {
        "name": "search_exercises",
        "description": "Find exercises for building or changing a training menu: alternatives when equipment is "
                       "missing or an exercise is uncomfortable, extra work for a muscle, cardio (category "
                       "'cardio'), or how to do a movement. "
                       "Only suggest exercises this returns or that are in the user's program.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "In English, e.g. 'horizontal row alternative'."},
                "muscles": {"type": "array", "items": {"type": "string", "enum": MUSCLES},
                            "description": "Primary muscle is any of these. Always set it when the user names a body "
                                           "part: back = lats, middle back, lower back, traps; legs = quadriceps, "
                                           "hamstrings, glutes, calves; arms = biceps, triceps, forearms."},
                "equipment": {"type": "array", "items": {"type": "string", "enum": EQUIPMENT},
                              "description": "Only what the user has. Leave out if any equipment is fine."},
                "level": {"type": "string", "enum": LEVELS, "description": "This level or easier."},
                "category": {"type": "string", "enum": sorted(CATEGORIES),
                             "description": "'training' (lifting, the default), 'cardio' (walking, running, cycling, "
                                            "rowing machine...), 'stretching', 'plyometrics' or 'strongman'."},
                "limit": {"type": "integer", "description": "1-10, default 5."},
            },
            "required": ["query"],
        },
    },
]


# ── Part B: running them ───────────────────────────────────────────────────────────────
def run_tool(name: str, args: dict[str, Any], store: Store, today: dt.date,
             kb: KnowledgeBase | None = None) -> tuple[str, bool]:
    """Run one tool call. Returns (content, is_error).

    The golden rule: NEVER raise. A tool error is information for the model ("no profile yet"),
    so return it as text and let the model react (ask the user, fix its input, retry).

    What to return:
    - success: (json.dumps(result), False)
    - failure: (f"Error: {e}", True)   for ValueError, KeyError, TypeError
    - unknown tool name: (f"Error: unknown tool {name}", True)

    What each tool returns (before json.dumps):
    - get_context          -> context.build_context(store, today): {"date", "weekday", "profile"}
    - update_profile       -> store.update_profile(**args)
    - get_daily_targets    -> coaching.daily_targets(profile)
    - log_meal             -> code finds the foods and calculates; the model never sends calories
                               {"logged": meal, "per_food", "assumptions", "say", "remaining_today"}
    - suggest_meal         -> rag/meals.py: grams solved to reach the protein goal; saves nothing
    - get_day_nutrition    -> {"meals": [...], "totals": coaching.day_totals(...),
                               "remaining": coaching.remaining(targets, meals)}
    - get_session_for      -> {"date": args["date"], "session": coaching.session_for(date, profile train_days)}
    - get_exercise_history -> store.exercise_history(args["exercise"], args.get("limit", 5))
    - suggest_next_weight  -> {"exercise": args["exercise"],
                               "weight_kg": coaching.next_weight([h["sets"] for h in last 2 history entries],
                                                                 args["rep_min"], args["rep_max"])}
    - log_workout          -> store.log_workout(...), then {"ok": True}
    - search_foods, get_food_nutrition, search_exercises -> rag/knowledge.py. `kb` is passed in by tests;
      otherwise the installed one is loaded on first use (so other tools never wait for it)

    Tools that need the profile (get_daily_targets, get_day_nutrition, get_session_for):
    - no profile yet -> raise ValueError("no profile yet: ask the user for their details and call update_profile")
      inside your try, so it comes back as an error string. The wording matters: it tells the
      model what to do next. (The test checks that the text contains "update_profile".)

    Example:
        run_tool("get_context", {}, store, dt.date(2026, 11, 23))
        -> ('{"date": "2026-11-23", "weekday": "Mon", "profile": null}', False)
    """
    try:
        if name == "get_context":
            if set(args) & PROFILE_FIELDS:              # seen: {"weight_kg": 58} sent here instead of update_profile
                return ("Error: get_context takes no arguments and saved nothing. If the user told you something new "
                        "about themselves, call update_profile with ONLY what they said."), True
            result = build_context(store, today)
        elif name == "update_profile":
            # Small models resend the whole profile; keep only real changes, so a reply can't silently rewrite it.
            saved = store.get_profile() or {}
            changed = {k: v for k, v in args.items() if saved.get(k) != v}
            result = store.update_profile(**changed) if changed else {"unchanged": True, "profile": saved}
        elif name == "get_daily_targets":
            result = coaching.daily_targets(_need_profile(store))
        elif name == "log_meal":
            result = _log_meal(args, store, today, kb or default_kb())
        elif name == "get_day_nutrition":
            targets = coaching.daily_targets(_need_profile(store))
            date = _date(args, today)
            eaten = store.meals_on(date)
            result = {"date": date, "targets": targets, "meals": eaten, "totals": _rounded(coaching.day_totals(eaten)),
                      "remaining": _rounded(coaching.remaining(targets, eaten))}
        elif name == "get_session_for":
            profile = _need_profile(store)
            result = {"date": args["date"],
                      "session": coaching.session_for(args["date"], profile.get("train_days", []))}
        elif name == "get_exercise_history":
            result = store.exercise_history(args["exercise"], args.get("limit", 5))
        elif name == "suggest_next_weight":
            history = [h["sets"] for h in store.exercise_history(args["exercise"], limit=2)]
            result = {"exercise": args["exercise"],
                      "weight_kg": coaching.next_weight(history, args["rep_min"], args["rep_max"])}
        elif name == "log_workout":
            date = _date(args, today)                       # models made up 2023-10-15; they don't send dates
            exercises = [_sets(e) for e in args["exercises"]]
            store.log_workout(date, exercises)
            result = {"ok": True, "date": date,
                      "logged": [f"{e['name']}: {len(e['sets'])}x{e['sets'][0]['reps']} at {e['sets'][0]['weight_kg']:g} kg"
                                 if len({(x['reps'], x['weight_kg']) for x in e['sets']}) == 1 else
                                 f"{e['name']}: " + ", ".join(f"{x['reps']} at {x['weight_kg']:g} kg" for x in e['sets'])
                                 for e in exercises]}
        elif name == "search_foods":
            result = (kb or default_kb()).search_foods(**args)
        elif name == "get_food_nutrition":
            result = meals.calculate(kb or default_kb(), args.get("items"))
            result = {k: v for k, v in result.items() if k != "items"}        # per_food says the same, shorter
        elif name == "suggest_meal":
            result = meals.suggest_meal(kb or default_kb(), args["meal"], _remaining_today(store, today),
                                        args.get("protein_g"), args.get("max_kcal"), args.get("avoid"))
        elif name == "search_exercises":
            result = (kb or default_kb()).search_exercises(**args)
        else:
            return f"Error: unknown tool {name}", True
        return json.dumps(result, ensure_ascii=False), False
    except (ValueError, KeyError, TypeError) as e:
        # e.g. KeyError('date') -> "Error: 'date'"; TypeError from **args with a missing/extra field
        return f"Error: {e}", True


def _need_profile(store: Store) -> dict[str, Any]:
    """The profile, or a ValueError whose text tells the MODEL what to do next."""
    profile = store.get_profile()
    if not profile:
        raise ValueError("no profile yet: ask the user for their details and call update_profile")
    return profile


def _log_meal(args: dict[str, Any], store: Store, today: dt.date, kb: KnowledgeBase) -> dict[str, Any]:
    """Calories and the date are worked out here, whatever the model says. Nothing is saved if any food is unclear."""
    date = _date(args, today)
    calc = meals.calculate(kb, args.get("items"))
    meal = store.log_meal(date=date,
                          name=", ".join(f"{r['label']} {r['grams']:g} g" for r in calc["items"]), **calc["total"])
    left = _remaining_today(store, today)
    goal = coaching.daily_targets(_need_profile(store)) if left else None
    say = f"Logged for {meal['date']}: {calc['say']}"
    if left:
        say += (f". Left today: {left['kcal']:g} of {goal['kcal']} kcal, "
                f"{left['protein_g']:g} of {goal['protein_g']} g protein")
    return {"logged": meal, "per_food": calc["per_food"], "assumptions": calc["assumptions"],
            **({"incomplete": calc["incomplete"]} if "incomplete" in calc else {}),
            "sources": calc["sources"], "say": say, "remaining_today": left, "targets": goal}


def _sets(exercise: dict[str, Any]) -> dict[str, Any]:
    """{name, sets: 3, reps: 8, weight_kg: 40} -> {name, sets: [{reps: 8, weight_kg: 40}] * 3}.
    A list of sets is accepted too (different reps per set). Small models wrote one set for "3 sets of 8"."""
    sets = exercise.get("sets")
    if isinstance(sets, list):
        rows = sets
    elif isinstance(sets, int) and not isinstance(sets, bool) and 1 <= sets <= 20:
        rows = [{"reps": exercise["reps"], "weight_kg": exercise["weight_kg"]}] * sets
    else:
        raise ValueError(f"sets for {exercise.get('name')!r} must be a number of sets from 1 to 20")
    for x in rows:
        if not (isinstance(x.get("reps"), int) and 0 < x["reps"] <= 100 and isinstance(x.get("weight_kg"), (int, float))
                and 0 <= x["weight_kg"] <= 500):
            raise ValueError(f"each set of {exercise.get('name')!r} needs reps (1-100) and weight_kg (0-500)")
    return {"name": exercise["name"], "sets": [dict(x) for x in rows]}


def _date(args: dict[str, Any], today: dt.date) -> str:
    """The date from days_ago. Models make up dates (2023-10-15, three times), so they don't get to send one."""
    if "date" in args:
        raise ValueError("there is no 'date': use days_ago (0 = today, 1 = yesterday)")
    days_ago = args.get("days_ago") or 0
    if isinstance(days_ago, bool) or not isinstance(days_ago, int) or not 0 <= days_ago <= 14:
        raise ValueError("days_ago must be a whole number from 0 (today) to 14")
    return (today - dt.timedelta(days=days_ago)).isoformat()


def _remaining_today(store: Store, today: dt.date) -> dict[str, float] | None:
    """What is left of today's targets, or None when the profile isn't complete enough to have targets."""
    try:
        targets = coaching.daily_targets(_need_profile(store))
    except (ValueError, KeyError):
        return None
    return _rounded(coaching.remaining(targets, store.meals_on(today.isoformat())))


def _rounded(macros: dict[str, float]) -> dict[str, float]:
    return {k: round(v) if k == "kcal" else round(v, 1) for k, v in macros.items()}
