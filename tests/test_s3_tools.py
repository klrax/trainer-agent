import datetime as dt
import json

import pytest

from trainer.store import Store
from trainer.tools import TOOLS, run_tool

TODAY = dt.date(2026, 11, 23)  # a Monday
KAT = {"sex": "female", "age": 30, "height_cm": 165, "weight_kg": 60, "activity": "moderate",
       "goal": "maintain", "train_days": ["Mon", "Wed", "Fri"]}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "data.json")


def call(store, tool, **args):
    content, is_error = run_tool(tool, args, store, TODAY)
    assert isinstance(content, str)
    return (json.loads(content) if not is_error else content), is_error


def test_s3_every_tool_has_a_schema():
    names = [t["name"] for t in TOOLS]
    assert sorted(names) == sorted([
        "get_context", "update_profile", "get_daily_targets", "log_meal",
        "get_day_nutrition", "get_session_for", "get_exercise_history", "suggest_next_weight", "log_workout"])
    for t in TOOLS:
        assert t["description"].strip(), t["name"]
        assert t["input_schema"]["type"] == "object", t["name"]
        assert set(t["input_schema"]["required"]) <= set(t["input_schema"]["properties"]), t["name"]
    req = {t["name"]: set(t["input_schema"]["required"]) for t in TOOLS}
    assert req["log_meal"] == {"date", "name", "kcal", "protein_g", "carbs_g", "fat_g"}
    assert req["suggest_next_weight"] == {"exercise", "rep_min", "rep_max"}
    assert req["log_workout"] == {"date", "exercises"}
    assert req["get_session_for"] == {"date"}


def test_s3_context_and_profile(store):
    ctx, err = call(store, "get_context")
    assert not err and (ctx["date"], ctx["weekday"], ctx["profile"]) == ("2026-11-23", "Mon", None)
    out, err = call(store, "update_profile", **KAT)
    assert not err and out["goal"] == "maintain"
    assert call(store, "get_daily_targets") == ({"kcal": 2046, "protein_g": 108, "fat_g": 57, "carbs_g": 275}, False)


def test_s3_errors_come_back_as_text_not_exceptions(store):
    out, err = call(store, "get_daily_targets")                   # no profile yet
    assert err and out.startswith("Error:") and "update_profile" in out
    out, err = call(store, "teleport")
    assert err and "unknown tool" in out
    out, err = call(store, "log_meal", date="2026-11-23", name="oats")   # missing fields
    assert err and out.startswith("Error:")
    out, err = call(store, "update_profile", favourite_colour="red")
    assert err


def test_s3_nutrition(store):
    call(store, "update_profile", **KAT)
    call(store, "log_meal", date="2026-11-23", name="oats", kcal=350, protein_g=12, carbs_g=60, fat_g=7)
    out, err = call(store, "get_day_nutrition", date="2026-11-23")
    assert not err
    assert out["totals"] == {"kcal": 350, "protein_g": 12, "carbs_g": 60, "fat_g": 7}
    assert out["remaining"] == {"kcal": 1696, "protein_g": 96, "carbs_g": 215, "fat_g": 50}
    assert [m["name"] for m in out["meals"]] == ["oats"]


def test_s3_training(store):
    call(store, "update_profile", **KAT)
    assert call(store, "get_session_for", date="2026-11-25") == ({"date": "2026-11-25", "session": "Full body B"}, False)
    assert call(store, "get_session_for", date="2026-11-24") == ({"date": "2026-11-24", "session": None}, False)
    assert call(store, "suggest_next_weight", exercise="Squat", rep_min=5, rep_max=8) == \
        ({"exercise": "Squat", "weight_kg": None}, False)
    out, err = call(store, "log_workout", date="2026-11-23",
                    exercises=[{"name": "Squat", "sets": [{"reps": 8, "weight_kg": 50}] * 3}])
    assert (out, err) == ({"ok": True}, False)
    assert call(store, "suggest_next_weight", exercise="squat", rep_min=5, rep_max=8) == \
        ({"exercise": "squat", "weight_kg": 52.5}, False)
    out, err = call(store, "get_exercise_history", exercise="Squat")
    assert out == [{"date": "2026-11-23", "sets": [{"reps": 8, "weight_kg": 50}] * 3}]
