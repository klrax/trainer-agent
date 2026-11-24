import datetime as dt

from trainer.context import build_context
from trainer.store import Store

MONDAY, TUESDAY = dt.date(2026, 11, 23), dt.date(2026, 11, 24)


def test_no_profile(tmp_path):
    ctx = build_context(Store(tmp_path / "d.json"), MONDAY)
    assert (ctx["date"], ctx["weekday"], ctx["profile"]) == ("2026-11-23", "Mon", None)
    assert ctx["today"]["session"] is None and ctx["today"]["exercises"] == []


def test_todays_plan_and_weights(tmp_path):
    s = Store(tmp_path / "d.json")
    s.update_profile(name="Kat", train_days=["Mon", "Wed", "Fri"])
    today = build_context(s, MONDAY)["today"]
    assert today["session"] == "Full body A"
    squat = today["exercises"][0]
    assert squat == {"name": "Back squat", "sets": 3, "reps": "5-8", "suggested_weight_kg": None}  # no history
    s.log_workout("2026-11-20", [{"name": "Back squat", "sets": [{"reps": 8, "weight_kg": 50}] * 3}])
    today = build_context(s, MONDAY)["today"]
    assert today["exercises"][0]["suggested_weight_kg"] == 52.5          # hit 3x8 -> +2.5
    plank = [e for e in today["exercises"] if e["name"] == "Plank"][0]
    assert plank["suggested_weight_kg"] is None                         # timed exercise: no weight
    assert build_context(s, TUESDAY)["today"] == {"session": None, "exercises": [], "note": "rest day"}
