import pytest

from trainer.store import Store


def test_s1_empty_store_and_profile(tmp_path):
    s = Store(tmp_path / "data.json")
    assert s.get_profile() is None
    assert s.update_profile(name="Kat", age=30) == {"name": "Kat", "age": 30}
    assert s.update_profile(age=31) == {"name": "Kat", "age": 31}          # merges, doesn't replace
    with pytest.raises(ValueError):
        s.update_profile(favourite_colour="red")                           # unknown field


def test_s1_get_profile_returns_a_copy(tmp_path):
    s = Store(tmp_path / "data.json")
    s.update_profile(name="Kat", train_days=["Mon", "Thu"])
    p = s.get_profile()
    p["train_days"].append("Sat")
    assert s.get_profile()["train_days"] == ["Mon", "Thu"]


def test_s1_data_survives_a_restart(tmp_path):
    path = tmp_path / "data.json"
    s = Store(path)
    s.update_profile(name="Kat")
    s.log_meal("2026-11-22", "oats", 350, 12, 60, 7)
    s.log_workout("2026-11-22", [{"name": "Squat", "sets": [{"reps": 5, "weight_kg": 60}]}])
    s2 = Store(path)                                                       # a new process, same file
    assert s2.get_profile() == {"name": "Kat"}
    assert s2.meals_on("2026-11-22")[0]["name"] == "oats"
    assert s2.exercise_history("Squat") == [{"date": "2026-11-22", "sets": [{"reps": 5, "weight_kg": 60}]}]


def test_s1_meals(tmp_path):
    s = Store(tmp_path / "data.json")
    m = s.log_meal("2026-11-22", "oats", 350, 12, 60, 7)
    assert m == {"date": "2026-11-22", "name": "oats", "kcal": 350, "protein_g": 12, "carbs_g": 60, "fat_g": 7}
    s.log_meal("2026-11-23", "rice", 200, 4, 45, 0)
    assert [x["name"] for x in s.meals_on("2026-11-22")] == ["oats"]
    assert s.meals_on("2026-01-01") == []
    with pytest.raises(ValueError):
        s.log_meal("2026-11-22", "bad", -1, 0, 0, 0)


def test_s1_exercise_history(tmp_path):
    s = Store(tmp_path / "data.json")
    sq = lambda w: {"name": "Squat", "sets": [{"reps": 5, "weight_kg": w}]}
    s.log_workout("2026-11-20", [sq(62.5), {"name": "Bench", "sets": [{"reps": 8, "weight_kg": 40}]}])
    s.log_workout("2026-11-13", [sq(60)])                                  # logged late, older date
    s.log_workout("2026-11-24", [{"name": "squat", "sets": [{"reps": 5, "weight_kg": 65}]}])
    hist = s.exercise_history("SQUAT")                                     # case-insensitive
    assert [h["date"] for h in hist] == ["2026-11-13", "2026-11-20", "2026-11-24"]   # oldest first
    assert [h["date"] for h in s.exercise_history("Squat", limit=2)] == ["2026-11-20", "2026-11-24"]
    assert s.exercise_history("Deadlift") == []
    with pytest.raises(ValueError):
        s.log_workout("2026-11-22", [])
