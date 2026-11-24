import pytest

from trainer.coaching import daily_targets, day_totals, next_weight, remaining, session_for, weekly_split

KAT = {"sex": "female", "age": 30, "height_cm": 165, "weight_kg": 60, "activity": "moderate", "goal": "maintain"}


def test_s2_daily_targets():
    assert daily_targets(KAT) == {"kcal": 2046, "protein_g": 108, "fat_g": 57, "carbs_g": 275}
    cut = daily_targets({**KAT, "goal": "cut"})
    assert cut["kcal"] == 1546
    male = daily_targets({**KAT, "sex": "male", "age": 35, "weight_kg": 80, "height_cm": 180, "goal": "bulk"})
    assert male == {"kcal": 3020, "protein_g": 144, "fat_g": 84, "carbs_g": 422}
    with pytest.raises(ValueError):
        daily_targets({**KAT, "activity": "couch"})
    with pytest.raises(ValueError):
        daily_targets({**KAT, "goal": "shred"})


def test_s2_totals_and_remaining():
    meals = [{"kcal": 350, "protein_g": 12, "carbs_g": 60, "fat_g": 7},
             {"kcal": 600, "protein_g": 45, "carbs_g": 50, "fat_g": 20}]
    assert day_totals(meals) == {"kcal": 950, "protein_g": 57, "carbs_g": 110, "fat_g": 27}
    assert day_totals([]) == {"kcal": 0, "protein_g": 0, "carbs_g": 0, "fat_g": 0}
    targets = {"kcal": 900, "protein_g": 100, "carbs_g": 100, "fat_g": 60}
    assert remaining(targets, meals) == {"kcal": -50, "protein_g": 43, "carbs_g": -10, "fat_g": 33}


def test_s2_weekly_split_and_session_for():
    assert weekly_split(3) == ["Full body A", "Full body B", "Full body A"]
    assert weekly_split(4) == ["Upper", "Lower", "Upper", "Lower"]
    with pytest.raises(ValueError):
        weekly_split(7)
    # 2026-11-23 is a Monday, 2026-11-25 a Wednesday, 2026-11-28 a Saturday
    days = ["Sat", "Mon", "Wed"]                               # any order: sorted Mon..Sun first
    assert session_for("2026-11-23", days) == "Full body A"    # Mon = 1st training day
    assert session_for("2026-11-25", days) == "Full body B"    # Wed = 2nd
    assert session_for("2026-11-28", days) == "Full body A"    # Sat = 3rd
    assert session_for("2026-11-24", days) is None             # Tue = rest
    with pytest.raises(ValueError):
        session_for("2026-11-23", ["Mon"])                     # 1 day/week isn't a supported split


def test_s2_next_weight():
    sets = lambda w, *reps: [{"reps": r, "weight_kg": w} for r in reps]
    assert next_weight([], 6, 8) is None
    assert next_weight([sets(40, 8, 8, 8)], 6, 8) == 42.5          # hit the top of the range: add
    assert next_weight([sets(40, 8, 8, 7)], 6, 8) == 40            # not every set at the top: same
    assert next_weight([sets(40, 8, 8, 8)], 6, 8, step=5) == 45
    assert next_weight([sets(40, 6, 5, 5)], 6, 8) == 40            # one bad session: keep trying
    assert next_weight([sets(40, 5, 5), sets(40, 6, 5)], 6, 8) == 35   # two bad in a row: floor(36/2.5)*2.5
    assert next_weight([sets(40, 5, 5), sets(40, 7, 7)], 6, 8) == 40
