"""All calorie and protein arithmetic is in code. No network, no Ollama, no data files."""
import datetime as dt
import json

import pytest

from trainer.rag import meals
from trainer.rag.index import Doc
from trainer.rag.knowledge import KnowledgeBase
from trainer.store import Store
from trainer.tools import run_tool

TODAY = dt.date(2026, 11, 23)
KAT = {"sex": "female", "age": 30, "height_cm": 165, "weight_kg": 60, "activity": "moderate", "goal": "maintain"}


def food(id_, name, kcal, p, f=1.0, c=1.0, portions=()):
    return Doc(id_, "food", name, name, {"name": name, "source": id_[:4].upper(), "group": "", "estimated": False,
               "portions": list(portions), "per_100g": {"kcal": kcal, "protein_g": p, "fat_g": f, "carbs_g": c}}, None)


@pytest.fixture
def kb():
    return KnowledgeBase([
        food("usda:172187", "Egg, whole, cooked, scrambled", 149, 10.0, 11.0, 1.6),
        food("mext:01174", "角形食パン 焼き", 269, 9.7, 4.5, 50.6),
        food("mext:11288", "むね 皮なし 焼き", 177, 38.8, 3.3, 0.1),
        food("mext:01088", "精白米 うるち米", 156, 2.5, 0.3, 37.1),
        food("mext:06264", "ブロッコリー 花序 ゆで", 30, 3.9, 0.4, 5.2),
        food("mext:10136", "しろさけ 焼き", 160, 29.1, 5.1, 0.1),
        food("mext:06268", "ほうれんそう ゆで", 23, 2.6, 0.5, 4.0),
        food("usda:1", "Cake, chocolate", 389, 3.5), food("usda:2", "Syrup, chocolate", 270, 2.0),
        food("usda:3", "Hummus, plain", 200, 8.0, portions=[{"portion": "2 tablespoons", "grams": 30.0}]),
        food("mext:9", "ラーメンスープ 濃縮", 157, 3.3)], [])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "d.json")
    s.update_profile(**KAT)
    return s


def call(store, kb, tool, **args):
    out, err = run_tool(tool, args, store, TODAY, kb)
    return (out if err else json.loads(out)), err


def item(food, quantity, unit):
    return {"food": food, "quantity": quantity, "unit": unit}


BREAKFAST = [item("scrambled eggs", 2, "piece"), item("toast", 1, "slice")]


def test_meals_logging_calculates_everything(store, kb):
    out, err = call(store, kb, "log_meal", items=BREAKFAST)
    assert not err
    # 2 x 61 g x 149/100 = 181.78;  55 g x 269/100 = 147.95;  sum 329.73
    assert out["logged"]["kcal"] == 330 and out["logged"]["protein_g"] == 17.5      # 12.2 + 5.335
    assert out["logged"]["date"] == "2026-11-23"
    assert out["remaining_today"]["kcal"] == 2046 - 330 and out["remaining_today"]["protein_g"] == 90.5
    assert "330 kcal" in out["say"] and "1 piece of scrambled eggs = 61 g" in out["say"]
    assert len(store.meals_on("2026-11-23")) == 1


def test_meals_model_numbers_and_dates_are_never_used(store, kb):
    out, _ = call(store, kb, "log_meal", items=BREAKFAST, kcal=2250, protein_g=99)
    assert out["logged"]["kcal"] == 330
    out, err = call(store, kb, "log_meal", items=BREAKFAST, date="2023-10-15")
    assert err and "days_ago" in out and len(store.store["meals"]) == 1
    out, _ = call(store, kb, "log_meal", items=BREAKFAST, days_ago=1)
    assert out["logged"]["date"] == "2026-11-22" and out["remaining_today"]["kcal"] == 2046 - 330


@pytest.mark.parametrize("items", [
    None, [], ["2 eggs"], [{}], [{"food": "toast"}], [item("2 toast", 1, "slice")], [item("toast and rice", 1, "g")],
    [item("toast", 0, "g")], [item("toast", -1, "g")], [item("toast", True, "g")], [item("toast", "1", "g")],
    [item("toast", float("nan"), "g")], [item("toast", 9999, "g")], [item("toast", 1, "handful")],
    [item("toast", 1, "bowl")], [item("12345", 1, "g")], [item("unicorn", 1, "g")],
    [item("toast", 1, "slice"), item("unicorn", 1, "g")]])
def test_meals_unclear_food_saves_nothing(store, kb, items):
    out, err = call(store, kb, "log_meal", items=items)
    assert err and out.startswith("Error:") and store.store["meals"] == []


def test_meals_ambiguous_or_partial_names_ask(kb):
    with pytest.raises(ValueError, match="several foods"):
        meals.resolve(kb, "chocolate")                      # a cake and a syrup, far apart
    assert meals.resolve(kb, "Cake, chocolate").id == "usda:1"
    with pytest.raises(ValueError, match="not found"):
        meals.resolve(kb, "ラーメン")                          # not ラーメンスープ


def test_meals_units(kb):
    hummus = kb.food_by_id["usda:3"]
    assert meals.grams_of(hummus, 3, "tbsp")[0] == 45       # the table says 2 tablespoons = 30 g
    assert meals.grams_of(hummus, 80, "g") == (80.0, None)
    with pytest.raises(ValueError, match="ask the user"):
        meals.grams_of(hummus, 1, "bowl")


def test_meals_lookup_saves_nothing(store, kb):
    out, err = call(store, kb, "get_food_nutrition", items=[item("chicken breast", 120, "g")])
    assert not err and out["total"]["protein_g"] == 46.6 and store.store["meals"] == []


def test_meals_suggestion_reaches_the_goal_and_saves_nothing(store, kb):
    out, err = call(store, kb, "suggest_meal", meal="lunch")
    assert not err and out["protein_goal_g"] == 50 and out["kcal_limit"] == 1023     # 108 / 2 -> kept to 50
    assert out["options"] and all(o["total"]["protein_g"] >= 50 and o["total"]["kcal"] <= 1023 for o in out["options"])
    assert store.store["meals"] == []
    out, _ = call(store, kb, "suggest_meal", meal="lunch", protein_g=30, max_kcal=300)
    assert all(o["total"]["protein_g"] >= 30 and o["total"]["kcal"] <= 300 for o in out["options"])


def test_meals_avoid(store, kb):
    out, _ = call(store, kb, "suggest_meal", meal="lunch", avoid=["meat"])
    assert out["options"] and not any("chicken" in o["say"] for o in out["options"])
    out, err = call(store, kb, "suggest_meal", meal="lunch", avoid=["meat", "fish"])
    assert err and "tell the user" in out
    out, err = call(store, kb, "suggest_meal", meal="lunch", avoid=["sesame"])
    assert err and "can't check" in out


def test_meals_own_goal_steps_down_but_the_users_number_does_not(kb):
    left = {"protein_g": 200, "kcal": 800}
    out = meals.suggest_meal(kb, "lunch", left)
    assert out["kcal_limit"] == 400
    tight = meals.suggest_meal(kb, "lunch", left, max_kcal=150)
    assert tight["protein_goal_g"] < 50 and "below_goal" in tight
    with pytest.raises(ValueError, match="lower protein goal"):
        meals.suggest_meal(kb, "lunch", left, protein_g=50, max_kcal=150)


@pytest.mark.parametrize("bad", [{"meal": "brunch"}, {"meal": "lunch", "protein_g": "lots"},
                                 {"meal": "lunch", "protein_g": 500}, {"meal": "lunch", "max_kcal": 5}])
def test_meals_bad_suggestion_input(store, kb, bad):
    assert call(store, kb, "suggest_meal", **bad)[1]


def test_meals_no_profile_still_works(tmp_path, kb):
    out, err = call(Store(tmp_path / "x.json"), kb, "suggest_meal", meal="dinner")
    assert not err and out["protein_goal_g"] == 30 and out["kcal_limit"] is None


def test_meals_day_totals_are_rounded(store, kb):
    for _ in range(3):
        store.log_meal("2026-11-23", "x", 100, 0.1, 0.1, 0.1)
    out, _ = call(store, kb, "get_day_nutrition")
    assert out["totals"]["protein_g"] == 0.3 and out["remaining"]["protein_g"] == 107.7


def test_meals_common_names_are_unique_and_menus_use_known_foods():
    seen = {}
    for s in meals.STAPLES:
        for a in s.aliases:
            assert seen.setdefault(meals._key(a), s.id) == s.id, a
        assert set(s.units) <= set(meals.UNITS)
    for _, main, _, sides in meals._MENUS:
        assert {main, *sides} <= set(meals._BY_ID)
