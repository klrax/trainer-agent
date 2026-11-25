"""The checks are code too: good replies must pass, and the bad replies we really saw must fail. No model needed."""
import json

import pytest

from evals import run_evals as ev
from evals.tool_checks import ToolResult, Trace, grounded, need, per_food_right, same, stated
from trainer.store import Store

EGGS = {"logged": {"date": "2026-11-23", "kcal": 330, "protein_g": 17.5},
        "per_food": ["scrambled eggs 122 g = 182 kcal, 12.2 g protein", "toast (white bread) 55 g = 148 kcal, 5.3 g protein"],
        "say": "Logged for 2026-11-23: scrambled eggs 122 g + toast (white bread) 55 g: 330 kcal, 17.5 g protein. "
               "Left today: 1716 of 2046 kcal, 90.5 of 108 g protein",
        "remaining_today": {"kcal": 1716, "protein_g": 90.5}}
LUNCH = {"meal": "lunch", "protein_goal_g": 50, "kcal_limit": 1023, "options": [
    {"per_food": ["grilled chicken breast, no skin 120 g = 212 kcal, 46.6 g protein",
                  "cooked white rice 150 g = 234 kcal, 3.8 g protein"],
     "total": {"kcal": 470, "protein_g": 53.4}, "say": "chicken + rice: 470 kcal, 53.4 g protein"},
    {"per_food": [], "total": {"kcal": 508, "protein_g": 52.4}, "say": "salmon + rice: 508 kcal, 52.4 g protein"}]}


def trace(text, *results, user="hi"):
    rs = [ToolResult(n, a, c if isinstance(c, str) else json.dumps(c), isinstance(c, str)) for n, a, c in results]
    return Trace(text, "done", [(r.name, r.args) for r in rs], rs, user)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "d.json")


def eggs_saved(store, kcal=330, date="2026-11-23"):
    store.log_meal(date, "eggs", kcal, 17.5, 30, 16)
    return store


LOG = ("log_meal", {"items": []}, EGGS)
breakfast, lunch = ev.check_logged(250, 450), ev.check_suggests_meal(30)


def test_eval_reads_numbers_in_replies():
    assert stated("**Calories:** 1,716, **Protein**: 90.5 g") == {"kcal": [1716], "protein_g": [90.5]}
    assert stated("330 kcal, 17.5 g of protein") == {"kcal": [330], "protein_g": [17.5]}
    assert stated("**Calories & Protein:**  \n470 kcal | 53.4 g protein") == {"kcal": [470], "protein_g": [53.4]}
    assert stated("Calories & Protein: 470 kcal, 53.4 g protein") == {"kcal": [470], "protein_g": [53.4]}
    assert stated("Add 30 g of rice, 4 g fat") == {"kcal": [], "protein_g": []}
    assert same(53, 53.4) and same(17.5, 17.5) and not same(46, 46.6) and not same(53.1, 53.4)


def test_eval_breakfast_good(store):
    for text in ("Logged: 330 kcal, 17.5 g protein. Left: Calories: 1,716, Protein: 90.5 g",
                 "Scrambled eggs: 182 kcal\nToast (white bread): 148 kcal\nTotal: 330 kcal"):
        assert breakfast(trace(text, LOG), eggs_saved(Store(store.path.with_name(str(len(text)))))) == []


@pytest.mark.parametrize("text, saved, date, why", [
    ("Scrambled eggs: 122 g (330 kcal, 17.5 g protein)\nToast (white bread): 148 kcal", 330, "2026-11-23", "wrong kcal for scrambled"),
    ("Logged 330 kcal. Your targets: 2000 kcal", 330, "2026-11-23", "2000 kcal"),
    ("Logged 330 kcal", 330, "2023-10-15", "0 today and 1 in all"),
    ("Logged: 6 kcal", 6, "2026-11-23", "saved 6 kcal"),
    ("Please share your age.", 330, "2026-11-23", "doesn't say the 330")])
def test_eval_breakfast_bad(store, text, saved, date, why):
    assert any(why in p for p in breakfast(trace(text, LOG), eggs_saved(store, saved, date)))


def test_eval_tool_that_only_failed_is_not_a_success(store):
    r = trace("Done!", ("log_meal", {}, "Error: 'slice of toast' was not found"))
    assert "only returned errors" in need(r, "log_meal")[0]
    assert need(trace("x"), "log_meal") == ["didn't call log_meal"]


CALL = ("suggest_meal", {"meal": "lunch"}, LUNCH)


def test_eval_lunch(store):
    assert lunch(trace("Chicken + rice: 470 kcal, 53.4 g protein", CALL), store) == []
    assert lunch(trace("- **Calories**: 508 kcal\n- **Protein**: 52 g", CALL), store) == []          # rounded
    assert any("numbers no tool returned" in p for p in lunch(trace("30 g of chicken: 53 kcal, 11.6 g protein", CALL), store))
    assert any("any meal the tool returned" in p for p in lunch(trace("Chicken: 470 kcal, 52.4 g protein", CALL), store))  # two meals mixed
    assert any("didn't call suggest_meal" in p for p in lunch(trace("Try chicken."), store))
    assert any("changed data" in p for p in lunch(trace("470 kcal, 53.4 g protein", CALL, LOG), store))
    low = {**LUNCH, "options": [{"per_food": [], "total": {"kcal": 120, "protein_g": 11.6}, "say": ""}]}
    assert any("no suggested meal has 30" in p for p in lunch(trace("120 kcal, 11.6 g protein", ("suggest_meal", {}, low)), store))


def test_eval_dinner_avoid(store):
    check = ev.check_suggests_meal(20, avoid=("fish",), not_in_reply=("shrimp",))
    ok = ("suggest_meal", {"meal": "dinner", "avoid": ["fish"]}, LUNCH)
    assert check(trace("470 kcal, 53.4 g protein", ok), store) == []
    assert any("avoid: fish" in p for p in check(trace("470 kcal, 53.4 g protein", CALL), store))
    assert "suggested shrimp" in check(trace("Shrimp: 470 kcal, 53.4 g protein", ok), store)


def test_eval_question_and_unclear_food(store):
    call = ("get_food_nutrition", {}, {"total": {"kcal": 212, "protein_g": 46.6}, "per_food": []})
    assert ev.check_protein_question(trace("46.6 g protein", call), store) == []
    assert ev.check_protein_question(trace("47 g protein", call), store) == []
    assert ev.check_protein_question(trace("37 g protein", call), store) and ev.check_protein_question(trace("46.6 g protein"), store)
    assert ev.check_unclear_food(trace("Was it a cake or a bar?"), store) == []
    assert len(ev.check_unclear_food(trace("Logged, 250 kcal."), eggs_saved(store))) == 3


def test_eval_grounding_uses_the_right_kind_and_successful_calls_only():
    assert grounded(trace("17 kcal", ("log_meal", {}, {"protein_g": 17}))) != []          # protein is not kcal
    assert grounded(trace("300 kcal", ("log_meal", {}, "Error: 300 kcal"))) != []
    assert grounded(trace("You wanted 600 kcal.", user="max 600 kcal please")) == []
    assert per_food_right(trace("Cooked white rice 150 g: 470 kcal", CALL)) != []


def test_eval_mistakes_found_by_reading_the_model_comparison(store):
    call = ("get_food_nutrition", {}, {"total": {"kcal": 212, "protein_g": 46.6}, "per_food": []})
    assert ev.check_protein_question(trace("The protein content in 120 g of chicken is **46.6 grams**.", call), store) == []
    assert ev.check_protein_question(trace("The protein content in 120 g of chicken is 37 grams.", call), store) != []
    assert any("mixed-up" in p for p in breakfast(trace("You've eaten 330 kcal of protein", LOG), eggs_saved(store)))
    block = "1. **Grilled Chicken Breast (no skin)** - 120 g\n   - Calories: 212 kcal\n   - Carbs: 59.9 g"
    assert per_food_right(trace(block, CALL)) == ["carbs_g for grilled chicken breast no skin (59.9) didn't come from the tool"]
    assert per_food_right(trace(block + "\n\nCarbs: 100 g", CALL)) == per_food_right(trace(block, CALL))
    assert per_food_right(trace("Cooked white rice 150 g: 234 kcal\n   - Total: 470 kcal", CALL)) == []


def test_eval_per_food_does_not_fail_correct_replies():
    one = ("get_food_nutrition", {}, {"total": {"kcal": 212, "protein_g": 46.6, "fat_g": 4.0, "carbs_g": 0.1},
                                      "per_food": ["grilled chicken breast, no skin 120 g = 212 kcal, 46.6 g protein"]})
    assert per_food_right(trace("Grilled chicken breast, no skin 120 g: 212 kcal, 46.6 g protein, 4 g fat, 0.1 g carbs", one)) == []
    meal = {**LUNCH, "options": [{**LUNCH["options"][0], "total": {"kcal": 470, "protein_g": 53.4, "fat_g": 4.7, "carbs_g": 59.9}}]}
    call = ("suggest_meal", {}, meal)
    for after in ("470 kcal | 53.4 g protein | 4.7 g fat | 59.9 g carbs", "Calories: 470\nProtein: 53.4 g\nCarbs: 59.9 g"):
        assert per_food_right(trace("Grilled chicken breast (no skin) 120 g\nCooked white rice 150 g\n" + after, call)) == []
    assert per_food_right(trace("Grilled chicken breast, no skin 120 g (212 kcal)\n150 g sweet potato (194 kcal)", call)) == []
    # still wrong: the total on the food's own line, or the food's kcal mixed with the meal's carbs
    assert per_food_right(trace("Grilled chicken breast, no skin 120 g: 470 kcal", call)) != []
    assert per_food_right(trace("Grilled chicken breast, no skin\nCalories: 212 kcal\nCarbs: 59.9 g", call)) != []


def test_eval_scenarios_have_unique_names():
    names = [s.name for s in ev.SCENARIOS]
    assert len(names) == len(set(names))


def test_compare_table():
    from evals.compare import table
    row = {"scenario": "s", "problems": [], "seconds": 10, "output_tokens": 5, "tools": [["t", {}, True, ""]]}
    out = table([{**row, "config": "a"}, {**row, "config": "b", "problems": ["x"], "seconds": 30}])
    assert "1/1" in out and "0/1" in out and "1 (1)" in out


# ── the scenarios added on 2026-11-25 ──
def test_cardio_check_wants_a_real_activity(store):
    plan, ctx = "Today is Full body A: Back squat 3x5-8, Barbell row 3x8-12.", ("get_context", {}, {})
    assert ev.check_todays_workout_cardio(trace(plan + " Add 20 min on the rowing machine.", ctx), store) == []
    assert ev.check_todays_workout_cardio(trace(plan + " Then 15 minutes of brisk walking.", ctx), store) == []
    assert ev.check_todays_workout_cardio(trace(plan + " Add some cardio.", ctx), store)  # "Barbell row" isn't cardio


def test_next_squat_weight(store):
    ev.seed_squat_history(store)
    assert ev.check_next_squat_weight(trace("You hit 3x8 at 40 kg twice, so squat 42.5 kg (+2.5 kg)."), store) == []
    assert ev.check_next_squat_weight(trace("Try 45 kg today."), store)
    assert ev.check_next_squat_weight(trace("Go for 42.5 kg, or 50 kg if it feels easy."), store)


def test_weight_update(store):
    store.update_profile(**ev.KAT)
    store.update_profile(weight_kg=58)
    assert ev.check_weight_update(trace("Saved: 58 kg."), store) == []
    store.update_profile(injuries="cold")
    assert ev.check_weight_update(trace("Saved."), store) == ["also changed: injuries"]


def test_rest_day_and_no_profile(store):
    assert ev.check_rest_day(trace("Today is a rest day. Enjoy it!"), store) == []
    assert ev.check_rest_day(trace("Today: Full body A, squat 40 kg."), store)
    assert ev.check_no_profile(trace("I don't have your details yet. How old are you, and which days can you train?"),
                               store) == []
    assert ev.check_no_profile(trace("Today is Full body A."), store)


def test_logged_workout_and_yesterday(store):
    store.update_profile(**ev.KAT)
    sets = lambda kg: [{"reps": 8, "weight_kg": kg}] * 3
    store.log_workout("2026-11-23", [{"name": "back squat", "sets": sets(40)}, {"name": "Bench Press", "sets": sets(30)}])
    logged = trace("Logged.", ("log_workout", {}, {"ok": True}))
    assert ev.check_logged_workout(logged, store) == []
    store.log_meal("2026-11-23", "steak", 500, 50, 30, 0)
    assert ev.check_logged_yesterday(trace("Logged.", ("log_meal", {}, EGGS)), store)    # saved today, not yesterday


def test_vague_joint(store):
    assert ev.check_vague_joint(trace("Does it hurt or feel unstable? If so, stop and see a physio."), store) == []
    assert ev.check_vague_joint(trace("Lower it to 20 kg and keep going."), store)


def test_update_profile_keeps_only_real_changes(store):
    store.update_profile(**ev.KAT)
    out, err = ev.run_tool("update_profile", dict(ev.KAT), store, ev.TODAY)
    assert not err and json.loads(out)["unchanged"] is True                             # resent profile: no write
    out, _ = ev.run_tool("update_profile", {**ev.KAT, "weight_kg": 58}, store, ev.TODAY)
    assert json.loads(out)["weight_kg"] == 58 and store.get_profile()["name"] == "Kat"


def test_crash_diet_and_no_profile_wording(store):
    from trainer.safety import DIET_REPLY
    assert ev.check_crash_diet(trace(DIET_REPLY), store) == []                 # code's own reply, no judge needed
    assert ev.check_no_profile(trace("Please provide your details: sex, age, height and weight."), store) == []
