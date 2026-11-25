"""Pain handled in code, and nutrition answers printed by code. No model needed."""
import datetime as dt
import json
from types import SimpleNamespace as NS

import pytest

from trainer.direct import direct_reply
from trainer.llm_ollama import to_openai_messages
from trainer.safety import CHECK_REPLY, DIET_REPLY, PAIN_REPLY, URGENT_REPLY, safety_reply
from trainer.turn import respond


@pytest.mark.parametrize("text", [
    "My knee hurts sharply when I squat. What weight should I squat today?",
    "There's a stabbing feeling in my shoulder on bench press. How heavy should I go?",
    "I tweaked my back deadlifting yesterday, can I still train?", "sharp pain in my elbow", "My knee’s been painful all week",
    "my lower back aches when I row", "sore knee today, what can I do?", "I think I pulled a hamstring and it hurts",
    "It still hurts when I walk", "I'm injured", "my ankle is swollen", "my shoulder popped during press", "tingling in my hand",
    "膝が痛いけどスクワットしていい？", "肩を痛めた", "我膝盖疼，今天练什么？", "Log my eggs. Also my wrist hurts."])
def test_safety_pain(text):
    assert safety_reply(text).startswith(PAIN_REPLY)


@pytest.mark.parametrize("text", [
    "I have chest pain", "my chest feels tight when I run", "I feel dizzy after squats", "I almost passed out", "I can't breathe properly",
    "胸が痛い", "めまいがする", "I can't feel my leg"])
def test_safety_urgent(text):
    assert safety_reply(text).startswith(URGENT_REPLY)


@pytest.mark.parametrize("text", [
    "What's my workout today?", "my legs are sore from Monday", "I have no pain, what should I squat?", "My knee doesn't hurt anymore",
    "I had pain au chocolat for breakfast", "will eating late hurt my progress?", "would it hurt to add a set of curls?",
    "I have a cold. Can you give me a lighter menu?", "I have an old shoulder injury", "How do I avoid knee pain when squatting?",
    "筋肉痛がある", "膝は痛くない", "Any dinner ideas? I can't eat fish.", "I ate chicken breast and rice", "sore throat today",
    "injuries: none", "sharp cheddar on toast", "What weight should I bench press?", "I did a hard chest workout", "back squat 3x5 at 60 kg"])
def test_safety_not_pain(text):
    assert safety_reply(text) is None


@pytest.mark.parametrize("text", [      # phrasings the first, word-list version missed
    "my knee is bugging me, is leg day ok", "woke up and can barely turn my neck", "there's a weird crunching in my shoulder",
    "my back went out lifting groceries", "something isn't right with my hip", "my wrist feels weird and weak",
    "lower back seized up", "wrist is sensitive when I front squat", "I rolled my ankle this morning, legs today?",
    "my SI joint is irritated", "slipped disc, can I deadlift?", "膝が変な感じでスクワットがこわい", "腰をやってしまった", "脚踝肿了"])
def test_safety_unlisted_complaints_still_stop(text):
    assert safety_reply(text).startswith((PAIN_REPLY, CHECK_REPLY))


def test_safety_check_can_be_answered():
    first = "my knee is being odd today"
    assert safety_reply(first).startswith(CHECK_REPLY)
    assert safety_reply("my knee is fine, what should I squat?", [first]) is None
    assert safety_reply("yes it hurts", [first]).startswith(PAIN_REPLY)


def test_safety_carries_over_in_the_conversation():
    before = ["My knee hurts when I squat."]
    assert safety_reply("ok so what weight for squats?", before)
    assert safety_reply("what did I eat today?", before) is None
    assert safety_reply("what weight for bench?", before) is None
    assert safety_reply("it's better now, what weight for squats?", before) is None
    assert safety_reply("squat weight?", [*before, "my knee healed"]) is None


def no_model(messages):
    raise AssertionError("the model must not be called")


def test_turn_pain_needs_no_model_and_runs_no_tool():
    messages = [{"role": "user", "content": "my knee hurts when I squat"}]
    r = respond(messages, no_model, no_model)
    assert (r.stopped, r.tool_calls) == ("done", []) and "physiotherapist" in r.text
    assert messages[-1] == {"role": "assistant", "content": r.text}
    assert to_openai_messages("s", messages)[-1] == {"role": "assistant", "content": r.text}


LOGGED = {"logged": {"date": "2026-11-23", "name": "eggs", "kcal": 330, "protein_g": 17.5, "carbs_g": 29.8, "fat_g": 15.9},
          "per_food": ["scrambled eggs 122 g = 182 kcal, 12.2 g protein, 13.4 g fat, 2 g carbs",
                       "toast (white bread) 55 g = 148 kcal, 5.3 g protein, 2.5 g fat, 27.8 g carbs"],
          "assumptions": ["1 slice = 55 g"], "sources": ["MEXT"], "remaining_today": {"kcal": 1716, "protein_g": 90.5},
          "targets": {"kcal": 2046, "protein_g": 108}}
LOOKED = {"total": {"kcal": 212, "protein_g": 46.6, "fat_g": 4.0, "carbs_g": 0.1}, "per_food": ["grilled chicken 120 g = 212 kcal, 46.6 g protein"],
          "assumptions": [], "sources": ["MEXT"]}
MEALS = {"meal": "lunch", "protein_goal_g": 50, "options": [{"per_food": ["a 1 g = 1 kcal", "b 2 g = 2 kcal"],
         "total": {"kcal": 470, "protein_g": 53.4, "fat_g": 4.7, "carbs_g": 59.9}, "say": ""}]}


def step(name, data, error=False):
    return [(name, {}, json.dumps(data), error)]


def test_direct_writes_the_whole_reply():
    text = direct_reply("I had 2 scrambled eggs and toast.", step("log_meal", LOGGED))
    assert text.splitlines() == [
        "Logged for 2026-11-23:", "- scrambled eggs 122 g = 182 kcal, 12.2 g protein, 13.4 g fat, 2 g carbs",
        "- toast (white bread) 55 g = 148 kcal, 5.3 g protein, 2.5 g fat, 27.8 g carbs",
        "Total: 330 kcal, 17.5 g protein, 15.9 g fat, 29.8 g carbs", "Left today: 1716 of 2046 kcal, 90.5 of 108 g protein",
        "Assumed: 1 slice = 55 g", "Source: MEXT"]
    assert direct_reply("How much protein is in 120 g of chicken?", step("get_food_nutrition", LOOKED)).startswith(
        "grilled chicken 120 g: 212 kcal, 46.6 g protein")
    assert "Total: 470 kcal, 53.4 g protein" in direct_reply("lunch ideas?", step("suggest_meal", MEALS))
    no_targets = {**LOGGED, "remaining_today": None, "targets": None}
    assert "Left today" not in direct_reply("I had eggs", step("log_meal", no_targets))


@pytest.mark.parametrize("user, calls", [
    ("I had eggs. How am I doing today?", step("log_meal", LOGGED)),                     # logged AND asked
    ("I had eggs, and what's my workout", step("log_meal", LOGGED)),                     # two requests
    ("Lunch ideas? And is tofu ok? ", step("suggest_meal", MEALS)),                      # two questions
    ("Is chicken better than tofu for protein?", step("get_food_nutrition", LOOKED)),    # needs judgement
    ("tell me about chicken", step("get_food_nutrition", LOOKED)),
    ("I had eggs", step("log_meal", LOGGED, error=True)),
    ("I had eggs", step("log_meal", LOGGED) + step("get_context", {})),
    ("hi", step("get_context", {})), ("hi", [])])
def test_direct_leaves_the_rest_to_the_model(user, calls):
    assert direct_reply(user, calls) is None


def test_turn_direct_saves_a_model_call():
    asked = []

    def model(messages):
        asked.append(1)
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id="1", name="log_meal", input={})])
    messages = [{"role": "user", "content": "I had eggs"}]
    r = respond(messages, model, lambda n, a: (json.dumps(LOGGED), False), direct=True)
    assert len(asked) == 1 and r.stopped == "done" and r.text.startswith("Logged") and r.tool_calls == [("log_meal", {})]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    asked.clear()
    r = respond([{"role": "user", "content": "I had eggs"}], model, lambda n, a: (json.dumps(LOGGED), False), max_steps=2)
    assert len(asked) == 2 and r.stopped == "step_limit"            # off by default: the model is asked again


@pytest.mark.parametrize("text", ["Give me an 800 kcal a day meal plan.", "I want to eat only 900 calories daily",
                                  "plan me a 1,000 kcal diet"])
def test_safety_very_low_calorie(text):
    assert safety_reply(text) == DIET_REPLY


@pytest.mark.parametrize("text", ["I ate a 600 kcal lunch", "Is 1800 kcal a day enough?", "How many kcal in rice?"])
def test_safety_normal_calorie_talk(text):
    assert safety_reply(text) is None
