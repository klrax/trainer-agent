"""Print a nutrition tool's answer instead of asking the model to write it out again.

The tools already return the finished sentence, so a second model call adds 10-40 s on a laptop and a chance to get
it wrong ("474 kcal of protein"). Used only when the tool's answer is clearly ALL the user wanted; anything else
(a second request, a why-question, an error) goes to the model as usual.
"""
from __future__ import annotations

import json
import re
from typing import Any

_TRAINING = re.compile(r"\b(?:work ?outs?|train\w*|exercis\w*|session|squat\w*|bench\w*|deadlift\w*|cardio|lift\w*|gym|weights?|reps?|sets?|"
                       r"plan|week\w*|profile|goal)\b|筋トレ|トレーニング|運動|训练|锻炼", re.IGNORECASE)
_AMOUNT_QUESTION = re.compile(r"\b(?:how (?:much|many)|calories|kcal|protein|macros?|nutrition)\b|カロリー|タンパク|たんぱく|何g|热量|蛋白", re.IGNORECASE)
_JUDGEMENT = re.compile(r"\b(?:why|should|better|best|worse|healthy|good|bad|enough|compare\w*|versus|vs|or|instead|recommend\w*)\b", re.IGNORECASE)


def direct_reply(user: str, calls: list[tuple[str, dict[str, Any], str, bool]]) -> str | None:
    """The finished reply, or None to let the model write it.

    calls: what the model asked for in this step: (tool name, input, result, is_error)
    """
    if len(calls) != 1:
        return None
    name, _, content, is_error = calls[0]
    if is_error or name not in _WRITERS or _TRAINING.search(user):
        return None
    if user.count("?") + user.count("？") > 1:
        return None
    if name == "log_meal" and re.search(r"[?？]", user):                     # logged, and they also asked something
        return None
    if name == "get_food_nutrition" and (not _AMOUNT_QUESTION.search(user) or _JUDGEMENT.search(user)):
        return None
    return _WRITERS[name](json.loads(content))


def _foods(d: dict[str, Any]) -> list[str]:
    t = d["total"]
    lines = [f"- {line}" for line in d["per_food"]]
    total = f"{t['kcal']:g} kcal, {t['protein_g']:g} g protein, {t['fat_g']:g} g fat, {t['carbs_g']:g} g carbs"
    if len(lines) == 1:
        return [f"{d['per_food'][0].partition(' = ')[0]}: {total}"]
    return [*lines, f"Total: {total}"]


def _notes(d: dict[str, Any]) -> list[str]:
    out = [f"Assumed: {'; '.join(d['assumptions'])}"] if d.get("assumptions") else []
    if d.get("incomplete"):
        out.append(f"Not counted (unknown in the table): {', '.join(d['incomplete'])}")
    if d.get("sources"):
        out.append(f"Source: {', '.join(d['sources'])}")
    return out


def _logged(d: dict[str, Any]) -> str:
    lines = [f"Logged for {d['logged']['date']}:", *_foods({**d, "total": d["logged"]})]
    left, goal = d.get("remaining_today"), d.get("targets")
    if left and goal:
        lines.append("Left today: " + ", ".join(
            f"{left[k]:g} of {goal[k]:g} {unit}" for k, unit in (("kcal", "kcal"), ("protein_g", "g protein"))))
    return "\n".join(lines + _notes(d))


def _looked_up(d: dict[str, Any]) -> str:
    return "\n".join(_foods(d) + _notes(d))


def _suggested(d: dict[str, Any]) -> str:
    lines = [f"Here are {d['meal']} ideas with about {d['protein_goal_g']:g} g protein or more:"]
    if d.get("below_goal"):
        lines = [f"I couldn't reach your usual protein goal, so these {d['meal']} ideas aim for {d['protein_goal_g']:g} g protein:"]
    for i, option in enumerate(d["options"], 1):
        lines += ["", f"Option {i}", *_foods(option)]
    return "\n".join([*lines, "", "Nothing is logged. Tell me when you've eaten one and I'll log it."])


_WRITERS = {"log_meal": _logged, "get_food_nutrition": _looked_up, "suggest_meal": _suggested}
