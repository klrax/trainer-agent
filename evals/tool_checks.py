"""Checks on what the tools were asked, what they returned, and whether the reply repeats it correctly.

No LLM judge here: the judge once passed a "30 g protein" lunch whose own tool result said 11.6 g. The tools
calculate every number, so plain code can compare the reply with them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from trainer.agent import AgentResult


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    content: str
    is_error: bool

    @property
    def data(self) -> Any:
        return json.loads(self.content)


@dataclass
class Trace(AgentResult):
    """An AgentResult plus every tool's result and the user's message."""
    results: list[ToolResult] = field(default_factory=list)
    user: str = ""
    seconds: float = 0.0


def succeeded(r: Trace, *tools: str) -> list[ToolResult]:
    """Calls to these tools that worked (no names = all tools). A call that returned an error doesn't count."""
    return [t for t in r.results if not t.is_error and (not tools or t.name in tools)]


def need(r: Trace, tool: str) -> list[str]:
    """[] if the tool worked at least once, otherwise what went wrong (not called / only errors)."""
    if succeeded(r, tool):
        return []
    errors = [t.content for t in r.results if t.name == tool]
    return [f"{tool} only returned errors: {errors[-1][:120]}" if errors else f"didn't call {tool}"]


# ── numbers in the reply ───────────────────────────────────────────────────────────────
_N = r"(\d[\d,]*(?:\.\d+)?)"
# "330 kcal", "17.5 g protein", "17.5 g of protein", "**Calories:** 1,716", "Protein: 90.5 g"
_KCAL = re.compile(rf"{_N}\s*(?:kcal|calories)\b|\b(?:kcal|calories|energy)[ \t]*[:=][ \t]*{_N}", re.IGNORECASE)
_PROTEIN = re.compile(rf"{_N}\s*g(?:rams)?\s+(?:of\s+)?protein\b|\bprotein[ \t]*[:=][ \t]*{_N}(?![\d.,]*\s*k?cal)", re.IGNORECASE)


_MACRO = {k: re.compile(rf"{_N}\s*g(?:rams)?\s+(?:of\s+)?{k}\b|\b{k}(?:ohydrates)?[ \t]*[:=][ \t]*{_N}", re.IGNORECASE)
          for k in ("fat", "carbs?")}


def _plain(text: str) -> str:
    return re.sub(r"[*_`~≈]", "", text)


def _found(pattern: re.Pattern, text: str) -> list[float]:
    return [float(next(g for g in m.groups() if g).replace(",", "")) for m in pattern.finditer(_plain(text))]


def stated(text: str) -> dict[str, list[float]]:
    """The calorie and protein numbers a text states: {"kcal": [...], "protein_g": [...]}"""
    return {"kcal": _found(_KCAL, text), "protein_g": _found(_PROTEIN, text)}


def stated_all(text: str) -> dict[str, list[float]]:
    """Also fat and carbs."""
    return {**stated(text), "fat_g": _found(_MACRO["fat"], text), "carbs_g": _found(_MACRO["carbs?"], text)}


def same(a: float, b: float) -> bool:
    """Equal, or the reply rounded to a whole number (53.4 -> 53). Nothing looser: 46 for 46.6 is wrong."""
    return abs(a - b) < 0.051 or (a == int(a) and abs(a - b) <= 0.5)


def _numbers(value: Any, key: str = "") -> list[tuple[str, float]]:
    """Every number in a tool result with its key; ready-made sentences ("say", "per_food") are read too."""
    if isinstance(value, dict):
        return [x for k, v in value.items() for x in _numbers(v, k)]
    if isinstance(value, list):
        return [x for v in value for x in _numbers(v, key)]
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [("kcal" if "kcal" in key else "protein_g" if "protein" in key else key, float(value))]
    return [(k, n) for k, ns in stated(value).items() for n in ns]


def grounded(r: Trace) -> list[str]:
    """Every calorie and protein number in the reply must come from a tool result or the user's own message.
    Calories are only compared with calories, protein with protein."""
    known = [x for t in succeeded(r) for x in _numbers(t.data)] + _numbers(r.user)
    problems = []
    for kind, label in (("kcal", "kcal"), ("protein_g", "g protein")):
        bad = [n for n in stated(r.text)[kind] if not any(k == kind and same(n, v) for k, v in known)]
        if bad:
            problems.append(f"numbers no tool returned: {', '.join(f'{n:g} {label}' for n in dict.fromkeys(bad))}")
    return problems


def _name(text: str) -> str:
    """"Grilled Chicken Breast (no skin)" and "grilled chicken breast, no skin" are the same food."""
    return " ".join(re.sub(r"[^a-z0-9. ]", " ", text.lower()).split())


def per_food_right(r: Trace) -> list[str]:
    """Numbers listed under one food must be THAT food's numbers. (Seen: the eggs given the whole meal's kcal, the
    chicken given the whole meal's carbs.)

    A food is named on a line, and the lines after it belong to it until another food, a total or a blank line.
    Those later lines may instead give the whole meal's total, if their kcal is that total. The naming line may not.
    """
    foods: dict[str, list[dict[str, list[float]]]] = {}
    totals: list[dict[str, float]] = []
    for t in succeeded(r, "log_meal", "get_food_nutrition", "suggest_meal"):
        for block in [t.data, *t.data.get("options", [])]:
            lines = block.get("per_food", [])
            total = block.get("total") or block.get("logged")
            if lines and total:
                totals.append(total)
            for line in lines:
                label, _, numbers = line.partition(" = ")
                values = stated_all(numbers)
                if len(lines) == 1 and total:              # one food: its numbers and the total are the same thing
                    values = {k: v + [total[k]] if k in total else v for k, v in values.items()}
                foods.setdefault(_name(re.sub(r"\s[\d.]+ g$", "", label)), []).append(values)
    problems, current, after = [], None, {}

    def check(food: str, values: dict[str, list[float]], may_be_total: bool) -> None:
        if may_be_total and any(values["kcal"] and all(same(v, total.get(k, -1)) for k, vs in values.items() for v in vs)
                                for total in totals):
            return
        for kind, given in values.items():
            right = [v for s in foods[food] for v in s[kind]]
            wrong = [v for v in given if not any(same(v, x) for x in right)]
            if wrong:
                problems.append(f"wrong {kind} for {food}: {wrong[0]:g}, the tool said {right[0]:g}" if right
                                else f"{kind} for {food} ({wrong[0]:g}) didn't come from the tool")

    def close() -> None:
        if current and any(after.values()):
            check(current, after, may_be_total=True)

    for line in _plain(r.text).splitlines():
        named = [f for f in foods if f in _name(line)]
        another = not named and re.search(r"\d\s*g\b(?!\w)(?!\s*(?:of\s+)?(?:protein|fat|carb))", line) and not any(stated_all(line).values()) \
            or not named and re.search(r"\d+\s*g\s+(?!of protein|of fat|of carb|protein|fat|carb)(?:of\s+)?[a-z]{3,}", line, re.IGNORECASE)
        if named or another or not line.strip() or re.search(r"total|left|remaining|target|logged", line, re.IGNORECASE):
            close()
            current, after = None, {}
            if len(named) == 1 and not re.search(r"total|left|remaining|target|logged", line, re.IGNORECASE):
                current = named[0]
                check(current, stated_all(line), may_be_total=False)
            continue
        if current:
            for kind, values in stated_all(line).items():
                after.setdefault(kind, []).extend(values)
            after.setdefault("kcal", [])
    close()
    return list(dict.fromkeys(problems))


def units_right(r: Trace) -> list[str]:
    """"474 kcal of protein" is nonsense even though 474 is a real number."""
    bad = re.search(r"[\d.,]+\s*(?:kcal|calories)\s+of\s+(?:protein|fat|carbs?)\b", _plain(r.text), re.IGNORECASE)
    return [f"mixed-up units: {bad.group(0)!r}"] if bad else []


def states(r: Trace, kind: str, value: float) -> bool:
    return any(same(n, value) for n in stated(r.text)[kind])


def numbers_right(r: Trace) -> list[str]:
    return grounded(r) + per_food_right(r) + units_right(r)
