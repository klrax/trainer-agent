"""Every calorie and protein calculation, in code.

The model names foods and amounts in plain words; this file finds the table entry, turns "2 pieces" into
grams, scales, adds up, and builds meals that reach a protein target. The model never sends a calorie
number and never needs a food_id (small models invent them: "12345").

    calculate(kb, [{"food": "scrambled eggs", "quantity": 2, "unit": "piece"}])
    -> {"items": [{"name", "grams", "kcal", ...}], "total": {...}, "assumptions": [...], "say": "..."}
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge import WEAK_COVERAGE, KnowledgeBase
from .text import is_japanese, normalize, tokenize

UNITS = ["g", "ml", "piece", "slice", "bowl", "cup", "tbsp", "tsp", "pack", "serving"]
MAX_QUANTITY = 3000
MIN_GRAMS = 10
MAX_SPREAD = 1.3        # the top matches may differ this much in kcal per 100 g before we ask which one
UNIT_WORDS = {"tbsp": "tbsp|tablespoon", "tsp": "tsp|teaspoon", "ml": "ml|milliliter", "pack": "pack|package|container",
              "piece": "piece|item|each|fruit|sandwich|bar"}


@dataclass(frozen=True)
class Staple:
    """A common food tied to one checked table entry. The numbers still come from the table.
    units: grams in one unit. These are ordinary sizes, so using one is always reported as an assumption."""
    id: str
    label: str
    aliases: tuple[str, ...]
    units: dict[str, float] = field(default_factory=dict)
    tags: frozenset[str] = frozenset()      # for "avoid"


def _s(id_: str, label: str, aliases: str, tags: str = "", **units: float) -> Staple:
    return Staple(id_, label, tuple(a.strip() for a in aliases.split(",")), units, frozenset(tags.split()))


STAPLES = [
    _s("mext:11288", "grilled chicken breast, no skin", "chicken breast, grilled chicken breast, chicken, grilled chicken, "
       "skinless chicken breast, 鶏むね肉, 鶏胸肉, 鶏むね肉 皮なし", "chicken", piece=150),
    _s("mext:11220", "raw chicken breast, no skin", "raw chicken breast, 鶏むね肉 皮なし 生", "chicken", piece=200),
    _s("mext:11225", "grilled chicken thigh, no skin", "chicken thigh, grilled chicken thigh, 鶏もも肉", "chicken", piece=150),
    _s("mext:11228", "grilled chicken tenderloin", "chicken tenderloin, sasami, ささみ", "chicken", piece=40),
    _s("usda:172187", "scrambled eggs", "scrambled egg, scrambled eggs, スクランブルエッグ", "egg", piece=61),
    _s("usda:173423", "fried egg", "fried egg, 目玉焼き", "egg", piece=46),
    _s("mext:12005", "boiled egg", "egg, boiled egg, hard boiled egg, ゆで卵, ゆでたまご, 卵", "egg", piece=50),
    _s("mext:12004", "raw egg", "raw egg, 生卵", "egg", piece=50),
    _s("mext:01174", "toast (white bread)", "toast, white toast, toasted bread, トースト",
       "wheat dairy", slice=55, piece=55),
    _s("mext:01026", "white bread", "bread, white bread, 食パン", "wheat dairy", slice=60, piece=60),
    _s("usda:172689", "whole-wheat toast", "whole wheat toast, whole grain toast, wholemeal toast", "wheat dairy", slice=25, piece=25),
    _s("mext:01088", "cooked white rice", "rice, white rice, cooked rice, cooked white rice, steamed rice, ご飯, ごはん, 白米, 白ご飯",
       "", bowl=150, cup=150, serving=150),
    _s("mext:01085", "cooked brown rice", "brown rice, cooked brown rice, 玄米, 玄米ご飯", "", bowl=150, cup=150, serving=150),
    _s("mext:01004", "dry oatmeal", "oats, oatmeal, dry oats, オートミール", "oats", bowl=40, cup=80, tbsp=6, serving=40),
    _s("mext:01039", "boiled udon", "udon, うどん", "wheat", bowl=220, pack=220, serving=220),
    _s("mext:01128", "boiled soba", "soba, そば, 蕎麦", "wheat buckwheat", bowl=200, pack=200, serving=200),
    _s("mext:01064", "boiled pasta", "pasta, spaghetti, boiled spaghetti, パスタ, スパゲッティ", "wheat", bowl=220, serving=220),
    _s("mext:02046", "steamed sweet potato", "sweet potato, steamed sweet potato, さつまいも", "", piece=200),
    _s("mext:04032", "firm tofu", "tofu, firm tofu, 豆腐, 木綿豆腐", "soy", pack=300, piece=300, serving=150),
    _s("mext:04033", "silken tofu", "silken tofu, soft tofu, 絹ごし豆腐", "soy", pack=300, piece=300, serving=150),
    _s("mext:04046", "natto", "natto, 納豆", "soy", pack=45, serving=45),
    _s("mext:06016", "boiled edamame", "edamame, 枝豆", "soy", bowl=80, cup=80, serving=80),
    _s("mext:10136", "grilled salmon", "salmon, grilled salmon, cooked salmon, 鮭, 焼き鮭", "fish", piece=80, serving=80),
    _s("mext:10156", "grilled mackerel", "mackerel, grilled mackerel, 鯖, 焼き鯖", "fish", piece=80, serving=80),
    _s("mext:10260", "canned tuna in water", "tuna, canned tuna, tuna can, ツナ缶, ツナ", "fish", pack=70, serving=70),
    _s("usda:175180", "boiled shrimp", "shrimp, cooked shrimp, prawn, えび", "shellfish", serving=85),
    _s("mext:11278", "grilled pork fillet", "pork fillet, pork tenderloin, 豚ヒレ肉", "pork", serving=100),
    _s("usda:171794", "lean ground beef, cooked", "ground beef, lean ground beef, beef mince", "beef", serving=85),
    _s("usda:170894", "plain nonfat Greek yogurt", "greek yogurt, plain greek yogurt, ギリシャヨーグルト", "dairy",
       pack=170, cup=170, bowl=170, serving=170),
    _s("mext:13025", "plain yogurt", "yogurt, plain yogurt, ヨーグルト", "dairy", cup=200, bowl=100, serving=100),
    _s("mext:14001", "olive oil", "olive oil, オリーブオイル", "", tbsp=12, tsp=4),
    _s("mext:14017", "salted butter", "butter, salted butter, バター", "dairy", tbsp=12, tsp=4),
    _s("mext:07006", "avocado", "avocado, アボカド", "", piece=140),
    _s("usda:170567", "almonds", "almond, almonds, アーモンド", "nuts", piece=1.2, serving=28),
    _s("mext:13040", "processed cheese", "cheese, processed cheese, sliced cheese, チーズ", "dairy", slice=18, piece=18),
    _s("mext:07042", "orange juice", "orange juice, オレンジジュース", "", cup=200, ml=1.0),
    _s("mext:16045", "black coffee", "coffee, black coffee, コーヒー, ブラックコーヒー", "", cup=200, ml=1.0),
    _s("mext:16006", "beer", "beer, ビール", "", cup=350, ml=1.0, serving=350),
    _s("usda:172470", "peanut butter", "peanut butter, ピーナッツバター", "peanut", tbsp=16),
    _s("mext:13003", "milk", "milk, whole milk, 牛乳", "dairy", cup=206, ml=1.03),
    _s("mext:13033", "cottage cheese", "cottage cheese, カッテージチーズ", "dairy", serving=100),
    _s("mext:07107", "banana", "banana, バナナ", "", piece=100),
    _s("mext:07148", "apple", "apple, りんご", "", piece=200),
    _s("mext:06264", "boiled broccoli", "broccoli, boiled broccoli, steamed broccoli, ブロッコリー", "", cup=80, serving=80),
    _s("mext:06268", "boiled spinach", "spinach, boiled spinach, ほうれん草", "", serving=80, bowl=80),
    _s("mext:06061", "raw cabbage", "cabbage, raw cabbage, キャベツ", "", serving=80, cup=80),
    _s("mext:06183", "cherry tomatoes", "cherry tomato, cherry tomatoes, ミニトマト", "", piece=15, serving=90),
]
_BY_ID = {s.id: s for s in STAPLES}


def _key(text: str) -> str:
    """Word order and plurals don't matter: "eggs scrambled" == "scrambled egg". Japanese: exact, no spaces."""
    if is_japanese(text):
        return "".join(normalize(text).split())
    return " ".join(sorted(tokenize(text)))


_ALIAS = {_key(a): s for s in STAPLES for a in s.aliases}


# ── finding the food ───────────────────────────────────────────────────────────────────
def resolve(kb: KnowledgeBase, food: Any):
    """The table entry for one food in plain words, or ValueError telling the model what to do.

    Order: a common food -> an entry's exact name -> search, but only if most of the words match.
    A doubtful match is refused, never used: a wrong food gives a confident wrong number. So is an ambiguous one:
    "chocolate" matches a cake, a bar and a drink, and their calories are far apart.
    """
    if not isinstance(food, str) or not food.strip():
        raise ValueError("each item needs 'food': one food in plain words, e.g. 'scrambled eggs'")
    food = re.sub(rf"^\s*(?:an?\s+|one\s+)?(?:{'|'.join(UNITS[2:])})s?\s+of\s+", "", food, flags=re.IGNORECASE)    # "slice of toast"
    staple = _ALIAS.get(_key(food))
    if staple and staple.id in kb.food_by_id and not re.match(r"\s*\d", food):
        return kb.food_by_id[staple.id]
    exact = normalize(food).strip()
    for doc in kb.food_by_id.values():
        if normalize(doc.data["name"]).strip() == exact:
            return doc
    if re.match(r"\s*\d", food):
        raise ValueError(f"{food!r}: put the number in 'quantity' and only the food in 'food'")
    if re.search(r"\b(and|with)\b|[&+＋]", food, re.IGNORECASE):
        raise ValueError(f"{food!r} is more than one food: send one item per food")
    japanese = is_japanese(food)
    hits = kb.foods.search(food, limit=3, keep=lambda d: not japanese or d.data["source"] == "MEXT")
    close = [h.doc.data["name"] for h in hits]
    good = [h for h in hits if h.coverage >= WEAK_COVERAGE and (not japanese or _whole_words(food, h.doc.text))]
    if good and good[0] is hits[0]:
        kcal = [h.doc.data["per_100g"]["kcal"] for h in good if h.doc.data["per_100g"]["kcal"] is not None]
        if kcal and max(kcal) > MAX_SPREAD * min(kcal):
            raise ValueError(f"{food!r} could mean several foods with different calories: {close}. Ask the user which "
                             "one, then send that exact name")
        return hits[0].doc
    raise ValueError(f"{food!r} was not found in the food tables. Closest: {close}. Ask the user which is right "
                     "and send that exact name, or ask for the ingredients. Do not make up a number")


def _whole_words(food: str, text: str) -> bool:
    """Japanese words run together, so ラーメン also matches ラーメンスープ (a soup concentrate). Require whole words."""
    words = set(normalize(text).split())
    return all(w in words for w in normalize(food).split())


def grams_of(doc: Any, quantity: Any, unit: Any) -> tuple[float, str | None]:
    """(grams, assumption). assumption is None when the user gave a weight."""
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or not 0 < quantity <= MAX_QUANTITY:
        raise ValueError(f"quantity must be a number above 0 and up to {MAX_QUANTITY}, got {quantity!r}")
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    staple = _BY_ID.get(doc.id)
    if unit == "g":
        # "2 eggs" once arrived as 2 g and 6 kcal was saved. Spoon foods (oil, butter) can really be this small.
        if quantity < MIN_GRAMS and staple and staple.units and not {"tbsp", "tsp"} & set(staple.units):
            raise ValueError(f"{quantity:g} g of {staple.label} is too little to be right. If the user counted them, use "
                             f"unit {' or '.join(repr(u) for u in staple.units)} with quantity {quantity:g}")
        return float(quantity), None
    if staple and unit in staple.units:
        one = staple.units[unit]
        return quantity * one, f"1 {unit} of {staple.label} = {one:g} g"
    for p in doc.data["portions"]:                         # e.g. {"portion": "2 tbsp", "grams": 32.0}
        m = re.search(rf"([\d.]+)\s+(?:{UNIT_WORDS.get(unit, unit)})s?\b", p["portion"].lower())
        if m and float(m.group(1)) > 0:
            one = p["grams"] / float(m.group(1))
            return quantity * one, f"1 {unit} of {doc.data['name']} = {one:g} g (table)"
    raise ValueError(f"the weight of 1 {unit} of {doc.data['name']!r} is unknown: ask the user for grams, then use unit 'g'")


# ── calculating ────────────────────────────────────────────────────────────────────────
def calculate(kb: KnowledgeBase, items: Any) -> dict[str, Any]:
    """kcal and PFC of each item and the total. One bad item raises: never a total that silently misses a food."""
    items = [_from_text(i) if isinstance(i, str) else i for i in items] if isinstance(items, list) else items
    if not items or not all(isinstance(i, dict) for i in items):
        raise ValueError('items must be a list like [{"food": "cooked rice", "quantity": 150, "unit": "g"}]')
    parts, assumptions, labels = [], [], []
    for item in items:
        doc = resolve(kb, item.get("food"))
        grams, assumed = grams_of(doc, item.get("quantity"), item.get("unit"))
        parts.append({"food_id": doc.id, "name": doc.data["name"], "grams": grams})
        labels.append(_label(doc))
        if assumed:
            assumptions.append(assumed)
        if _key(item["food"]) not in (_key(labels[-1]), _key(doc.data["name"])):
            assumptions.append(f"'{item['food']}' counted as {labels[-1]}")
    out = kb.nutrition(parts)
    for row, label in zip(out["items"], labels):
        row["label"] = label
        row["grams"] = round(row["grams"])
    out["assumptions"] = list(dict.fromkeys(assumptions))
    out["per_food"] = _per_food(out["items"])
    out["say"] = _say(out)
    return out


def _from_text(text: str) -> dict[str, Any]:
    """"2 scrambled eggs" / "1 slice of toast" / "150g rice": small models often send text. No number -> refused later."""
    m = re.match(rf"\s*(\d+(?:\.\d+)?)\s*(?:({'|'.join(UNITS)})s?\b)?\s*(?:of\s+)?(.*)", text, re.IGNORECASE)
    if not m or not m.group(3):
        return {"food": text}
    return {"food": m.group(3), "quantity": float(m.group(1)), "unit": (m.group(2) or "piece").lower()}


def _label(doc: Any) -> str:
    return _BY_ID[doc.id].label if doc.id in _BY_ID else doc.data["name"]


def _per_food(rows: list[dict[str, Any]]) -> list[str]:
    """One labelled line per food. Given only a total, the model shared it out among the foods and got it wrong."""
    def n(v, unit):
        return f"unknown {unit}" if v is None else f"{v:g} {unit}"
    return [f"{r['label']} {r['grams']:g} g = " + ", ".join(n(r[k], unit) for k, unit in (
        ("kcal", "kcal"), ("protein_g", "g protein"), ("fat_g", "g fat"), ("carbs_g", "g carbs"))) for r in rows]


def _say(calc: dict[str, Any]) -> str:
    """One ready-made sentence, so the model copies numbers instead of working them out."""
    t = calc["total"]
    foods = " + ".join(f"{r['label']} {r['grams']:g} g" for r in calc["items"])
    text = f"{foods}: {t['kcal']:g} kcal, {t['protein_g']:g} g protein, {t['fat_g']:g} g fat, {t['carbs_g']:g} g carbs"
    if calc.get("incomplete"):
        text += f" (not counted, unknown in the table: {', '.join(calc['incomplete'])})"
    text += f" (source: {', '.join(calc['sources'])})"
    if calc.get("assumptions"):
        text += f". Assumed: {'; '.join(calc['assumptions'])}"
    return text


# ── suggesting a meal ──────────────────────────────────────────────────────────────────
MEALS_LEFT = {"breakfast": 3, "lunch": 2, "dinner": 1, "snack": 3}      # this one included
AVOID = ["meat", "chicken", "pork", "beef", "fish", "shellfish", "egg", "dairy", "soy", "wheat", "buckwheat", "oats"]
_MEAT = {"chicken", "pork", "beef"}

# (meal types, main food whose grams are solved, (min g, max g), {side: grams}). The FIRST side is the carb: it is
# halved, then dropped, when the meal is over the calorie limit.
_MENUS = [
    ("lunch dinner", "mext:11288", (60, 250), {"mext:01088": 150, "mext:06264": 80}),
    ("lunch dinner", "mext:10136", (60, 200), {"mext:01088": 150, "mext:06268": 80}),
    ("lunch dinner", "mext:10260", (70, 210), {"mext:01064": 180, "mext:06183": 90}),
    ("lunch dinner", "mext:11278", (60, 200), {"mext:02046": 150, "mext:06061": 80}),
    ("lunch dinner", "usda:175180", (60, 250), {"mext:01128": 200, "mext:06268": 80}),
    ("lunch dinner", "mext:04032", (150, 450), {"mext:01088": 150, "mext:06016": 80}),
    ("lunch dinner", "mext:13033", (100, 250), {"mext:01174": 110, "mext:06183": 90}),
    ("breakfast", "usda:170894", (100, 400), {"mext:01004": 40, "mext:07107": 100}),
    ("breakfast", "usda:172187", (61, 305), {"mext:01174": 55, "mext:06183": 90}),
    ("breakfast", "mext:10136", (60, 160), {"mext:01088": 150, "mext:04046": 45}),
    ("breakfast", "mext:04032", (150, 450), {"mext:01088": 150, "mext:04046": 45}),
    ("snack", "usda:170894", (100, 400), {"mext:07107": 100}),
    ("snack", "mext:04032", (100, 450), {"mext:06016": 80}),
    ("snack", "mext:11228", (40, 200), {"mext:07148": 100}),
]


def meal_goal(meal: str, remaining: dict[str, float] | None) -> tuple[float, float | None]:
    """(protein goal, kcal limit) for this meal: what is left today shared over the meals left.
    Protein is kept to 20-50 g (snack 15-25 g). No profile yet: 30 g (snack 15 g) and no calorie limit."""
    if meal not in MEALS_LEFT:
        raise ValueError(f"meal must be one of {list(MEALS_LEFT)}")
    low, high = (15, 25) if meal == "snack" else (20, 50)
    if remaining is None:
        return (15 if meal == "snack" else 30), None
    share = MEALS_LEFT[meal]
    return (min(high, max(low, round(remaining["protein_g"] / share))),
            max(200 if meal == "snack" else 350, round(remaining["kcal"] / share)))


def suggest_meal(kb: KnowledgeBase, meal: str, remaining: dict[str, float] | None = None,
                 protein_g: float | None = None, max_kcal: float | None = None,
                 avoid: list[str] | None = None, limit: int = 3) -> dict[str, Any]:
    """Meals that reach the protein goal within the calorie limit, with grams worked out. Saves nothing."""
    goal, kcal_limit = meal_goal(meal, remaining)
    for name, value, low, high in (("protein_g", protein_g, 5, 100), ("max_kcal", max_kcal, 100, 2000)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be a number from {low} to {high}, got {value!r}")
    # A named protein number is a minimum: it can raise our goal, never lower it (a model sent 25 unasked).
    wanted = goal if protein_g is None else max(goal, protein_g)
    kcal_limit = kcal_limit if max_kcal is None else max_kcal
    unknown = set(avoid or []) - set(AVOID)
    if unknown:
        raise ValueError(f"can't check for {sorted(unknown)}; avoid can be {AVOID}. If it is an allergy, tell the "
                         "user you can't check for it")
    banned = set(avoid or [])
    if "meat" in banned:
        banned |= _MEAT
    if "fish" in banned:
        banned.add("shellfish")       # "I can't eat fish" often means seafood; leaving out too much is the safe mistake
    menus = [m for m in _MENUS if meal in m[0].split()
             and not any(i not in kb.food_by_id or _BY_ID[i].tags & banned for i in [m[1], *m[3]])]
    # Our own goal steps down by 5 g until something fits, but never below a number the user named.
    goals = [g for g in (wanted, *range(int(wanted) - 5, 9, -5)) if protein_g is None or g >= protein_g or g == wanted]
    if protein_g is not None and protein_g not in goals:
        goals.append(protein_g)
    options: list[dict[str, Any]] = []
    for goal in goals:
        options = [o for _, main, (low, high), sides in menus
                   if (o := _fit(kb, main, low, high, sides, goal, kcal_limit))][:limit]
        if options:
            break
    if not options:
        raise ValueError(f"no meal in the list has {wanted:g} g protein"
                         + (f" within {kcal_limit:g} kcal" if kcal_limit else "")
                         + (f" without {sorted(banned)}" if banned else "") + ": tell the user, and offer a lower protein goal")
    out = {"meal": meal, "protein_goal_g": goal, "kcal_limit": kcal_limit, "options": options,
           "note": "Suggestions only, nothing was logged. Copy the 'say' line; don't change grams or numbers, and don't split a total between foods (per_food has each food)."}
    if goal < wanted:
        out["below_goal"] = f"nothing reached {wanted:g} g, so these aim for {goal:g} g: tell the user"
    return out


def _fit(kb, main, low, high, sides, goal, kcal_limit) -> dict[str, Any] | None:
    carb = next(iter(sides))
    for scale in (1, 0.5, 0):
        fixed = {i: g * (scale if i == carb else 1) for i, g in sides.items()}
        fixed = {i: g for i, g in fixed.items() if g}
        have = sum(_per_100g(kb, i, "protein_g") * g / 100 for i, g in fixed.items())
        need = max(0.0, goal - have) / _per_100g(kb, main, "protein_g") * 100
        grams = max(low, math.ceil(need / 10 - 1e-9) * 10)
        if grams > high:
            continue
        calc = kb.nutrition([{"food_id": i, "name": kb.food_by_id[i].data["name"], "grams": g}
                             for i, g in {main: grams, **fixed}.items()])
        if calc["total"]["protein_g"] >= goal and (kcal_limit is None or calc["total"]["kcal"] <= kcal_limit):
            for row in calc["items"]:
                row["label"] = _BY_ID[row["food_id"]].label
            return {"per_food": _per_food(calc["items"]), "total": calc["total"], "say": _say(calc)}
    return None


def _per_100g(kb, food_id: str, key: str) -> float:
    return kb.food_by_id[food_id].data["per_100g"][key] or 0.0
