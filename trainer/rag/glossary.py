"""Everyday names for the official Japanese food names.

The MEXT table says "にわとり［若どり・主品目］むね 皮なし 生"; people say "鶏むね肉" or "chicken breast".
Each rule adds everyday words to a food's searchable text when every part appears in its official name.
This only changes what can be FOUND; it never changes a number.
"""
from __future__ import annotations

# Everyday cut names go only to 若どり (young chicken, what shops sell), not 親 (old hen).
RULES: list[tuple[tuple[str, ...], str]] = [
    (("にわとり",), "鶏 鶏肉 とり肉 チキン chicken"),
    (("にわとり", "若どり", "むね"), "鶏むね肉 鶏胸肉 chicken breast"),
    (("にわとり", "若どり", "もも"), "鶏もも肉 chicken thigh"),
    (("にわとり", "若どり", "ささみ"), "鶏ささみ chicken tenderloin"),
    (("ぶた",), "豚 豚肉 ポーク pork"),
    (("ぶた", "ばら"), "豚バラ肉 pork belly"),
    (("ぶた", "ロース"), "豚ロース肉 pork loin"),
    (("うし",), "牛 牛肉 ビーフ beef"),
    (("うし", "ひき肉"), "牛ひき肉 ground beef"),
    (("鶏卵",), "卵 たまご 玉子 egg"),
    (("鶏卵", "全卵", "ゆで"), "ゆで卵 ゆでたまご boiled egg"),
    (("水稲めし", "うるち米"), "ご飯 ごはん 白米 白ご飯 ライス cooked white rice"),
    (("水稲めし", "玄米"), "玄米ご飯 cooked brown rice"),
    (("おにぎり",), "おむすび rice ball onigiri"),
    (("糸引き納豆",), "納豆 natto fermented soybeans"),
    (("木綿豆腐",), "豆腐 もめん豆腐 firm tofu"),
    (("絹ごし豆腐",), "豆腐 silken tofu"),
    (("しろさけ",), "鮭 さけ サーモン salmon"),
    (("たいせいようさけ",), "鮭 サーモン アトランティックサーモン atlantic salmon"),
    (("まぐろ",), "鮪 マグロ ツナ tuna"),
    (("まさば",), "鯖 さば サバ mackerel"),
    (("食パン",), "トースト パン white bread toast"),
    (("普通牛乳",), "牛乳 ミルク whole milk"),
    (("ヨーグルト",), "yogurt"),
    (("プロセスチーズ",), "チーズ cheese"),
    (("さつまいも",), "薩摩芋 焼き芋 sweet potato"),
    (("じゃがいも",), "じゃが芋 ジャガイモ potato"),
    (("バナナ",), "banana"),
    (("りんご",), "林檎 リンゴ apple"),
    (("ブロッコリー",), "broccoli"),
    (("オートミール",), "oatmeal oats"),
    (("うどん", "ゆで"), "うどん udon noodles"),
    (("そば", "ゆで"), "そば 蕎麦 soba noodles"),
    (("マカロニ", "スパゲッティ"), "パスタ pasta spaghetti"),
    (("アボカド",), "avocado"),
    (("アーモンド",), "almond"),
]


def everyday_names(official_name: str) -> str:
    """All everyday words that apply to this official name, joined by spaces ("" if none)."""
    return " ".join(words for parts, words in RULES if all(p in official_name for p in parts))
