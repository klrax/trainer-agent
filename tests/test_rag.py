"""Retrieval tests: no network, no Ollama, no data files (a tiny made-up knowledge base)."""
import datetime as dt
import json
import sqlite3

import pytest

from trainer.rag import ingest
from trainer.rag.bm25 import BM25
from trainer.rag.embed import DIMS, EmbeddingUnavailable, unit
from trainer.rag.glossary import everyday_names
from trainer.rag.index import Doc, Index
from trainer.rag.ingest import mext_value
from trainer.rag.knowledge import KnowledgeBase, body_part_muscles
from trainer.rag.text import shared_stretch, tokenize
from trainer.store import Store
from trainer.tools import run_tool


def food(id_, name, kcal, p, f, c, source="MEXT", vector=None):
    return Doc(id_, "food", name, f"{name} {everyday_names(name)}",
               {"name": name, "source": source, "group": "", "estimated": False, "portions": [],
                "per_100g": {"kcal": kcal, "protein_g": p, "fat_g": f, "carbs_g": c}}, vector)


def exercise(name, muscle, equipment, level="beginner", category="strength"):
    return Doc(f"ex:{name}", "exercise", name, f"{name} {muscle} {equipment}",
               {"name": name, "source": "t", "primary_muscles": [muscle], "secondary_muscles": [], "mechanic": None,
                "equipment": equipment, "level": level, "category": category, "instructions": ["Do it."]})


@pytest.fixture
def kb():
    return KnowledgeBase(
        [food("mext:1", "にわとり 若どり むね 皮なし 生", 105, 23.3, 1.9, 0.1),
         food("mext:2", "にわとり 若どり むね 皮つき 生", 133, 21.3, 5.9, 0.1),
         food("mext:3", "鶏卵 全卵 生", 142, 12.2, 10.2, 0.4),
         food("mext:4", "なぞの食品", 50, None, 1.0, 2.0),
         food("usda:1", "Egg, whole, raw", 143, 12.6, 9.5, 0.7, source="USDA")],
        [exercise("Pullups", "lats", "body only"),
         exercise("Lat Pulldown", "lats", "cable"),
         exercise("One Arm Chin", "lats", "body only", level="expert"),
         exercise("Lat Stretch", "lats", "body only", category="stretching")])


def test_rag_tokenize():
    assert tokenize("鶏むね") == ["鶏む", "むね"]
    assert tokenize("卵") == ["卵"]
    assert tokenize("Two Eggs, ＬＧ２１") == ["egg", "lg21"]
    assert tokenize("potatoes berries molasses glass") == ["potato", "berry", "molasse", "glass"]
    assert tokenize("、。!") == []


def test_rag_bm25_rare_words_and_coverage():
    b = BM25([["chicken", "raw"], ["egg", "raw"], ["rice", "raw"]])
    s = b.scores(["egg", "raw"])
    assert max(s, key=s.get) == 1
    assert b.coverage(["egg", "raw"], 1) == 1.0
    assert 0 < b.coverage(["egg", "unicorn"], 1) < 0.5      # a word nobody has counts against it
    assert BM25([]).scores(["x"]) == {}


def test_rag_mext_cell_values():
    assert mext_value("12.2") == (12.2, False)
    assert mext_value("10.199999999999999") == (10.2, False)
    assert mext_value("(3.3)") == (3.3, True)
    assert mext_value("Tr") == (0.0, False) and mext_value("(Tr)") == (0.0, True)
    assert mext_value("-") == (None, False) and mext_value("") == (None, False)     # unknown is not zero


def test_rag_everyday_name_finds_official_name(kb):
    top = kb.search_foods("鶏むね肉 皮なし")["results"][0]
    assert top["food_id"] == "mext:1" and top["match"] == "good"
    assert top["per_100g"] == {"kcal": 105, "protein_g": 23.3, "fat_g": 1.9, "carbs_g": 0.1}


def test_rag_language_picks_the_table(kb):
    assert {r["source"] for r in kb.search_foods("卵")["results"]} == {"MEXT"}
    assert "USDA" in {r["source"] for r in kb.search_foods("raw egg")["results"]}


def test_rag_nutrition_math_adds_before_rounding(kb):
    out = kb.nutrition([{"food_id": "mext:1", "name": "にわとり 若どり むね 皮なし 生", "grams": 200},
                        {"food_id": "mext:3", "name": "鶏卵 全卵 生", "grams": 55}])
    assert out["items"][0]["kcal"] == 210 and out["items"][0]["protein_g"] == 46.6
    assert out["total"] == {"kcal": 288, "protein_g": 53.3, "fat_g": 9.4, "carbs_g": 0.4}
    twice = kb.nutrition([{"food_id": "mext:1", "name": "にわとり 若どり むね 皮なし 生", "grams": 50}] * 2)      # 52.5 each, shown as 52
    assert [i["kcal"] for i in twice["items"]] == [52, 52] and twice["total"]["kcal"] == 105
    assert "incomplete" not in out


def test_rag_unknown_value_is_reported_not_zero(kb):
    out = kb.nutrition([{"food_id": "mext:4", "name": "なぞの食品", "grams": 100}])
    assert out["items"][0]["protein_g"] is None and out["incomplete"] == ["protein_g of なぞの食品"]


N = "にわとり 若どり むね 皮なし 生"


@pytest.mark.parametrize("items", [[], [{"food_id": "mext:999", "name": N, "grams": 10}],
                                   [{"food_id": "mext:1", "name": N, "grams": 0}],
                                   [{"food_id": "mext:1", "name": N, "grams": 5000}],
                                   [{"food_id": "mext:1", "name": N, "grams": "100"}],
                                   [{"food_id": "mext:1", "name": N, "grams": float("nan")}],
                                   [{"food_id": "mext:1", "name": N}], [{"food_id": "mext:1", "grams": 100}],
                                   [{"food_id": "mext:1", "name": "", "grams": 100}]])
def test_rag_bad_amounts_raise(kb, items):
    with pytest.raises((ValueError, KeyError)):
        kb.nutrition(items)


def test_rag_exercise_filters(kb):
    def names(**kw):
        return {r["name"] for r in kb.search_exercises("lats", **kw)["results"]}
    assert names(equipment=["body only"]) == {"Pullups", "One Arm Chin"}
    assert names(equipment=["body only"], level="beginner") == {"Pullups"}
    assert "Lat Stretch" in names(category="stretching")
    assert "Lat Stretch" not in names()                                     # lifting only by default
    assert names(muscles=["chest"]) == set()
    assert names(muscles=["chest", "lats"], level="beginner") == {"Pullups", "Lat Pulldown"}
    with pytest.raises(ValueError):
        kb.search_exercises("x", level="godlike")
    with pytest.raises(ValueError):
        kb.search_exercises("x", category="yoga")


def test_rag_tools_and_japanese_output(kb, tmp_path):
    out, err = run_tool("search_foods", {"query": "卵"}, Store(tmp_path / "d.json"), dt.date(2026, 11, 23), kb)
    assert not err and "鶏卵" in out and json.loads(out)["search"] == "keyword"
    out, err = run_tool("search_foods", {"query": "  "}, Store(tmp_path / "d.json"), dt.date(2026, 11, 23), kb)
    assert err


def test_rag_missing_index_tells_the_model(tmp_path):
    with pytest.raises(ValueError, match="trainer.rag.ingest"):
        KnowledgeBase.load(tmp_path / "none.db")


def test_rag_hybrid_vector_and_fallback():
    docs = [food("a", "apple", 1, 1, 1, 1, vector=[1.0, 0.0]), food("b", "plum", 1, 1, 1, 1, vector=[0.0, 1.0])]
    ix = Index(docs, embed_query=lambda q: [0.0, 1.0])
    assert ix.search("zzz")[0].doc.id == "b" and ix.last_mode == "hybrid"       # no keyword match: meaning decides
    assert [h.doc.id for h in ix.search("zzz", keep=lambda d: d.id == "a")] == ["a"]

    def down(q):
        raise EmbeddingUnavailable("ollama is off")
    ix = Index(docs, embed_query=down)
    assert ix.search("apple")[0].doc.id == "a" and ix.last_mode == "keyword"
    with pytest.raises(EmbeddingUnavailable):
        ix.search("apple", mode="vector")


def test_rag_ingest_resumes_and_reembeds_changed_text():
    con = sqlite3.connect(":memory:")
    con.executescript(ingest.SCHEMA)
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return [[1.0] * DIMS for _ in texts]
    docs = [food(str(i), f"food {i}", 1, 1, 1, 1) for i in range(5)]
    ingest.save_docs(con, docs)
    assert ingest.embed_missing(con, batch=2, embed=fake, log=lambda m: None) == 5 and calls == [2, 2, 1]
    ingest.save_docs(con, docs)
    assert ingest.embed_missing(con, embed=fake, log=lambda m: None) == 0
    docs[0].text = "changed"
    ingest.save_docs(con, docs)
    assert ingest.embed_missing(con, embed=fake, log=lambda m: None) == 1
    loaded = ingest.load_docs(con, "food")
    assert [d.id for d in loaded] == ["0", "1", "2", "3", "4"] and len(loaded[0].vector) == DIMS


def test_rag_unit_vector():
    v = unit([3.0, 4.0] + [0.0] * 1000)
    assert len(v) == DIMS and v[:2] == [0.6, 0.8]


def test_rag_synonyms_do_not_push_a_food_down():
    common = food("mext:1", "だいず 糸引き納豆", 184, 16.5, 10.0, 12.1)      # gets many everyday words
    rare = [food(f"mext:{i}", "だいず 寺納豆", 248, 18.6, 8.1, 31.5) for i in (2, 3)]
    assert len(common.text) > 2 * len(rare[0].text)
    assert KnowledgeBase([*rare, common], []).search_foods("糸引き納豆")["results"][0]["food_id"] == "mext:1"
    assert KnowledgeBase([*rare, common], []).search_foods("納豆")["results"][0]["food_id"] == "mext:1"


def test_rag_shared_stretch():
    assert shared_stretch("鶏むね肉 皮なし", "にわとり むね 皮なし 生 鶏むね肉") == 1.0
    assert shared_stretch("サラダチキン", "サラダな") == 0.5
    assert shared_stretch("スターバックス", "プレミックス粉 バッター") == 3 / 7
    assert shared_stretch("", "x") == 0.0


def test_rag_scattered_katakana_pairs_are_weak():
    kb = KnowledgeBase([food("mext:1", "プレミックス粉 天ぷら用 バッター", 1, 1, 1, 1),
                        food("mext:2", "こめ", 1, 1, 1, 1)], [])
    assert all(r["match"] == "weak" for r in kb.search_foods("スターバックス ラテ")["results"])


def test_rag_meaning_alone_must_be_close_for_foods():
    def kb(cos):
        return KnowledgeBase([food("a", "ribeye", 1, 1, 1, 1, vector=[cos, 0.0])], [], embed_query=lambda q: [1.0, 0.0])
    assert kb(0.51).search_foods("unicorn")["results"][0]["match"] == "weak"
    assert kb(0.65).search_foods("unicorn")["results"][0]["match"] == "good"


def test_rag_wrong_id_for_the_food_is_rejected(kb):
    # seen live: qwen3 searched chicken, then sent the egg's id
    with pytest.raises(ValueError, match="鶏卵 全卵 生"):
        kb.nutrition([{"food_id": "mext:3", "name": "にわとり 若どり むね 皮なし 生", "grams": 150}])


def test_rag_no_real_food_id_in_tool_descriptions():
    from trainer.tools import TOOLS
    assert "mext:" not in json.dumps(TOOLS) and "usda:" not in json.dumps(TOOLS)   # models copy examples


def test_rag_body_part_is_filled_in():
    assert body_part_muscles("back exercises") == ["lats", "middle back", "lower back", "traps"]
    assert body_part_muscles("背中を鍛える種目")[0] == "lats"
    assert body_part_muscles("back squat") is None and body_part_muscles("leg press") is None
    assert body_part_muscles("stable movement") is None       # "ab" inside a word doesn't count
