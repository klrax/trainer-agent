"""Build the knowledge base: parse the raw files, embed every document, save to data/rag/index.db.

    uv run python -m trainer.rag.ingest              # parse + embed (safe to stop and rerun: it resumes)
    uv run python -m trainer.rag.ingest --no-embed   # keyword search only, takes seconds

One food or exercise = one document ("chunk"). They are short and self-contained, so splitting them
further would separate a name from its numbers.

Raw files (see README for the download commands):
- data/raw/mext/mext_food_table_2023.xlsx   Standard Tables of Food Composition in Japan (MEXT)
- data/raw/usda/sr_legacy_csv.zip           USDA FoodData Central, SR Legacy
- data/raw/exercises/exercises.json         free-exercise-db
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from . import embed as emb
from .glossary import everyday_names
from .index import Doc

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "rag" / "index.db"
PFC = ("kcal", "protein_g", "fat_g", "carbs_g")

MEXT_GROUPS = {"01": "穀類", "02": "いも及びでん粉類", "03": "砂糖及び甘味類", "04": "豆類", "05": "種実類",
               "06": "野菜類", "07": "果実類", "08": "きのこ類", "09": "藻類", "10": "魚介類", "11": "肉類",
               "12": "卵類", "13": "乳類", "14": "油脂類", "15": "菓子類", "16": "し好飲料類",
               "17": "調味料及び香辛料類", "18": "調理済み流通食品類"}
# columns found by their code in the table's own header row, not by position
MEXT_COLUMNS = {"ENERC_KCAL": "kcal", "PROT-": "protein_g", "FAT-": "fat_g", "CHOCDF-": "carbs_g"}


# ── MEXT (Japanese foods) ──────────────────────────────────────────────────────────────
def mext_value(raw: str) -> tuple[float | None, bool]:
    """(value, is_estimated) for one table cell.

    "12.2" -> (12.2, False)   "(3.3)" -> (3.3, True)   "Tr" -> (0.0, False), a trace
    "(Tr)" -> (0.0, True)     "-" or "" -> (None, False), not measured (NOT the same as zero)
    """
    s = raw.strip()
    estimated = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    if s in ("", "-", "*"):
        return None, False
    if s == "Tr":
        return 0.0, estimated
    return round(float(s), 1), estimated


def parse_mext(path: Path) -> Iterator[Doc]:
    rows = _xlsx_rows(path)
    columns: dict[str, str] = {}
    for row in rows:
        if not columns:
            found = {code: col for col, code in row.items() if code in MEXT_COLUMNS}
            if len(found) == len(MEXT_COLUMNS):
                columns = {MEXT_COLUMNS[code]: col for code, col in found.items()}
            continue
        number, name = row.get("B", ""), row.get("D", "")
        if not re.fullmatch(r"\d{5}", number) or not name:
            continue
        per_100g, estimated = {}, False
        for key in PFC:
            per_100g[key], est = mext_value(row.get(columns[key], ""))
            estimated = estimated or est
        title = re.sub(r"\s+", " ", name.replace("　", " ")).strip()
        group = MEXT_GROUPS.get(row.get("A", ""), "")
        yield Doc(id=f"mext:{number}", kind="food", title=title,
                  text=" ".join(x for x in (title, everyday_names(title)) if x),
                  data={"name": title, "source": "MEXT", "group": group, "per_100g": per_100g,
                        "estimated": estimated, "portions": []})
    if not columns:
        raise ValueError(f"{path.name}: header row with {sorted(MEXT_COLUMNS)} not found")


def _xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    """Rows of the first sheet as {"A": "...", "B": "..."} (an .xlsx is a zip of XML files)."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as z:
        strings = []
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
            parts = [si.find(ns + "t")] + [r.find(ns + "t") for r in si.findall(ns + "r")]   # skips furigana
            strings.append("".join(t.text or "" for t in parts if t is not None))
        for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
            cells = {}
            for c in row.findall(ns + "c"):
                v = c.find(ns + "v")
                if v is not None and v.text is not None:
                    col = re.match(r"[A-Z]+", c.get("r")).group()
                    cells[col] = strings[int(v.text)] if c.get("t") == "s" else v.text
            yield cells


# ── USDA (English foods) ───────────────────────────────────────────────────────────────
USDA_NUTRIENTS = {"1008": "kcal", "1003": "protein_g", "1004": "fat_g", "1005": "carbs_g"}


def parse_usda(path: Path, max_portions: int = 3) -> Iterator[Doc]:
    with zipfile.ZipFile(path) as z:
        def table(name: str) -> Iterator[dict[str, str]]:
            member = next(n for n in z.namelist() if n.endswith("/" + name))
            with z.open(member) as f:
                yield from csv.DictReader(io.TextIOWrapper(f, "utf-8-sig"))

        groups = {r["id"]: r["description"] for r in table("food_category.csv")}
        units = {r["id"]: r["name"] for r in table("measure_unit.csv")}
        nutrients: dict[str, dict[str, float]] = defaultdict(dict)
        for r in table("food_nutrient.csv"):
            if r["nutrient_id"] in USDA_NUTRIENTS and r["amount"]:
                nutrients[r["fdc_id"]][USDA_NUTRIENTS[r["nutrient_id"]]] = round(float(r["amount"]), 1)
        portions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in table("food_portion.csv"):
            if len(portions[r["fdc_id"]]) < max_portions and r["gram_weight"]:
                unit = "" if r["measure_unit_id"] == "9999" else units.get(r["measure_unit_id"], "")
                label = " ".join(x for x in (r["amount"], unit, r["portion_description"], r["modifier"]) if x)
                portions[r["fdc_id"]].append({"portion": label, "grams": float(r["gram_weight"])})
        for r in table("food.csv"):
            fid, name = r["fdc_id"], r["description"].strip()
            yield Doc(id=f"usda:{fid}", kind="food", title=name, text=name,
                      data={"name": name, "source": "USDA", "group": groups.get(r["food_category_id"], ""),
                            "per_100g": {k: nutrients[fid].get(k) for k in PFC},
                            "estimated": False, "portions": portions[fid]})


# ── exercises ──────────────────────────────────────────────────────────────────────────
def parse_exercises(path: Path) -> Iterator[Doc]:
    for e in json.loads(path.read_text()):
        muscles = e["primaryMuscles"]
        facts = [f"muscles: {', '.join(muscles)}"]
        if e["secondaryMuscles"]:
            facts.append(f"also works: {', '.join(e['secondaryMuscles'])}")
        for key in ("equipment", "mechanic", "force", "level", "category"):
            if e.get(key):
                facts.append(f"{key}: {e[key]}")
        text = f"{e['name']}. {'; '.join(facts)}. {' '.join(e['instructions'])[:600]}"
        yield Doc(id=f"ex:{e['id']}", kind="exercise", title=e["name"], text=text,
                  data={"name": e["name"], "source": "free-exercise-db", "primary_muscles": muscles,
                        "secondary_muscles": e["secondaryMuscles"], "equipment": e.get("equipment"),
                        "level": e.get("level"), "mechanic": e.get("mechanic"), "category": e.get("category"),
                        "instructions": e["instructions"]})


# ── the database ───────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, kind TEXT, title TEXT, text TEXT, data TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS vectors (hash TEXT PRIMARY KEY, vec BLOB);
"""


def doc_hash(doc: Doc) -> str:
    """Changes when the text, the model or the size changes, so a stale vector is never reused."""
    key = f"{emb.MODEL}|{emb.DIMS}|{emb.document_text(doc.title, doc.text)}"
    return hashlib.sha256(key.encode()).hexdigest()


def save_docs(con: sqlite3.Connection, docs: list[Doc]) -> None:
    with con:                                           # one transaction: all of it or none of it
        con.execute("DELETE FROM docs")
        con.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?)",
                        [(d.id, d.kind, d.title, d.text, json.dumps(d.data, ensure_ascii=False), doc_hash(d))
                         for d in docs])


def embed_missing(con: sqlite3.Connection, batch: int = 32, embed=emb.embed, log=print) -> int:
    """Embed documents that have no vector yet. Commits per batch, so stopping loses at most one batch."""
    todo = con.execute("SELECT DISTINCT hash, title, text FROM docs "
                       "WHERE hash NOT IN (SELECT hash FROM vectors)").fetchall()
    start = time.monotonic()
    for n in range(0, len(todo), batch):
        part = todo[n:n + batch]
        vectors = embed([emb.document_text(title, text) for _, title, text in part])
        with con:
            con.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?)",
                            [(h, array("f", v).tobytes()) for (h, _, _), v in zip(part, vectors, strict=True)])
        done = n + len(part)
        if (n // batch) % 10 == 0 or done == len(todo):
            rate = done / (time.monotonic() - start)
            log(f"  embedded {done}/{len(todo)} · {rate:.1f}/s · ~{(len(todo) - done) / rate / 60:.0f} min left")
    return len(todo)


def load_docs(con: sqlite3.Connection, kind: str) -> list[Doc]:
    rows = con.execute("SELECT d.id, d.kind, d.title, d.text, d.data, v.vec FROM docs d "
                       "LEFT JOIN vectors v ON v.hash = d.hash WHERE d.kind = ? ORDER BY d.rowid", (kind,))
    docs = []
    for id_, kind_, title, text, data, blob in rows:
        vec = None
        if blob is not None:
            vec = array("f")
            vec.frombytes(blob)
            vec = vec.tolist()
        docs.append(Doc(id_, kind_, title, text, json.loads(data), vec))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the knowledge base.")
    ap.add_argument("--no-embed", action="store_true", help="skip embeddings (keyword search only)")
    opts = ap.parse_args()

    docs: list[Doc] = []
    for label, path, parse in [("MEXT foods", RAW / "mext" / "mext_food_table_2023.xlsx", parse_mext),
                               ("USDA foods", RAW / "usda" / "sr_legacy_csv.zip", parse_usda),
                               ("exercises", RAW / "exercises" / "exercises.json", parse_exercises)]:
        if not path.exists():
            print(f"{label}: missing {path.relative_to(ROOT)}, skipped")
            continue
        part = list(parse(path))
        print(f"{label}: {len(part)}")
        docs += part
    if not docs:
        raise SystemExit("no raw files found; see the README for download commands")

    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    save_docs(con, docs)
    print(f"saved {len(docs)} documents to {DB.relative_to(ROOT)}")
    if not opts.no_embed:
        try:
            embed_missing(con)
        except emb.EmbeddingUnavailable as e:
            raise SystemExit(f"embeddings stopped ({e}). Keyword search works; start Ollama and rerun to finish.")


if __name__ == "__main__":
    main()
