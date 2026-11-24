# Trainer Agent

A tool-using AI strength coach. It plans each training session, suggests working weights by
progressive overload, and tracks meals against calorie and macro targets. It runs on a hosted
model (Claude API) or on a local model through Ollama.

**Status: in progress.** It works end to end; evals and hosting are being built out.

```
you> what's my workout today?
  … coach is thinking (model call 1)
  → get_context({}): {"date": "2026-11-23", "weekday": "Mon", "profile": {...}, "today": {"session": "Full body A", ...}}
coach> Today is Full body A: Back squat 3x5-8, Bench press 3x6-10, Barbell row 3x8-12, Plank 3x30-60s.
       No history yet for these, so start light and log what you lift.
```

## How it works

```
cli.py ──► agent.run_turn ──► call_model(messages) ── llm.py         (Claude API)
                │                                  └─ llm_ollama.py  (OpenAI-format adapter → Ollama)
                └──► tools.run_tool ──► context.py / coaching.py / store.py
                                    └─► rag/ (food tables, exercises)
```

- **The agent loop** (`trainer/agent.py`): call the model; if it asks for tools, run them all and
  send every result back in one message; repeat until it answers. Handles parallel tool calls, tool
  errors (returned to the model as text, never raised), `max_tokens`, `pause_turn`, refusals, and a step limit.
- **Tools** (`trainer/tools.py`): 12 tools with JSON schemas. `get_context` returns the date, the
  profile and today's plan with suggested weights in one call, because small models often skipped
  separate lookups.
- **Deterministic logic in code** (`trainer/coaching.py`): calorie and macro targets (Mifflin-St Jeor),
  weekly split, the session on a given date, and double-progression weight suggestions, all unit-tested.
  The model handles the conversation, not the arithmetic.
- **State** (`trainer/store.py`): a JSON file with atomic writes. The model is stateless and reaches
  data only through tools.
- **Provider abstraction**: the loop only knows `call_model(messages)`. `llm_ollama.py` translates to
  and from the OpenAI chat format, so the same agent runs on any OpenAI-compatible server (Ollama, vLLM).

## Retrieval (RAG)

Food numbers and exercise suggestions come from reference data, not from the model's memory.

```
question ─► keyword search (BM25) ─┐
        └─► meaning search (embeddings) ─┴─► merge by rank ─► entries with numbers and source
                                                                 └─► code scales to grams and adds up
```

- **Data (11,207 entries, one document each):** 2,538 Japanese foods (MEXT), 7,793 US foods (USDA SR
  Legacy), 876 exercises. Not stored in this repo; the build reads the original files.
- **Finding:** hybrid search with Reciprocal Rank Fusion. Filters (equipment, level, muscle) are exact
  and run before ranking. A small glossary links everyday words to official names
  (鶏むね肉 → にわとり むね); it affects what can be found, never a number.
- **Arithmetic in code:** `get_food_nutrition` scales per-100 g values and sums them. A value the table
  didn't measure is reported as missing, never counted as zero.
- **When it's unsure or down:** poor matches are marked `weak` so the coach asks. Without Ollama it
  falls back to keyword search and says so in the result.
- **Vector store:** SQLite plus a full scan. At about 11,000 vectors of 256 numbers there is no need
  for a vector database.
- **Evaluation:** `evals/retrieval_eval.py` compares keyword, vector and hybrid search (hit@1, hit@5, MRR).

```bash
mkdir -p data/raw/mext data/raw/usda data/raw/exercises
curl -fL -o data/raw/mext/mext_food_table_2023.xlsx https://www.mext.go.jp/content/20260327-mxt_kagsei-mext-000029402_02.xlsx
curl -fL -o data/raw/usda/sr_legacy_csv.zip https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip
curl -fL -o data/raw/exercises/exercises.json https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json
ollama pull embeddinggemma:300m          # optional: keyword search works without it
uv run python -m trainer.rag.ingest      # add --no-embed for keyword search only
```

Sources: 「日本食品標準成分表（八訂）増補2023年」（文部科学省）, values extracted and reformatted;
U.S. Department of Agriculture, FoodData Central (SR Legacy); free-exercise-db (The Unlicense).

## Run it

```bash
uv sync
uv run pytest                                          # 57 tests, no model or API key needed

# Local model (free; slow on CPU):
ollama pull qwen3:1.7b
uv run python -m trainer.cli --provider ollama

# Claude API:
export ANTHROPIC_API_KEY=...
uv run python -m trainer.cli
```

## Testing

- **Unit tests** cover the store, the coaching rules, the tools, the agent loop (with a fake model)
  and the format adapter.
- **Evals** (`evals/run_evals.py`) run the whole agent against the real model: each scenario seeds
  data, sends one message, and checks the outcome and the tool calls. Scenarios run several times to
  get a pass rate.

```bash
uv run python -m evals.run_evals --provider ollama -n 3
```

## Lessons so far (local qwen3:1.7b)

- The model skipped profile lookups and asked for details that were already saved. Merging the
  lookups into one `get_context` tool fixed it.
- It invented weights when there was no history. Weights are now computed in code and returned by
  `get_context`, and the prompt forbids making them up.
- One run logged a fake workout in response to a question. Next steps: validation in write tools,
  and eval checks for unwanted side effects.
- About 60 s per model call on a laptop CPU. A hosted model is needed for use from a phone.
