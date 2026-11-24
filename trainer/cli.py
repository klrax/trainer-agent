"""PROVIDED · the chat app. Run:  uv run python -m trainer.cli

Commands inside the chat:  /usage  /reset (forget this conversation, keep the data)  /quit
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

from .agent import run_turn
from .llm import UsageLog, make_call_model
from .prompts import SYSTEM
from .store import Store
from .tools import TOOLS, run_tool

DIM, RESET = "\033[2m", "\033[0m"


def build_call_model(provider, system, tools, *, usage=None, effort="medium", model=None, think=False):
    """The one place that picks the model provider. Both return the same call_model(messages)."""
    if provider == "ollama":
        from . import llm_ollama
        return llm_ollama.make_call_model(system, tools, model=model or llm_ollama.DEFAULT_MODEL, usage=usage,
                                          think=think)
    return make_call_model(system, tools, effort=effort, usage=usage)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with your AI strength coach.")
    ap.add_argument("--data", default="data/trainer.json", help="where your profile and logs are saved")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="claude only")
    ap.add_argument("--provider", default="claude", choices=["claude", "ollama"])
    ap.add_argument("--model", default=None, help="ollama model name (default: llm_ollama.DEFAULT_MODEL)")
    ap.add_argument("--think", action="store_true", help="ollama only: let qwen3 reason before answering (slower)")
    opts = ap.parse_args()

    Path(opts.data).parent.mkdir(parents=True, exist_ok=True)
    store = Store(opts.data)
    usage = UsageLog()
    call_model = build_call_model(opts.provider, SYSTEM, TOOLS, usage=usage, effort=opts.effort, model=opts.model,
                                  think=opts.think)

    if hasattr(call_model, "warm_up"):
        print(f"{DIM}Loading {opts.model or 'the local model'} and reading the instructions (one time only)...{RESET}",
              flush=True)
        start = time.monotonic()
        prompt_tokens = call_model.warm_up()
        print(f"{DIM}  ready in {time.monotonic() - start:.0f} s · system prompt + tools = {prompt_tokens} tokens{RESET}")

    calls = {"n": 0}

    def timed_call_model(messages):
        calls["n"] += 1
        print(f"{DIM}  … coach is thinking (model call {calls['n']}){RESET}", flush=True)
        start = time.monotonic()
        response = call_model(messages)
        print(f"{DIM}    {time.monotonic() - start:.1f} s{RESET}", flush=True)
        return response

    def execute_tool(name, args):
        content, is_error = run_tool(name, args, store, dt.date.today())
        flag = " ERROR" if is_error else ""
        print(f"{DIM}  → {name}({json.dumps(args)}){flag}: {content[:120]}{RESET}")   # watch the agent think
        return content, is_error

    messages: list = []
    print("Coach ready. Try: 'what's my workout today?'   (/usage /reset /quit)")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/usage":
            print(usage)
            continue
        if user == "/reset":
            messages = []
            print("(conversation cleared; your saved data is kept)")
            continue

        messages.append({"role": "user", "content": user})
        calls["n"] = 0
        result = run_turn(messages, timed_call_model, execute_tool)
        if result.stopped == "refusal":
            messages.pop()          # drop the declined message so the next turn starts clean
        print(f"\ncoach> {result.text or '(no text)'}")
        if result.stopped not in ("done", "refusal"):
            print(f"{DIM}  [stopped: {result.stopped}]{RESET}")
    print(usage)


if __name__ == "__main__":
    main()
