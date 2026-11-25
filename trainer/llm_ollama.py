"""Run the agent on a local model with Ollama (an adapter between two API formats).

Your loop speaks the Claude message format. Ollama (like vLLM and most in-house LLM servers)
speaks the OpenAI chat-completions format. You write the translation in both directions,
and agent.py, tools.py and all the tests stay unchanged.
Design: the provider sits behind one function, so models can be swapped and compared with the same evals.

The two formats, side by side:

    Claude-style (your `messages`)                    OpenAI-style (what Ollama wants)
    system=... (a separate argument)                  {"role": "system", "content": ...} first
    {"role": "user", "content": "hi"}                 {"role": "user", "content": "hi"}
    {"role": "assistant", "content": [blocks]}        {"role": "assistant", "content": "text",
        text block, tool_use block (id, name, input)       "tool_calls": [{"id", "type": "function",
                                                           "function": {"name", "arguments": JSON STRING}}]}
    {"role": "user", "content": [tool_result, ...]}   one {"role": "tool", "tool_call_id", "content"} PER result
    stop_reason "tool_use" / "end_turn" / ...         finish_reason "tool_calls" / "stop" / "length"

Run:  uv run pytest -k s4b      (no Ollama needed: the tests use plain dicts)
"""
from __future__ import annotations

import json
import re
import urllib.request
from types import SimpleNamespace as NS
from typing import Any

DEFAULT_MODEL = "qwen3:1.7b"
BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible endpoint


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each Claude tool {"name", "description", "input_schema"} becomes
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": <the input_schema>}}
    - to_openai_tools([]) -> []
    """
    return [
        {"type": "function",
         "function": {"name": tool["name"], "description": tool["description"], "parameters": tool["input_schema"]}}
        for tool in tools
    ]


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate your whole history (Claude-style) to an OpenAI-style message list.

    1. first: {"role": "system", "content": system}
    2. a user message whose content is a str -> {"role": "user", "content": that str}
    3. a user message whose content is a list (tool results) -> for EACH item, in order:
       - {"type": "tool_result", "tool_use_id": X, "content": C, ...}
             -> {"role": "tool", "tool_call_id": X, "content": C}     (is_error is dropped: no such field)
       - {"type": "text", "text": T} -> {"role": "user", "content": T}
    4. an assistant message: its content is a list of BLOCK OBJECTS (use b.type, b.text: attributes, not ["..."])
       - "content": the .text of its text blocks joined with "\\n"  ("" if there are none)
       - "tool_calls": one per tool_use block, in order:
             {"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
       - no tool_use blocks -> leave the "tool_calls" key out entirely
       - other block types (e.g. "thinking") are skipped
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        content = message["content"]
        if message["role"] == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            for item in content:                     # a list: tool results (one OpenAI message EACH)
                if item["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": item["tool_use_id"], "content": item["content"]})
                elif item["type"] == "text":
                    out.append({"role": "user", "content": item["text"]})
            continue
        if isinstance(content, str):                     # a reply written by code (safety.py, direct.py)
            out.append({"role": "assistant", "content": content})
            continue
        # assistant: a list of block objects; thinking blocks are skipped (Ollama can't use them)
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": "\n".join(b.text for b in content if b.type == "text"),
        }
        tool_calls = [
            {"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}}
            for b in content if b.type == "tool_use"
        ]
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        out.append(assistant)
    return out


def from_openai(message: dict[str, Any], finish_reason: str | None) -> Any:
    """Translate one OpenAI-style reply into what run_turn expects: an object with
    .stop_reason and .content (a list of blocks). Build objects with NS(...), e.g.
        NS(type="text", text="hi")   NS(type="tool_use", id="call_0", name="get_context", input={})
        return NS(stop_reason="end_turn", content=[...])

    Text:
    - message["content"] may be None, "", or text
    - remove any "<think>...</think>" section (some local models print their reasoning there):
          re.sub(r"<think>.*?</think>", "", text, flags=re.S)
      then .strip()
    - non-empty after that -> ONE text block, first in the list; empty -> no text block

    Tool calls (message.get("tool_calls") may be missing or None):
    - one tool_use block per call, in order; name = call["function"]["name"]
    - id: call.get("id"); if missing/empty -> f"call_{i}" (i = its position in tool_calls)
    - input: call["function"]["arguments"] is USUALLY a JSON string -> json.loads it
          - already a dict (Ollama sometimes does this) -> use it as is
          - "" or None -> {}
          - not valid JSON -> {"_invalid_json": <the raw string>}   (don't crash: run_tool will
            report the bad input back to the model, which can then retry)

    stop_reason:
    - at least one tool_use block -> "tool_use"   (even if finish_reason says "stop": local servers vary)
    - else finish_reason == "length" -> "max_tokens"
    - else -> "end_turn"

    Rule: be strict in what you send, forgiving in what you accept.
    """
    content: list[Any] = []

    text = re.sub(r"<think>.*?</think>", "", message.get("content") or "", flags=re.S).strip()
    if text:
        content.append(NS(type="text", text=text))

    for i, call in enumerate(message.get("tool_calls") or []):
        content.append(NS(type="tool_use", id=call.get("id") or f"call_{i}",
                          name=call["function"]["name"], input=_parse_arguments(call["function"].get("arguments"))))

    if any(b.type == "tool_use" for b in content):
        stop_reason = "tool_use"          # trust the tool calls, not finish_reason: local servers vary
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return NS(stop_reason=stop_reason, content=content)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    """Arguments usually arrive as a JSON string, sometimes already as a dict. Never raise."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"_invalid_json": arguments}   # run_tool reports this back to the model, which can retry


# ── PROVIDED: the HTTP call. Uses only the standard library (no extra package). ──────
def make_call_model(system: str, tools: list[dict[str, Any]], *, model: str = DEFAULT_MODEL,
                    base_url: str = BASE_URL, usage: Any = None, timeout: float = 600, think: bool = False):
    """Same contract as llm.make_call_model: returns call_model(messages) -> response.

    think=False sends reasoning_effort "none", which turns the reasoning step off. (The "/no_think" text this used
    before did nothing with this Ollama version: a one-word answer still took 192 tokens instead of 2.)
    Much faster on a CPU; may be worse at multi-step tool use. Measure both. Models without reasoning ignore it.
    """
    oa_tools = to_openai_tools(tools)

    def post(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": to_openai_messages(system, messages), **extra}
        if not think:
            body["reasoning_effort"] = "none"
        if oa_tools:
            body["tools"] = oa_tools
        req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def call_model(messages: list[dict[str, Any]]):
        data = post(messages)
        choice = data["choices"][0]
        if usage is not None:
            u = data.get("usage") or {}
            usage.add(NS(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0),
                         cache_read_input_tokens=0))
        return from_openai(choice["message"], choice.get("finish_reason"))

    def warm_up() -> int:
        """Load the model and process the system prompt + tools once (a 1-token request).

        Ollama keeps the processed prompt prefix, so the first real call only has to read the new
        messages. Returns the prompt size in tokens, a number worth knowing on a small context window.
        """
        data = post([{"role": "user", "content": "hi"}], max_tokens=1)
        return (data.get("usage") or {}).get("prompt_tokens", 0)

    call_model.warm_up = warm_up  # type: ignore[attr-defined]
    return call_model
