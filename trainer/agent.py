"""The agent loop.

    user message
         │
         ▼
    ┌─► call the model ──► stop_reason?
    │        "end_turn"   → done: return the text
    │        "tool_use"   → run every tool it asked for
    │                        send ALL the results back in ONE user message
    └──────────────────────── and call the model again
             "max_tokens" → cut off: return what we have
             "pause_turn" → call again (no tool results)
             "refusal"    → the model declined: return REFUSAL_TEXT
    + a step limit, so a confused model can't loop (and spend) forever

The model never runs anything. It only ASKS for a tool; your code decides whether and how to run it.

Run:  uv run pytest -k s4      (uses a fake model: free and repeatable)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

REFUSAL_TEXT = "Sorry, I can't help with that one."


@dataclass
class AgentResult:
    text: str                      # what to show the user
    stopped: str                   # "done" | "max_tokens" | "refusal" | "step_limit"
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)   # (name, input), in call order


def run_turn(
    messages: list[dict[str, Any]],
    call_model: Callable[[list[dict[str, Any]]], Any],
    execute_tool: Callable[[str, dict[str, Any]], tuple[str, bool]],
    max_steps: int = 8,
    finish: Callable[[list[tuple[str, dict[str, Any], str, bool]]], str | None] | None = None,
) -> AgentResult:
    """Run one user turn to the end. `messages` already ends with the user's message.

    Inputs:
    - call_model(messages) -> a response with .stop_reason and .content (a list of blocks)
    - each block has .type: "text" (.text), "tool_use" (.id, .name, .input), "thinking", or others
    - execute_tool(name, input) -> (content_str, is_error)      (in the real app: tools.run_tool)
    Passing these two in as functions is what lets the tests use a fake model.

    Each step:
    1. response = call_model(messages)                     (counts as 1 step)
    2. unless stop_reason is "refusal": append {"role": "assistant", "content": response.content}
       - append content UNCHANGED (the whole list, thinking blocks too). The API needs them
         back exactly as it sent them; don't rebuild the list or keep only the text.
    3. then by stop_reason:
       - "end_turn"   -> return AgentResult(text, "done", tool_calls)
       - "max_tokens" -> return AgentResult(text, "max_tokens", tool_calls)
       - "refusal"    -> return AgentResult(REFUSAL_TEXT, "refusal", tool_calls)   (nothing appended)
       - "pause_turn" -> go to the next step (no tool results to send)
       - "tool_use"   -> for EACH tool_use block, in order:
             - add (block.name, block.input) to tool_calls
             - content, is_error = execute_tool(block.name, block.input)
             - if execute_tool RAISES: content = f"Error: {e}", is_error = True  (never crash the chat)
             - make {"type": "tool_result", "tool_use_id": block.id, "content": content, "is_error": is_error}
         then append ONE message {"role": "user", "content": [all the tool_result dicts]}
         and go to the next step
       - then, if finish is given: text = finish([(name, input, content, is_error), ...] of this step). Some text ->
         append {"role": "assistant", "content": text} and return it as "done" (the tool's answer is the reply)
    4. after max_steps model calls without finishing -> return AgentResult("", "step_limit", tool_calls)

    Words: "text" = the .text of every "text" block in response.content, joined with "\\n".

    Example (one tool round trip), messages afterwards:
        [user "what's today's workout?",
         assistant [text "Checking.", tool_use t1 get_context],
         user [tool_result t1 ...],
         assistant [text "Full body A."]]
    """
    tool_calls: list[tuple[str, dict[str, Any]]] = []

    for _ in range(max_steps):
        response = call_model(messages)

        # Refusal: the model declined. Add nothing to the history and stop.
        if response.stop_reason == "refusal":
            return AgentResult(REFUSAL_TEXT, "refusal", tool_calls)

        # Keep the reply EXACTLY as it came back (thinking blocks included): the API needs it unchanged.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return AgentResult(_text(response.content), "done", tool_calls)
        if response.stop_reason == "max_tokens":
            return AgentResult(_text(response.content), "max_tokens", tool_calls)
        if response.stop_reason == "pause_turn":
            continue                        # nothing to run; just ask the model to carry on

        # "tool_use": run every tool the model asked for, in order.
        results, step = [], []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls.append((block.name, block.input))
            try:
                content, is_error = execute_tool(block.name, block.input)
            except Exception as e:          # a crashing tool must not crash the chat
                content, is_error = f"Error: {e}", True
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": content, "is_error": is_error})
            step.append((block.name, block.input, content, is_error))

        # ALL results go back in ONE user message (separate messages teach the model to stop calling tools in parallel).
        messages.append({"role": "user", "content": results})

        # The tool's answer may already be the whole reply: then skip the model call that would only copy it.
        text = finish(step) if finish else None
        if text:
            messages.append({"role": "assistant", "content": text})
            return AgentResult(text, "done", tool_calls)

    # Ran out of steps while the model was still calling tools: stop, to cap cost and latency.
    return AgentResult("", "step_limit", tool_calls)


def _text(content: list[Any]) -> str:
    """The text the user should see: every text block, joined. Skips thinking and tool_use blocks."""
    return "\n".join(block.text for block in content if block.type == "text")
