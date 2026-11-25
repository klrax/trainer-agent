"""One user message, start to finish: the single path the chat and the evals both use.

    user message ─► pain? ──yes──► fixed reply, no model call          (safety.py)
                      │no
                      ▼
                  agent loop ─► nutrition tool answered everything? ──► print the tool's own answer   (direct.py)
                      └──────► otherwise the model writes the reply
"""
from __future__ import annotations

from typing import Any, Callable

from .agent import AgentResult, run_turn
from .direct import direct_reply
from .safety import safety_reply


def respond(messages: list[dict[str, Any]], call_model: Callable, execute_tool: Callable, *, direct: bool = False,
            max_steps: int = 8) -> AgentResult:
    """Like run_turn (messages already ends with the user's message), plus the safety check and direct replies.
    direct: print nutrition tool answers without another model call (worth it for slow local models)."""
    user = messages[-1]["content"]
    earlier = [m["content"] for m in messages[:-1] if m["role"] == "user" and isinstance(m["content"], str)]
    fixed = safety_reply(user, earlier)
    if fixed:
        messages.append({"role": "assistant", "content": fixed})
        return AgentResult(fixed, "done")
    finish = (lambda calls: direct_reply(user, calls)) if direct else None
    return run_turn(messages, call_model, execute_tool, max_steps, finish)
