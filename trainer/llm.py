"""PROVIDED · the only file that talks to the Claude API.

Everything else gets a plain `call_model(messages)` function, so the loop can be tested
with a fake and the provider details stay in one place.
"""
from __future__ import annotations

from typing import Any

import anthropic

MODEL = "claude-opus-5-5"


class UsageLog:
    """Adds up tokens across calls, so you can quote real numbers ("a turn costs ~N tokens")."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def __str__(self) -> str:
        return (f"{self.calls} model calls · {self.input_tokens} in · {self.output_tokens} out"
                f" · {self.cache_read_tokens} cache-read")


def make_call_model(system: str, tools: list[dict[str, Any]], *, effort: str = "medium",
                    usage: UsageLog | None = None, client: anthropic.Anthropic | None = None):
    """Returns call_model(messages) -> response, with everything but `messages` fixed.

    - effort: "low" | "medium" | "high" ... how hard the model thinks. Opus 5.5 defaults to medium.
    - fallbacks="default": if a safety classifier declines the request, the API retries it on
      another model inside the same call, so a false positive doesn't end the conversation.
      (Needs the beta endpoint and the beta flag below.)
    """
    client = client or anthropic.Anthropic()

    def call_model(messages: list[dict[str, Any]]):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            **({"tools": tools} if tools else {}),
            messages=messages,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if usage is not None:
            usage.add(response.usage)
        return response

    return call_model
