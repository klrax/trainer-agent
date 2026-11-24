"""The Ollama adapter. Pure functions, so no Ollama needed to run these."""
import json
from types import SimpleNamespace as NS

from trainer.llm_ollama import from_openai, to_openai_messages, to_openai_tools

TOOL = {"name": "get_session_for", "description": "Session for a date.",
        "input_schema": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}}


def test_s4b_tools():
    assert to_openai_tools([TOOL]) == [{"type": "function", "function": {
        "name": "get_session_for", "description": "Session for a date.", "parameters": TOOL["input_schema"]}}]
    assert to_openai_tools([]) == []


def test_s4b_messages_round_trip_shape():
    history = [
        {"role": "user", "content": "what's today?"},
        {"role": "assistant", "content": [NS(type="thinking", thinking="", signature="s"),
                                          NS(type="text", text="Checking."),
                                          NS(type="tool_use", id="c1", name="get_context", input={}),
                                          NS(type="tool_use", id="c2", name="get_session_for", input={"date": "2026-11-23"})]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": '{"date": "2026-11-23"}', "is_error": False},
            {"type": "tool_result", "tool_use_id": "c2", "content": "Error: no profile yet", "is_error": True}]},
        {"role": "assistant", "content": [NS(type="text", text="Rest day.")]},
    ]
    out = to_openai_messages("SYS", history)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "what's today?"}
    assert out[2]["role"] == "assistant" and out[2]["content"] == "Checking."      # thinking dropped
    assert out[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "get_context", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "get_session_for", "arguments": json.dumps({"date": "2026-11-23"})}}]
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"date": "2026-11-23"}'}   # one message PER result
    assert out[4] == {"role": "tool", "tool_call_id": "c2", "content": "Error: no profile yet"}
    assert out[5] == {"role": "assistant", "content": "Rest day."}                  # no "tool_calls" key at all
    assert len(out) == 6


def test_s4b_plain_text_reply():
    r = from_openai({"role": "assistant", "content": "  Full body A today.  "}, "stop")
    assert r.stop_reason == "end_turn"
    assert [(b.type, b.text) for b in r.content] == [("text", "Full body A today.")]
    r = from_openai({"role": "assistant", "content": "Here is your pla"}, "length")
    assert r.stop_reason == "max_tokens"
    r = from_openai({"role": "assistant", "content": "<think>\nuser wants plan\n</think>\n\nPlan: squat."}, "stop")
    assert [b.text for b in r.content] == ["Plan: squat."]                          # <think> section removed


def test_s4b_tool_calls():
    msg = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_x", "type": "function", "function": {"name": "get_session_for", "arguments": '{"date": "2026-11-23"}'}},
        {"type": "function", "function": {"name": "get_context", "arguments": {}}},          # no id, args already a dict
        {"id": "call_z", "type": "function", "function": {"name": "log_meal", "arguments": "{oops"}}]}
    r = from_openai(msg, "stop")                       # some local servers say "stop" even with tool calls
    assert r.stop_reason == "tool_use"
    assert [b.type for b in r.content] == ["tool_use"] * 3                         # empty text -> no text block
    a, b, c = r.content
    assert (a.id, a.name, a.input) == ("call_x", "get_session_for", {"date": "2026-11-23"})
    assert (b.id, b.name, b.input) == ("call_1", "get_context", {})                  # made-up id: "call_<index>"
    assert c.input == {"_invalid_json": "{oops"}                                   # don't crash; run_tool will report it


def test_s4b_works_with_your_loop():
    """The adapter's output must drive YOUR run_turn unchanged."""
    from trainer.agent import run_turn
    replies = [from_openai({"content": None, "tool_calls": [
                   {"id": "t1", "function": {"name": "get_context", "arguments": "{}"}}]}, "tool_calls"),
               from_openai({"content": "Monday: Full body A."}, "stop")]
    msgs = [{"role": "user", "content": "today?"}]
    r = run_turn(msgs, lambda m: replies.pop(0), lambda n, a: ('{"weekday": "Mon"}', False))
    assert r.text == "Monday: Full body A." and r.tool_calls == [("get_context", {})]
    assert to_openai_messages("S", msgs)[-2] == {"role": "tool", "tool_call_id": "t1", "content": '{"weekday": "Mon"}'}
