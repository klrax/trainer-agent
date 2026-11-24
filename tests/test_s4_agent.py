"""The agent loop is tested with a FAKE model: no API key, no cost, fully repeatable.

A fake response looks like the SDK's: .stop_reason and .content (a list of blocks with .type).
"""
from types import SimpleNamespace as NS

from trainer.agent import REFUSAL_TEXT, run_turn


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, **input):
    return NS(type="tool_use", id=id, name=name, input=input)


def thinking():
    return NS(type="thinking", thinking="", signature="sig")


def fake_model(*responses):
    """Returns call_model(messages) that replays responses in order and records what it was sent."""
    queue = list(responses)
    seen = []

    def call_model(messages):
        seen.append([dict(m) for m in messages])
        return queue.pop(0)

    call_model.seen = seen
    return call_model


def tools_ok(name, args):
    return f"result of {name}", False


def test_s4_no_tools_just_answers():
    msgs = [{"role": "user", "content": "hi"}]
    model = fake_model(NS(stop_reason="end_turn", content=[thinking(), text("Hello!"), text("Ready?")]))
    r = run_turn(msgs, model, tools_ok)
    assert (r.text, r.stopped, r.tool_calls) == ("Hello!\nReady?", "done", [])
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"][0].type == "thinking"      # content appended UNCHANGED, thinking kept


def test_s4_one_tool_round_trip():
    msgs = [{"role": "user", "content": "what's today's workout?"}]
    first = NS(stop_reason="tool_use", content=[text("Checking."), tool_use("t1", "get_context")])
    model = fake_model(first, NS(stop_reason="end_turn", content=[text("Full body A.")]))
    r = run_turn(msgs, model, tools_ok)
    assert r.text == "Full body A." and r.stopped == "done"
    assert r.tool_calls == [("get_context", {})]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "t1",
                                   "content": "result of get_context", "is_error": False}]
    assert len(model.seen) == 2 and model.seen[1][-1]["role"] == "user"   # 2nd call saw the result


def test_s4_parallel_calls_go_back_in_ONE_message():
    msgs = [{"role": "user", "content": "plan today"}]
    first = NS(stop_reason="tool_use", content=[tool_use("a", "get_context"),
                                                tool_use("b", "get_profile")])
    model = fake_model(first, NS(stop_reason="end_turn", content=[text("ok")]))
    run_turn(msgs, model, tools_ok)
    results = msgs[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]      # same order as the calls
    assert msgs[3]["role"] == "assistant"                         # not two separate user messages


def test_s4_tool_errors_and_crashes_are_reported_not_raised():
    def tools(name, args):
        if name == "boom":
            raise RuntimeError("disk full")
        return "Error: no profile yet", True

    msgs = [{"role": "user", "content": "x"}]
    first = NS(stop_reason="tool_use", content=[tool_use("a", "get_daily_targets"), tool_use("b", "boom")])
    model = fake_model(first, NS(stop_reason="end_turn", content=[text("I need your details first.")]))
    r = run_turn(msgs, model, tools)
    assert r.stopped == "done"
    a, b = msgs[2]["content"]
    assert a["is_error"] is True and a["content"] == "Error: no profile yet"
    assert b["is_error"] is True and b["content"] == "Error: disk full"


def test_s4_step_limit():
    loop_forever = [NS(stop_reason="tool_use", content=[tool_use(f"t{i}", "get_context")]) for i in range(10)]
    msgs = [{"role": "user", "content": "x"}]
    model = fake_model(*loop_forever)
    r = run_turn(msgs, model, tools_ok, max_steps=3)
    assert r.stopped == "step_limit" and len(model.seen) == 3 and len(r.tool_calls) == 3


def test_s4_max_tokens_and_pause_turn():
    msgs = [{"role": "user", "content": "x"}]
    r = run_turn(msgs, fake_model(NS(stop_reason="max_tokens", content=[text("Here is your pla")])), tools_ok)
    assert (r.text, r.stopped) == ("Here is your pla", "max_tokens")

    msgs = [{"role": "user", "content": "x"}]
    model = fake_model(NS(stop_reason="pause_turn", content=[text("...")]),
                       NS(stop_reason="end_turn", content=[text("done")]))
    r = run_turn(msgs, model, tools_ok)
    assert r.text == "done" and len(model.seen) == 2
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant"]   # paused turn kept, no tool results


def test_s4_refusal_is_not_appended():
    msgs = [{"role": "user", "content": "x"}]
    r = run_turn(msgs, fake_model(NS(stop_reason="refusal", content=[])), tools_ok)
    assert (r.text, r.stopped) == (REFUSAL_TEXT, "refusal")
    assert len(msgs) == 1
