"""Agent runs: recording, linking logs into runs, replay with recorded tool results,
and the step comparison."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from llmshift import cli
from llmshift.agents import compare_steps, link_runs, lookup, recorded_results, run_agent, trace
from llmshift.compare import run_compare
from llmshift.logparse import parse_logs
from llmshift.providers import resolve
from llmshift.providers.mock import MockProvider
from llmshift.record import agent_run, wrap
from llmshift.testset import TestCase, TestFileError, load_tests

SYSTEM = "You are Acme's support agent. Look orders up before answering."
TOOLS = [{"type": "function", "function": {"name": "lookup_order", "description": "Find an order by id", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
ORDERS = {
    "4521": {"order_id": "4521", "status": "shipped", "eta": "March 14"},
    "7788": {"order_id": "7788", "status": "delivered", "eta": "March 2"},
    "9001": {"order_id": "9001", "status": "pending", "eta": "unknown"},
}


class _Completions:
    """An OpenAI-style client backed by a simulated model."""

    def __init__(self, spec: str):
        self.provider = resolve(spec)

    def create(self, **kw):
        resp = self.provider.complete({"messages": kw["messages"], "params": {"tools": kw.get("tools")}})
        message: dict = {"role": "assistant", "content": resp.text or None}
        if resp.tool_calls:
            message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in resp.tool_calls]
        return {"model": "gpt-4o-2024-08-06", "choices": [{"index": 0, "message": message, "finish_reason": resp.finish_reason}],
                "usage": {"prompt_tokens": resp.usage["input_tokens"], "completion_tokens": resp.usage["output_tokens"]}}


class _Chat:
    def __init__(self, spec: str):
        self.completions = _Completions(spec)


class FakeClient:
    def __init__(self, spec: str = "mock:legacy"):
        self.chat = _Chat(spec)


def support_agent(client, question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for _ in range(6):
        out = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = out["choices"][0]["message"]
        if not msg.get("tool_calls"):
            return msg["content"]
        messages.append(msg)
        for tc in msg["tool_calls"]:
            order = ORDERS[json.loads(tc["function"]["arguments"])["order_id"]]
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(order)})
    raise AssertionError("agent did not finish")


QUESTIONS = ["Where is my order #4521?", "Has order #7788 arrived?", "What is happening with order #9001?"]


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "logs" / "agent.jsonl"
    client = wrap(FakeClient(), path=log)
    for q in QUESTIONS:
        with agent_run("support") as run_id:
            assert run_id
            support_agent(client, q)
    assert cli.main(["snapshot", "logs"]) == 0
    return load_tests(".llmshift/tests.jsonl")


def test_recorded_runs_become_one_case_each(recorded):
    assert len(recorded) == 3 and all(c.agent for c in recorded)
    case = next(c for c in recorded if "4521" in c.user_text)
    steps = case.agent["steps"]
    assert [s["tool_calls"][0]["name"] if s["tool_calls"] else "answer" for s in steps] == ["lookup_order", "answer"]
    assert json.loads(steps[0]["results"][0]["content"]) == ORDERS["4521"]
    assert steps[0]["results"][0]["tool_call_id"] == steps[0]["tool_calls"][0]["id"]
    assert "shipped" in case.baseline_output and case.checks[0]["type"] == "agent"
    assert not any(c["type"] == "tool_calls" for c in case.checks)
    assert case.group.endswith("-agent") and case.baseline["usage"]["input_tokens"] > 0
    assert TestCase.from_dict(case.to_dict()).agent == case.agent  # round trip through the test file


def test_replay_on_the_same_model_reproduces_the_run(recorded):
    res = run_compare(recorded, "mock:legacy", show_progress=False)
    assert res.summary["passed"] == 3, [(c.id, [r.message for r in c.failing]) for c in res.cases]
    agent_checks = [r for c in res.cases for r in c.checks if r.type == "agent"]
    assert all(r.status == "pass" and "same tool calls (1) in 2 steps" in r.message for r in agent_checks)
    assert res.cases[0].trace == res.cases[0].baseline_trace


def test_replay_finds_extra_calls_loops_and_cost_per_task(recorded):
    res = run_compare(recorded, "mock:nextgen", show_progress=False)
    by_order = {next(o for o in ORDERS if o in (c.input_preview or "")): c for c in res.cases}
    shipped = next(r for r in by_order["4521"].checks if r.type == "agent")
    assert shipped.status == "fail" and shipped.severity == "warn" and "extra tool call(s): lookup_order" in shipped.message
    assert by_order["4521"].trace == ["lookup_order(order_id=4521)", "lookup_order(order_id=4521)", "answer"]
    pending = by_order["9001"]
    loop = next(r for r in pending.checks if r.type == "agent")
    assert pending.status == "fail" and loop.severity == "error"
    assert "looped" in loop.message and "without a final answer" in loop.message
    agent = res.summary["agent"]
    assert agent["tasks"] == 3 and agent["out_of_steps"] == 1 and agent["looped"] == 1
    assert agent["avg_steps"]["baseline"] == 2.0 and agent["avg_steps"]["candidate"] > 2.0
    assert agent["cost_per_task"]["baseline"] and agent["cost_per_task"]["candidate"]
    assert any(n.startswith("Agent tasks (3)") and "cost per task" in n and "looped" in n for n in res.insights)


def test_terminal_and_html_show_the_steps(recorded, capsys):
    cli.main(["compare", "--to", "mock:nextgen", "--show", "all", "--html", "r.html"])
    out = capsys.readouterr().out
    assert "Steps (old): lookup_order(order_id=4521)" in out and "Steps (new):" in out
    assert "<th>Old model</th><th>New model</th>" in Path("r.html").read_text(encoding="utf-8")


def test_replays_are_cached(recorded, tmp_path):
    from llmshift.cache import ResponseCache

    calls = {"n": 0}
    legacy = resolve("mock:legacy")

    def counting(request):
        calls["n"] += 1
        return legacy.complete(request)

    cache = ResponseCache(tmp_path / "c")
    run_compare(recorded, MockProvider("counting", fn=counting), cache=cache, show_progress=False)
    first = calls["n"]
    run_compare(recorded, MockProvider("counting", fn=counting), cache=cache, show_progress=False)
    assert first == 6 and calls["n"] == first


def test_logs_without_run_ids_are_linked_by_their_messages(tmp_path):
    client = wrap(FakeClient(), path=tmp_path / "plain.jsonl")
    support_agent(client, "Where is my order #4521?")
    lines = [json.loads(x) for x in (tmp_path / "plain.jsonl").read_text(encoding="utf-8").splitlines()]
    for rec in lines:
        rec.pop("agent_run", None)  # as from any other logger
    plain = tmp_path / "logs.jsonl"
    plain.write_text("\n".join(json.dumps({"request": r["request"], "response": r["response"]}) for r in lines) + "\n", encoding="utf-8")
    inters, _ = parse_logs([str(plain)])
    linked = link_runs(inters)
    assert len(inters) == 2 and len(linked) == 1 and linked[0].agent and "shipped" in linked[0].output
    # A log that stops mid-task (no final answer) is left as it was.
    assert len(link_runs(inters[:1])) == 1 and link_runs(inters[:1])[0].agent is None


def test_run_ids_keep_separate_runs_apart(tmp_path):
    client = wrap(FakeClient(), path=tmp_path / "a.jsonl")
    with agent_run(run_id="one"):
        support_agent(client, "Where is my order #4521?")
    with agent_run(run_id="two"):
        support_agent(client, "Where is my order #4521?")  # identical messages, different run
    inters, _ = parse_logs([str(tmp_path / "a.jsonl")])
    linked = link_runs(inters)
    assert len(linked) == 2 and all(x.agent for x in linked)
    assert [x.agent_run["id"] for x in linked] == ["one", "two"]


def test_agent_run_follows_async_tasks(tmp_path):
    class AsyncCompletions:
        def __init__(self):
            self.sync = _Completions("mock:legacy")

        def create(self, **kw):
            async def go():
                return self.sync.create(**kw)

            return go()

    class AsyncClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": AsyncCompletions()})()

    client = wrap(AsyncClient(), path=tmp_path / "async.jsonl")

    async def task(q):
        await client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": q}], tools=TOOLS)

    async def main():
        with agent_run(run_id="async-run"):
            await asyncio.gather(task("Where is #4521?"), task("Where is #7788?"))

    asyncio.run(main())
    recs = [json.loads(x) for x in (tmp_path / "async.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["agent_run"]["id"] for r in recs] == ["async-run", "async-run"]
    assert sorted(r["agent_run"]["step"] for r in recs) == [1, 2]


def test_lookup_and_compare_steps_helpers():
    agent = {"steps": [{"text": "", "tool_calls": [{"id": "a", "name": "search", "arguments": {"q": "x"}}], "results": [{"tool_call_id": "a", "content": "R1"}]}, {"text": "done", "tool_calls": []}]}
    rec = recorded_results(agent)
    used: set = set()
    assert lookup(rec, {"name": "search", "arguments": {"q": "x"}}, used) == ("R1", "recorded")
    assert lookup(rec, {"name": "search", "arguments": {"q": "y"}}, used) == ("R1", "same tool, other arguments")
    assert lookup(rec, {"name": "other", "arguments": {}}, used)[1] == "unrecorded"
    new = [{"text": "", "tool_calls": [{"name": "search", "arguments": {"q": "z"}}], "results": [{"replay": "same tool, other arguments"}]}, {"text": "done", "tool_calls": []}]
    failures, changes, details = compare_steps(agent["steps"], new)
    assert failures == ['search.q: "x" -> "z"'] and not changes and details["trace"] == ["search(q=z)", "answer"]
    assert trace(agent["steps"]) == ["search(q=x)", "answer"]


def test_agent_definitions_are_validated():
    base = {"id": "a", "input": {"messages": [{"role": "user", "content": "hi"}]}}
    for bad in ({"steps": []}, {"steps": "x"}, {"steps": [1]}, {"steps": [{"text": "a"}], "max_steps": 0}, {"steps": [{"results": ["x"]}]}):
        with pytest.raises(TestFileError):
            TestCase.from_dict({**base, "agent": bad})
    ok = TestCase.from_dict({**base, "agent": {"steps": [{"text": "", "tool_calls": [{"name": "f", "arguments": "{\"a\": 1}"}], "results": [{"content": {"x": 1}}]}, {"text": "done"}], "max_steps": 4}})
    assert ok.agent["steps"][0]["tool_calls"][0]["arguments"] == {"a": 1} and ok.agent["steps"][0]["results"][0]["content"] == '{"x": 1}'


def test_errors_mid_run_are_case_errors(recorded):
    calls = {"n": 0}
    legacy = resolve("mock:legacy")

    def fails_second(request):
        calls["n"] += 1
        if request["messages"][-1]["role"] == "tool":
            from llmshift.providers.base import ProviderError

            raise ProviderError("upstream timeout", code="timeout")
        return legacy.complete(request)

    res = run_compare(recorded, MockProvider("flaky", fn=fails_second), show_progress=False)
    assert res.summary["errors"] == 3 and all("upstream timeout" in c.error for c in res.cases)
    assert run_agent(resolve("mock:legacy"), recorded[0], {"messages": recorded[0].messages, "params": recorded[0].params}).finish_reason == "stop"
