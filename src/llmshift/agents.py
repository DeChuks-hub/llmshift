"""Agents: multi-step runs (model calls plus tool results) as test cases.

Recording: wrap your client with `llmshift.record.wrap` and run each task inside
`with agent_run("refund-flow"):` so its calls are linked. Logs without run ids are
linked too, when each request continues the previous one with its tool results.

Replay: the new model starts from the run's first request. Whenever it calls a tool
it gets the result recorded for that call, so both models see the same world and no
real tool runs. The `agent` check compares the steps (missing, extra or changed tool
calls, loops, running out of steps); the usual checks compare the final answer.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from . import messages as msgs
from .providers.base import Response
from .textutil import json_loads, stable_hash

LOOP_REPEATS = 3  # the same call this many times (and more than the old model) is a loop


def canonical_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json_loads(args) if args.strip() else {}
        except (ValueError, RecursionError):
            return args.strip()
    return json.dumps(args if args is not None else {}, sort_keys=True, ensure_ascii=False, default=str)


def call_key(call: dict) -> tuple[str, str]:
    return str(call.get("name") or ""), canonical_args(call.get("arguments"))


def describe_call(call: dict, limit: int = 60) -> str:
    """lookup_order(order_id=4521)"""
    args = call.get("arguments")
    if isinstance(args, dict):
        inner = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v}" for k, v in args.items())
    else:
        inner = str(args or "")
    inner = inner if len(inner) <= limit else inner[: limit - 3] + "..."
    return f"{call.get('name')}({inner})"


def trace(steps: list[dict]) -> list[str]:
    """One short line per step: the tool calls made, or the final answer."""
    out = []
    for step in steps or []:
        calls = step.get("tool_calls") or []
        out.append(", ".join(describe_call(c) for c in calls) if calls else "answer")
    return out


# ------------------------------------------------------------- linking logs


def _continuation_split(messages: list[dict]) -> int | None:
    """For a request that ends [..., assistant with tool calls, tool, tool], the index of
    that assistant turn; None for anything else (a new user turn is a new task)."""
    i = len(messages)
    while i > 0 and messages[i - 1].get("role") == "tool":
        i -= 1
    if i == len(messages) or i == 0:
        return None
    turn = messages[i - 1]
    if turn.get("role") != "assistant" or not turn.get("tool_calls"):
        return None
    return i - 1


def _key(messages: list[dict]) -> str:
    return stable_hash(messages, 24)


def _run_id(inter: Any) -> str | None:
    run = getattr(inter, "agent_run", None)
    return str(run.get("id")) if isinstance(run, dict) and run.get("id") is not None else None


def link_runs(interactions: list) -> list:
    """Merge chains of agent steps into one interaction per run: the first step, carrying
    the recorded `agent` steps, the final answer as output and the summed usage. Chains
    without a final answer, and everything else, are returned unchanged."""
    by_key: dict[str, list[int]] = {}
    for i, inter in enumerate(interactions):
        by_key.setdefault(_key(inter.messages), []).append(i)
    parent: dict[int, int] = {}
    child: dict[int, int] = {}
    for j, inter in enumerate(interactions):
        split = _continuation_split(inter.messages)
        if split is None:
            continue
        wanted = Counter(tc.get("name") for tc in inter.messages[split].get("tool_calls") or [])
        for i in by_key.get(_key(inter.messages[:split]), []):
            prev = interactions[i]
            if i == j or i in child or Counter(tc.get("name") for tc in prev.tool_calls or []) != wanted:
                continue
            if _run_id(prev) is not None and _run_id(inter) is not None and _run_id(prev) != _run_id(inter):
                continue
            parent[j], child[i] = i, j
            break
    out, consumed = [], set()
    for i, inter in enumerate(interactions):
        if i in consumed:
            continue
        if i in parent or i not in child:
            if i not in parent:
                out.append(inter)
            continue
        chain = [i]
        while chain[-1] in child:
            chain.append(child[chain[-1]])
        steps = _steps([interactions[k] for k in chain])
        if steps[-1]["tool_calls"]:  # the log ends mid-task: keep the steps as they were
            continue
        consumed.update(chain)
        out.append(_merge([interactions[k] for k in chain], steps))
    # Members of incomplete chains are kept as ordinary interactions.
    linked = {id(x) for x in out}
    for i, inter in enumerate(interactions):
        if i not in consumed and id(inter) not in linked and i in parent:
            out.append(inter)
    return out


def _steps(chain: list) -> list[dict]:
    steps = []
    for n, inter in enumerate(chain):
        step: dict = {"text": inter.output or "", "tool_calls": [c for c in (msgs.normalize_tool_call(tc) for tc in inter.tool_calls or []) if c]}
        if n + 1 < len(chain):
            nxt = chain[n + 1].messages
            split = _continuation_split(nxt)
            sent = nxt[split]  # the assistant turn as the app sent it back, with the call ids it used
            step["tool_calls"] = [c for c in (msgs.normalize_tool_call(tc) for tc in sent.get("tool_calls") or []) if c]
            step["results"] = [
                {"tool_call_id": m.get("tool_call_id"), "name": m.get("name"), "content": msgs.content_text(m.get("content"))}
                for m in nxt[split + 1:]
            ]
        steps.append(step)
    return steps


def _merge(chain: list, steps: list[dict]) -> Any:
    head = copy.copy(chain[0])
    last = chain[-1]
    head.output = last.output or ""
    head.tool_calls = []
    head.finish_reason = last.finish_reason
    head.error = next((x.error for x in chain if x.error), None)
    usage: dict = {}
    for key in ("input_tokens", "output_tokens"):
        values = [x.usage.get(key) for x in chain if isinstance(x.usage, dict)]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            usage[key] = int(sum(values))
    head.usage = usage
    head.metadata = dict(head.metadata or {})
    head.metadata["agent_steps"] = len(steps)
    head.agent = {"steps": steps}
    return head


# ------------------------------------------------------------------ replay


def recorded_results(agent: dict) -> list[dict]:
    out = []
    for step in (agent or {}).get("steps") or []:
        results = step.get("results") or []
        by_id = {r.get("tool_call_id"): r for r in results if r.get("tool_call_id")}
        for i, call in enumerate(step.get("tool_calls") or []):
            res = by_id.get(call.get("id")) or (results[i] if i < len(results) else None)
            if res is not None:
                out.append({"name": call.get("name"), "args": canonical_args(call.get("arguments")), "content": res.get("content", "")})
    return out


def lookup(recorded: list[dict], call: dict, used: set) -> tuple[str, str]:
    """The recorded result for a call: the same call, else the same tool's next unused
    result, else a stub saying nothing was recorded."""
    name, args = call_key(call)
    for i, r in enumerate(recorded):
        if r["name"] == name and r["args"] == args:
            used.add(i)  # taken: a call with other arguments gets the next unused result
            return r["content"], "recorded"
    same = [i for i, r in enumerate(recorded) if r["name"] == name]
    fresh = [i for i in same if i not in used] or same
    if fresh:
        used.add(fresh[0])
        return recorded[fresh[0]]["content"], "same tool, other arguments"
    return json.dumps({"error": "no recorded result for this tool call (llmshift replay)"}), "unrecorded"


def max_steps_for(agent: dict) -> int:
    recorded = len((agent or {}).get("steps") or [])
    limit = agent.get("max_steps") if isinstance(agent, dict) else None
    return int(limit) if isinstance(limit, int) and limit > 0 else max(6, 2 * recorded + 2)


def run_agent(provider: Any, case: Any, request: dict) -> Response:
    """Replay one recorded run on `provider`; ProviderError propagates like a single call."""
    agent = case.agent or {}
    recorded = recorded_results(agent)
    messages = copy.deepcopy(request.get("messages") or [])
    params = request.get("params") or {}
    steps: list[dict] = []
    used: set = set()
    totals = {"input_tokens": 0, "output_tokens": 0}
    known = True
    latency = 0.0
    last: Response | None = None
    for n in range(max_steps_for(agent)):
        resp = provider.complete({"messages": messages, "params": params})
        last = resp
        latency += resp.latency_ms or 0.0
        for key in totals:
            value = (resp.usage or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)
            else:
                known = False
        calls = []
        for i, call in enumerate(resp.tool_calls or []):
            call = dict(call)
            call["id"] = call.get("id") or f"call_{n}_{i}"
            calls.append(call)
        step: dict = {"text": resp.text or "", "tool_calls": calls}
        steps.append(step)
        if not calls:
            break
        messages.append({"role": "assistant", "content": resp.text or "", "tool_calls": calls})
        step["results"] = []
        for call in calls:
            content, how = lookup(recorded, call, used)
            step["results"].append({"tool_call_id": call["id"], "name": call.get("name"), "content": content, "replay": how})
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call.get("name"), "content": content})
    finished = bool(steps) and not steps[-1]["tool_calls"]
    return Response(
        text=(last.text or "") if (last is not None and finished) else "",
        model=last.model if last else None,
        finish_reason=(last.finish_reason if last and finished else "step_limit"),
        usage=totals if known else {},
        latency_ms=round(latency, 1),
        fingerprint=last.fingerprint if last else None,
        refusal=last.refusal if last and finished else None,
        steps=steps,
    )


# ------------------------------------------------------------------- check


def compare_steps(base_steps: list[dict], new_steps: list[dict]) -> tuple[list[str], list[str], dict]:
    """(failures, changes, details) between a recorded run and a replay."""
    from .checks import _pair_calls

    base_calls = [c for s in base_steps for c in s.get("tool_calls") or []]
    new_calls = [c for s in new_steps for c in s.get("tool_calls") or []]
    failures: list[str] = []
    changes: list[str] = []
    base_counts = Counter(call_key(c) for c in base_calls)
    looped = [(k, n) for k, n in Counter(call_key(c) for c in new_calls).items() if n >= LOOP_REPEATS and n > base_counts.get(k, 0)]
    for (name, _), n in looped:
        example = next(c for c in new_calls if c.get("name") == name)
        failures.append(f"looped: called {describe_call(example)} {n} times")
    if new_steps and new_steps[-1].get("tool_calls"):
        failures.append(f"stopped after {len(new_steps)} steps without a final answer")
    base_names = Counter(c.get("name") for c in base_calls)
    new_names = Counter(c.get("name") for c in new_calls)
    missing = base_names - new_names
    extra = new_names - base_names
    looped_names = {name for (name, _), _ in looped}
    if missing:
        failures.append("missing tool call(s): " + ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(missing.items())))
    extra_shown = {n: c for n, c in extra.items() if n not in looped_names}
    if extra_shown:
        changes.append("extra tool call(s): " + ", ".join(f"{n} x{c}" if c > 1 else n for n, c in sorted(extra_shown.items())))
    for name in sorted(base_names & new_names):
        olds = [c for c in base_calls if c.get("name") == name]
        news = [c for c in new_calls if c.get("name") == name]
        # Calls of one step run in parallel, in any order: pair them by their arguments.
        failures.extend(_pair_calls(name, olds, news, "auto"))

    def order(steps: list[dict]) -> list[list[str]]:  # step by step; within a step the order is free
        return [sorted(str(c.get("name")) for c in s.get("tool_calls") or []) for s in steps if s.get("tool_calls")]

    if not missing and not extra and order(base_steps) != order(new_steps):
        changes.append("tools called in a different order: " + " -> ".join(str(c.get("name")) for c in new_calls))
    unrecorded = sum(1 for s in new_steps for r in s.get("results") or [] if r.get("replay") == "unrecorded")
    if unrecorded:
        changes.append(f"{unrecorded} call(s) had no recorded result, so the model saw a placeholder")
    details = {
        "baseline_steps": len(base_steps),
        "steps": len(new_steps),
        "missing": sorted(missing.elements()),
        "extra": sorted(extra.elements()),
        "looped": [name for (name, _), _ in looped],
        "unfinished": bool(new_steps and new_steps[-1].get("tool_calls")),
        "unrecorded": unrecorded,
        "trace": trace(new_steps),
        "baseline_trace": trace(base_steps),
    }
    return failures, changes, details


def c_agent(check: dict, sample: Any, case: Any, ctx: Any) -> Any:
    from .checks import FAIL, PASS, Outcome

    base_steps = (case.agent or {}).get("steps") or []
    new_steps = getattr(sample, "steps", None) or []
    if not base_steps:
        return Outcome("error", "the agent check needs a recorded run (an 'agent' with steps in the test case)")
    if not new_steps:
        return Outcome("error", "no agent steps in the answer; the case was not replayed as an agent run")
    failures, changes, details = compare_steps(base_steps, new_steps)
    steps_note = f"{len(base_steps)} -> {len(new_steps)} steps"
    if failures:
        return FAIL("; ".join(failures[:4]) + f" ({steps_note})", score=0.0, **details)
    if changes:
        return FAIL("; ".join(changes[:3]) + f" ({steps_note})", score=0.5, soft=True, **details)
    calls = sum(len(s.get("tool_calls") or []) for s in base_steps)
    return PASS(f"same tool calls ({calls}) in {len(new_steps)} steps", **details)
