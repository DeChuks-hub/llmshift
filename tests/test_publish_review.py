"""What the review before publishing 0.1.1 found: one bad thinking case stripped thinking from
the whole run and blamed the model wrongly, pick recommended a candidate it could not check,
a process-wide regex breaker, racing python: providers, a verdict that contradicted the exit
code, false facts failures, a retired replacement, no py.typed, long-context and 1-hour cache
prices, a Gemini retry on every case, and a PR workflow that failed on forks."""

from __future__ import annotations

import threading

import pytest

from llmshift.compare import run_compare
from llmshift.testset import TestCase

# ------------------------------------------------------------------ thinking


def _anthropic_like(server, seen):
    """Answers as the Anthropic API would, with its 400s for what a model refuses."""

    def reply(req):
        body = req["json"]
        seen.append(body)
        thinking = (body.get("thinking") or {}).get("type")
        effort = (body.get("output_config") or {}).get("effort")
        refuse = None
        if thinking in ("enabled", "adaptive") and body.get("temperature", 1) != 1:
            refuse = "temperature may only be set to 1 when thinking is enabled"
        elif thinking in ("enabled", "adaptive") and "top_k" in body:
            refuse = "top_k is not supported when thinking is enabled"
        elif thinking == "adaptive" and body["model"] == "claude-sonnet-4-5":
            refuse = '"thinking.type.adaptive" is not supported for this model'
        elif effort == "xhigh" and body["model"] == "claude-opus-4-6":
            refuse = "output_config.effort: 'xhigh' is not supported for this model"
        elif effort in ("xhigh", "max") and thinking == "disabled" and body["model"] == "claude-opus-4-7":  # a rule llmshift does not know
            refuse = f"output_config.effort '{effort}' is not supported when thinking is disabled on this model. Use effort 'high' or below, or enable thinking."
        if refuse:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": refuse}}, {}
        return 200, {"id": "m", "type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": "ok"}],
                     "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}, {}

    server.route("POST", "/v1/messages", reply)


def _case(cid, params):
    return TestCase(id=cid, input={"messages": [{"role": "user", "content": "q"}], "params": params}, baseline={"output": "ok"}, checks=[{"type": "contains", "value": "ok"}])


def _run(server, model, cases, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return run_compare(cases, f"anthropic:{model}", config={"providers": {"anthropic": {"base_url": server.url}}}, show_progress=False, concurrency=1)


def test_one_bad_thinking_case_spoils_nothing_and_is_told_the_truth(server, monkeypatch):
    seen = []
    _anthropic_like(server, seen)
    budget = {"type": "enabled", "budget_tokens": 2048}
    cases = [_case("bad", {"thinking": budget, "temperature": 0.3}), _case("good1", {"thinking": budget}), _case("good2", {"thinking": budget}),
             _case("topk", {"thinking": budget, "top_k": 5})]
    result = _run(server, "claude-sonnet-4-5", cases, monkeypatch)
    by_id = {c.id: c for c in result.cases}
    assert by_id["good1"].status == by_id["good2"].status == "pass"  # they kept their thinking
    assert all(body.get("thinking") == budget for body in seen)  # never dropped, not even for the bad cases
    bad = next(r for r in by_id["bad"].checks if r.type == "request").message
    assert "temperature=0.3" in bad and "temperature must be 1" in bad and "thinking" not in bad.split(" as this request")[0].split("rejects ")[1]
    assert "adaptive" not in bad  # Sonnet 4.5 has no adaptive thinking; nothing adaptive was sent
    assert "top_k=5" in next(r for r in by_id["topk"].checks if r.type == "request").message


def test_dynamic_thinking_on_an_older_claude_becomes_a_budget(server, monkeypatch):
    seen = []
    _anthropic_like(server, seen)
    cases = [_case("dynamic", {"thinking": {"type": "adaptive"}}), _case("budget", {"thinking": {"type": "enabled", "budget_tokens": 2048}})]
    by_id = {c.id: c for c in _run(server, "claude-sonnet-4-5", cases, monkeypatch).cases}
    assert seen[0]["thinking"]["type"] == "enabled" and seen[1]["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    message = next(r for r in by_id["dynamic"].checks if r.type == "request").message
    assert 'send {"type": "enabled", "budget_tokens": N}' in message and by_id["budget"].status == "pass"


def test_xhigh_maps_to_high_where_it_does_not_exist_and_spoils_no_other_level(server, monkeypatch):
    seen = []
    _anthropic_like(server, seen)
    cases = [_case("xhigh", {"reasoning_effort": "xhigh"}), _case("low", {"reasoning_effort": "low"})]
    by_id = {c.id: c for c in _run(server, "claude-opus-4-6", cases, monkeypatch).cases}
    assert [b["output_config"]["effort"] for b in seen] == ["high", "low"]  # no 400 spent
    assert "has no xhigh effort" in next(r for r in by_id["xhigh"].checks if r.type == "request").message
    assert by_id["low"].status == "pass"


def test_an_effort_refused_beside_thinking_off_stays_with_that_case(server, monkeypatch):
    seen = []
    _anthropic_like(server, seen)
    cases = [_case("off", {"thinking": {"type": "disabled"}, "reasoning_effort": "xhigh"}), _case("on", {"thinking": {"type": "adaptive"}, "reasoning_effort": "xhigh"})]
    by_id = {c.id: c for c in _run(server, "claude-opus-4-7", cases, monkeypatch).cases}
    assert [(b["thinking"]["type"], b["output_config"]["effort"]) for b in seen] == [("disabled", "xhigh"), ("disabled", "high"), ("adaptive", "xhigh")]
    assert "rejects reasoning_effort" in next(r for r in by_id["off"].checks if r.type == "request").message
    assert by_id["on"].status == "pass" and not [r for r in by_id["on"].checks if r.type == "request"]  # its xhigh was never refused


def test_turning_thinking_off_is_translated_per_model():
    from llmshift import logparse
    from llmshift.providers.anthropic import plan_thinking

    _, params = logparse.anthropic_request({"model": "claude-sonnet-5-5", "max_tokens": 100, "thinking": {"type": "between_tools"}, "messages": [{"role": "user", "content": "hi"}]})
    assert params["thinking"] == {"type": "between_tools"}  # kept: it is how Sonnet 5.5 turns thinking off
    sonnet = plan_thinking("claude-sonnet-5-5", {"thinking": {"type": "disabled"}})
    assert sonnet["thinking"] == {"type": "between_tools"} and sonnet["rejected"] == ["thinking"]
    assert plan_thinking("claude-sonnet-5-5", {"thinking": {"type": "between_tools"}})["rejected"] == []
    opus = plan_thinking("claude-opus-4-8", {"thinking": {"type": "between_tools"}})
    assert opus["thinking"] == {"type": "disabled"} and "Sonnet 5.5 only" in opus["advice"][0]
    fable = plan_thinking("claude-fable-5-1", {"thinking": {"type": "disabled"}})  # always a 400 on Fable
    assert fable["thinking"] is None and fable["rejected"] == ["thinking"] and "always thinks" in fable["advice"][0]


def test_gemini_does_not_retry_a_refused_thinking_config_on_every_case(server, monkeypatch):
    from llmshift.providers import resolve

    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    calls = []

    def reply(req):
        calls.append(req["json"].get("generationConfig", {}))
        if "thinkingConfig" in calls[-1]:
            return 400, {"error": {"code": 400, "message": "Budget 0 is invalid. This model only works in thinking mode.", "status": "INVALID_ARGUMENT"}}, {}
        return 200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}, {}

    server.route("POST", "/models/", reply)
    gemini = resolve("gemini:gemini-2.5-pro", {"providers": {"gemini": {"base_url": server.url}}})
    request = {"messages": [{"role": "user", "content": "hi"}], "params": {"thinking": {"type": "disabled"}}}
    for _ in range(3):
        assert gemini.complete(request).rejected == ["thinking"]
    assert len(calls) == 4  # two for the first case, then one each


# --------------------------------------------------------------------- pick


def test_pick_recommends_no_candidate_it_could_not_check(make_case, monkeypatch):
    from llmshift import userregex
    from llmshift.pick import _rate_cell, run_pick

    monkeypatch.setattr(userregex, "TIMEOUT", 0.3)
    pattern = r"^(\w+\s?)*$"
    cases = [make_case("x", user=f"q{i}", id=f"ok{i}", checks=[{"type": "regex", "pattern": pattern}]) for i in range(3)]
    cases += [make_case("x", user="a" * 60 + f"!{i}", id=f"odd{i}", checks=[{"type": "regex", "pattern": pattern}]) for i in range(2)]
    res = run_pick(cases, ["mock:echo"], min_pass_rate=0.95, concurrency=2)
    cand = res.candidates[0]
    assert cand.error is None and cand.unchecked == 2 and cand.pass_rate == 1.0  # of the three it could check
    assert not cand.meets_bar and res.recommended is None and res.inconclusive
    assert cand.low is not None and cand.low > 0.25  # the range is over the same three cases, not 8%-72%
    assert _rate_cell(cand) == "100% (3/3), 2 not checked"


def test_a_regex_breaker_lives_for_one_run_and_one_model(make_case, monkeypatch):
    from llmshift import userregex

    monkeypatch.setattr(userregex, "TIMEOUT", 0.3)
    pattern = r"^(\w+\s?)*$"
    odd = [make_case("x", user="a" * 60 + f"!{i}", id=f"odd{i}", checks=[{"type": "regex", "pattern": pattern}]) for i in range(3)]
    fine = [make_case("x", user=f"fine {i}", id=f"fine{i}", checks=[{"type": "regex", "pattern": pattern}]) for i in range(3)]
    first = run_compare(odd, "mock:echo", show_progress=False, concurrency=1).summary
    assert first["unchecked"] == 3
    second = run_compare(fine, "mock:echo", show_progress=False, concurrency=1).summary  # a later run (an MCP call) tries it again
    assert second["passed"] == 3 and second["unchecked"] == 0
    assert userregex.search(pattern, "fine words")  # outside a run, nothing is remembered


# --------------------------------------------------------- providers, reports


def test_two_python_providers_on_one_file_do_not_race(tmp_path):
    import sys

    from llmshift.providers.custom import PythonProvider

    module = tmp_path / "slowmod.py"
    module.write_text("import time\ntime.sleep(0.2)\n\ndef steady(request):\n    return 'steady'\n\ndef other(request):\n    return 'other'\n", encoding="utf-8")
    for _ in range(3):
        for name in [k for k in sys.modules if k.startswith("llmshift_user_slowmod")]:
            del sys.modules[name]
        errors = []

        def load(fn):
            try:
                PythonProvider(f"{module.as_posix()}:{fn}")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=load, args=(fn,)) for fn in ("steady", "other", "steady", "other")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors  # "has no callable 'steady'" in 4 of 6 pick runs


def test_the_headline_says_inconclusive_when_the_exit_code_does(make_case):
    from llmshift.report import compare_markdown, compare_terminal

    cases = [make_case(f"q{i}", user=f"q{i}", id=f"c{i}", checks=[{"type": "contains", "value": "q"}]) for i in range(3)]
    cases[0].checks = [{"type": "regex", "pattern": "(unclosed"}]
    result = run_compare(cases, "mock:echo", show_progress=False, min_pass_rate=0.95)
    assert result.passed  # what could be checked passed...
    assert "inconclusive: 1 not checked" in compare_terminal(result) and "meets" not in compare_terminal(result)
    assert "inconclusive: 1 not checked" in compare_markdown(result) and "meets threshold" not in compare_markdown(result)


@pytest.mark.parametrize("baseline, answer", [
    ("The meeting is at 2:30 PM.", "The meeting is at 14:30."),
    ("The temperature drops to -5 degrees.", "The temperature drops to minus 5 degrees."),
    ("It is minus 5 outside.", "It is -5 outside."),
    ("The balance is negative 30 dollars.", "The balance is -30 dollars."),
    ("20 minus 5 is 15.", "20 - 5 = 15."),  # a subtraction stays three positive numbers
])
def test_facts_read_times_and_signs_as_people_write_them(baseline, answer):
    from llmshift.checks import CheckContext, Sample, evaluate
    from llmshift.facts import extract

    case = TestCase(id="c", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": baseline}, checks=[])
    assert evaluate({"type": "facts", "facts": extract(baseline)}, Sample(text=answer), case, CheckContext()).status == "pass"


def test_a_retired_replacement_is_not_recommended():
    from llmshift.registry import Registry

    lookup = Registry.load().lookup("labs-leanstral-2603")
    assert lookup.replacement is None
    assert "its listed replacement, labs-leanstral-1-5, retired too on 2026-09-30, and no successor is listed" in lookup.message
    chain = Registry([
        {"id": "a-1", "provider": "x", "retires": "2026-01-01", "replacement": "a-2"},
        {"id": "a-2", "provider": "x", "retires": "2026-03-01", "replacement": "a-3"},
        {"id": "a-3", "provider": "x", "status": "retired", "retires": "2026-05-01"},
    ]).lookup("a-1")
    assert chain.replacement is None
    assert "its listed replacement, a-2, leads to a-3, which retired too on 2026-05-01, and no successor is listed" in chain.message


def test_the_package_is_typed():
    from pathlib import Path

    import llmshift

    assert (Path(llmshift.__file__).parent / "py.typed").is_file()  # mypy reads the annotations of llmshift.record


# --------------------------------------------------------------------- prices


def test_long_prompts_and_hour_long_cache_writes_cost_what_is_billed():
    from llmshift import pricing
    from llmshift.providers.anthropic import AnthropicProvider

    table = {
        "h": {"mode": "chat", "input_cost_per_token": 1e-7, "output_cost_per_token": 5e-7, "cache_creation_input_token_cost_above_1hr": 2e-7,
              "input_cost_per_token_above_100k_tokens": 5e-7, "output_cost_per_token_above_100k_tokens": 2.5e-6},
        "q": {"mode": "chat", "input_cost_per_token": 5e-8, "output_cost_per_token": 4e-7, "tiered_pricing": [
            {"input_cost_per_token": 5e-8, "output_cost_per_token": 4e-7, "range": [0, 256000.0]},
            {"input_cost_per_token": 2.5e-7, "output_cost_per_token": 2e-6, "range": [256000.0, 1000000.0]}]},
    }
    converted = pricing.convert_litellm(table)
    assert converted["h"] == [0.1, 0.5, None, None, 0.2, [[100000, 0.5, 2.5]]] and converted["q"][5] == [[256000, 0.25, 2.0]]
    assert pricing.prices_problems({"fetched": "2026-10-20", "prices": converted}) == []
    haiku = pricing.price_for("anthropic:claude-haiku-5-5")  # the bundled list: five times the price above 100K
    small, large = pricing.usage_cost(haiku, {"input_tokens": 50_000, "output_tokens": 0})[0], pricing.usage_cost(haiku, {"input_tokens": 150_000, "output_tokens": 0})[0]
    assert large / 150_000 == pytest.approx(5 * small / 50_000)
    usage = AnthropicProvider.parse({"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {
        "input_tokens": 10, "output_tokens": 10, "cache_creation_input_tokens": 1000, "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1000}}}).usage
    assert usage["cache_creation_1h_input_tokens"] == 1000
    plain = pricing.Price(1.0, 1.0, "t")
    assert pricing.usage_cost(plain, usage)[0] == pytest.approx((10 + 1000 * 2.0 + 10) / 1e6)  # 2x the input price, not 1.25x


# ------------------------------------------------------------------ workflows


def test_the_pr_workflows_skip_the_comparison_on_forks(tmp_path, monkeypatch):
    from pathlib import Path

    from llmshift import cli

    monkeypatch.chdir(tmp_path)
    cli.main(["init", "--github"])
    written = (tmp_path / ".github" / "workflows" / "llmshift.yml").read_text(encoding="utf-8")
    examples = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    for text in (written, (examples / "llmshift-pr.yml").read_text(encoding="utf-8"), (examples / "llmshift-check.yml").read_text(encoding="utf-8")):
        assert "LLMSHIFT_HAS_KEY: ${{ secrets." in text and "env.LLMSHIFT_HAS_KEY == 'true'" in text and "forks get no secrets" in text
