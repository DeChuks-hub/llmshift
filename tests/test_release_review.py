"""What the review before tagging 0.1.1 found: scan dates for Vertex, Bedrock and Azure ids,
costs that favoured Gemini, a not_equals that was not the opposite of equals, broken checks
blamed on the model, thinking left out of tests, a slow redaction rule, misquoted regexes."""

from __future__ import annotations

import time

import pytest

from llmshift.checks import CheckContext, Sample, evaluate
from llmshift.compare import CaseResult, combine_runs, failure_patterns, paired_comparison, run_compare
from llmshift.providers.base import ProviderError
from llmshift.providers.mock import MockProvider
from llmshift.registry import Registry
from llmshift.testset import TestCase

# ------------------------------------------------------------------ scan dates


@pytest.mark.parametrize("model, retires", [
    ("claude-3-haiku@20240307", "2026-08-23"),  # Vertex's own date, not the Anthropic API's 2026-04-20
    ("claude-3-5-sonnet-v2@20241022", "2026-02-19"),
    ("claude-3-7-sonnet@20250219", "2026-05-11"),
    ("claude-3-opus@20240229", "2025-08-01"),
    ("vertex_ai/claude-3-haiku@20240307", "2026-08-23"),
])
def test_a_vertex_id_gets_vertex_dates(model, retires):
    lookup = Registry.load().lookup(model)
    assert lookup.platform == "vertex" and lookup.provider == "google-vertex" and lookup.retires.isoformat() == retires


def test_in_the_gap_a_model_vertex_still_serves_is_not_retired(monkeypatch):
    monkeypatch.setenv("LLMSHIFT_TODAY", "2026-06-01")  # after the Anthropic API's date, before Vertex's
    reg = Registry.load()
    assert reg.lookup("claude-3-haiku@20240307").status == "retiring"
    assert reg.lookup("claude-3-haiku-20240307").status == "retired"


def test_what_an_id_says_wins_over_the_files_hint():
    reg = Registry.load()
    bedrock = reg.lookup("anthropic.claude-3-haiku-20240307-v1:0", platform="vertex")  # in a file that also uses AnthropicVertex
    assert bedrock.platform == "bedrock" and "Amazon Bedrock sets its own schedule" in bedrock.message and "Vertex" not in bedrock.message
    routed = reg.lookup("openrouter/anthropic/claude-3-opus", platform="vertex")
    assert routed.provider == "anthropic" and routed.retires.isoformat() == "2026-01-05"  # not Vertex's 2025-08-01
    assert reg.lookup("claude-3-opus", platform="vertex").provider == "google-vertex"  # an id that does not say takes the hint


def test_scan_of_a_file_that_uses_both_clouds(tmp_path, monkeypatch):
    from llmshift.scanner import scan_paths

    (tmp_path / "app.py").write_text(
        "from anthropic import AnthropicBedrock, AnthropicVertex\n"
        "vertex = AnthropicVertex(region='us-east5', project_id='p')\n"
        "bedrock = AnthropicBedrock()\n"
        "vertex.messages.create(model='claude-3-haiku@20240307', max_tokens=10, messages=[])\n"
        "bedrock.messages.create(model='anthropic.claude-3-haiku-20240307-v1:0', max_tokens=10, messages=[])\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    found = {f.model: f.lookup for f in scan_paths(["."]).findings}
    assert found["claude-3-haiku@20240307"].retires.isoformat() == "2026-08-23"
    assert found["anthropic.claude-3-haiku-20240307-v1:0"].platform == "bedrock"


def test_azures_gpt_35_spelling_is_known():
    lookup = Registry.load().lookup("gpt-35-turbo-0613")
    assert lookup.status == "retired" and lookup.platform == "azure" and "Azure sets its own schedule" in lookup.message


# ---------------------------------------------------------------------- costs


def test_gemini_thinking_and_cached_tokens_are_costed_as_billed():
    from llmshift import pricing
    from llmshift.providers.gemini import GeminiProvider
    from llmshift.providers.openai import OpenAIProvider

    gemini = GeminiProvider.parse({
        "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 120, "thoughtsTokenCount": 900, "cachedContentTokenCount": 800},
    })
    assert gemini.usage == {"input_tokens": 1000, "output_tokens": 1020, "cached_input_tokens": 800}  # 120 visible + 900 thinking
    chat = OpenAIProvider.parse_chat({
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 9000}},
    })
    responses = OpenAIProvider.parse_responses({
        "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 10000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 9000}},
    })
    assert chat.usage == responses.usage == {"input_tokens": 10000, "output_tokens": 50, "cached_input_tokens": 9000}
    price = pricing.Price(1.0, 10.0, "t", cache_read=0.1)
    assert pricing.usage_cost(price, chat.usage)[0] == pytest.approx((1000 + 9000 * 0.1 + 50 * 10.0) / 1e6)
    assert pricing.usage_cost(price, gemini.usage)[0] == pytest.approx((200 + 800 * 0.1 + 1020 * 10.0) / 1e6)
    assert pricing.usage_cost(pricing.Price(1.0, 10.0, "t"), chat.usage)[0] == pytest.approx((10000 + 500) / 1e6)  # no cache price listed


def test_the_price_list_carries_cache_prices():
    from llmshift import pricing

    table = {
        "m": {"mode": "chat", "input_cost_per_token": 2e-6, "output_cost_per_token": 8e-6, "cache_read_input_token_cost": 5e-7},
        "c": {"mode": "chat", "input_cost_per_token": 3e-6, "output_cost_per_token": 1.5e-5, "cache_read_input_token_cost": 3e-7, "cache_creation_input_token_cost": 3.75e-6},
        "plain": {"mode": "chat", "input_cost_per_token": 1e-6, "output_cost_per_token": 2e-6},
    }
    converted = pricing.convert_litellm(table)
    assert converted == {"c": [3.0, 15.0, 0.3, 3.75], "m": [2.0, 8.0, 0.5, None], "plain": [1.0, 2.0]}  # readers before 0.1.1 use the first two
    assert pricing.prices_problems({"fetched": "2026-10-20", "prices": converted}) == []
    gpt = pricing.price_for("openai:gpt-4o")  # the bundled list
    assert gpt.cache_read is not None and gpt.cache_read < gpt.input
    custom = pricing.overrides_from_config({"pricing": {"azure:prod": {"input": 2.5, "output": 10, "cached_input": 1.25}}})
    assert custom["azure:prod"].cache_read == 1.25


# --------------------------------------------------------------- not_equals


@pytest.mark.parametrize("check, text, status", [
    ({"type": "not_equals", "value": "N/A"}, "N/A\n", "fail"),  # equals: N/A passes it, so not_equals must not
    ({"type": "not_equals", "value": "n/a", "case_sensitive": False}, "N/A", "fail"),
    ({"type": "not_equals", "value": "42", "numeric": True}, "42.0", "fail"),
    ({"type": "not_equals", "value": "N/A", "strip": False}, "N/A\n", "pass"),
    ({"type": "not_equals", "value": "N/A"}, "Paris", "pass"),
])
def test_not_equals_is_the_opposite_of_equals(make_case, check, text, status):
    case, ctx = make_case("x"), CheckContext()
    assert evaluate(check, Sample(text=text), case, ctx).status == status
    assert evaluate({**check, "type": "equals"}, Sample(text=text), case, ctx).status == ("fail" if status == "pass" else "pass")


def test_an_imported_string_check_is_exact_as_openai_compares(make_case):
    from llmshift.importers.openai_evals import convert_grader

    for op, kind in (("eq", "equals"), ("ne", "not_equals")):
        check = convert_grader({"type": "string_check", "name": op, "operation": op, "reference": "Hardware", "input": "{{sample.output_text}}"}, {}, [])
        assert check["type"] == kind and check["strip"] is False
        assert evaluate(check, Sample(text="Hardware\n"), make_case("x"), CheckContext()).status == ("fail" if op == "eq" else "pass")


# ------------------------------------------------------------- not checked


def _cases(make_case, checks, n=3):
    return [make_case(f"q{i}", user=f"q{i}", id=f"c{i}", checks=checks) for i in range(n)]


def test_a_check_that_cannot_run_is_not_the_models_failure(make_case):
    result = run_compare(_cases(make_case, [{"type": "regex", "pattern": "(unclosed"}]), "mock:echo", show_progress=False, min_pass_rate=0.9)
    s = result.summary
    assert (s["unchecked"], s["failed"], s["errors"], s["evaluated"]) == (3, 0, 0, 0)
    assert all(c.status == "error" and c.unchecked and c.error.startswith("could not check: regex: invalid regex") for c in result.cases)
    assert s["by_category"]["unchecked"]["fail"] == 3 and "errors" not in s["by_category"]
    assert any("could not be checked" in note for note in result.insights)
    assert CaseResult.from_dict(result.cases[0].to_dict()).unchecked  # kept in compare JSON


def test_what_could_be_checked_decides_the_pass_rate(make_case):
    cases = _cases(make_case, [{"type": "contains", "value": "q"}], n=4)
    cases[0].checks = [{"type": "regex", "pattern": "(unclosed"}]  # cannot run
    cases[1].checks = [{"type": "regex", "pattern": "(unclosed"}, {"type": "contains", "value": "zzz"}]  # a real failure still counts
    s = run_compare(cases, "mock:echo", show_progress=False).summary
    assert (s["unchecked"], s["failed"], s["passed"]) == (1, 1, 2)
    assert s["pass_rate"] == pytest.approx(2 / 3)  # of the three cases that could be checked


def test_a_judge_that_cannot_be_reached_is_reported_as_such(make_case):
    def down(request):
        raise ProviderError("401 invalid key", code="invalid_api_key")

    cases = [TestCase(id=f"j{i}", input={"messages": [{"role": "user", "content": f"Tell me about refund {i}"}], "params": {}},
                      baseline={"output": f"Refunds take {i + 3} business days."}, checks=[{"type": "llm_judge"}]) for i in range(2)]
    s = run_compare(cases, "mock:echo", judge=MockProvider("judge", fn=down), show_progress=False).summary
    assert s["unchecked"] == 2 and s["failed"] == 0
    assert any("grading model failed" in p for p in s["by_category"]["unchecked"]["patterns"])


def test_the_exit_code_says_the_run_is_inconclusive(make_case, tmp_path, monkeypatch, capsys):
    from llmshift import cli
    from llmshift.testset import save_tests

    monkeypatch.chdir(tmp_path)
    cases = _cases(make_case, [{"type": "contains", "value": "q"}])
    save_tests(cases, tmp_path / "ok.jsonl")
    cases[0].checks = [{"type": "regex", "pattern": "(unclosed"}]
    save_tests(cases, tmp_path / "broken.jsonl")
    assert cli.main(["compare", "--to", "mock:echo", "--tests", "ok.jsonl", "--no-report", "--min-pass-rate", "0.9"]) == 0
    assert cli.main(["compare", "--to", "mock:echo", "--tests", "broken.jsonl", "--no-report", "--min-pass-rate", "0.9", "--junit", "j.xml"]) == 2
    out = capsys.readouterr().out
    assert "1 not checked" in out and "Could not check" in out and "1 failed" not in out
    junit = (tmp_path / "j.xml").read_text(encoding="utf-8")
    assert 'failures="0" errors="1"' in junit and 'type="not_checked"' in junit


def test_repeats_and_the_paired_test_leave_out_runs_that_could_not_be_checked():
    def run(status, unchecked=False):
        return CaseResult(id="c", group="g", status=status, unchecked=unchecked, raw_status=status, baseline_live_status="pass")

    combined = combine_runs([run("pass"), run("error", unchecked=True), run("pass")])
    assert combined.pass_fraction == 1.0 and combined.run_statuses == ["pass", "unchecked", "pass"] and not combined.unchecked
    assert paired_comparison([run("error", unchecked=True)])["compared"] == 0  # not a regression


def test_a_runaway_pattern_is_given_up_after_two_tries(make_case, monkeypatch):
    from llmshift import userregex

    monkeypatch.setattr(userregex, "TIMEOUT", 0.5)
    monkeypatch.setattr(userregex, "_timeouts", {})
    cases = [make_case(f"q{i}", user="a" * 60 + f"!{i}", id=f"c{i}", checks=[{"type": "regex", "pattern": r"^(\w+\s?)*$"}]) for i in range(10)]
    start = time.perf_counter()
    s = run_compare(cases, "mock:echo", show_progress=False, concurrency=1).summary
    assert s["unchecked"] == 10 and time.perf_counter() - start < 10  # 2 x 0.5 s, not 10 x 2 s
    assert any("not tried again" in p for p in s["by_category"]["unchecked"]["patterns"])


def test_a_worker_that_died_between_searches_is_not_blamed_on_the_pattern():
    from llmshift import userregex

    assert userregex.search(r"o+", "foo")
    userregex._worker.kill()
    userregex._worker.wait()  # killed from outside: an antivirus, a Ctrl+C in a long-running MCP server
    assert userregex.search(r"a+b", "xaab")  # a fresh worker, no "backtracks without end"


# ----------------------------------------------------------------- thinking


def test_snapshot_keeps_claude_and_gemini_thinking():
    from llmshift import logparse

    _, claude = logparse.anthropic_request({
        "model": "claude-opus-5-5", "max_tokens": 16000, "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"}, "messages": [{"role": "user", "content": "hi"}],
    })
    assert claude["thinking"] == {"type": "adaptive", "display": "summarized"} and claude["reasoning_effort"] == "high"
    _, budget = logparse.anthropic_request({"model": "claude-sonnet-4-5", "max_tokens": 9000, "thinking": {"type": "enabled", "budget_tokens": 8000}, "messages": [{"role": "user", "content": "hi"}]})
    assert budget["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    _, gemini = logparse.gemini_request({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingBudget": 2048, "includeThoughts": True}}})
    assert gemini["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    _, gemini3 = logparse.gemini_request({"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}}})
    assert gemini3["reasoning_effort"] == "low"


def _claude_body(model, params, monkeypatch):
    from llmshift.providers import resolve

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    return resolve(f"anthropic:{model}").build_body({"messages": [{"role": "user", "content": "hi"}], "params": params})


def test_claude_is_sent_the_thinking_the_app_asked_for(monkeypatch):
    body = _claude_body("claude-sonnet-4-5", {"thinking": {"type": "enabled", "budget_tokens": 8000}, "max_tokens": 4000}, monkeypatch)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8000} and body["max_tokens"] > 8000  # the budget must be below max_tokens
    body = _claude_body("claude-opus-5-5", {"thinking": {"type": "adaptive"}, "reasoning_effort": "high"}, monkeypatch)
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "high"}
    assert "output_config" not in _claude_body("claude-3-7-sonnet-20250219", {"reasoning_effort": "high"}, monkeypatch)  # no effort before 4.6


def test_a_thinking_budget_on_claude_4_7_and_later_is_a_rejected_setting(make_case, monkeypatch):
    from llmshift.compare import request_rejections
    from llmshift.providers.base import Response

    body = _claude_body("claude-opus-5-5", {"thinking": {"type": "enabled", "budget_tokens": 8000}}, monkeypatch)
    assert body["thinking"] == {"type": "adaptive"}  # how the answer is tested
    rejected = request_rejections("claude-opus-5-5", {"thinking": {"type": "enabled", "budget_tokens": 8000}}, Response(text="x"))
    assert rejected and rejected[0].startswith("thinking=")
    assert not request_rejections("claude-sonnet-4-5", {"thinking": {"type": "enabled", "budget_tokens": 8000}}, Response(text="x"))
    assert not request_rejections("anthropic/claude-opus-5-5", {"thinking": {"type": "enabled", "budget_tokens": 8000}}, Response(text="x"), "openai")  # OpenRouter never sends it
    assert "thinking" not in _claude_body("claude-opus-5-5", {"thinking": {"type": "disabled"}}, monkeypatch)  # disabled 400s on Opus 5.5


def test_claude_gets_thinking_on_the_wire_and_an_old_model_is_retried_without_it(server, monkeypatch):
    from llmshift.providers import resolve

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    sent = []

    def reply(req):
        sent.append(req["json"])
        if "thinking" in sent[-1] and sent[-1]["model"].startswith("claude-3-5"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking: Extra inputs are not permitted"}}, {}
        return 200, {"id": "m", "type": "message", "role": "assistant", "model": sent[-1]["model"], "content": [{"type": "text", "text": "ok"}],
                     "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 2}}, {}

    server.route("POST", "/v1/messages", reply)
    request = {"messages": [{"role": "user", "content": "hi"}], "params": {"thinking": {"type": "adaptive"}, "reasoning_effort": "high", "max_tokens": 4000}}
    new = resolve("anthropic:claude-opus-5-5", {"providers": {"anthropic": {"base_url": server.url}}}).complete(request)
    assert sent[-1]["thinking"] == {"type": "adaptive"} and sent[-1]["output_config"] == {"effort": "high"} and new.rejected == []
    old = resolve("anthropic:claude-3-5-haiku-20241022", {"providers": {"anthropic": {"base_url": server.url}}})
    response = old.complete(request)
    assert "thinking" not in sent[-1] and response.text == "ok" and response.rejected == ["thinking"] and old.adaptations


def test_gemini_is_sent_thinking_and_a_model_without_it_is_retried(server, monkeypatch):
    from llmshift.providers import resolve

    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    calls = []

    def reply(req):
        calls.append(req["json"].get("generationConfig", {}))
        if "thinkingConfig" in calls[-1] and len(calls) == 1:
            return 400, {"error": {"code": 400, "message": "Budget 0 is invalid. This model only works in thinking mode.", "status": "INVALID_ARGUMENT"}}, {}
        return 200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}, {}

    server.route("POST", "/models/", reply)
    gemini = resolve("gemini:gemini-2.5-pro", {"providers": {"gemini": {"base_url": server.url}}})
    response = gemini.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"thinking": {"type": "disabled"}}})
    assert calls[0]["thinkingConfig"] == {"thinkingBudget": 0} and "thinkingConfig" not in calls[1]
    assert response.text == "ok" and response.rejected == ["thinking"] and gemini.adaptations


# --------------------------------------------------------- redaction, summary


def test_dense_xml_password_text_is_redacted_in_linear_time():
    from llmshift.redact import Redactor

    for unit in ("password>a1", "pwd>x9>"):
        text = unit * (1_000_000 // len(unit))
        start = time.perf_counter()
        Redactor().text(text)
        assert time.perf_counter() - start < 3, unit  # 2.8 s per MB before, quadratic
    assert Redactor().text("<password>Hu7nter22</password>") == "<password><SECRET_1></password>"


def test_the_summary_quotes_a_regex_whole():
    from llmshift.checks import CheckResult

    result = CheckResult("regex", "format", "error", "fail", "output does not match /^(Yes|No)$/")
    assert failure_patterns(result) == ["output does not match /^(Yes|No)$/"]  # it showed /^$/
    forbidden = CheckResult("regex", "format", "error", "fail", 'output matches forbidden pattern /"(api|sk)-\\w+"/')
    assert "(api|sk)" in failure_patterns(forbidden)[0]
