"""Settings the new model rejects: Claude 4.7 and later take only the default sampling, and
an OpenAI reasoning model takes only temperature 1. A test that quietly left them out would
pass while the real call fails with a 400 error, so compare fails those cases instead,
upgrade lists where the code sets them, and --drop-param tests without them."""

from __future__ import annotations

import json

import pytest

from llmshift import cli
from llmshift.providers import resolve
from llmshift.providers.anthropic import AnthropicProvider, default_sampling_only, rejected_sampling


@pytest.mark.parametrize(
    "model, strict",
    [
        ("claude-sonnet-5-5", True), ("claude-opus-4-7", True), ("claude-opus-4-8", True), ("claude-opus-5", True),
        ("claude-haiku-5-5", True), ("claude-fable-5-1", True), ("claude-mythos-5-1", True), ("anthropic:claude-sonnet-5-5", True),
        ("anthropic.claude-opus-4-7", True), ("claude-opus-4-7@20260101", True),
        ("claude-sonnet-4-6", False), ("claude-opus-4-6", False), ("claude-haiku-4-5", False), ("claude-sonnet-4-20250514", False),
        ("claude-opus-4-1-20250805", False), ("claude-3-7-sonnet-20250219", False), ("claude-3-5-haiku-20241022", False),
        ("claude-2.1", False), ("gpt-5.6-sol", False), ("mock:nextgen", False),
    ],
)
def test_which_claude_models_take_only_the_default_sampling(model, strict):
    assert default_sampling_only(model) is strict


def test_rejected_sampling_names_each_setting_and_value():
    assert rejected_sampling("claude-sonnet-5-5", {"temperature": 0, "top_p": 0.9, "top_k": 40, "max_tokens": 300}) == ["temperature=0", "top_p=0.9", "top_k=40"]
    assert rejected_sampling("claude-sonnet-5-5", {"temperature": 1, "max_tokens": 300}) == []  # the default is accepted
    assert rejected_sampling("claude-sonnet-4-6", {"temperature": 0, "top_p": 0.9}) == []


def _claude_api(strict: set[str], answer: str = "The order ships on Monday."):
    """A Messages API that rejects what Claude 4.7 and later reject."""

    def handler(req):
        body = req["json"]
        if body["model"] in strict:
            for key in ("top_p", "top_k"):
                if key in body:
                    return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": f"`{key}` is not supported for this model."}}
            if body.get("temperature", 1) != 1:
                return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is not supported for this model; only the default value 1."}}
        text = answer
        return {"type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 6}}

    return handler


def _project(tmp_path, server, params: dict) -> None:
    (tmp_path / ".llmshift").mkdir()
    rows = [
        {"id": f"c{i}", "input": {"messages": [{"role": "user", "content": f"When does order {i} ship?"}], "params": params},
         "baseline": {"output": "The order ships on Monday."}, "checks": [{"type": "contains", "value": "Monday"}]}
        for i in range(3)
    ]
    (tmp_path / ".llmshift" / "tests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {"tests": ".llmshift/tests.jsonl", "providers": {"anthropic": {"base_url": server.url, "api_key": "k", "max_retries": 0}}}
    (tmp_path / "llmshift.yaml").write_text(json.dumps(config), encoding="utf-8")


def _run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out + out.err


def test_compare_fails_cases_whose_settings_the_new_model_rejects(in_tmp, server, capsys):
    server.route("POST", "/v1/messages", _claude_api({"claude-sonnet-5-5"}))
    _project(in_tmp, server, {"temperature": 0, "top_p": 0.9, "max_tokens": 300})
    code, out = _run(capsys, "compare", "--to", "anthropic:claude-sonnet-5-5", "--json", "r.json")
    data = json.loads((in_tmp / "r.json").read_text(encoding="utf-8"))
    assert code == 1 and [c["status"] for c in data["cases"]] == ["fail"] * 3
    check = next(c for c in data["cases"][0]["checks"] if c["type"] == "request")
    assert check["details"]["rejected"] == ["temperature=0", "top_p=0.9"] and "400 error" in check["message"]
    assert "--drop-param temperature --drop-param top_p" in check["message"]
    assert data["insights"][0].startswith("3 requests send settings anthropic:claude-sonnet-5-5 rejects (temperature, top_p)")
    assert "Rejected request settings" in out and "temperature rejected" in out
    # Answered from the cache, the cases still fail: the answers remember what was refused.
    sent = len(server.requests)
    code, _ = _run(capsys, "compare", "--to", "anthropic:claude-sonnet-5-5", "--json", "r2.json")
    again = json.loads((in_tmp / "r2.json").read_text(encoding="utf-8"))
    assert code == 1 and len(server.requests) == sent and [c["status"] for c in again["cases"]] == ["fail"] * 3
    # Once the code no longer sends them, the tests leave them out too.
    code, out = _run(capsys, "compare", "--to", "anthropic:claude-sonnet-5-5", "--drop-param", "temperature", "--drop-param", "top_p", "--json", "r3.json")
    fixed = json.loads((in_tmp / "r3.json").read_text(encoding="utf-8"))
    assert code == 0 and [c["status"] for c in fixed["cases"]] == ["pass"] * 3
    assert all("temperature" not in r["json"] and "top_p" not in r["json"] for r in server.requests[sent:])
    # A model that takes these settings is not affected.
    code, _ = _run(capsys, "compare", "--to", "anthropic:claude-sonnet-4-6", "--json", "r4.json")
    assert code == 0


def test_drop_params_in_the_config_and_its_type(in_tmp, server, capsys):
    from llmshift.config import ConfigError, load_config

    server.route("POST", "/v1/messages", _claude_api({"claude-opus-4-7"}))
    _project(in_tmp, server, {"temperature": 0.2, "top_k": 40, "max_tokens": 300})
    code, out = _run(capsys, "compare", "--to", "anthropic:claude-opus-4-7", "--no-report")
    assert code == 1 and "top_k rejected" in out
    config = json.loads((in_tmp / "llmshift.yaml").read_text(encoding="utf-8"))
    config["drop_params"] = ["temperature", "top_k"]
    (in_tmp / "llmshift.yaml").write_text(json.dumps(config), encoding="utf-8")
    code, out = _run(capsys, "compare", "--to", "anthropic:claude-opus-4-7", "--no-report")
    assert code == 0 and "rejected" not in out
    config["drop_params"] = "temperature"
    (in_tmp / "llmshift.yaml").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="drop_params must be a list"):
        load_config("llmshift.yaml")
    with pytest.raises(SystemExit):
        cli.main(["compare", "--drop-param", "temperature=0"])
    assert "expected a setting name" in capsys.readouterr().err


def test_a_proxy_that_drops_settings_can_keep_the_note_without_failing():
    from llmshift.compare import run_compare
    from llmshift.providers.base import Response
    from llmshift.testset import TestCase

    class Proxy:  # e.g. LiteLLM with drop_params: production never sends the setting
        kind, model, spec, adaptations = "test", "claude-sonnet-5-5", "proxy:claude-sonnet-5-5", []

        def identity(self):
            return {"kind": "test"}

        def complete(self, request):
            return Response(text="The order ships on Monday.")

    case = TestCase(id="c1", input={"messages": [{"role": "user", "content": "When?"}], "params": {"temperature": 0}},
                    baseline={"output": "The order ships on Monday."}, checks=[{"type": "contains", "value": "Monday"}])
    assert run_compare([case], Proxy(), show_progress=False).cases[0].status == "fail"
    quiet = run_compare([case], Proxy(), show_progress=False, severity={"request": "info"}).cases[0]
    assert quiet.status == "pass" and [c.type for c in quiet.checks if c.status == "fail"] == ["request"]


class _Claude:
    """Answers as the old model did; its id is a model that rejects a non-default temperature."""

    kind, model, spec, adaptations = "test", "claude-sonnet-5-5", "anthropic:claude-sonnet-5-5", []

    def __init__(self):
        self.calls = 0

    def identity(self):
        return {"kind": "test"}

    def complete(self, request):
        self.calls += 1
        return __import__("llmshift.providers.base", fromlist=["Response"]).Response(text="The order ships on Monday.")


def _cases():
    from llmshift.testset import TestCase

    return [
        TestCase(id=f"c{i}", input={"messages": [{"role": "system", "content": "You answer order questions."}, {"role": "user", "content": f"When does order {i} ship?"}], "params": {"temperature": 0}},
                 baseline={"output": "The order ships on Monday."}, checks=[{"type": "contains", "value": "Monday"}])
        for i in range(3)
    ]


def test_fix_says_what_no_prompt_can_fix_and_spends_no_runs_on_it():
    from llmshift.fix import fix_markdown, run_fix

    model = _Claude()
    result = run_fix(_cases(), model, show_progress=False)
    assert result.groups == [] and model.calls == 3  # the answers match: no prompt was tried
    assert result.notes and "rejects temperature as the requests send them" in result.notes[0] and "--drop-param temperature" in result.notes[0]
    assert "**Note:**" in fix_markdown(result)


def test_watch_says_why_the_pass_rate_fell(tmp_path):
    from llmshift.watch import run_watch

    outcome = run_watch(_cases(), _Claude(), history_dir=str(tmp_path), alerts=[f"file:{tmp_path / 'alerts.md'}"], alert_on="info", min_pass_rate=0.9)
    assert outcome.summary["pass_rate"] == 0 and "rejects temperature" in outcome.notes[0]
    alert = json.loads((tmp_path / "alerts.md").read_text(encoding="utf-8"))
    assert "rejects temperature" in alert["lines"][0]  # said first, before the pass rate's fall


def test_upgrade_lists_the_sampling_settings_a_new_claude_rejects(in_tmp, capsys):
    (in_tmp / "app.py").write_text(
        "import anthropic\n\nclient = anthropic.Anthropic()\nMODEL = \"claude-3-7-sonnet-20250219\"\n\n\ndef summarize(text):\n"
        "    return client.messages.create(\n        model=MODEL,\n        max_tokens=300,\n        temperature=0,\n        top_p=0.9,\n"
        "        messages=[{\"role\": \"user\", \"content\": text}],\n    )\n",
        encoding="utf-8",
    )
    (in_tmp / "settings.yaml").write_text("llm:\n  provider: anthropic\n  temperature: 0.2\n", encoding="utf-8")
    code, out = _run(capsys, "upgrade", "--from", "claude-3-7-sonnet-20250219", "--to", "claude-sonnet-5-5")
    assert code == 0 and '+MODEL = "claude-sonnet-5-5"' in out.replace("\r", "")
    assert "temperature, top_p or top_k is set at app.py:11, app.py:12, settings.yaml:3: claude-sonnet-5-5 rejects" in " ".join(out.split())
    code, out = _run(capsys, "upgrade", "--from", "claude-3-7-sonnet-20250219", "--to", "claude-sonnet-4-6")
    assert code == 0 and "top_p or top_k is set" not in out  # Sonnet 4.6 still takes them


def test_providers_learn_rejected_values_not_whole_settings(server):
    server.route("POST", "/v1/messages", _claude_api({"claude-sonnet-5-5"}))
    p = AnthropicProvider("claude-sonnet-5-5", options={"base_url": server.url, "api_key": "k", "max_retries": 0})

    def ask(**params):
        before = len(server.requests)
        resp = p.complete({"messages": [{"role": "user", "content": "hi"}], "params": params})
        return resp.rejected, len(server.requests) - before

    assert ask(temperature=0) == (["temperature"], 2)  # refused, then answered without it
    assert ask(temperature=1) == ([], 1)  # the default is still sent: this model takes it
    assert ask(temperature=0) == (["temperature"], 1)  # learnt: left out at once
    assert ask(temperature=0, top_k=40) == (["temperature", "top_k"], 2)
    assert server.requests[-2]["json"].get("top_k") == 40  # top_k is passed on, so its rejection shows


def test_openai_reasoning_models_record_a_refused_temperature(server):
    def handler(req):
        body = req["json"]
        if body.get("temperature", 1) != 1:
            return 400, {"error": {"message": "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.",
                                   "type": "invalid_request_error", "param": "temperature", "code": "unsupported_value"}}
        return {"id": "x", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    server.route("POST", "/v1/chat/completions", handler)
    p = resolve("openai:gpt-5.6-sol", {"providers": {"openai": {"base_url": server.url + "/v1", "api_key": "k", "max_retries": 0}}})
    assert p.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 0}}).rejected == ["temperature"]
    before = len(server.requests)
    assert p.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 1}}).rejected == []
    assert p.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 0}}).rejected == ["temperature"]
    assert len(server.requests) - before == 2 and server.requests[-2]["json"]["temperature"] == 1


def test_a_refused_prefill_or_forced_tool_choice_is_recorded(server):
    def handler(req):
        body = req["json"]
        if body["messages"][-1]["role"] == "assistant":
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "This model does not support assistant message prefill."}}
        if (body.get("tool_choice") or {}).get("type") == "any":
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "tool_choice: type \"tool\" and \"any\" are not supported for this model."}}
        return {"type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [{"type": "text", "text": "{}"}], "stop_reason": "end_turn", "usage": {}}

    server.route("POST", "/v1/messages", handler)
    p = AnthropicProvider("claude-opus-5-5", options={"base_url": server.url, "api_key": "k", "max_retries": 0})
    prefill = {"messages": [{"role": "user", "content": "JSON please"}, {"role": "assistant", "content": "{"}], "params": {}}
    assert p.complete(prefill).rejected == ["assistant prefill"]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    forced = {"messages": [{"role": "user", "content": "use f"}], "params": {"tools": tools, "tool_choice": "required"}}
    assert p.complete(forced).rejected == ["tool_choice"]


def test_top_k_is_kept_from_logs_and_passed_on():
    from llmshift.logparse import anthropic_request
    from llmshift.providers.gemini import GeminiProvider

    _messages, params = anthropic_request({"model": "claude-3-7-sonnet", "max_tokens": 10, "top_k": 40, "messages": [{"role": "user", "content": "hi"}]})
    assert params["top_k"] == 40
    body = GeminiProvider("gemini-2.5-flash", options={"api_key": "k"}).build_body({"messages": [{"role": "user", "content": "hi"}], "params": {"top_k": 40}})
    assert body["generationConfig"]["topK"] == 40


def test_shadow_answers_keep_what_the_live_model_refused(tmp_path):
    from llmshift.shadow import build_cases

    record = {
        "llmshift_shadow": 1, "ts": "2026-10-23T10:00:00Z", "api": "chat",
        "request": {"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 0}},
        "baseline": {"output": "hello"}, "shadow": {"spec": "anthropic:claude-sonnet-5-5", "output": "hello", "rejected": ["temperature"]},
    }
    _cases, answers, _spec = build_cases([record])
    assert next(iter(answers.values())).rejected == ["temperature"]
