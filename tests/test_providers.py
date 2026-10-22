"""Provider request/response translation, adaptation and error handling over real HTTP."""

from __future__ import annotations

import sys

import pytest

from conftest import chat_response
from llmshift.providers import infer_provider, parse_spec, resolve
from llmshift.providers.anthropic import to_anthropic_messages
from llmshift.providers.base import ProviderError
from llmshift.providers.gemini import sanitize_schema
from llmshift.runner import RunAborted, run_cases
from llmshift.testset import TestCase


def req(text="hi", system="Be brief.", **params):
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}], "params": params}


def openai(server, model="gpt-test", **opts):
    return resolve(f"openai:{model}", {"providers": {"openai": {"base_url": server.url + "/v1", "api_key": "sk-test", **opts}}, "max_retries": 2})


# ------------------------------------------------------------------ specs


def test_parse_spec_and_infer():
    assert parse_spec("openai:gpt-5") == ("openai", "gpt-5")
    assert parse_spec("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert parse_spec("ft:gpt-4o-mini:org::x") == (None, "ft:gpt-4o-mini:org::x")
    assert parse_spec("google:gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert infer_provider("gpt-6-astra") == "openai"
    assert infer_provider("o3") == "openai"
    assert infer_provider("claude-opus-5-5") == "anthropic"
    assert infer_provider("gemini-4-argon") == "gemini"
    assert infer_provider("mistral-large-latest") == "mistral"
    assert infer_provider("llama3") is None
    with pytest.raises(ProviderError):
        resolve("llama3")
    assert resolve("claude-sonnet-5-5").kind == "anthropic"
    custom = resolve("local:my-model", {"providers": {"local": {"type": "openai", "base_url": "http://localhost:9/v1"}}})
    assert custom.kind == "openai" and custom.base_url == "http://localhost:9/v1" and custom.spec == "local:my-model"


# ----------------------------------------------------------------- openai


def test_openai_chat_roundtrip(server):
    server.route("POST", "/v1/chat/completions", lambda r: chat_response("hello there", tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]))
    p = openai(server)
    resp = p.complete(req(temperature=0, max_tokens=50, tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}]))
    assert resp.text == "hello there" and resp.tool_calls == [{"name": "f", "arguments": {"x": 1}, "id": "call_1"}]
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5} and resp.fingerprint == "fp_test" and resp.latency_ms is not None
    sent = server.requests[0]
    assert sent["headers"]["authorization"] == "Bearer sk-test"
    assert sent["json"]["model"] == "gpt-test" and sent["json"]["temperature"] == 0
    assert sent["json"]["max_tokens"] == 50  # non-OpenAI base URL: keep max_tokens


def test_openai_adapts_to_rejected_parameters(server):
    def handler(r):
        body = r["json"]
        if "temperature" in body:
            return 400, {"error": {"message": "Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.", "type": "invalid_request_error", "param": "temperature", "code": "unsupported_value"}}
        if "max_tokens" in body:
            return 400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.", "type": "invalid_request_error", "param": "max_tokens", "code": "unsupported_parameter"}}
        return chat_response("ok")

    server.route("POST", "/v1/chat/completions", handler)
    p = openai(server, "reasoner")
    assert p.complete(req(temperature=0, max_tokens=20)).text == "ok"
    assert any("temperature" in a for a in p.adaptations) and any("max_completion_tokens" in a for a in p.adaptations)
    n = len(server.requests)
    assert p.complete(req(temperature=0, max_tokens=20)).text == "ok"
    assert len(server.requests) == n + 1  # learned: no failed round-trips the second time
    assert "max_completion_tokens" in server.requests[-1]["json"]


def test_openai_switches_to_responses_api(server):
    server.route("POST", "/v1/chat/completions", lambda r: (404, {"error": {"message": "This model is only supported in v1/responses and not in v1/chat/completions.", "type": "invalid_request_error", "param": "model", "code": None}}))
    server.route(
        "POST",
        "/v1/responses",
        lambda r: {"object": "response", "status": "completed", "model": "pro-x", "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "from responses"}]}], "usage": {"input_tokens": 3, "output_tokens": 2}},
    )
    p = resolve("openai:pro-x", {"providers": {"openai": {"base_url": server.url + "/v1", "api_key": "k"}}})
    p.official = True  # behave like api.openai.com
    resp = p.complete(req(response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}, "strict": True}}, max_tokens=99))
    assert resp.text == "from responses"
    body = server.requests[-1]["json"]
    assert body["input"][0]["role"] == "system" and body["max_output_tokens"] == 99 and body["store"] is False
    assert body["text"]["format"] == {"type": "json_schema", "name": "x", "schema": {"type": "object"}, "strict": True}


def test_openai_retries_with_retry_after(server):
    state = {"n": 0}

    def handler(r):
        state["n"] += 1
        if state["n"] == 1:
            return 429, {"error": {"message": "slow down", "type": "rate_limit_error"}}, {"retry-after-ms": "10"}
        return chat_response("after retry")

    server.route("POST", "/v1/chat/completions", handler)
    assert openai(server).complete(req()).text == "after retry"
    assert state["n"] == 2


def test_openai_errors(server):
    server.route("POST", "/v1/chat/completions", lambda r: (401, {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}))
    with pytest.raises(ProviderError) as info:
        openai(server).complete(req())
    assert info.value.status == 401 and info.value.fatal and "Incorrect API key" in str(info.value)
    p = resolve("openai:gpt-x", {"providers": {"openai": {"base_url": "https://api.openai.com/v1"}}})
    with pytest.raises(ProviderError) as info:
        p.complete(req())
    assert info.value.code == "missing_api_key"


def test_run_cases_aborts_on_fatal_errors(server):
    server.route("POST", "/v1/chat/completions", lambda r: (404, {"error": {"message": "The model does not exist", "code": "model_not_found"}}))
    cases = [TestCase(id=f"c{i}", input=req(str(i))) for i in range(8)]
    with pytest.raises(RunAborted) as info:
        run_cases(openai(server), cases, concurrency=1, show_progress=False)
    assert "does not exist" in str(info.value)
    assert len(server.requests) <= 3


def test_azure_urls(server):
    server.route("POST", "/openai/deployments/my-dep/chat/completions", lambda r: chat_response("azure ok"))
    p = resolve("azure:my-dep", {"providers": {"azure": {"endpoint": server.url, "api_key": "az"}}})
    assert p.complete(req()).text == "azure ok"
    sent = server.requests[-1]
    assert sent["path"] == "/openai/deployments/my-dep/chat/completions?api-version=2024-10-21"
    assert sent["headers"]["api-key"] == "az" and "model" not in sent["json"]
    server.route("POST", "/openai/v1/chat/completions", lambda r: chat_response("v1 ok"))
    p = resolve("azure:my-dep", {"providers": {"azure": {"endpoint": server.url, "api_key": "az", "api_version": "v1"}}})
    assert p.complete(req()).text == "v1 ok" and server.requests[-1]["json"]["model"] == "my-dep"


# -------------------------------------------------------------- anthropic


def test_anthropic_translation_and_adaptation(server):
    def handler(r):
        body = r["json"]
        if "temperature" in body:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is deprecated for this model."}}
        return {"type": "message", "model": "claude-x", "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Hi"}, {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}], "stop_reason": "tool_use", "usage": {"input_tokens": 4, "output_tokens": 2}}

    server.route("POST", "/v1/messages", handler)
    p = resolve("anthropic:claude-x", {"providers": {"anthropic": {"base_url": server.url, "api_key": "ak"}}})
    resp = p.complete(req(temperature=0.2, response_format={"type": "json_object"}, tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}], tool_choice="required"))
    assert resp.text == "Hi" and resp.tool_calls == [{"id": "toolu_1", "name": "f", "arguments": {"a": 1}}] and resp.finish_reason == "tool_calls"
    body = server.requests[-1]["json"]
    assert server.requests[-1]["headers"]["x-api-key"] == "ak" and server.requests[-1]["headers"]["anthropic-version"] == "2023-06-01"
    assert "temperature" not in body and body["max_tokens"] == 4096 and "valid JSON object" in body["system"]
    assert body["tools"] == [{"name": "f", "input_schema": {"type": "object"}}] and body["tool_choice"] == {"type": "any"}
    assert p.adaptations


def test_anthropic_message_conversion():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "f", "arguments": '{"x": 1}'}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]},
    ]
    system, out = to_anthropic_messages(messages)
    assert system == "sys"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "call_1", "name": "f", "input": {"x": 1}}
    assert out[2]["content"][0]["type"] == "tool_result" and out[2]["content"][-1]["source"]["media_type"] == "image/png"
    sys_only, only = to_anthropic_messages([{"role": "system", "content": "just this"}])
    assert sys_only == "" and only[0]["content"][0]["text"] == "just this"


# ----------------------------------------------------------------- gemini


def test_gemini_translation(server):
    server.route(
        "POST",
        "/v1beta/models/gemini-x:generateContent",
        lambda r: {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking", "thought": True}, {"text": "answer"}, {"functionCall": {"name": "f", "args": {"q": "x"}}}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}, "modelVersion": "gemini-x-001"},
    )
    p = resolve("gemini:models/gemini-x", {"providers": {"gemini": {"base_url": server.url + "/v1beta", "api_key": "gk"}}})
    resp = p.complete(req(temperature=0.1, max_tokens=64, response_format={"type": "json_object"}, tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"q": {"type": ["string", "null"]}}, "additionalProperties": False}}}]))
    assert resp.text == "answer" and resp.tool_calls == [{"name": "f", "arguments": {"q": "x"}}] and resp.finish_reason == "tool_calls"
    sent = server.requests[-1]
    assert sent["headers"]["x-goog-api-key"] == "gk"
    body = sent["json"]
    assert body["systemInstruction"]["parts"][0]["text"].startswith("Be brief.")
    assert body["generationConfig"] == {"temperature": 0.1, "maxOutputTokens": 64, "responseMimeType": "application/json"}
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["parameters"] == {"type": "OBJECT", "properties": {"q": {"type": "STRING", "nullable": True}}}


def test_gemini_blocked_prompt(server):
    server.route("POST", "/v1beta/models/g:generateContent", lambda r: {"promptFeedback": {"blockReason": "SAFETY"}})
    p = resolve("gemini:g", {"providers": {"gemini": {"base_url": server.url + "/v1beta", "api_key": "k"}}})
    resp = p.complete(req())
    assert resp.text == "" and resp.refusal == "SAFETY"


def test_sanitize_schema_refs():
    schema = {"type": "object", "$defs": {"x": {"type": "integer", "default": 3}}, "properties": {"a": {"$ref": "#/$defs/x"}}}
    assert sanitize_schema(schema) == {"type": "OBJECT", "properties": {"a": {"type": "INTEGER"}}}


# ------------------------------------------------------- custom providers


def test_python_and_command_providers(tmp_path, monkeypatch):
    mod = tmp_path / "myllm.py"
    mod.write_text("def answer(request):\n    return {'text': 'echo: ' + request['messages'][-1]['content']}\n", encoding="utf-8")
    p = resolve(f"python:{mod.as_posix()}:answer")
    assert p.complete(req("abc")).text == "echo: abc"
    monkeypatch.chdir(tmp_path)
    assert resolve("python:myllm:answer").complete(req("x")).text == "echo: x"
    with pytest.raises(ProviderError):
        resolve("python:myllm:nope")
    script = tmp_path / "cmd.py"
    script.write_text("import json, sys\nreq = json.load(sys.stdin)\nprint(json.dumps({'text': req['model'] + ':' + req['messages'][-1]['content']}))\n", encoding="utf-8")
    cfg = {"providers": {"app": {"type": "command", "command": [sys.executable, str(script)]}}}
    assert resolve("app:v1", cfg).complete(req("hey")).text == "v1:hey"


def test_mock_provider_unknown():
    with pytest.raises(ProviderError):
        resolve("mock:nope")
    assert resolve("mock:echo").complete(req("same")).text == "same"
