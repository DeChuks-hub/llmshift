"""Regression tests for review findings in the provider, runner, cache and recording layers."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import pytest

from llmshift.cache import ResponseCache
from llmshift.providers import resolve
from llmshift.providers.anthropic import AnthropicProvider, to_anthropic_messages
from llmshift.providers.base import ProviderError, Response
from llmshift.providers.gemini import GeminiProvider
from llmshift.providers.mock import coerce_response
from llmshift.record import _StreamAccumulator, log, wrap
from llmshift.runner import RunAborted, run_cases
from llmshift.testset import TestCase


def _anthropic_ok(text: str = "ok") -> dict:
    return {"type": "message", "role": "assistant", "model": "claude-test", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}}


def test_anthropic_only_drops_parameters_on_capability_errors(server):
    calls = []

    def handler(req):
        calls.append(req["json"])
        if "temperature" in req["json"]:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "temperature: range error, must be <= 1.0"}}
        return _anthropic_ok()

    server.route("POST", "/v1/messages", handler)
    p = AnthropicProvider("claude-test", options={"base_url": server.url, "api_key": "k", "max_retries": 0})
    with pytest.raises(ProviderError):  # a bad value is the caller's problem, not a capability gap
        p.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 0.5}})
    assert len(calls) == 1 and not p.adaptations

    def deprecated(req):
        calls.append(req["json"])
        if "temperature" in req["json"]:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "`temperature` is deprecated for this model."}}
        return _anthropic_ok("adapted")

    server.route("POST", "/v1/messages", deprecated)
    assert p.complete({"messages": [{"role": "user", "content": "hi"}], "params": {"temperature": 0.5}}).text == "adapted"
    assert "temperature" not in calls[-1] and p.adaptations


def test_anthropic_identity_and_response_validation():
    a = AnthropicProvider("claude-test", options={"api_key": "k", "max_tokens": 100})
    b = AnthropicProvider("claude-test", options={"api_key": "other", "max_tokens": 200})
    c = AnthropicProvider("claude-test", options={"api_key": "other", "max_tokens": 100})
    assert a.identity() != b.identity() and a.identity() == c.identity()  # the key never matters
    with pytest.raises(ProviderError) as exc:
        AnthropicProvider.parse({"id": "x"})
    assert exc.value.code == "invalid_response" and exc.value.retryable
    with pytest.raises(ProviderError):
        AnthropicProvider.parse(["not", "an", "object"])


def test_anthropic_links_tool_calls_without_ids():
    messages = [
        {"role": "user", "content": "order 7?"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "arguments": {"order": 7}}]},
        {"role": "tool", "content": "shipped"},
    ]
    _, out = to_anthropic_messages(messages)
    use = next(b for m in out for b in m["content"] if b["type"] == "tool_use")
    result = next(b for m in out for b in m["content"] if b["type"] == "tool_result")
    assert use["id"] == result["tool_use_id"] and all(ch.isalnum() or ch in "_-" for ch in use["id"])


def test_gemini_tool_results_get_the_function_name_and_keep_nan_as_text():
    p = GeminiProvider("gemini-test", options={"api_key": "k"})
    body = p.build_body(
        {
            "messages": [
                {"role": "user", "content": "check"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup", "arguments": {"q": 1}}]},
                {"role": "tool", "content": '{"score": NaN}'},
            ],
            "params": {},
        }
    )
    call = body["contents"][1]["parts"][0]["functionCall"]
    response = body["contents"][2]["parts"][0]["functionResponse"]
    assert response["name"] == "lookup" and "id" not in call and "id" not in response  # ids were not in the request
    assert response["response"] == {"score": "NaN"}
    json.dumps(body, allow_nan=False)  # the request body is valid JSON


def test_gemini_parse_is_defensive():
    with pytest.raises(ProviderError) as exc:
        GeminiProvider.parse({"unexpected": True})
    assert exc.value.code == "invalid_response"
    r = GeminiProvider.parse({"candidates": [{"content": {"parts": [{"text": None}, {"text": "hi"}, "junk"]}, "finishReason": "STOP"}]})
    assert r.text == "hi" and r.finish_reason == "stop"


def test_coerce_response_never_returns_none_text():
    assert coerce_response({"text": None, "output": "fallback"}).text == "fallback"
    assert coerce_response({"text": None}).text == ""
    assert coerce_response(Response(text=None)).text == ""  # type: ignore[arg-type]
    assert coerce_response({"text": "x", "usage": "bogus"}).usage == {}


def test_python_provider_async_function_and_dataclass_module(tmp_path):
    mod = tmp_path / "user_fn.py"
    mod.write_text(
        "import asyncio\nfrom dataclasses import dataclass\n\n@dataclass\nclass Box:\n    text: str\n\n"
        "async def answer(request):\n    await asyncio.sleep(0)\n    return Box('hi ' + request['messages'][-1]['content']).text\n",
        encoding="utf-8",
    )
    p = resolve(f"python:{mod}:answer")
    assert p.complete({"messages": [{"role": "user", "content": "there"}], "params": {}}).text == "hi there"
    before = p.identity()
    mod.write_text(mod.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    assert resolve(f"python:{mod}:answer").identity() != before  # edited code is not served from cache


def test_command_provider_encoding_and_timeout(tmp_path):
    script = tmp_path / "echo_cmd.py"
    script.write_text("import json, sys\nreq = json.load(sys.stdin)\nprint(req['messages'][-1]['content'] + ' ' + '\\u00e9')\n", encoding="utf-8")
    cfg = {"providers": {"cmd": {"type": "command", "command": [sys.executable, str(script)]}}}
    assert resolve("cmd:x", cfg).complete({"messages": [{"role": "user", "content": "caf\u00e9 \u2603"}], "params": {}}).text == "caf\u00e9 \u2603 \u00e9"
    slow = tmp_path / "slow_cmd.py"
    slow.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    cfg = {"providers": {"cmd": {"type": "command", "command": [sys.executable, str(slow)], "timeout": 1}}}
    start = time.time()
    with pytest.raises(ProviderError) as exc:
        resolve("cmd:x", cfg).complete({"messages": [{"role": "user", "content": "x"}], "params": {}})
    assert exc.value.code == "timeout" and time.time() - start < 20


def test_provider_config_under_an_alias_name():
    cfg = {"providers": {"claude": {"base_url": "http://example.invalid", "api_key": "x"}}}
    for spec in ("claude:claude-sonnet-4-5", "anthropic:claude-sonnet-4-5", "claude-sonnet-4-5"):
        p = resolve(spec, cfg)
        assert isinstance(p, AnthropicProvider) and p.base_url == "http://example.invalid", spec


def test_connection_errors_abort_the_run_quickly():
    p = resolve("openai:gpt-test", {"providers": {"openai": {"base_url": "http://127.0.0.1:9", "api_key": "k"}}, "max_retries": 0})
    cases = [TestCase(id=f"c{i}", input={"messages": [{"role": "user", "content": str(i)}], "params": {}}) for i in range(20)]
    with pytest.raises(RunAborted):
        run_cases(p, cases, concurrency=2, show_progress=False)


def test_cache_survives_corrupt_entries_and_odd_values(tmp_path):
    cache = ResponseCache(tmp_path)
    key = cache.key({"kind": "x"}, {"messages": []})
    path = cache._path(key)
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2]", encoding="utf-8")
    assert cache.get(key) is None and cache.misses == 1
    cache.put(key, Response(text="ok", tool_calls=[{"name": "f", "arguments": {"when": object()}}]))
    assert cache.get(key).text == "ok"
    assert not [f for f in path.parent.iterdir() if f.name.startswith(".tmp-")]


def test_runner_treats_a_broken_cache_as_a_miss(tmp_path):
    class Broken(ResponseCache):
        def get(self, key):
            raise RuntimeError("disk on fire")

        def put(self, key, response):
            raise RuntimeError("disk on fire")

    p = resolve("mock:echo")
    cases = [TestCase(id="a", input={"messages": [{"role": "user", "content": "hi"}], "params": {}})]
    assert run_cases(p, cases, cache=Broken(tmp_path), show_progress=False)["a"].text == "hi"


# ------------------------------------------------------------------ record


class _Completions:
    def create(self, **kw):
        if kw.get("stream"):
            def gen():
                yield {"model": "gpt-x", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"or'}}]}}]}
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'der": 7}'}}]}}]}
                yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
                yield {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

            return gen()
        return {"model": "gpt-x", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class _AsyncCompletions:
    def create(self, **kw):  # a plain function returning a coroutine, as the SDK's decorators produce
        async def run():
            return {"model": "gpt-a", "choices": [{"message": {"role": "assistant", "content": "async ok"}}]}

        return run()


class _Messages:
    def create(self, **kw):
        events = [
            {"type": "message_start", "message": {"model": "claude-x", "usage": {"input_tokens": 9}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check."}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"order":'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " 7}"}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
        ]
        return iter(events)


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _Client:
    def __init__(self, completions):
        self.chat = _Chat(completions)
        self.messages = _Messages()
        self.base_url = "https://api.example.com/v1"
        self.timeout = 30.0

    def with_options(self, **kw):
        return self


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_recorder_streams_tool_calls_and_usage(tmp_path):
    path = tmp_path / "rec.jsonl"
    client = wrap(_Client(_Completions()), path=path)
    assert client.base_url == "https://api.example.com/v1" and client.timeout == 30.0  # attributes pass through untouched
    stream = client.with_options(timeout=5).chat.completions.create(model="gpt-x", messages=[], stream=True)
    assert next(stream)["model"] == "gpt-x"  # next() works as on the SDK's stream objects
    list(stream)
    list(client.messages.create(model="claude-x", messages=[], stream=True, max_tokens=5))
    chat, msg = _records(path)
    message = chat["response"]["choices"][0]["message"]
    assert message["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"order": 7}'}
    assert chat["response"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    assert msg["response"]["content"][1] == {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"order": 7}}
    assert msg["response"]["usage"] == {"input_tokens": 9, "output_tokens": 12} and msg["response"]["stop_reason"] == "tool_use"


def test_recorder_handles_async_methods_that_are_not_coroutine_functions(tmp_path):
    path = tmp_path / "rec.jsonl"
    client = wrap(_Client(_AsyncCompletions()), path=path)
    result = asyncio.run(client.chat.completions.create(model="gpt-a", messages=[]))
    assert result["choices"][0]["message"]["content"] == "async ok"
    assert _records(path)[0]["response"]["choices"][0]["message"]["content"] == "async ok"


def test_concurrent_log_calls_keep_lines_whole(tmp_path):
    path = tmp_path / "rec.jsonl"

    def worker(i):
        for j in range(40):
            log({"model": "m", "messages": [{"role": "user", "content": f"{i}-{j}" * 200}]}, {"text": "x"}, path=path)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(_records(path)) == 240


def test_stream_accumulator_rebuilds_gemini_streams():
    acc = _StreamAccumulator()
    acc.add({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}], "modelVersion": "gemini-x"})
    acc.add({"candidates": [{"content": {"parts": [{"text": "lo"}, {"functionCall": {"name": "f", "args": {"a": 1}}}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 2}})
    out = acc.response("gemini")
    r = GeminiProvider.parse(out)
    assert r.text == "Hello" and r.tool_calls[0]["arguments"] == {"a": 1} and r.model == "gemini-x"
    assert r.finish_reason == "tool_calls" and out["usageMetadata"] == {"promptTokenCount": 2}


def test_certificate_errors_are_explained_and_not_retried(monkeypatch):
    import ssl
    import urllib.error

    from llmshift.providers import http
    from llmshift.providers.base import ProviderError

    class Opener:
        def open(self, req, timeout=None):
            raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    monkeypatch.setattr(http, "_OPENER", Opener())
    waits = []
    with pytest.raises(ProviderError) as info:
        http.request("POST", "https://api.example.com/v1/chat/completions", body={}, max_retries=4, sleep=waits.append)
    err = info.value
    assert waits == [] and err.code == "ssl_error" and err.fatal  # no pointless retries; the run stops early
    assert "api.example.com" in str(err) and "SSL_CERT_FILE" in str(err) and "Install Certificates" in str(err)
