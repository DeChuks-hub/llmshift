from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

PROVIDER_ENV = [
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_ORGANIZATION", "OPENAI_PROJECT_ID",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_AD_TOKEN",
    "LLMSHIFT_REGISTRY", "GITHUB_ACTIONS", "GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "SLACK_WEBHOOK_URL",
    "OLLAMA_HOST", "LLMSHIFT_LOG",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch real providers; freeze 'today' so lifecycle statuses are stable."""
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLMSHIFT_TODAY", "2026-10-21")
    monkeypatch.setenv("NO_COLOR", "1")
    from llmshift import console

    console.set_color(False)


@pytest.fixture
def in_tmp(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class MockServer:
    """A tiny HTTP server: routes map (METHOD, path-prefix) -> handler(req) -> (status, body, headers)."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], Callable[[dict], Any]] = {}
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw.decode("utf-8", "replace")
                req = {"method": method, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "json": body}
                server.requests.append(req)
                handler = None
                for (m, prefix), fn in sorted(server.routes.items(), key=lambda kv: -len(kv[0][1])):
                    if m == method and self.path.startswith(prefix):
                        handler = fn
                        break
                if handler is None:
                    status, payload, headers = 404, {"error": {"message": f"no route for {method} {self.path}", "type": "not_found"}}, {}
                else:
                    result = handler(req)
                    if isinstance(result, tuple):
                        status, payload = result[0], result[1]
                        headers = result[2] if len(result) > 2 else {}
                    else:
                        status, payload, headers = 200, result, {}
                data = payload if isinstance(payload, bytes) else (payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8"))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                self._handle("POST")

            def do_GET(self) -> None:
                self._handle("GET")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def route(self, method: str, prefix: str, fn: Callable[[dict], Any]) -> None:
        self.routes[(method, prefix)] = fn

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = MockServer()
    yield s
    s.close()


def chat_response(text: str, model: str = "gpt-test", **extra: Any) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "system_fingerprint": extra.pop("fingerprint", "fp_test"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text, **extra}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def make_case():
    from llmshift.testset import TestCase

    def _make(output: str, user: str = "hello", system: str = "You are a test assistant.", checks: list | None = None, params: dict | None = None, **kw: Any) -> TestCase:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return TestCase(
            id=kw.pop("id", f"c_{abs(hash((output, user))) % 10**8}"),
            input={"messages": messages, "params": params or {}},
            baseline={"output": output},
            checks=checks or [],
            group=kw.pop("group", "g1"),
            **kw,
        )

    return _make


def write_jsonl(path: os.PathLike, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
