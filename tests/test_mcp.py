"""MCP server: JSON-RPC dispatch, the stdio transport and the llmshift tools."""

from __future__ import annotations

import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llmshift import cli, mcp
from llmshift.demo import create_project
from llmshift.providers import http
from llmshift.snapshot import build_snapshot
from llmshift.testset import save_tests

SRC = Path(__file__).resolve().parents[1] / "src"
TOOL_NAMES = {"llmshift_scan", "llmshift_models", "llmshift_snapshot", "llmshift_compare", "llmshift_fix", "llmshift_doctor"}


def rpc(server, method, params=None, rid=1):
    message = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def call(server, name, arguments=None, rid=1):
    return rpc(server, "tools/call", {"name": name, "arguments": arguments or {}}, rid)


def payload(response):
    result = response["result"]
    assert result["isError"] is False, result["content"][0]["text"]
    data = json.loads(result["content"][0]["text"])
    assert result["structuredContent"] == data
    return data


def error_text(response):
    result = response["result"]
    assert result["isError"] is True and "structuredContent" not in result
    return result["content"][0]["text"]


@pytest.fixture
def project(tmp_path):
    root = create_project(str(tmp_path / "proj"))
    save_tests(build_snapshot([str(root / "logs")]).cases, root / ".llmshift" / "tests.jsonl")
    return root


@pytest.fixture
def server(tmp_path):
    return mcp.Server(root=tmp_path)


# ---------------------------------------------------------------- protocol


def test_initialize_negotiates_the_protocol_version(server):
    for asked, agreed in (("2025-03-26", "2025-03-26"), ("2024-11-05", "2024-11-05"), ("1999-01-01", "2025-06-18"), (None, "2025-06-18")):
        params = {"protocolVersion": asked, "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}} if asked else {}
        result = rpc(server, "initialize", params)["result"]
        assert result["protocolVersion"] == agreed
        assert result["capabilities"] == {"tools": {"listChanged": False}}
        assert result["serverInfo"]["name"] == "llmshift" and result["instructions"]
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None and server.initialized


def test_tools_list_schemas(server):
    for version in mcp.PROTOCOL_VERSIONS:
        rpc(server, "initialize", {"protocolVersion": version})
        tools = {t["name"]: t for t in rpc(server, "tools/list")["result"]["tools"]}
        assert TOOL_NAMES <= set(tools)
        for spec in tools.values():
            schema = spec["inputSchema"]
            assert schema["type"] == "object" and schema["additionalProperties"] is False
            assert set(schema.get("required", [])) <= set(schema["properties"])
            assert all(isinstance(p, dict) and ("type" in p or "enum" in p) for p in schema["properties"].values())
            assert len(spec["description"]) > 80
            assert ("annotations" in spec) == (version >= "2025-03-26") and ("title" in spec) == (version >= "2025-06-18")
        if version >= "2025-03-26":
            assert tools["llmshift_scan"]["annotations"]["readOnlyHint"] and tools["llmshift_compare"]["annotations"]["openWorldHint"]
            assert tools["llmshift_snapshot"]["annotations"]["destructiveHint"]


def test_protocol_errors(server):
    def code(message):
        return server.handle(message)["error"]["code"]

    assert code({"id": 1, "method": "ping"}) == mcp.INVALID_REQUEST  # no "jsonrpc"
    assert code({"jsonrpc": "1.0", "id": 1, "method": "ping"}) == mcp.INVALID_REQUEST
    assert code({"jsonrpc": "2.0", "id": None, "method": "ping"}) == mcp.INVALID_REQUEST
    assert code({"jsonrpc": "2.0", "id": True, "method": "ping"}) == mcp.INVALID_REQUEST
    assert code({"jsonrpc": "2.0", "id": 2.5, "method": "ping"}) == mcp.INVALID_REQUEST
    assert code({"jsonrpc": "2.0", "id": 1, "method": 7}) == mcp.INVALID_REQUEST
    assert code({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": [1]}) == mcp.INVALID_PARAMS
    unknown = server.handle({"jsonrpc": "2.0", "id": "abc", "method": "resources/list"})
    assert unknown["id"] == "abc" and unknown["error"]["code"] == mcp.METHOD_NOT_FOUND
    assert server.handle([])["error"]["code"] == mcp.INVALID_REQUEST
    assert server.handle(42)["error"]["code"] == mcp.INVALID_REQUEST
    # Notifications and stray responses get no reply.
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/whatever"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None
    batch = server.handle([{"jsonrpc": "2.0", "id": 1, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"jsonrpc": "2.0", "id": 2, "method": "nope"}])
    assert [r["id"] for r in batch] == [1, 2] and batch[0]["result"] == {} and batch[1]["error"]["code"] == mcp.METHOD_NOT_FOUND
    assert rpc(server, "ping")["result"] == {}


def test_tools_call_argument_errors(server):
    def err(params):
        reply = rpc(server, "tools/call", params)
        return reply["error"]["code"], reply["error"]["message"]

    assert err({"name": "llmshift_nope"})[0] == mcp.INVALID_PARAMS
    assert err({})[0] == mcp.INVALID_PARAMS
    assert err({"name": "llmshift_scan", "arguments": [1]})[0] == mcp.INVALID_PARAMS
    code, message = err({"name": "llmshift_scan", "arguments": {"warn_days": "soon"}})
    assert code == mcp.INVALID_PARAMS and "warn_days" in message
    code, message = err({"name": "llmshift_scan", "arguments": {"pathz": ["."]}})
    assert code == mcp.INVALID_PARAMS and "pathz" in message
    code, message = err({"name": "llmshift_snapshot", "arguments": {}})
    assert code == mcp.INVALID_PARAMS and "sources" in message
    assert err({"name": "llmshift_compare", "arguments": {"min_pass_rate": 1.5}})[0] == mcp.INVALID_PARAMS


# ------------------------------------------------------------------- tools


def test_scan_and_models_tools(project, server):
    rpc(server, "initialize", {"protocolVersion": "2025-06-18"})
    data = payload(call(server, "llmshift_scan", {"cwd": str(project)}))
    by_model = {m["model"]: m for m in data["models"]}
    retired = by_model["claude-3-5-haiku-20241022"]
    assert retired["status"] == "retired" and retired["replacement"] and "app/summaries.py:10" in retired["locations"]
    assert data["would_fail_ci"] is True and data["summary"]["retired"] == 1 and "next" in data
    assert payload(call(server, "llmshift_scan", {"cwd": str(project), "fail_on": "never"}))["would_fail_ci"] is False
    one = payload(call(server, "llmshift_models", {"model": "gpt-4o-2024-05-13"}))["lookup"]
    assert one["status"] == "retiring" and one["days_left"] == 2
    found = payload(call(server, "llmshift_models", {"query": "claude-3-5", "status": "retired", "limit": 2}))
    assert found["matches"] >= 2 and len(found["models"]) == 2 and found["not_shown"] == found["matches"] - 2
    soon = payload(call(server, "llmshift_models", {"retiring_within_days": 7}))
    assert all(m["status"] == "retiring" and m["days_left"] <= 7 for m in soon["models"])
    rpc(server, "initialize", {"protocolVersion": "2024-11-05"})  # older clients get text content only
    old = call(server, "llmshift_models", {"model": "gpt-4o"})["result"]
    assert "structuredContent" not in old and json.loads(old["content"][0]["text"])["lookup"]["model"] == "gpt-4o"


def test_compare_fix_and_doctor_tools(project, server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value-123")
    rpc(server, "initialize", {"protocolVersion": "2025-06-18"})
    changed = payload(call(server, "llmshift_compare", {"cwd": str(project), "to": "mock:nextgen", "max_cases_shown": 3}))
    assert changed["ok"] is False and "below the 95% threshold" in changed["verdict"]
    assert changed["summary"]["total"] == 18 and changed["summary"]["failed"] == 12
    assert len(changed["cases"]) == 3 and changed["cases_not_shown"] == 15 and changed["cases"][0]["problems"]
    assert "format" in changed["what_changed"] and "next" in changed
    assert all((project / r).exists() for r in changed["reports"]) and len(changed["reports"]) == 2
    same = payload(call(server, "llmshift_compare", {"cwd": str(project), "to": "mock:legacy"}))
    assert same["ok"] is True and same["summary"]["passed"] == 18 and same["cases"] == []
    subset = payload(call(server, "llmshift_compare", {"cwd": str(project), "to": "mock:nextgen", "groups": ["triage"], "limit": 2}))
    assert subset["summary"]["total"] == 2
    fixed = payload(call(server, "llmshift_fix", {"cwd": str(project), "to": "mock:nextgen", "write_tests": "fixed.jsonl"}))
    triage = next(g for g in fixed["groups"] if g["group"].startswith("support-ticket-triage"))
    assert triage["improved"] and any("code fences" in line for line in triage["added_instructions"])
    assert any(loc.startswith("app/triage.py:") for loc in triage["prompt_locations"]) and triage["suggested_prompt"]
    assert fixed["after"]["pass_rate"] > fixed["before"]["pass_rate"] and (project / "fixed.jsonl").exists()
    again = error_text(call(server, "llmshift_fix", {"cwd": str(project), "to": "mock:nextgen", "write_tests": "fixed.jsonl"}))
    assert "already exists" in again
    reply = call(server, "llmshift_doctor", {"cwd": str(project)})
    doctor = payload(reply)
    assert doctor["tests"] == {"path": ".llmshift/tests.jsonl", "exists": True, "cases": 18}
    assert "OPENAI_API_KEY" in doctor["credentials_set"] and doctor["models"]["candidate"]["provider"] == "MockProvider"
    assert "sk-secret-value-123" not in json.dumps(reply)


def test_snapshot_tool_modes(tmp_path, server):
    root = create_project(str(tmp_path / "proj"))
    args = {"cwd": str(root), "sources": ["logs"]}
    first = payload(call(server, "llmshift_snapshot", args))
    assert first["cases_written"] == 18 and first["tests_file"] == ".llmshift/tests.jsonl" and first["prompt_templates"] == 3
    assert "already exists" in error_text(call(server, "llmshift_snapshot", args))
    appended = payload(call(server, "llmshift_snapshot", {**args, "mode": "append"}))
    assert appended["cases_written"] == 0 and appended["total_cases"] == 18
    assert payload(call(server, "llmshift_snapshot", {**args, "mode": "overwrite"}))["cases_written"] == 18
    other = payload(call(server, "llmshift_snapshot", {**args, "output": "only-openai.jsonl", "models": ["gpt-4o*"]}))
    assert other["cases_written"] == 14
    nothing = error_text(call(server, "llmshift_snapshot", {**args, "output": "none.jsonl", "models": ["no-such-model"]}))
    assert "no test cases were created" in nothing and "different model" in nothing


def test_tool_failures_are_error_results(tmp_path, server):
    assert "not found" in error_text(call(server, "llmshift_compare", {"to": "mock:legacy"}))
    assert "is not a directory" in error_text(call(server, "llmshift_scan", {"cwd": "no/such/dir"}))
    assert "no such file or directory" in error_text(call(server, "llmshift_scan", {"paths": ["missing.py"]}))
    root = create_project(str(tmp_path / "proj"))
    cases = build_snapshot([str(root / "logs")]).cases
    save_tests(cases, root / ".llmshift" / "tests.jsonl")
    save_tests(cases, tmp_path / ".llmshift" / "tests.jsonl")  # tests, but no llmshift.yaml naming a candidate
    assert "which model" in error_text(call(server, "llmshift_compare", {}))
    assert "cannot tell which provider" in error_text(call(server, "llmshift_compare", {"cwd": str(root), "to": "mystery-model-9"}))
    assert os.getcwd() != str(root)  # the working directory is restored


# --------------------------------------------------------------- transport


class Feeder:
    """A blocking line source standing in for stdin."""

    def __init__(self):
        self.lines: queue.Queue = queue.Queue()

    def put(self, data):
        self.lines.put(data)

    def close(self):
        self.lines.put(b"")

    def readline(self, limit=-1):
        return self.lines.get(timeout=30)


def _lines(raw: bytes) -> list[dict]:
    out = []
    for line in raw.split(b"\n"):
        if line:
            message = json.loads(line)
            assert message["jsonrpc"] == "2.0" and ("result" in message) != ("error" in message)
            out.append(message)
    return out


def test_stdio_transport_keeps_stdout_clean(server, monkeypatch, capsys):
    noisy = mcp.Tool("noisy", "noisy", "prints to stdout", {"type": "object", "properties": {}, "additionalProperties": False},
                     lambda args, cfg: print("stray print") or mcp.con.out("console line") or {"done": True})
    monkeypatch.setitem(mcp.TOOLS, "noisy", noisy)
    monkeypatch.setattr(mcp, "MAX_MESSAGE_BYTES", 2000)
    lines = [
        b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}\r\n',
        b"\r\n",
        b"   \n",
        b"\xef\xbb\xbf" + b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n',  # UTF-8 BOM
        b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"noisy","arguments":{}}}\n',
        b"{not json\n",
        b'{"jsonrpc":"2.0","id":3,"method":"ping","params":{"note":"\xff\xfe broken utf-8"}}\n',
        b'{"jsonrpc":"2.0","id":4,"method":"ping","params":{"pad":"' + b"x" * 5000 + b'"}}\n',
        b"[" * 1500 + b"\n",  # deeper than the JSON parser allows
        b'{"jsonrpc":"2.0","id":7,"method":"ping"}',  # no final newline before EOF
    ]
    out = io.BytesIO()
    server.serve(io.BytesIO(b"".join(lines)), out)
    replies = _lines(out.getvalue())
    by_id = {r["id"]: r for r in replies if r.get("id") is not None}
    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    assert json.loads(by_id[2]["result"]["content"][0]["text"]) == {"done": True}
    assert by_id[3]["result"] == {} and by_id[7]["result"] == {}
    errors = [r["error"] for r in replies if r.get("id") is None]
    assert sorted(e["code"] for e in errors) == [mcp.PARSE_ERROR, mcp.PARSE_ERROR, mcp.INVALID_REQUEST]  # bad JSON, too deep, too large
    assert 4 not in by_id
    err = capsys.readouterr().err
    assert "stray print" in err and "console line" in err


def test_stdio_batches(server):
    out = io.BytesIO()
    server.serve(io.BytesIO(b'[{"jsonrpc":"2.0","id":5,"method":"ping"},{"jsonrpc":"2.0","id":6,"method":"tools/list"}]\n'), out)
    (batch,) = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["id"] for r in batch] == [5, 6] and batch[1]["result"]["tools"]


def test_cancelling_a_running_tool(server, monkeypatch):
    started = threading.Event()

    def slow(args, cfg):
        started.set()
        deadline = time.time() + 20
        while not http.CANCEL.is_set() and time.time() < deadline:
            time.sleep(0.01)
        return {"finished": True}

    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    monkeypatch.setitem(mcp.TOOLS, "slow", mcp.Tool("slow", "slow", "waits until cancelled", schema, slow))
    feeder, out = Feeder(), io.BytesIO()
    worker = threading.Thread(target=server.serve, args=(feeder, out))
    worker.start()
    try:
        _cancel_flow(feeder, out, started)
    finally:
        feeder.close()
        worker.join(30)
        http.CANCEL.clear()
    assert not worker.is_alive()
    ids = [r["id"] for r in _lines(out.getvalue())]
    assert "call-1" not in ids and 3 not in ids and ids.count(2) == 1 and 4 in ids  # no reply for cancelled requests


def _cancel_flow(feeder, out, started):
    feeder.put(b'{"jsonrpc":"2.0","id":"call-1","method":"tools/call","params":{"name":"slow"}}\n')
    assert started.wait(10)
    feeder.put(b'{"jsonrpc":"2.0","id":2,"method":"ping"}\n')  # answered while the tool runs
    for _ in range(200):
        if b'"id":2' in out.getvalue():
            break
        time.sleep(0.01)
    assert b'"id":2' in out.getvalue() and b"call-1" not in out.getvalue()
    feeder.put(b'{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"call-1","reason":"user"}}\n')
    feeder.put(b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"slow"}}\n')
    feeder.put(b'{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":3}}\n')
    feeder.put(b'{"jsonrpc":"2.0","id":4,"method":"ping"}\n')
    for _ in range(3000):
        if b'"id":4' in out.getvalue():
            return
        time.sleep(0.01)


def test_cli_registration(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "EXTENSIONS", ("mcp",))
    args = cli.build_parser().parse_args(["mcp", "--cwd", str(tmp_path)])
    assert args.func is mcp.cmd_mcp and args.cwd == str(tmp_path)
    assert cli.main(["mcp", "--cwd", str(tmp_path / "missing")]) == cli.EXIT_USAGE


# --------------------------------------------------------------- subprocess


class Client:
    def __init__(self, cwd: Path):
        env = dict(os.environ, NO_COLOR="1", LLMSHIFT_TODAY="2026-10-21", PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
        self.proc = subprocess.Popen([sys.executable, "-m", "llmshift.mcp"], cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.raw: list[bytes] = []
        self.replies: queue.Queue = queue.Queue()
        self.stderr: list[bytes] = []
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=lambda: self.stderr.extend(iter(self.proc.stderr.readline, b"")), daemon=True).start()

    def _read(self):
        for line in iter(self.proc.stdout.readline, b""):
            self.raw.append(line)
            self.replies.put(json.loads(line))

    def send(self, data):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode() + b"\n"
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def reply(self, timeout=120):
        return self.replies.get(timeout=timeout)

    def request(self, method, params=None, rid=1):
        message = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.reply()
        assert response["id"] == rid
        return response


def test_stdio_server_end_to_end(project):
    client = Client(project)
    try:
        init = client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}})
        assert init["result"]["serverInfo"]["name"] == "llmshift"
        client.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = client.request("tools/list", rid=2)["result"]["tools"]
        assert TOOL_NAMES <= {t["name"] for t in tools}
        scan = payload(client.request("tools/call", {"name": "llmshift_scan", "arguments": {}}, rid=3))
        assert any(m["model"] == "claude-3-5-haiku-20241022" and m["status"] == "retired" for m in scan["models"])
        changed = payload(client.request("tools/call", {"name": "llmshift_compare", "arguments": {"to": "mock:nextgen"}}, rid=4))
        assert changed["ok"] is False and changed["what_changed"]["format"]["failed"] == 8
        same = payload(client.request("tools/call", {"name": "llmshift_compare", "arguments": {"to": "mock:legacy"}}, rid=5))
        assert same["ok"] is True and same["summary"]["passed"] == 18
        client.send(b'{"jsonrpc": "2.0", "id": 6, "method": \n')
        assert client.reply()["error"]["code"] == mcp.PARSE_ERROR
        assert client.request("tools/destroy", rid=7)["error"]["code"] == mcp.METHOD_NOT_FOUND
        bad = client.request("tools/call", {"name": "llmshift_compare", "arguments": {"to": 5}}, rid=8)
        assert bad["error"]["code"] == mcp.INVALID_PARAMS
        assert client.request("ping", rid=9)["result"] == {}
        client.proc.stdin.close()
        assert client.proc.wait(timeout=60) == 0
    finally:
        if client.proc.poll() is None:
            client.proc.kill()
    time.sleep(0.2)
    assert client.raw and all(line.endswith(b"\n") for line in client.raw)
    for line in client.raw:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0" and ("result" in message) != ("error" in message)
    assert b"MCP server on stdio" in b"".join(client.stderr)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_tools_that_run_git_answer_promptly_over_stdio(project):
    # On Windows a child process that inherits the server's stdin, a pipe the reader
    # thread is blocked on, hangs at startup: scan waited out git's 30 s timeout, and
    # check (git without a timeout) never answered.
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "x"], cwd=project, check=True)
    client = Client(project)
    try:
        client.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}})
        start = time.monotonic()
        scan = payload(client.request("tools/call", {"name": "llmshift_scan", "arguments": {}}, rid=2))
        check = client.request("tools/call", {"name": "llmshift_check", "arguments": {"base": "HEAD", "model": "mock:legacy"}}, rid=3)
        assert time.monotonic() - start < 20
        assert scan["models"] and "result" in check
        assert client.request("ping", rid=4)["result"] == {}  # the protocol stream is intact
        client.proc.stdin.close()
        assert client.proc.wait(timeout=60) == 0
    finally:
        if client.proc.poll() is None:
            client.proc.kill()


CHILD_IN_SERVER = """
import subprocess, sys
from llmshift import mcp
reader, writer = mcp._protocol_streams()
# A child started with inherited streams, as any library might start one.
subprocess.run([sys.executable, "-c", "import sys; print('LEAK'); print('child read', repr(sys.stdin.read()), file=sys.stderr)"], timeout=60)
writer.write(b"protocol\\n")
writer.flush()
print("server read", reader.readline().decode().strip(), file=sys.stderr)
"""


def test_child_processes_can_neither_read_nor_write_the_protocol_streams():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    proc = subprocess.run([sys.executable, "-c", CHILD_IN_SERVER], input=b'{"jsonrpc": "2.0"}\n', capture_output=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == b"protocol\n"  # the child's print went to stderr
    err = proc.stderr.decode()
    assert "LEAK" in err and "child read ''" in err  # the child got the null device
    assert 'server read {"jsonrpc": "2.0"}' in err  # and did not steal the next message
