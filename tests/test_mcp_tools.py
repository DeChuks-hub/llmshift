"""The MCP tools for the newer commands: pick, safety, check, upgrade, and compare repeats."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from llmshift import mcp
from llmshift.demo import create_project
from llmshift.snapshot import build_snapshot
from llmshift.testset import save_tests
from test_mcp import call, payload, rpc


@pytest.fixture
def project(tmp_path):
    root = create_project(str(tmp_path / "proj"))
    save_tests(build_snapshot([str(root / "logs")]).cases, root / ".llmshift" / "tests.jsonl")
    return root


@pytest.fixture
def server(project):
    srv = mcp.Server(root=project)
    rpc(srv, "initialize", {"protocolVersion": "2025-06-18"})
    return srv


def test_new_tools_are_listed(server):
    names = {t["name"] for t in rpc(server, "tools/list")["result"]["tools"]}
    assert {"llmshift_pick", "llmshift_safety", "llmshift_check", "llmshift_upgrade"} <= names


def test_pick_tool_recommends_the_cheapest_passing_model(server):
    data = payload(call(server, "llmshift_pick", {"models": ["mock:nextgen", "mock:frugal"], "volume": 100000}))
    assert data["recommended"] == "mock:frugal" and data["monthly_saving"] > 0


def test_pick_tool_reads_the_volume_as_written_in_llmshift_yaml(server, project):
    with open(project / "llmshift.yaml", "a", encoding="utf-8") as fh:
        fh.write("pick:\n  volume: 100k\n")
    data = payload(call(server, "llmshift_pick", {"models": ["mock:frugal"]}))
    assert data["volume_per_month"] == 100_000 and data["monthly_saving"] > 0


def test_safety_tool_finds_the_riskier_model(server):
    data = payload(call(server, "llmshift_safety", {"to": "mock:nextgen", "baseline_model": "mock:legacy", "categories": ["injection"], "probes": 6}))
    assert data["riskier"] == ["injection"]


def test_compare_tool_repeats(server):
    data = payload(call(server, "llmshift_compare", {"to": "mock:legacy", "repeat": 2, "limit": 4}))
    assert data["summary"]["repeats"] == 2 and data["summary"]["flaky"] == 0 and len(data["summary"]["pass_rate_range"]) == 2


def test_check_tool_outside_git_is_a_tool_error(server):
    result = call(server, "llmshift_check", {})["result"]
    assert result["isError"] is True and "git" in result["content"][0]["text"].lower()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_check_tool_runs_only_the_edited_prompt(server, project):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args], cwd=project, check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    triage = project / "app" / "triage.py"
    text = triage.read_text(encoding="utf-8")
    assert "Respond with JSON" in text or "JSON" in text
    triage.write_text(text.replace("JSON", "raw JSON (no markdown)", 1), encoding="utf-8")
    data = payload(call(server, "llmshift_check", {"model": "mock:nextgen", "base": "HEAD"}))
    assert len(data["changes"]) == 1 and "triage" in data["changes"][0]["group"]


def test_upgrade_tool_plans_then_applies(server, project):
    plan = payload(call(server, "llmshift_upgrade", {"from": "gpt-4o-2024-05-13", "to": "gpt-5.6-sol"}))
    assert "+" in plan["diff"] and "gpt-5.6-sol" in plan["diff"] and "written" not in plan
    assert "gpt-4o-2024-05-13" in (project / "app" / "triage.py").read_text(encoding="utf-8")
    applied = payload(call(server, "llmshift_upgrade", {"from": "gpt-4o-2024-05-13", "to": "gpt-5.6-sol", "apply": True}))
    assert applied["written"] and "gpt-5.6-sol" in (project / "app" / "triage.py").read_text(encoding="utf-8")
    result = call(server, "llmshift_upgrade", {})["result"]
    assert result["isError"] is True


def test_tools_read_llmshift_yaml_like_the_command_line(server, project, monkeypatch, capsys):
    import json

    from llmshift import cli

    with open(project / "llmshift.yaml", "a", encoding="utf-8") as fh:
        fh.write("repeat: 2\nfix:\n  target_pass_rate:\nseverity:\n  json_valid: info\n  json_schema: info\n  json_match: info\n  structure: info\n")
    data = payload(call(server, "llmshift_compare", {"to": "mock:legacy", "limit": 3}))
    assert data["summary"]["repeats"] == 2
    payload(call(server, "llmshift_fix", {"to": "mock:nextgen", "limit": 4}))  # a null target is the default, not a crash
    mcp_pick = {c["model"]: c["pass_rate"] for c in payload(call(server, "llmshift_pick", {"models": ["mock:nextgen"]}))["candidates"]}
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "EXTENSIONS", ("pick",))
    capsys.readouterr()
    cli.main(["pick", "--models", "mock:nextgen", "--format", "json"])
    cli_pick = {c["model"]: c["pass_rate"] for c in json.loads(capsys.readouterr().out)["candidates"]}
    assert mcp_pick == cli_pick and mcp_pick["mock:nextgen"] > 0.4  # the severities apply to both
