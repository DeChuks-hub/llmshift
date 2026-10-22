"""Public Python API, the fix optimizer path, and shipped YAML files."""

from __future__ import annotations

import json
from pathlib import Path

from llmshift._vendor import yaml

import llmshift
from llmshift.fix import optimize_prompt, run_fix
from llmshift.providers.mock import MockProvider
from llmshift.testset import TestCase, load_tests, save_tests

ROOT = Path(__file__).resolve().parents[1]


def test_python_api(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text('client.chat.completions.create(model="gpt-4o-2024-05-13", messages=[])\n', encoding="utf-8")
    result = llmshift.scan(str(tmp_path))
    assert [f.model for f in result.findings] == ["gpt-4o-2024-05-13"]
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"prompt": "Say hi", "response": "Hello there, friend."}) + "\n", encoding="utf-8")
    cases = llmshift.snapshot(str(log))
    assert len(cases) == 1 and cases[0].baseline_output == "Hello there, friend."
    res = llmshift.compare(cases, MockProvider("same", fn=lambda r: "Hello there, friend."), show_progress=False)
    assert res.summary["passed"] == 1
    save_tests(cases, tmp_path / "t.yaml")
    assert llmshift.load_tests(tmp_path / "t.yaml")[0].id == cases[0].id
    save_tests(cases, tmp_path / "t.json")
    assert load_tests(tmp_path / "t.json")[0].baseline_output == "Hello there, friend."


def test_hand_written_yaml_tests(tmp_path):
    (tmp_path / "cases.yaml").write_text(
        """
- id: refund-window
  system: You are a support agent.
  prompt: Can I get a refund after 40 days?
  baseline: No, refunds are only available within 30 days.
  checks:
    - type: contains
      value: "30 days"
- prompt: Ping
  checks: [{type: not_empty}]
""",
        encoding="utf-8",
    )
    cases = load_tests(tmp_path / "cases.yaml")
    assert cases[0].system_prompt == "You are a support agent." and cases[0].baseline_output.startswith("No,")
    assert cases[1].id.startswith("c_") and cases[1].messages == [{"role": "user", "content": "Ping"}]


def _cases():
    out = []
    for i in range(4):
        msgs = [{"role": "system", "content": "Answer with one word."}, {"role": "user", "content": f"Question {i}"}]
        out.append(TestCase(id=f"q{i}", input={"messages": msgs, "params": {}}, baseline={"output": "yes"}, checks=[{"type": "similarity", "metric": "token_f1", "threshold": 0.99}], group="g"))
    return out


def test_fix_with_optimizer():
    seen = []

    def candidate(req):
        system = req["messages"][0]["content"]
        return "yes" if "LOWERCASE YES" in system else "Yes, absolutely!"

    def optimizer(req):
        seen.append(req["messages"][1]["content"])
        return "Here you go:\n<prompt>\nAnswer with one word. Always reply with LOWERCASE YES.\n</prompt>"

    result = run_fix(_cases(), MockProvider("cand", fn=candidate), optimizer=MockProvider("opt", fn=optimizer), show_progress=False, rounds=2)
    group = result.groups[0]
    assert group.suggested_prompt == "Answer with one word. Always reply with LOWERCASE YES."
    assert group.strategy == "optimizer" and group.after["passed"] == group.after["cases"]
    assert "Expected output (old model):\nyes" in seen[0] and "Yes, absolutely!" in seen[0]
    assert result.after["pass_rate"] == 1.0


def test_optimizer_junk_is_ignored():
    case = _cases()[0]
    from llmshift.compare import run_compare

    res = run_compare([case], MockProvider("c", fn=lambda r: "no"), show_progress=False)
    junk = MockProvider("o", fn=lambda r: "ok")
    assert optimize_prompt(junk, "Answer.", res.cases, {case.id: case}) is None


def test_shipped_yaml_is_valid():
    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite" and {"command", "args", "version"} <= set(action["inputs"])
    run_step = next(s for s in action["runs"]["steps"] if s.get("id") == "run")
    assert "${{" not in run_step["run"]  # inputs are passed via env, never interpolated into the script
    for wf in list((ROOT / "examples" / "workflows").glob("*.yml")) + [ROOT / ".github" / "workflows" / "ci.yml"]:
        assert "jobs" in yaml.safe_load(wf.read_text(encoding="utf-8")), wf
    cfg = yaml.safe_load((ROOT / "examples" / "llmshift.yaml").read_text(encoding="utf-8"))
    assert cfg["watch"]["alerts"][0]["type"] == "slack"
