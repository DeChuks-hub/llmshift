"""scan: model detection across languages + lifecycle registry lookups."""

from __future__ import annotations

import json

import pytest

from llmshift import report
from llmshift.registry import Registry
from llmshift.scanner import is_model_key, looks_like_known_model, scan_paths, should_fail

PY = '''"""Docstring mentioning gpt-4 must be ignored."""
import os
from openai import OpenAI

MODEL = "gpt-4o-2024-05-13"
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
client = OpenAI()

def run(text):
    # model="gpt-4" in a comment is ignored
    client.chat.completions.create(model=MODEL, messages=[])
    anthropic_client.messages.create(model="claude-3-5-sonnet-20241022", max_tokens=5, messages=[])
    payload = {"model": "o3", "input": text}
    candidates = ["gpt-6-astra", "claude-opus-5-5"]
    llm = ChatOpenAI(model=os.environ.get("LLM", "gpt-4o-mini"))
    GenerativeModel("gemini-2.0-flash")
    Agent("Support bot")  # not a model
    client.responses.create(model="ft:gpt-4o-mini-2024-07-18:acme::x1", input=text)
'''

TS = """import OpenAI from "openai";
const models = [
  'gpt-4.1-nano',
  "gemini-2.0-flash",
];
// const legacy = "text-davinci-003";
export const ask = () => client.chat.completions.create({ model: process.env.MODEL ?? "gpt-5", messages: [] });
const pkg = "gpt-3-encoder";
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "web").mkdir()
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "src" / "app.py").write_text(PY, encoding="utf-8")
    (tmp_path / "web" / "chat.ts").write_text(TS, encoding="utf-8")
    (tmp_path / "config.yaml").write_text("llm:\n  model: claude-sonnet-4-5\n  embedding_model: text-embedding-3-small\n  deployment_name: prod-gpt4o\n  v-model: email\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENAI_MODEL=gpt-4-turbo\nMODEL_PATH=./weights.bin\nBEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("We used gpt-3.5-turbo-0301 once.\n", encoding="utf-8")
    (tmp_path / "node_modules" / "x" / "index.js").write_text('const m = "gpt-4";\n', encoding="utf-8")
    nb = {"cells": [{"cell_type": "markdown", "source": ["gpt-4o"]}, {"cell_type": "code", "source": ["%pip install openai\n", "client.chat.completions.create(model=\"gpt-4-0613\", messages=[])\n"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (tmp_path / "explore.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_scan_finds_models(project):
    res = scan_paths(["."])
    found = {(f.file, f.line, f.model, f.kind) for f in res.findings}
    expected = {
        ("src/app.py", 5, "gpt-4o-2024-05-13", "constant"),
        ("src/app.py", 6, "gpt-3.5-turbo", "env-default"),
        ("src/app.py", 11, "gpt-4o-2024-05-13", "call"),
        ("src/app.py", 12, "claude-3-5-sonnet-20241022", "call"),
        ("src/app.py", 13, "o3", "call"),
        ("src/app.py", 14, "gpt-6-astra", "string"),
        ("src/app.py", 14, "claude-opus-5-5", "string"),
        ("src/app.py", 15, "gpt-4o-mini", "env-default"),
        ("src/app.py", 16, "gemini-2.0-flash", "call"),
        ("src/app.py", 18, "ft:gpt-4o-mini-2024-07-18:acme::x1", "call"),
        ("web/chat.ts", 3, "gpt-4.1-nano", "string"),
        ("web/chat.ts", 4, "gemini-2.0-flash", "string"),
        ("web/chat.ts", 7, "gpt-5", "call"),
        ("config.yaml", 2, "claude-sonnet-4-5", "config"),
        ("config.yaml", 3, "text-embedding-3-small", "config"),
        ("config.yaml", 4, "prod-gpt4o", "config"),
        (".env", 1, "gpt-4-turbo", "config"),
        (".env", 3, "anthropic.claude-3-haiku-20240307-v1:0", "config"),
        ("explore.ipynb", 2, "gpt-4-0613", "call"),
    }
    assert expected <= found, expected - found
    models = {f.model for f in res.findings}
    for absent in ("gpt-4", "text-davinci-003", "gpt-3-encoder", "email", "./weights.bin", "Support bot", "gpt-3.5-turbo-0301"):
        assert absent not in models  # docstrings, comments, packages, UI bindings, paths, docs (off by default), node_modules
    via = next(f for f in res.findings if f.line == 11)
    assert via.via == "MODEL, line 5" and via.call == "client.chat.completions.create"
    nb = next(f for f in res.findings if f.file == "explore.ipynb")
    assert nb.cell == 2 and nb.location == "explore.ipynb:2 (cell 2)"
    statuses = {f.model: f.lookup.status for f in res.findings}
    assert statuses["claude-3-5-sonnet-20241022"] == "retired"
    assert statuses["gpt-4o-2024-05-13"] == "retiring"
    assert statuses["anthropic.claude-3-haiku-20240307-v1:0"] == "retired"
    assert statuses["gpt-6-astra"] == "active" and statuses["prod-gpt4o"] == "unknown"
    with_docs = scan_paths(["."], include_docs=True)
    assert "gpt-3.5-turbo-0301" in {f.model for f in with_docs.findings}
    excluded = scan_paths(["."], exclude=["web"])
    assert not any(f.file.startswith("web/") for f in excluded.findings)


def test_fail_on_levels(project):
    res = scan_paths(["."])
    assert should_fail(res, "retired")
    assert not should_fail(res, "never")
    healthy = project / "healthy"
    healthy.mkdir()
    (healthy / "a.py").write_text('client.create(model="gpt-4o")\n', encoding="utf-8")  # retires 2027-01-14 (85 days)
    res = scan_paths([str(healthy)])
    assert not should_fail(res, "retired")
    assert should_fail(res, "retiring")
    assert not should_fail(scan_paths([str(healthy)], warn_days=30), "retiring")
    assert should_fail(scan_paths([str(healthy)], warn_days=30), "deprecated")


def test_renderers(project, monkeypatch):
    res = scan_paths(["."])
    text = report.scan_terminal(res)
    assert "RETIRED" in text and "claude-3-5-sonnet-20241022" in text and "2 DAYS LEFT" in text
    md = report.scan_markdown(res)
    assert "| Status | Model |" in md and "`claude-3-5-sonnet-20241022`" in md
    sarif = report.scan_sarif(res)
    rules = {r["ruleId"] for r in sarif["runs"][0]["results"]}
    assert {"retired-model", "retiring-model", "unknown-model"} <= rules
    notes = report.github_annotations(res)
    assert any(n.startswith("::error file=src/app.py,line=12") for n in notes)
    assert all("%0A" not in n.split("::", 2)[1] for n in notes)
    data = res.to_dict()
    assert data["summary"]["retired"] >= 2 and data["models"][0]["lifecycle"]["status"] == "retired"


def test_key_and_model_heuristics():
    for key in ("model", "MODEL", "modelName", "OPENAI_MODEL_NAME", "chatModel", "deployment_name", "BEDROCK_MODEL_ID", "base_model"):
        assert is_model_key(key), key
    for key in ("v-model", "model_path", "model_config", "modelValue", "data_model", "remodel"):
        assert not is_model_key(key), key
    for value in ("gpt-4o", "o4-mini", "claude-3-5-haiku@20241022", "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "gemini-2.5-pro", "models/gemini-1.5-flash", "mistral-large-latest", "command-r-plus", "deepseek-chat", "grok-4.7", "meta-llama/Llama-3.1-8B-Instruct", "openrouter/openai/gpt-4o"):
        assert looks_like_known_model(value), value
    for value in ("gpt-3-encoder", "claude-code", "gemini-cli", "command-line", "mistral-common", "o1-is-a-variable", "llama-index"):
        assert not looks_like_known_model(value), value


def test_registry_lookups(monkeypatch):
    reg = Registry.load()
    assert reg.updated == "2026-10-21" and len(reg.entries) > 500
    lk = reg.lookup("gpt-4o-2024-05-13")
    assert lk.status == "retiring" and lk.days_left == 2 and lk.replacement == "gpt-5.6-sol" and lk.severity == "warning"
    assert reg.lookup("openrouter/openai/gpt-4o-mini").provider == "openai"
    ft = reg.lookup("ft:gpt-4o-mini-2024-07-18:acme::abc")
    assert ft.fine_tune_of == "gpt-4o-mini-2024-07-18" and ft.message.startswith("fine-tune of")
    bedrock = reg.lookup("anthropic.claude-v2:1")
    assert bedrock.model == "claude-2.1" and bedrock.status == "retired" and "Amazon Bedrock" in bedrock.message
    vertex = reg.lookup("claude-3-5-haiku@20241022")
    assert vertex.provider == "google-vertex" and vertex.status == "retired"
    gem = reg.lookup("gemini-2.5-flash")
    assert gem.status == "active" and gem.severity == "notice" and any("Vertex AI" in p for p in gem.other_platforms)
    assert reg.lookup("gemini-2.5-flash", platform="vertex").status == "retired"
    alias = reg.lookup("gpt-5.6")
    assert alias.alias and alias.severity == "info"
    snap = reg.lookup("gpt-4o-2099-01-01")
    assert snap.status == "unknown" and "unknown snapshot of gpt-4o" in snap.message
    monkeypatch.setenv("LLMSHIFT_TODAY", "2026-10-23")
    assert reg.lookup("gpt-4o-2024-05-13").status == "retired"  # flips on its shutdown day


def test_registry_override_file(tmp_path, monkeypatch):
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps({"models": [{"id": "my-ft-model", "provider": "openai", "status": "deprecated", "retires": "2026-11-01", "replacement": "gpt-6-luna"}, {"id": "gpt-6-astra", "provider": "openai", "status": "deprecated", "retires": "2026-12-01"}]}), encoding="utf-8")
    reg = Registry.load([str(extra)])
    assert reg.lookup("my-ft-model").status == "retiring"
    assert reg.lookup("gpt-6-astra").retires.isoformat() == "2026-12-01"  # overrides bundled entry
    monkeypatch.setenv("LLMSHIFT_REGISTRY", str(extra))
    assert Registry.load().lookup("my-ft-model").replacement == "gpt-6-luna"
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(ValueError):
        Registry.load([str(bad)])


def test_registry_search():
    reg = Registry.load()
    retiring = reg.search("gpt-4o", provider="openai", status="retiring")
    assert retiring and all(lk.status == "retiring" and "gpt-4o" in lk.model for lk in retiring)
    assert retiring == sorted(retiring, key=lambda lk: lk.retires)


def test_models_named_in_comments_are_not_calls(in_tmp, capsys):
    """Commented-out settings, trailing notes and /* blocks */ are not what the code
    calls: scan must not fail CI on them, and upgrade must not rewrite them."""
    from llmshift import cli

    (in_tmp / "settings.yaml").write_text("# model: gpt-4-turbo (old)\nmodel: gpt-4o-2024-05-13  # was claude-3-opus-20240229\n", encoding="utf-8")
    (in_tmp / "app.ini").write_text("[llm]\n; model = gpt-4-turbo\nmodel = gpt-4o-2024-05-13 ; old gpt-4-0613\n", encoding="utf-8")
    (in_tmp / ".env").write_text("OPENAI_MODEL=gpt-4o-2024-05-13\n# OLD_MODEL=gpt-4-turbo\n", encoding="utf-8")
    (in_tmp / "app.ts").write_text('const a = { model: "gpt-4o-2024-05-13" }; // was "gpt-4-turbo"\n/* old:\n   model: "claude-3-opus-20240229"\n*/\nconst b = { model: "gpt-4o-mini" };\n', encoding="utf-8")
    result = scan_paths(["."])
    assert sorted((f.file, f.line, f.model) for f in result.findings) == [
        (".env", 1, "gpt-4o-2024-05-13"), ("app.ini", 3, "gpt-4o-2024-05-13"), ("app.ts", 1, "gpt-4o-2024-05-13"),
        ("app.ts", 5, "gpt-4o-mini"), ("settings.yaml", 2, "gpt-4o-2024-05-13"),
    ]
    assert cli.main(["upgrade", "--from", "claude-3-opus-20240229", "--to", "claude-opus-4-8", "--apply"]) == 0
    assert "claude-3-opus-20240229" in (in_tmp / "settings.yaml").read_text(encoding="utf-8")  # the comment is untouched


def test_provider_prefixes_arns_routers_and_labels(in_tmp):
    (in_tmp / "agents.py").write_text(
        "from pydantic_ai import Agent\nfrom langchain.chat_models import init_chat_model\n"
        'agent = Agent("openai:gpt-4o-2024-05-13")\nllm = init_chat_model("anthropic:claude-3-7-sonnet-20250219")\n'
        'other = init_chat_model(model="google_genai:gemini-2.0-flash")\nGPT4_LABEL = "GPT-4"\n', encoding="utf-8")
    arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    (in_tmp / "config.yaml").write_text(f"llm:\n  modelId: {arn}\n  router: anthropic/claude-3.5-sonnet\n", encoding="utf-8")
    (in_tmp / "NOTES.md").write_text("We used claude-3-opus-20240229. It worked.\n", encoding="utf-8")
    found = {f.model: f.lookup for f in scan_paths(["."], include_docs=True).findings}
    assert set(found) == {"openai:gpt-4o-2024-05-13", "anthropic:claude-3-7-sonnet-20250219", "google_genai:gemini-2.0-flash", arn, "anthropic/claude-3.5-sonnet", "claude-3-opus-20240229"}
    assert all(lk.status in ("retired", "retiring") for lk in found.values())
    router = found["anthropic/claude-3.5-sonnet"]
    assert router.provider == "anthropic" and router.replacement == "claude-sonnet-5-5"  # not Vertex's undated entry
    assert found[arn].platform == "bedrock"


def test_llmshift_reports_are_not_scanned(in_tmp, capsys, monkeypatch):
    """Calendars, compare results, JUnit and SARIF files name hundreds of models: they are
    llmshift's own findings, not models the code calls."""
    from llmshift import cli
    from llmshift.demo import create_project

    root = create_project(str(in_tmp / "app"))
    monkeypatch.chdir(root)
    before = {(f.file, f.model) for f in scan_paths(["."]).findings}
    assert cli.main(["snapshot", "logs"]) == 0
    assert cli.main(["calendar", "--out", "site"]) == 0
    assert cli.main(["compare", "--to", "mock:frugal", "--json", "results.json", "--junit", "junit.xml"]) == 0
    capsys.readouterr()
    assert cli.main(["scan", ".", "--format", "sarif"]) in (0, 1)
    (root / "scan.sarif").write_text(capsys.readouterr().out, encoding="utf-8")
    assert {(f.file, f.model) for f in scan_paths(["."]).findings} == before


def test_every_known_model_id_passes_the_trigger_prefilter():
    """has_trigger runs before the big model-id regex: an id with none of its triggers
    would never be found."""
    from llmshift.scanner import KNOWN_RE, has_trigger

    ids = {v for e in Registry.load().entries for v in (e.get("id"), e.get("replacement")) if isinstance(v, str) and v}
    known = sorted(i for i in ids if KNOWN_RE.fullmatch(i))
    assert len(known) > 300
    assert [i for i in known if not has_trigger(i.lower())] == []


def test_comment_stripping_skips_lines_that_cannot_hold_a_model(monkeypatch):
    """Only lines with model words, or a /* block */ to follow, go through strip_comment:
    walking every character of every line made scan twice as slow on large repositories."""
    from llmshift import scanner

    seen = []
    real = scanner.strip_comment
    monkeypatch.setattr(scanner, "strip_comment", lambda line, markers, state: seen.append(line) or real(line, markers, state))
    lines = [
        "const a = 1;",
        "let total = price * count;",
        "/* the old settings",
        '   model: "gpt-4-turbo" */',
        "const b = 2;",
        '// model: "gpt-4o-mini"',
        'const model = "gpt-4o"; // was gpt-4',
    ]
    found = scanner.scan_generic("\n".join(lines) + "\n", code=True, comments=scanner.COMMENT_SYNTAX[".ts"])
    assert [(lineno, value) for lineno, _col, value, *_ in found] == [(7, "gpt-4o")]
    assert seen == [lines[2], lines[3], lines[6]]
