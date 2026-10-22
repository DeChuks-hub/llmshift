"""Regression tests for review findings: scanning, registry, log parsing, redaction,
config, watch, CLI behaviour, test files and the OpenAI Evals importer."""

from __future__ import annotations

import gzip
import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import write_jsonl
from llmshift import cli, console, facts, formats, graders, textutil
from llmshift import jsonschema_lite as js
from llmshift.config import load_config
from llmshift.demo import create_project
from llmshift.importers import openai_evals as oe
from llmshift.logparse import parse_logs, parse_time
from llmshift.redact import Redactor
from llmshift.registry import Registry
from llmshift.scanner import looks_like_known_model, scan_paths
from llmshift.snapshot import build_snapshot
from llmshift.testset import TestCase, TestFileError, append_tests, load_tests, save_tests
from llmshift.watch import detect_drift, load_history, parse_every, run_watch
from test_importer import EVAL, ITEMS, TEMPLATE_RUN, output_item

# ------------------------------------------------------------- text & facts


def test_lcs_matches_dynamic_programming():
    def dp(a, b):
        t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                t[i + 1][j + 1] = t[i][j] + 1 if a[i] == b[j] else max(t[i][j + 1], t[i + 1][j])
        return t[-1][-1]

    rng = random.Random(3)
    for _ in range(200):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 40))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 40))]
        assert textutil.lcs_length(a, b) == dp(a, b)


def test_numbers_inside_bigger_numbers_and_ids_do_not_count():
    found = facts.extract("We shipped 10 items and 5 boxes.")
    result = facts.match(found, "We shipped 10,000 items for order #4521.")
    assert sorted(f["text"] for f in result["missing"]) == ["10", "5"]


def test_refusals_json_and_labels():
    assert formats.is_refusal("I can't help with that.")
    assert not formats.is_refusal("I can't provide a refund until the item is returned, but here is what to do.")
    assert formats.find_embedded_json("x" + "[" * 5000 + "]" * 5000) is None
    with pytest.raises(ValueError):
        formats._loads("[" * 100000 + "]" * 100000)
    assert not js.json_equal(True, 1) and js.json_equal([1, {"a": 2}], [1.0, {"a": 2.0}])
    assert graders.find_label("Label: billing", ["billing", "bill"]) == "billing"
    assert graders.score_from_text("Score: 7/10", 0, 10) == 7.0


# ------------------------------------------------------------------- scan


def test_scan_ignores_django_and_pandas_engine_settings(in_tmp):
    (in_tmp / "settings.py").write_text(
        'DATABASES = {"default": {"ENGINE": "django.db.backends.postgresql"}}\n'
        'AUTH_USER_MODEL = "accounts.User"\n'
        'df = pd.read_parquet("x.parquet", engine="pyarrow")\n'
        'DEFAULT_MODEL = "resnet50"\n',
        encoding="utf-8",
    )
    (in_tmp / "llm.py").write_text(
        'from openai import OpenAI\nDEFAULT_MODEL = "my-finetune-v2"\n'
        'client.completions.create(engine="text-curie-001", prompt="x")\n'
        'OLD = ["text-ada-001", "code-cushman-001", "pixtral-12b-2409", "voxtral-mini-latest"]\n',
        encoding="utf-8",
    )
    models = {f.model: f.file for f in scan_paths(["."]).findings}
    assert "settings.py" not in models.values()
    assert {"my-finetune-v2", "text-curie-001", "text-ada-001", "code-cushman-001", "pixtral-12b-2409", "voxtral-mini-latest"} <= set(models)


def test_every_known_pattern_passes_the_fast_prefilter():
    examples = [
        "gpt-4o", "chatgpt-4o-latest", "gpt-5-chat-latest", "o3-mini", "text-curie-001", "text-ada-001", "code-cushman-001",
        "davinci-002", "text-similarity-ada-001", "text-search-ada-doc-001", "omni-moderation-latest", "dall-e-3", "whisper-1",
        "tts-1-hd", "codex-mini-latest", "computer-use-preview", "sora-2", "ft:gpt-4o-mini:org::id1",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-haiku-20240307", "gemini-2.5-pro", "chat-bison@001",
        "textembedding-gecko@003", "text-multilingual-embedding-002", "pixtral-12b-2409", "voxtral-mini-latest", "magistral-medium-latest",
        "command-r-plus", "embed-english-v3.0", "rerank-english-v3.0", "grok-4", "deepseek-chat", "llama-3.1-70b", "openai/gpt-4o",
    ]
    for model in examples:
        assert looks_like_known_model(model), model


def test_scan_shell_long_lines_utf16_and_notebook_lines(in_tmp):
    (in_tmp / "run.sh").write_text("export OPENAI_MODEL=gpt-4o-mini\npython app.py --model claude-3-haiku-20240307\n", encoding="utf-8")
    (in_tmp / "bundle.min.js").write_text("var a=1;" * 400 + 'var m={model:"gpt-3.5-turbo-0613"};', encoding="utf-8")
    (in_tmp / "script.ps1").write_bytes('$model = "gpt-4-0613"\r\n'.encode("utf-16"))
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
            {"cell_type": "code", "metadata": {}, "execution_count": 1, "outputs": [], "source": ["import openai\n", "%pip install x\n", 'client.chat.completions.create(model="gpt-4-0314")\n']},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (in_tmp / "nb.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (in_tmp / "broken.ipynb").write_text("[1, 2]", encoding="utf-8")
    result = scan_paths(["."])
    by_model = {f.model: f for f in result.findings}
    assert {"gpt-4o-mini", "claude-3-haiku-20240307", "gpt-3.5-turbo-0613", "gpt-4-0613", "gpt-4-0314"} <= set(by_model)
    nb_finding = by_model["gpt-4-0314"]
    lines = (in_tmp / "nb.ipynb").read_text(encoding="utf-8").splitlines()
    assert nb_finding.cell == 2 and "gpt-4-0314" in lines[nb_finding.file_line - 1]
    assert any(e.startswith("broken.ipynb:") for e in result.errors)


def test_scan_excludes_and_globs(in_tmp):
    (in_tmp / "keep.py").write_text('model = "gpt-4o"\n', encoding="utf-8")
    (in_tmp / "src" / "legacy").mkdir(parents=True)
    (in_tmp / "src" / "legacy" / "old.py").write_text('model = "gpt-4-32k"\n', encoding="utf-8")
    (in_tmp / "src" / "new.py").write_text('model = "gpt-4.1"\n', encoding="utf-8")
    assert {f.model for f in scan_paths(["src"], exclude=["src/legacy"]).findings} == {"gpt-4.1"}  # cwd-relative
    assert {f.model for f in scan_paths(["."], exclude=["legacy"]).findings} == {"gpt-4o", "gpt-4.1"}  # a name at any depth
    assert not scan_paths(["."], exclude=["**/keep.py", "src"]).findings  # **/ also matches the top level
    assert not scan_paths(["keep.py"], exclude=["keep.py"]).findings  # explicit files honour --exclude
    assert {f.file for f in scan_paths(["src/*.py"]).findings} == {"src/new.py"}  # globs the shell did not expand


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_scan_in_git_repos(in_tmp):
    subprocess.run(["git", "init", "-q"], cwd=in_tmp, check=True)
    (in_tmp / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (in_tmp / "build").mkdir()
    (in_tmp / "build" / "cfg.yaml").write_text("model: gpt-4-0613\n", encoding="utf-8")
    (in_tmp / "ignored").mkdir()
    (in_tmp / "ignored" / "x.py").write_text('model = "gpt-4-0613"\n', encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=in_tmp, check=True)
    assert [f.file for f in scan_paths(["."]).findings] == ["build/cfg.yaml"]  # tracked, so scanned; ignored stays out
    assert [f.file for f in scan_paths(["ignored"]).findings] == ["ignored/x.py"]  # named explicitly, so scanned


def test_registry_resource_names_variants_and_overrides(tmp_path):
    reg = Registry.load()
    arn = reg.lookup("arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0")
    assert arn.model == "claude-3-sonnet-20240229" and arn.platform == "bedrock"
    assert reg.lookup("arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0").model == "claude-3-5-sonnet-20241022"
    assert reg.lookup("anthropic.claude-3-haiku-20240307-v1:0:200k").model == "claude-3-haiku-20240307"
    assert reg.lookup("anthropic.claude-v2:1").model == "claude-2.1"
    vertex = reg.lookup("projects/p/locations/us-central1/publishers/google/models/gemini-1.5-pro-002")
    assert vertex.model == "gemini-1.5-pro-002" and vertex.platform == "vertex"
    assert reg.lookup("anthropic/claude-3.5-sonnet").model == "claude-3-5-sonnet"
    assert reg.lookup("claude-sonnet-4.5").model == "claude-sonnet-4-5"
    override = tmp_path / "reg.json"
    override.write_bytes(b"\xef\xbb\xbf" + json.dumps([{"id": "gpt-4o", "status": "deprecated", "note": "internal policy"}]).encode())
    custom = Registry.load([str(override)])
    assert custom.lookup("gpt-4o").status == "deprecated" and len(custom.by_id["gpt-4o"]) == 1


# ------------------------------------------------------------------- logs


def test_batch_files_with_colliding_ids_pair_by_file_name(tmp_path):
    for batch, answer in (("run1", "first"), ("run2", "second")):
        write_jsonl(tmp_path / f"{batch}_input.jsonl", [{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": batch}]}}])
        write_jsonl(tmp_path / f"{batch}_output.jsonl", [{"custom_id": "request-1", "response": {"status_code": 200, "body": {"model": "gpt-4o-2024-08-06", "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]}}}])
    inters, _ = parse_logs([str(tmp_path)])
    pairs = {i.messages[-1]["content"]: i.output for i in inters}
    assert pairs == {"run1": "first", "run2": "second"}
    assert all(i.model == "gpt-4o" and i.metadata["served_model"] == "gpt-4o-2024-08-06" for i in inters)


def test_log_formats_from_review(tmp_path):
    records = [
        {"request": {"model": "text-embedding-3-small", "input": "hello"}, "response": {"object": "list", "data": [{"embedding": [0.1]}]}},
        {
            "request": {"model": "gpt-4o", "messages": [{"role": "user", "content": "stream me"}], "stream": True},
            "response": 'data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n',
        },
        {"request": {"model": "gpt-5", "input": "hi"}, "response": {"object": "response", "created_at": 1760000000, "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "yo"}]}]}},
        {"input": "classify: charged twice", "output": {"category": "billing", "urgency": "high"}},
    ]
    write_jsonl(tmp_path / "mixed.jsonl", records)
    (tmp_path / "sheet.csv").write_text("System Prompt,User Input,Model Output\nBe brief,What is 2+2?," + "4" * 200000 + "\n", encoding="utf-8")
    (tmp_path / "broken.jsonl.gz").write_bytes(b"\x1f\x8b\x08\x00garbage")
    inters, stats = parse_logs([str(tmp_path)])
    outputs = {i.messages[-1]["content"]: i for i in inters}
    assert outputs["stream me"].output == "Hello"
    assert outputs["hi"].timestamp.startswith("2025-10-09")
    assert json.loads(outputs["classify: charged twice"].output) == {"category": "billing", "urgency": "high"}
    sheet = outputs["What is 2+2?"]
    assert sheet.messages[0] == {"role": "system", "content": "Be brief"} and len(sheet.output) == 200000
    assert stats.skipped["unreadable file"] == 1
    assert any("embedding" in reason for reason in stats.skipped)


def test_parse_time_edge_cases():
    assert parse_time("2024") is None  # a year is not a unix timestamp
    assert parse_time("20240115").isoformat() == "2024-01-15T00:00:00+00:00"
    assert parse_time("2024-01-15 10:00:00 UTC").isoformat() == "2024-01-15T10:00:00+00:00"
    assert parse_time("2024-01-15T10:00:00+0530").isoformat() == "2024-01-15T10:00:00+05:30"
    assert parse_time("2024-01-15garbage") is None


def test_until_date_includes_that_whole_day(tmp_path):
    rows = [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": f"question {i}"}], "choices": [{"message": {"role": "assistant", "content": f"answer {i}"}}], "timestamp": ts}
        for i, ts in enumerate(["2024-01-30T12:00:00Z", "2024-01-31T18:30:00Z", "2024-02-01T00:00:01Z"])
    ]
    write_jsonl(tmp_path / "log.jsonl", rows)
    result = build_snapshot([str(tmp_path / "log.jsonl")], until="2024-01-31")
    assert sorted(c.messages[-1]["content"] for c in result.cases) == ["question 0", "question 1"]


def test_redaction_of_text_starting_with_data_and_consistent_placeholders():
    r = Redactor(pii=True)
    out = r.text("data: mail John@Example.com or john@example.com, call (555) 123-4567 or 555.123.4567")
    assert out == "data: mail <EMAIL_1> or <EMAIL_1>, call <PHONE_1> or <PHONE_1>"
    image = "data:image/png;base64,iVBORw0KGgo="
    assert r.text(image) == image


# ----------------------------------------------------------------- config


def test_config_lists_env_defaults_and_paths_from_a_subdirectory(in_tmp, monkeypatch):
    (in_tmp / ".git").mkdir()
    (in_tmp / "llmshift.yaml").write_text(
        "tests: ${TESTS_FILE:-.llmshift/tests.jsonl}\nscan:\n  paths: src\n  exclude: vendor\nwatch:\n  min_pass_rate: null\n  alerts:\n    type: file\n    path: alerts/out.json\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TESTS_FILE", "")
    (in_tmp / "src" / "pkg").mkdir(parents=True)
    monkeypatch.chdir(in_tmp / "src" / "pkg")
    cfg = load_config()
    assert Path(cfg["tests"]) == Path("../../.llmshift/tests.jsonl")
    assert cfg["scan"]["paths"] == [".."] and cfg["scan"]["exclude"] == ["vendor"]  # src, seen from src/pkg
    assert cfg["watch"]["alerts"][0]["path"] == str(Path("../../alerts/out.json"))
    assert cfg.user_value("watch", "min_pass_rate") == (True, None) and cfg.user_value("min_pass_rate") == (False, None)


def test_pyproject_config_is_found_from_a_subdirectory(in_tmp, monkeypatch):
    try:
        import tomllib  # noqa: F401
    except ImportError:  # Python < 3.11 reads pyproject.toml only when tomli is installed
        pytest.importorskip("tomli")
    (in_tmp / ".git").mkdir()
    (in_tmp / "pyproject.toml").write_text('[tool.llmshift]\ncandidate = "mock:nextgen"\n', encoding="utf-8")
    (in_tmp / "sub").mkdir()
    monkeypatch.chdir(in_tmp / "sub")
    cfg = load_config()
    assert cfg["candidate"] == "mock:nextgen" and cfg.path.name == "pyproject.toml"


# ------------------------------------------------------------------ watch


def _summary(rate: float, n: int = 10) -> dict:
    return {"total": n, "evaluated": n, "passed": round(rate * n), "warned": 0, "failed": n - round(rate * n), "errors": 0, "pass_rate": rate, "error_rate": 0.0}


def test_watch_drop_needs_significance_on_paired_cases():
    history = [{"run_id": "r1", "tests": "fp", "summary": _summary(1.0)}]
    prev = {f"c{i}": {"status": "pass", "case": f"h{i}"} for i in range(10)}

    def run(failing: int):
        cur = {f"c{i}": {"status": "fail" if i < failing else "pass", "case": f"h{i}"} for i in range(10)}
        return detect_drift(_summary(1 - failing / 10), cur, history, prev, model="m", min_pass_rate=None, max_pass_rate_drop=0.05,
                            max_similarity_drop=0.1, retirement_warn_days=30, fingerprint="fp")

    assert not any("dropped" in d.title for d in run(1))  # one flip is noise (p = 0.5)
    assert any("dropped" in d.title and d.severity == "critical" for d in run(6))
    changed_tests = {f"c{i}": {"status": "fail", "case": f"other{i}"} for i in range(10)}
    drifts = detect_drift(_summary(0.0), changed_tests, history, prev, model="m", min_pass_rate=None, max_pass_rate_drop=0.05,
                          max_similarity_drop=0.1, retirement_warn_days=30, fingerprint="fp")
    assert not any("dropped" in d.title for d in drifts)  # edited tests are not compared with their old results


def test_watch_new_test_set_and_first_run_warning():
    history = [{"run_id": "r1", "tests": "old", "summary": _summary(1.0)}]
    drifts = detect_drift(_summary(0.3), {}, history, {}, model="m", min_pass_rate=None, max_pass_rate_drop=0.05,
                          max_similarity_drop=0.1, retirement_warn_days=30, fingerprint="new")
    titles = [d.title for d in drifts]
    assert any("test set changed" in t for t in titles) and any("Only 30%" in t for t in titles)


def test_watch_interval_validation():
    assert parse_every("1m") == 60
    for bad in ("0", "30s", "0.5m"):
        with pytest.raises(ValueError):
            parse_every(bad)


def test_watch_history_append_and_info_alerts(tmp_path, make_case):
    hist = tmp_path / "hist"
    first = [make_case("hello", user="hello", id="a")]
    run_watch(first, "mock:echo", history_dir=str(hist), show_progress=False)
    hist_file = next(hist.rglob("history.jsonl"))
    hist_file.write_bytes(hist_file.read_bytes().rstrip(b"\n"))  # e.g. hand-edited
    alert_file = tmp_path / "alert.json"
    second = [make_case("bye", user="bye", id="b")]
    out = run_watch(second, "mock:echo", history_dir=str(hist), alerts=[{"type": "file", "path": str(alert_file)}], alert_on="info", show_progress=False)
    assert any("test set changed" in d.title for d in out.drifts)
    assert len(load_history(str(hist), "mock:echo")) == 2
    assert alert_file.exists()


# -------------------------------------------------------------------- cli


def test_scan_json_on_stdout_stays_valid_in_github_actions(in_tmp, monkeypatch, capsys):
    (in_tmp / "app.py").write_text('client.chat.completions.create(model="gpt-4-0613")\n', encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    cli.main(["scan", "--format", "json", "--fail-on", "never", "."])
    data = json.loads(capsys.readouterr().out)
    assert data["findings"][0]["model"] == "gpt-4-0613"


def test_scan_output_file_has_no_colour_codes(in_tmp, monkeypatch):
    (in_tmp / "app.py").write_text('model = "gpt-4-0613"\n', encoding="utf-8")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    cli.main(["scan", "-o", "scan.txt", "--fail-on", "never", "."])
    text = (in_tmp / "scan.txt").read_text(encoding="utf-8")
    assert "gpt-4-0613" in text and "\x1b[" not in text


def test_snapshot_append_reports_only_new_cases(in_tmp, capsys):
    rows = [{"model": "gpt-4o", "messages": [{"role": "user", "content": f"question {i}"}], "choices": [{"message": {"role": "assistant", "content": f"answer {i}"}}]} for i in range(3)]
    write_jsonl(in_tmp / "a.jsonl", rows[:2])
    write_jsonl(in_tmp / "b.jsonl", rows)
    assert cli.main(["snapshot", "a.jsonl", "-o", "tests.jsonl"]) == 0
    before = (in_tmp / "tests.jsonl").read_bytes()
    capsys.readouterr()
    assert cli.main(["snapshot", "b.jsonl", "-o", "tests.jsonl", "--append"]) == 0
    out = capsys.readouterr().out
    assert "added 1 new test case(s)" in out and "3 total" in out
    assert (in_tmp / "tests.jsonl").read_bytes().startswith(before) and len(load_tests("tests.jsonl")) == 3


def test_import_does_not_replace_tests_without_force(in_tmp):
    (in_tmp / "eval.json").write_text(json.dumps({"eval": EVAL, "runs": [TEMPLATE_RUN], "output_items": [output_item(0, ITEMS[0], "Hardware", True)]}), encoding="utf-8")
    (in_tmp / "tests.jsonl").write_text("keep me\n", encoding="utf-8")
    assert cli.main(["import", "openai-evals", "eval.json", "-o", "tests.jsonl", "--no-verify"]) == cli.EXIT_USAGE
    assert (in_tmp / "tests.jsonl").read_text(encoding="utf-8") == "keep me\n"
    assert cli.main(["import", "openai-evals", "eval.json", "-o", "tests.jsonl", "--no-verify", "--force"]) == 0
    assert len(load_tests("tests.jsonl")) == 1


def test_report_rejects_other_json_and_accepts_a_bom(in_tmp, capsys):
    (in_tmp / "scan.json").write_text(json.dumps({"root": ".", "findings": []}), encoding="utf-8")
    assert cli.main(["report", "scan.json"]) == cli.EXIT_USAGE
    assert "not a compare result" in capsys.readouterr().err
    (in_tmp / "res.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"candidate": "mock:x", "summary": {"total": 0}, "cases": []}).encode())
    assert cli.main(["report", "res.json"]) == 0


def test_init_keeps_a_non_utf8_gitignore(in_tmp):
    original = "# caf\xe9\n*.log\n".encode("latin-1")
    (in_tmp / ".gitignore").write_bytes(original)
    assert cli.main(["init"]) == 0
    data = (in_tmp / ".gitignore").read_bytes()
    assert data.startswith(original) and b".llmshift/cache/" in data


def test_demo_refuses_a_non_empty_directory(tmp_path):
    (tmp_path / "important.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_project(str(tmp_path))
    assert cli.main(["demo", "--dir", str(tmp_path)]) == cli.EXIT_USAGE
    assert (tmp_path / "important.txt").read_text(encoding="utf-8") == "mine"


def test_console_stops_quietly_on_a_closed_pipe(monkeypatch):
    class Closed:
        encoding = "utf-8"

        def write(self, text):
            raise BrokenPipeError()

        def flush(self):
            pass

        def fileno(self):
            raise OSError("no descriptor")

    monkeypatch.setattr("sys.stdout", Closed())
    with pytest.raises(SystemExit) as exc:
        console.out("hello")
    assert exc.value.code == 1


# ------------------------------------------------------------- test files


def _case(i: int) -> TestCase:
    return TestCase(id=f"c{i}", input={"messages": [{"role": "user", "content": f"q{i}"}], "params": {}}, baseline={"output": f"a{i}"})


def test_test_files_json_lines_in_json_and_directories(tmp_path):
    lines = tmp_path / "cases.json"
    lines.write_text("\n".join(json.dumps(_case(i).to_dict()) for i in range(2)) + "\n", encoding="utf-8")
    assert [c.id for c in load_tests(lines)] == ["c0", "c1"]
    with pytest.raises(TestFileError, match="directory"):
        save_tests([_case(0)], tmp_path)
    target = tmp_path / "t.jsonl"
    target.write_text(json.dumps(_case(0).to_dict()), encoding="utf-8")  # no trailing newline
    append_tests([_case(1)], target)
    assert [c.id for c in load_tests(target)] == ["c0", "c1"]


def test_importer_keeps_outputs_of_one_run(tmp_path):
    run2 = dict(TEMPLATE_RUN, id="evalrun_2", created_at=200)
    other = dict(output_item(1, ITEMS[1], "Software", True), run_id="evalrun_2")
    (tmp_path / "eval.json").write_text(
        json.dumps({"eval": EVAL, "runs": [TEMPLATE_RUN, run2], "output_items": [output_item(0, ITEMS[0], "Hardware", True), other]}), encoding="utf-8"
    )
    result = oe.import_from_export(str(tmp_path / "eval.json"))
    assert [c.baseline_output for c in result.cases] == ["Software"] and result.evals[0].run_id == "evalrun_2"


def test_importer_finds_the_registry_in_a_repo_checkout(tmp_path):
    reg = tmp_path / "evals" / "registry"
    (reg / "evals").mkdir(parents=True)
    (reg / "evals" / "x.yaml").write_text("x:\n  id: x.dev.v0\n", encoding="utf-8")
    (tmp_path / "evals" / "elsewhere.yaml").write_text("not: an eval\n", encoding="utf-8")
    root, files = oe._registry_root(tmp_path)
    assert root == reg and files == [reg / "evals" / "x.yaml"]


def test_gzip_logs_still_parse(tmp_path):
    rows = [{"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}], "choices": [{"message": {"role": "assistant", "content": "a"}}]}]
    with gzip.open(tmp_path / "log.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rows[0]) + "\n")
    inters, stats = parse_logs([str(tmp_path)])
    assert [i.output for i in inters] == ["a"] and not stats.skipped
