"""`llmshift upgrade`: swap retiring models in code, rename changed settings, open a PR."""

from __future__ import annotations

import ast
import codecs
import json
import subprocess
from pathlib import Path

import pytest

from llmshift import cli
from llmshift.demo import REPLY_PROMPT, TRIAGE_PROMPT, create_project
from llmshift.upgrade import _branch_name, apply_edits, Edit, UpgradePlan, Swap, needs_completion_tokens, plan_upgrade

PY_CRLF = (
    b"from openai import OpenAI\r\n"
    b"client = OpenAI()\r\n"
    b'MODEL = "gpt-4o-2024-05-13"\r\n'
    b"\r\n"
    b"def ask(q):\r\n"
    b'    return client.chat.completions.create(model=MODEL, max_tokens=200, temperature=0.2, messages=[{"role": "user", "content": q}])\r\n'
    b"\r\n"
    b"def other(q):\r\n"
    b'    return client.chat.completions.create(model="gpt-4o-mini", max_tokens=50, messages=[])\r\n'
    b'TUNED = "ft:gpt-4o-2024-05-13:acme::x1"\r\n'
)
TS = (
    "const res = await openai.chat.completions.create({\n"
    '  model: "gpt-4o-2024-05-13",\n'
    "  max_tokens: 300,\n"
    "  messages,\n"
    "});\n"
    'const other = { max_tokens: 5, model: "gpt-4o-mini" };\n'
    'const note = "max_tokens: in a string";\n'
    "// <p>Don't</p>\n"
)


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args], cwd=cwd, check=True, capture_output=True)


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture(autouse=True)
def _commands(monkeypatch):
    monkeypatch.setattr(cli, "EXTENSIONS", ("upgrade",))


@pytest.fixture
def project(in_tmp):
    (in_tmp / "app").mkdir()
    (in_tmp / "web").mkdir()
    (in_tmp / "config").mkdir()
    (in_tmp / "app" / "chat.py").write_bytes(PY_CRLF)
    (in_tmp / "web" / "client.ts").write_bytes(TS.encode())
    (in_tmp / "config" / "models.yaml").write_bytes(codecs.BOM_UTF16_LE + "llm:\r\n  model: gpt-4o-2024-05-13\r\n".encode("utf-16-le"))
    (in_tmp / "app" / "untouched.py").write_bytes(b'"""No models here."""\r\nX = 1\r\n')
    (in_tmp / "llmshift.yaml").write_bytes(b"baseline: openai:gpt-4o-2024-05-13\ncandidate: mock:nextgen\n")
    return in_tmp


def test_rules():
    assert needs_completion_tokens("gpt-5.6-sol") and needs_completion_tokens("o3-mini") and needs_completion_tokens("openai/gpt-6-luna")
    assert not needs_completion_tokens("gpt-4o-2024-05-13") and not needs_completion_tokens("gpt-4.1") and not needs_completion_tokens("claude-sonnet-5-5")
    plan = UpgradePlan(root=Path("."), swaps=[Swap("gpt-4o", "gpt-5.6-sol")])
    assert _branch_name(plan, None) == "llmshift/upgrade-gpt-4o-to-gpt-5.6-sol"
    assert _branch_name(plan, "feat/../x y.lock") == "feat/x-y-lock"
    data = "a\r\nb\rc\n".encode()
    assert apply_edits(data, [Edit(2, 0, "b", "B", "model")]) == ("a\r\nB\rc\n".encode(), [])
    assert apply_edits(data, [Edit(2, 0, "x", "B", "model")])[1]  # the expected text must be there


def test_dry_run_shows_a_diff_and_writes_nothing(project, capsys):
    before = {p: p.read_bytes() for p in project.rglob("*") if p.is_file()}
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13")
    assert code == 0 and "gpt-4o-2024-05-13 -> gpt-5.6-sol" in out and "Dry run: nothing was changed" in out
    assert "+MODEL = \"gpt-5.6-sol\"" in out.replace("\r", "")
    assert {p: p.read_bytes() for p in project.rglob("*") if p.is_file()} == before


def test_apply_changes_exactly_the_model_and_its_settings(project, capsys):
    untouched = (project / "app" / "untouched.py").read_bytes()
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--apply")
    assert code == 0 and "updated 3 file(s)" in out
    py = (project / "app" / "chat.py").read_bytes()
    assert py == (
        PY_CRLF.replace(b'MODEL = "gpt-4o-2024-05-13"', b'MODEL = "gpt-5.6-sol"')
        .replace(b"model=MODEL, max_tokens=200", b"model=MODEL, max_completion_tokens=200")
    )  # CRLF kept; the gpt-4o-mini call and the fine-tune are untouched
    ast.parse(py.decode())
    assert "temperature" in out  # the new model may reject a custom temperature: flagged, not changed
    ts = (project / "web" / "client.ts").read_text(encoding="utf-8")
    assert ts == TS.replace('"gpt-4o-2024-05-13"', '"gpt-5.6-sol"').replace("max_tokens: 300", "max_completion_tokens: 300")
    raw = (project / "config" / "models.yaml").read_bytes()
    assert raw.startswith(codecs.BOM_UTF16_LE) and raw[2:].decode("utf-16-le") == "llm:\r\n  model: gpt-5.6-sol\r\n"
    assert (project / "app" / "untouched.py").read_bytes() == untouched
    assert b"openai:gpt-4o-2024-05-13" in (project / "llmshift.yaml").read_bytes()  # llmshift's own config is yours to edit
    assert "llmshift's own configuration" in out


def test_guards(project, capsys):
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--to", "claude-sonnet-5-5")
    assert code == 0 and "different providers" in out and "Nothing to change" in out
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--to", "claude-sonnet-5-5", "--force")
    assert code == 0 and '+MODEL = "claude-sonnet-5-5"' in out.replace("\r", "")
    assert "max_completion_tokens" not in out  # only OpenAI's newer models need the rename
    code, _, err = run(capsys, "upgrade", "--from", "my-house-model")
    assert code == 2 and "no known replacement" in err
    code, _, err = run(capsys, "upgrade")
    assert code == 2 and "--auto" in err
    code, _, err = run(capsys, "upgrade", "--auto", "--pr")
    assert code == 2 and "git repository" in err


def test_notebooks_prefixed_and_platform_ids(in_tmp, capsys):
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": 1, "outputs": [], "source": ["from openai import OpenAI\n", 'client.chat.completions.create(model="gpt-4o-2024-05-13", messages=[])\n']}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (in_tmp / "nb.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (in_tmp / "router.yaml").write_text("openai_model: openai/gpt-4o-2024-05-13\n", encoding="utf-8")
    (in_tmp / "bedrock.py").write_text('import boto3\nMODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"\n', encoding="utf-8")
    assert run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--apply")[0] == 0
    cell = json.loads((in_tmp / "nb.ipynb").read_text(encoding="utf-8"))["cells"][0]["source"][1]
    assert cell == 'client.chat.completions.create(model="gpt-5.6-sol", messages=[])\n'
    assert (in_tmp / "router.yaml").read_text(encoding="utf-8") == "openai_model: openai/gpt-5.6-sol\n"
    code, out, _ = run(capsys, "upgrade", "--from", "claude-3-5-haiku-20241022", "--apply")
    assert code == 0 and "platform-specific" in out and "us.anthropic.claude-3-5-haiku-20241022-v1:0" in (in_tmp / "bedrock.py").read_text(encoding="utf-8")


def test_auto_compare_and_prompt_fixes(in_tmp, capsys, monkeypatch):
    root = create_project(str(in_tmp / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    capsys.readouterr()
    plan = plan_upgrade(["."], auto=True)
    assert {(s.old, s.new) for s in plan.swaps} == {
        ("gpt-4o-2024-05-13", "gpt-5.6-sol"),
        ("claude-3-5-haiku-20241022", "claude-haiku-4-5-20251001"),
        ("gpt-4o-mini", "gpt-6-luna"),
    }
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--compare", "--compare-with", "mock:nextgen", "--fix", "--apply", "--min-pass-rate", "0.8", "--no-cache", "--markdown", "body.md")
    assert code == 0 and "tests on mock:nextgen: pass rate" in out
    body = (root / "body.md").read_text(encoding="utf-8")
    assert "### Tests on `mock:nextgen`" in body and "Prompt changes suggested by `llmshift fix`" in body and "gpt-4o-2024-05-13" in body
    # The fix was appended to the prompt literals in the code (also the implicitly concatenated one).
    values = {n.value for n in ast.walk(ast.parse((root / "app" / "triage.py").read_text(encoding="utf-8"))) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert any(v.startswith(TRIAGE_PROMPT) and len(v) > len(TRIAGE_PROMPT) and "code fence" in v.lower() for v in values)
    replies = (root / "app" / "replies.py").read_text(encoding="utf-8")
    assert 'model=os.getenv("REPLY_MODEL", "gpt-5.6-sol")' in replies
    assert any(v.startswith(REPLY_PROMPT) and len(v) > len(REPLY_PROMPT) for v in (n.value for n in ast.walk(ast.parse(replies)) if isinstance(n, ast.Constant) and isinstance(n.value, str)))


def test_pull_request_flow(in_tmp, capsys, server, monkeypatch):
    empty = in_tmp / "empty.gitconfig"
    empty.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))  # no git identity, as on a fresh CI runner
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = create_project(str(in_tmp / "app"))
    remote = in_tmp / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    monkeypatch.chdir(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "origin", "main")
    pulls: list[dict] = []

    def create(req):
        if pulls:
            return 422, {"message": "Validation Failed", "errors": [{"message": "A pull request already exists for acme:x."}]}
        pulls.append(req["json"])
        return 201, {"number": 3, "html_url": "https://github.com/acme/app/pull/3"}

    def update(req):
        pulls.append(req["json"])
        return 200, {"number": 3, "html_url": "https://github.com/acme/app/pull/3"}

    server.route("GET", "/repos/acme/app/pulls", lambda r: [{"number": 3}])
    server.route("GET", "/repos/acme/app", lambda r: {"default_branch": "main"})
    server.route("POST", "/repos/acme/app/pulls", create)
    server.route("PATCH", "/repos/acme/app/pulls/3", update)
    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_PATCH", lambda self: self._handle("PATCH"), raising=False)
    for key, value in {"GITHUB_TOKEN": "t0k", "GITHUB_REPOSITORY": "acme/app", "GITHUB_API_URL": server.url, "GITHUB_ACTIONS": "true"}.items():
        monkeypatch.setenv(key, value)
    before = (root / "app" / "triage.py").read_bytes()
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr")
    assert code == 0 and "pull request: https://github.com/acme/app/pull/3" in out
    branch = "llmshift/upgrade-gpt-4o-2024-05-13-to-gpt-5.6-sol"
    pushed = subprocess.run(["git", "show", f"{branch}:app/triage.py"], cwd=remote, capture_output=True, check=True).stdout
    assert b'TRIAGE_MODEL = "gpt-5.6-sol"' in pushed
    author = subprocess.run(["git", "log", "-1", "--format=%an <%ae>", branch], cwd=remote, capture_output=True, text=True, check=True).stdout.strip()
    assert author == "github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"  # no identity configured: the Actions bot
    assert (root / "app" / "triage.py").read_bytes() == before  # back on main, working tree as it was
    assert pulls[0]["head"] == branch and pulls[0]["base"] == "main" and "gpt-5.6-sol" in pulls[0]["title"]
    assert "| `gpt-4o-2024-05-13` | retiring | 2026-10-23 |" in pulls[0]["body"] and "```diff" in pulls[0]["body"]
    # A second run updates the same branch and pull request instead of opening another.
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr")
    assert code == 0 and len(pulls) == 2 and "title" in pulls[1] and "head" not in pulls[1]
    # What real GitHub answers while "Allow GitHub Actions to create and approve pull
    # requests" is off (the default for new repositories): say how to fix it.
    server.route("POST", "/repos/acme/app/pulls", lambda r: (403, {"message": "GitHub Actions is not permitted to create or approve pull requests.", "status": "403"}))
    code, _, err = run(capsys, "upgrade", "--from", "claude-3-5-haiku-20241022", "--pr")
    assert code == 2 and "Allow GitHub Actions to create and approve pull requests" in err
    assert "https://github.com/acme/app/compare/main...llmshift/upgrade-claude-3-5-haiku-20241022" in err and "403, 403" not in err
    (root / "app" / "triage.py").write_bytes(before + b"# local edit\n")
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr")
    assert code == 2 and "uncommitted changes" in err
