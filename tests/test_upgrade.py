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


def _remote_log(remote, ref):
    return subprocess.run(["git", "log", "--format=%s", ref], cwd=remote, capture_output=True, text=True, check=True).stdout.split("\n")[:-1]


@pytest.fixture
def pr_repo(in_tmp, monkeypatch):
    """The demo app as a git repository with an origin, and no GitHub token."""
    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_REPOSITORY", "GITHUB_ACTIONS", "GITHUB_EVENT_PATH"):
        monkeypatch.delenv(key, raising=False)
    root = create_project(str(in_tmp / "app"))
    remote = in_tmp / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    monkeypatch.chdir(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Dev")
    _git(root, "config", "user.email", "dev@example.com")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "origin", "main")
    return root, remote


def test_pr_never_overwrites_commits_it_did_not_make(pr_repo, capsys, in_tmp):
    root, remote = pr_repo
    branch = "llmshift/upgrade-gpt-4o-2024-05-13-to-gpt-5.6-sol"
    code, out, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr")
    assert code == 0 and f"pushed 2 file(s) to branch {branch}" in out and "open the pull request" in err
    # A reviewer adds a commit to the open upgrade branch; the weekly re-run must keep it.
    reviewer = in_tmp / "reviewer"
    subprocess.run(["git", "clone", "-q", "-b", branch, str(remote), str(reviewer)], check=True, capture_output=True)
    (reviewer / "NOTES.md").write_text("reviewed\n", encoding="utf-8")
    _git(reviewer, "add", "-A")
    _git(reviewer, "commit", "-q", "-m", "reviewer fix")
    _git(reviewer, "push", "-q", "origin", branch)
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr")
    assert code == 2 and "has commits llmshift did not make" in err and "--branch NEW" in err
    assert _remote_log(remote, branch)[0] == "reviewer fix"
    # Never onto the base, default or current branch.
    main_before = _remote_log(remote, "main")
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "main")
    assert code == 2 and "--branch main is the base, default or current branch" in err
    assert _remote_log(remote, "main") == main_before
    # Its own earlier commit is replaced, not stacked on.
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "llmshift/again")
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "llmshift/again")
    assert code == 0 and _remote_log(remote, "llmshift/again") == ["Upgrade gpt-4o-2024-05-13 to gpt-5.6-sol", "base"]


def test_pr_leaves_the_checkout_alone_and_never_pushes_uncommitted_work(pr_repo, capsys):
    root, remote = pr_repo
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
    _git(root, "checkout", "-q", "--detach")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "llmshift/detached")
    assert code == 0 and "pushed 2 file(s)" in out
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout == head
    assert subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout == ""
    assert subprocess.run(["git", "branch", "--list", "llmshift/*"], cwd=root, capture_output=True, text=True, check=True).stdout == ""
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts} == snapshot
    pushed = subprocess.run(["git", "show", "llmshift/detached:app/triage.py"], cwd=remote, capture_output=True, check=True).stdout
    assert b'TRIAGE_MODEL = "gpt-5.6-sol"' in pushed
    # An uncommitted edit in a file the upgrade changes is never pushed, even with --allow-dirty.
    triage = root / "app" / "triage.py"
    triage.write_bytes(triage.read_bytes() + b"# my unfinished work\n")
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--allow-dirty", "--branch", "llmshift/dirty")
    assert code == 2 and "app/triage.py has uncommitted changes" in err and "never pushed" in err
    assert subprocess.run(["git", "ls-remote", "--heads", str(remote), "llmshift/dirty"], capture_output=True, text=True, check=True).stdout == ""
    # Unrelated uncommitted files are fine with --allow-dirty: they are simply not part of the commit.
    _git(root, "checkout", "-q", "--", "app/triage.py")
    (root / "README.md").write_text("local notes\n", encoding="utf-8")
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--allow-dirty", "--branch", "llmshift/other")
    files = subprocess.run(["git", "show", "--name-only", "--format=", "llmshift/other"], cwd=remote, capture_output=True, text=True, check=True).stdout.split()
    assert code == 0 and sorted(files) == ["app/replies.py", "app/triage.py"]


def test_github_helpers(tmp_path, monkeypatch):
    from llmshift import github as gh

    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "remote", "add", "origin", "https://example.com/x.git")
    for url, want in [
        ("https://github.com/acme/app.git", "acme/app"),
        ("git@github.com:acme/app.git", "acme/app"),
        ("ssh://git@github.com/acme/app", "acme/app"),
        ("https://notgithub.com/acme/app.git", None),
        ("git@github.com-work:acme/app.git", None),
    ]:
        _git(repo, "remote", "set-url", "origin", url)
        assert gh.remote_repo(cwd=repo) == want, url
    _git(repo, "remote", "set-url", "origin", "https://ghe.example.com/team/svc.git")
    assert gh.remote_repo(cwd=repo) is None and gh.remote_repo(cwd=repo, host="ghe.example.com") == "team/svc"
    ctx = gh.context({"GITHUB_SERVER_URL": "https://ghe.example.com/", "GITHUB_REPOSITORY": "team/svc"})
    assert ctx.host == "ghe.example.com" and ctx.compare_url("main", "llmshift/x y") == "https://ghe.example.com/team/svc/compare/main...llmshift/x%20y?expand=1"
    # A body cut to GitHub's limit keeps its Markdown closed; emoji count twice.
    body = "<details><summary>Diff</summary>\n\n```diff\n" + "+ line \U0001F600\n" * 20_000 + "```\n</details>\n"
    cut = gh.fit_markdown(body)
    assert len(cut.encode("utf-16-le")) // 2 <= gh.MAX_COMMENT and cut.count("```") == 2 and cut.count("</details>") == 1
    assert gh.fit_markdown("short") == "short"


def test_pull_request_errors_explain_and_link(pr_repo, capsys, server, monkeypatch):
    for key, value in {"GITHUB_TOKEN": "t0k", "GITHUB_REPOSITORY": "acme/app", "GITHUB_API_URL": server.url, "GITHUB_SERVER_URL": "https://ghe.example.com"}.items():
        monkeypatch.setenv(key, value)
    server.route("GET", "/repos/acme/app", lambda r: {"default_branch": "main"})
    server.route("GET", "/repos/acme/app/pulls", lambda r: [])
    validation = {"message": "Validation Failed", "errors": [{"resource": "PullRequest", "field": "base", "code": "invalid"}]}
    server.route("POST", "/repos/acme/app/pulls", lambda r: (422, validation))
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "llmshift/a")
    assert code == 2 and "Validation Failed: PullRequest base invalid" in err
    assert "https://ghe.example.com/acme/app/compare/main...llmshift/a?expand=1" in err
    server.route("POST", "/repos/acme/app/pulls", lambda r: (403, {"message": "API rate limit exceeded for installation."}, {"x-ratelimit-remaining": "0"}))
    code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--pr", "--branch", "llmshift/b")
    assert code == 2 and "rate limit" in err and "permissions" not in err and "llmshift/b?expand=1" in err


def test_provider_specs_vendor_prefixes_and_deployments(in_tmp, capsys):
    (in_tmp / "app.py").write_text(
        'from openai import OpenAI\nclient = OpenAI()\nA = "gpt-4o-2024-05-13"\nB = "openai/gpt-4o-2024-05-13"\n'
        'client.chat.completions.create(model=A, messages=[])\n', encoding="utf-8")
    (in_tmp / "azure.py").write_text('from openai import AzureOpenAI\nAzureOpenAI(azure_deployment="gpt-4o-2024-05-13")\n', encoding="utf-8")
    # llmshift's provider:model form: the code gets the bare id, a router prefix in the code stays.
    code, out, _ = run(capsys, "upgrade", "--from", "openai:gpt-4o-2024-05-13", "--to", "openai:gpt-5.6-sol", "--apply")
    app = (in_tmp / "app.py").read_text(encoding="utf-8")
    assert code == 0 and 'A = "gpt-5.6-sol"' in app and 'B = "openai/gpt-5.6-sol"' in app and "openai:" not in app
    # An Azure deployment name is a resource of yours: noted, not renamed.
    assert 'azure_deployment="gpt-4o-2024-05-13"' in (in_tmp / "azure.py").read_text(encoding="utf-8")
    assert "names an Azure deployment" in out
    # No note may claim an id was left alone when the diff changes it.
    assert "built on" not in out


def test_max_tokens_set_away_from_the_model_is_flagged(in_tmp, capsys):
    (in_tmp / "config.py").write_text('MODEL = "gpt-4o-2024-05-13"\n', encoding="utf-8")
    (in_tmp / "app.py").write_text(
        "from openai import OpenAI\nfrom config import MODEL\nclient = OpenAI()\n\n"
        "def ask(q):\n    return client.chat.completions.create(model=MODEL, max_tokens=200, messages=[])\n", encoding="utf-8")
    (in_tmp / "claude.py").write_text("import anthropic\nanthropic.Anthropic().messages.create(model='claude-sonnet-5-5', max_tokens=50, messages=[])\n", encoding="utf-8")
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13")
    assert code == 0 and "max_tokens is still set at app.py:6" in out and "claude.py" not in out


def test_line_numbers_agree_with_git_after_form_feeds(in_tmp, capsys):
    """A form feed or U+2028 earlier in the file is not a line end for git or Python:
    the scanner must count lines the same way, or upgrade edits the wrong line."""
    before = b'const LEGACY_NOTE = "page\x0cbreak \xe2\x80\xa8 here";\nconst X = 1;\nconst MODEL = "gpt-4o-2024-05-13";\n'
    (in_tmp / "app.js").write_bytes(before)
    code, _, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--apply")
    assert code == 0 and (in_tmp / "app.js").read_bytes() == before.replace(b"gpt-4o-2024-05-13", b"gpt-5.6-sol")


def test_rewrites_are_exact_and_safe(in_tmp, capsys, monkeypatch):
    # A form feed is not a line break for git: the diff must not invent one.
    (in_tmp / "a.py").write_bytes(b'X = "gpt-4o-2024-05-13"\n# page\x0cbreak\nY = 1\n')
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13")
    assert code == 0 and "No newline" not in out
    # Syntax newer than this Python cannot be checked by parsing, so it is not refused for that.
    (in_tmp / "a.py").unlink()
    (in_tmp / "new.py").write_bytes(b'MODEL = "gpt-4o-2024-05-13"\ntype Alias = list[int]\nmatch MODEL:\n    case _:\n        pass\n')
    code, out, _ = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--apply")
    assert code == 0 and b'MODEL = "gpt-5.6-sol"' in (in_tmp / "new.py").read_bytes() and "would not parse" not in out
    # A read-only file stops the upgrade before any file is written.
    (in_tmp / "b.py").write_text('M = "gpt-4o-2024-05-13"\n', encoding="utf-8")
    (in_tmp / "c.py").write_text('M = "gpt-4o-2024-05-13"\n', encoding="utf-8")
    (in_tmp / "c.py").chmod(0o444)
    try:
        code, _, err = run(capsys, "upgrade", "--from", "gpt-4o-2024-05-13", "--to", "gpt-6-luna", "--apply")
    finally:
        (in_tmp / "c.py").chmod(0o644)
    assert code == 2 and "cannot write c.py" in err and "nothing was changed" in err
    assert (in_tmp / "b.py").read_text(encoding="utf-8") == 'M = "gpt-4o-2024-05-13"\n'
    assert not list(in_tmp.glob(".*.llmshift-tmp"))


def test_a_replacement_that_retires_too_is_skipped(in_tmp, capsys):
    """o1-mini's listed replacement, o4-mini, retires itself: moving there would need a
    second migration, so the chain is followed to a model with no retirement date."""
    from llmshift.registry import Registry

    lk = Registry.load().lookup("o1-mini")
    assert lk.replacement == "gpt-5.6-terra" and "the listed replacement, o4-mini, is retiring too" in lk.message
    (in_tmp / "app.py").write_text('from openai import OpenAI\nOpenAI().chat.completions.create(model="o1-mini", messages=[])\n', encoding="utf-8")
    code, out, _ = run(capsys, "upgrade", "--from", "o1-mini")
    assert code == 0 and "o1-mini -> gpt-5.6-terra" in out
