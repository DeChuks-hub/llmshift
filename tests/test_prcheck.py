"""`llmshift check`: find the prompts a change edits and test only those."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from llmshift import cli
from llmshift import github as gh
from llmshift.demo import REPLY_PROMPT, SUMMARY_PROMPT, TRIAGE_PROMPT, create_project
from llmshift.prcheck import PromptChange, carry_edit, find_prompt_changes, prompt_candidates


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """The demo app in a git repo: tests snapshotted on main, then a feature branch."""
    root = create_project(str(tmp_path / "app"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli, "EXTENSIONS", ("prcheck",))
    (root / "prompts").mkdir()
    (root / "prompts" / "summary.txt").write_bytes((SUMMARY_PROMPT + "\n").encode())
    assert cli.main(["snapshot", "logs"]) == 0
    (root / ".gitignore").write_bytes(b".llmshift/cache/\n.llmshift/reports/\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "checkout", "-q", "-b", "feature")
    return root


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_bytes().decode("utf-8")
    assert old in text
    path.write_bytes(text.replace(old, new, 1).encode("utf-8"))


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# --------------------------------------------------------------- extraction


def test_prompt_candidates_in_many_file_types():
    py = 'P = (\n    "You are a triage assistant. "\n    \'Return "json".\'\n)\nQ = f"Hello {name}, you are a helpful bot"\nR = "one part " + "two part prompt"\n'
    cands = prompt_candidates(py, "app.py")
    assert 'You are a triage assistant. Return "json".' in cands  # implicit concatenation is one string
    assert "Hello {name}, you are a helpful bot" in cands and "one part two part prompt" in cands
    js = 'const P = "You are a helpful " +\n  "support agent.";\nconst T = `Multi\nline prompt here`;\n'
    assert {"You are a helpful support agent.", "Multi\nline prompt here"} <= set(prompt_candidates(js, "a.ts"))
    yml = "prompts:\n  system: |\n    You are terse.\n    Answer in one line.\n"
    assert "You are terse.\nAnswer in one line.\n" in prompt_candidates(yml, "p.yaml")
    assert prompt_candidates(json.dumps({"system": "You are a JSON-only bot"}), "p.json") == ["You are a JSON-only bot"]
    assert prompt_candidates("Whole file is the prompt.\n", "p.prompt") == ["Whole file is the prompt.\n"]
    assert prompt_candidates("def broken(:\n  'a broken file with a long string'", "x.py")  # falls back to quoted strings


def test_edits_keep_the_recorded_shape():
    base = '\n    You are a bot.\n    Be brief.\n'
    new = '\n    You are a bot.\n    Be very brief.\n'
    assert carry_edit("You are a bot.\nBe brief.", base, new) == ("You are a bot.\nBe very brief.", False)
    assert carry_edit("Intro. You are a bot. Outro.", "You are a bot.", "You are a robot.") == ("Intro. You are a robot. Outro.", False)
    assert carry_edit("Totally different", "abc template {x}", "abd template {x}")[1] is True


def test_find_prompt_changes_cases():
    old_a, old_b = "You are agent A. Answer politely and briefly.", "You are agent B. Always answer in JSON."
    base = {"a.py": f'P = "{old_a}"\n', "b.py": f'Q = "{old_b}"\n', "c.py": 'X = 1\n'}
    now = {
        "a.py": 'P = "You are agent A. Answer politely and very briefly."\n',  # edited
        "b.py": "# moved out\n",
        "c.py": f'Q = "{old_b}"\n',  # moved unchanged
    }
    diffs = [gh.FileDiff("M", p, p) for p in base]
    changes = find_prompt_changes({"ga": old_a, "gb": old_b}, diffs, base.get, now.get)
    by = {c.group: c for c in changes}
    assert by["ga"].status == "changed" and by["ga"].new == "You are agent A. Answer politely and very briefly." and by["ga"].files == ["a.py"]
    assert by["gb"].status == "moved" and by["gb"].files == ["c.py"]
    now["c.py"] = "X = 2\n"
    assert {c.group: c.status for c in find_prompt_changes({"gb": old_b}, diffs, base.get, now.get)} == {"gb": "removed"}
    # Unrelated edits in a file that holds a prompt are not prompt changes.
    assert find_prompt_changes({"ga": old_a}, [gh.FileDiff("M", "a.py", "a.py")], base.get, {"a.py": base["a.py"] + "Y = 2\n"}.get) == []
    # The same prompt edited differently in two files: both versions are tested.
    base2 = {"x.py": f'P = "{old_a}"\n', "y.txt": old_a}
    now2 = {"x.py": 'P = "You are agent A. Answer politely and briefly!"\n', "y.txt": "You are agent A. Answer politely, briefly."}
    two = find_prompt_changes({"ga": old_a}, [gh.FileDiff("M", p, p) for p in base2], base2.get, now2.get)
    assert len(two) == 2 and all(c.status == "changed" and "several places" in c.note for c in two)
    # Without history, a file holds a prompt when it holds something close to it.
    nohist = find_prompt_changes({"ga": old_a}, [gh.FileDiff("M", "a.py", "a.py")], None, now.get)
    assert [c.status for c in nohist] == ["changed"]
    assert PromptChange("g", "o", None, [], "removed").passed(0.95)


# ------------------------------------------------------------- end to end


def test_only_the_edited_prompt_is_tested_with_its_new_text(repo, capsys):
    _edit(repo / "app" / "triage.py", "(one short sentence).'", "(one short sentence). Return raw JSON only.'")
    code, out, _ = run(capsys, "check", "--model", "mock:nextgen", "--base", "main", "--min-pass-rate", "0.7", "--no-cache", "--json", "check.json")
    data = json.loads((repo / "check.json").read_text(encoding="utf-8"))
    changed = [c for c in data["changes"] if c["status"] == "changed"]
    assert code == 0 and len(changed) == 1 and changed[0]["files"] == ["app/triage.py"]
    assert changed[0]["new_prompt"] == TRIAGE_PROMPT + " Return raw JSON only."
    s = changed[0]["summary"]
    # nextgen stops fencing JSON when told "raw JSON": only its urgency changes remain (2 of 8).
    assert s["total"] == 8 and s["pass_rate"] == 0.75
    assert "pass rate 75%" in out
    code, out, _ = run(capsys, "check", "--model", "mock:nextgen", "--base", "main", "--no-cache")
    assert code == 1 and "below 95%" in out


def test_prompt_file_edits_and_the_production_model(repo, capsys):
    _edit(repo / "prompts" / "summary.txt", "Keep the key figures.", "Keep the key figures exactly as written.")
    code, out, _ = run(capsys, "check", "--model", "mock:nextgen", "--base", "main", "--no-cache", "--json", "c.json")
    change = next(c for c in json.loads((repo / "c.json").read_text(encoding="utf-8"))["changes"] if c["status"] == "changed")
    assert code == 0 and change["files"] == ["prompts/summary.txt"] and change["summary"]["pass_rate"] == 1.0
    assert change["new_prompt"] == SUMMARY_PROMPT.replace("Keep the key figures.", "Keep the key figures exactly as written.")
    # Default model: the baseline in llmshift.yaml (the model in production).
    code, out, _ = run(capsys, "check", "--base", "main", "--no-cache")
    assert code == 0 and "model mock:legacy" in out


def test_nothing_to_check_and_removed_prompts(repo, capsys):
    code, out, _ = run(capsys, "check", "--base", "main")
    assert code == 0 and "No prompt changes found; nothing to check." in out
    _edit(repo / "app" / "triage.py", "import json", "import json  # unrelated edit")
    code, out, _ = run(capsys, "check", "--base", "main")
    assert code == 0 and "No prompt changes found" in out
    path = repo / "app" / "replies.py"
    text = path.read_bytes().decode()
    start, end = text.index("REPLY_PROMPT = ("), text.index("def draft_reply")
    path.write_bytes((text[:start] + 'REPLY_PROMPT = load_prompt("reply")\n\n\n' + text[end:]).encode())
    code, out, _ = run(capsys, "check", "--base", "main", "--json", "r.json")
    removed = [c for c in json.loads((repo / "r.json").read_text(encoding="utf-8"))["changes"] if c["status"] == "removed"]
    assert code == 0 and removed and removed[0]["files"] == ["app/replies.py"] and REPLY_PROMPT == removed[0]["old_prompt"]


def test_committed_changes_and_all_groups(repo, capsys):
    _edit(repo / "app" / "summaries.py", "Keep the key figures.", "Keep the key figures verbatim.")
    _git(repo, "commit", "-q", "-am", "edit prompt")
    code, out, _ = run(capsys, "check", "--model", "mock:nextgen", "--base", "main", "--min-pass-rate", "0", "--all", "--json", "a.json")
    data = json.loads((repo / "a.json").read_text(encoding="utf-8"))
    statuses = sorted(c["status"] for c in data["changes"])
    assert code == 0 and statuses == ["changed", "unchanged", "unchanged"]


def test_pull_request_comment_is_created_then_updated(repo, capsys, server, monkeypatch):
    comments: list[dict] = []

    def list_comments(req):
        return 200, comments

    def create(req):
        comments.append({"id": 11, "body": req["json"]["body"]})
        return 201, comments[-1]

    def update(req):
        comments[0]["body"] = req["json"]["body"]
        return 200, comments[0]

    server.route("GET", "/repos/acme/app/issues/7/comments", list_comments)
    server.route("POST", "/repos/acme/app/issues/7/comments", create)
    server.route("PATCH", "/repos/acme/app/issues/comments/11", update)
    event = repo / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7, "base": {"ref": "main", "sha": "x"}}}), encoding="utf-8")
    for key, value in {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_PATH": str(event), "GITHUB_REPOSITORY": "acme/app", "GITHUB_TOKEN": "t0k", "GITHUB_API_URL": server.url, "GITHUB_BASE_REF": "main"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_PATCH", lambda self: self._handle("PATCH"), raising=False)
    _edit(repo / "app" / "triage.py", "(one short sentence).'", "(one short sentence). Return raw JSON only.'")
    assert run(capsys, "check", "--model", "mock:nextgen", "--min-pass-rate", "0.7")[0] == 0
    assert len(comments) == 1 and comments[0]["body"].startswith(gh.CHECK_MARKER) and "pass" in comments[0]["body"]
    assert run(capsys, "check", "--model", "mock:nextgen", "--min-pass-rate", "0.9")[0] == 1
    assert len(comments) == 1 and "below 90%" in comments[0]["body"]  # edited, not a second comment
    methods = [r["method"] for r in server.requests]
    assert methods.count("POST") == 1 and methods.count("PATCH") == 1
    assert all(r["headers"]["authorization"] == "Bearer t0k" for r in server.requests)


def test_helpful_errors(tmp_path, monkeypatch, capsys, repo):
    code, _, err = run(capsys, "check", "--base", "does-not-exist")
    assert code == 2 and "fetch-depth: 0" in err
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "llmshift.yaml").write_text("tests: tests.jsonl\nbaseline: mock:legacy\n", encoding="utf-8")
    (plain / "tests.jsonl").write_bytes((repo / ".llmshift" / "tests.jsonl").read_bytes())
    (plain / "p.txt").write_text(SUMMARY_PROMPT.replace("Keep the key figures.", "Keep every figure exactly."), encoding="utf-8")
    monkeypatch.chdir(plain)
    code, _, err = run(capsys, "check")
    assert code == 2 and "not a git repository" in err
    code, out, _ = run(capsys, "check", "--files", "p.txt", "--no-cache")
    assert code == 0 and "p.txt" in out and "pass rate 100%" in out
