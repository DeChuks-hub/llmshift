"""GitHub and git helpers for `llmshift check` and `llmshift upgrade` (REST API over
plain HTTPS, git through subprocess; no SDKs)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import UsageError
from .providers import http
from .providers.base import ProviderError

API_URL = "https://api.github.com"
CHECK_MARKER = "<!-- llmshift:check -->"
MAX_COMMENT = 65000  # GitHub rejects bodies over 65,536 characters


class GitError(RuntimeError):
    pass


# ---------------------------------------------------------------------- git


def git(*args: str, cwd: str | os.PathLike | None = None, check: bool = True) -> str:
    """Run git with an argument list (never through a shell) and return stdout."""
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        raise GitError("git is not installed or not on PATH") from None
    if check and proc.returncode != 0:
        raise GitError((proc.stderr or proc.stdout).strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def git_bytes(*args: str, cwd: str | os.PathLike | None = None) -> bytes | None:
    try:
        proc = subprocess.run(["git", *args], cwd=str(cwd) if cwd else None, capture_output=True)
    except FileNotFoundError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def repo_root(cwd: str | os.PathLike | None = None) -> Path | None:
    try:
        out = git("rev-parse", "--show-toplevel", cwd=cwd).strip()
    except GitError:
        return None
    return Path(out) if out else None


def ref_exists(ref: str, cwd: str | os.PathLike | None = None) -> bool:
    try:
        git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=cwd)
        return True
    except GitError:
        return False


def merge_base(a: str, b: str = "HEAD", cwd: str | os.PathLike | None = None) -> str:
    return git("merge-base", a, b, cwd=cwd).strip()


def show(ref: str, path: str, cwd: str | os.PathLike | None = None) -> bytes | None:
    """A file's content at a commit, or None when it did not exist there."""
    return git_bytes("show", f"{ref}:{path}", cwd=cwd)


@dataclass
class FileDiff:
    status: str  # A added, M modified, D deleted, R renamed, C copied
    old_path: str | None
    new_path: str | None


def changed_files(base: str, cwd: str | os.PathLike | None = None) -> list[FileDiff]:
    """Files that differ between `base` and the working tree, plus untracked files."""
    raw = git("diff", "--name-status", "-M", "-z", base, cwd=cwd)
    parts = [p for p in raw.split("\0")]
    out: list[FileDiff] = []
    i = 0
    while i < len(parts):
        status = parts[i]
        if not status:
            i += 1
            continue
        kind = status[0]
        if kind in ("R", "C") and i + 2 < len(parts):
            out.append(FileDiff(kind, parts[i + 1] if kind == "R" else None, parts[i + 2]))
            i += 3
            continue
        if i + 1 >= len(parts):
            break
        path = parts[i + 1]
        if kind == "A":
            out.append(FileDiff("A", None, path))
        elif kind == "D":
            out.append(FileDiff("D", path, None))
        else:
            out.append(FileDiff("M", path, path))
        i += 2
    for path in git("ls-files", "--others", "--exclude-standard", "-z", cwd=cwd).split("\0"):
        if path:
            out.append(FileDiff("A", None, path))
    return out


def is_dirty(cwd: str | os.PathLike | None = None) -> bool:
    return bool(git("status", "--porcelain", cwd=cwd).strip())


def current_branch(cwd: str | os.PathLike | None = None) -> str | None:
    out = git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd, check=False).strip()
    return None if out in ("", "HEAD") else out


_REMOTE_RE = re.compile(r"github\.com[:/]+(?P<repo>[^/\s]+/[^/\s]+?)(?:\.git)?/?$")


def remote_repo(remote: str = "origin", cwd: str | os.PathLike | None = None) -> str | None:
    """owner/name of a GitHub remote (https or ssh URL)."""
    url = git("remote", "get-url", remote, cwd=cwd, check=False).strip()
    m = _REMOTE_RE.search(url)
    return m.group("repo") if m else None


def has_identity(cwd: str | os.PathLike | None = None) -> bool:
    return bool(git("config", "user.email", cwd=cwd, check=False).strip()) and bool(git("config", "user.name", cwd=cwd, check=False).strip())


# ------------------------------------------------------------------ context


@dataclass
class Context:
    """What a GitHub Actions run (or the environment) tells us."""

    token: str | None = None
    repo: str | None = None  # owner/name
    api_url: str = API_URL
    pr_number: int | None = None
    base_ref: str | None = None
    base_sha: str | None = None
    in_actions: bool = False
    event: dict = field(default_factory=dict)

    @property
    def owner(self) -> str | None:
        return self.repo.split("/", 1)[0] if self.repo else None


def context(env: dict | None = None) -> Context:
    env = os.environ if env is None else env
    event: dict = {}
    path = env.get("GITHUB_EVENT_PATH")
    if path:
        try:
            event = json.loads(Path(path).read_text(encoding="utf-8-sig")) or {}
        except (OSError, ValueError):
            event = {}
    if not isinstance(event, dict):
        event = {}
    pr = event.get("pull_request") if isinstance(event.get("pull_request"), dict) else {}
    number = pr.get("number")
    if number is None and isinstance(event.get("issue"), dict) and event["issue"].get("pull_request"):
        number = event["issue"].get("number")
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    return Context(
        token=env.get("GITHUB_TOKEN") or env.get("GH_TOKEN") or None,
        repo=env.get("GITHUB_REPOSITORY") or None,
        api_url=(env.get("GITHUB_API_URL") or API_URL).rstrip("/"),
        pr_number=int(number) if isinstance(number, int) or (isinstance(number, str) and number.isdigit()) else None,
        base_ref=env.get("GITHUB_BASE_REF") or base.get("ref") or None,
        base_sha=base.get("sha") or None,
        in_actions=env.get("GITHUB_ACTIONS") == "true",
        event=event,
    )


# ---------------------------------------------------------------------- API


def api(ctx: Context, method: str, path: str, body: Any = None) -> Any:
    if not ctx.token:
        raise UsageError("no GitHub token: set GITHUB_TOKEN (in Actions: env GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }})")
    headers = {
        "Authorization": f"Bearer {ctx.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    return http.request(method, f"{ctx.api_url}{path}", headers=headers, body=body, timeout=60, max_retries=3)


def _paged(ctx: Context, path: str, limit: int = 1000) -> list:
    out: list = []
    page = 1
    sep = "&" if "?" in path else "?"
    while len(out) < limit:
        items = api(ctx, "GET", f"{path}{sep}per_page=100&page={page}")
        if not isinstance(items, list) or not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        page += 1
    return out


def upsert_comment(ctx: Context, number: int, body: str, marker: str = CHECK_MARKER) -> dict:
    """Create the PR comment that carries `marker`, or edit it if it exists."""
    if marker not in body:
        body = f"{marker}\n{body}"
    if len(body) > MAX_COMMENT:
        body = body[: MAX_COMMENT - 80] + "\n\n_(truncated: the full report is in the job summary)_\n"
    repo = urllib.parse.quote(ctx.repo or "", safe="/")
    for comment in _paged(ctx, f"/repos/{repo}/issues/{number}/comments"):
        if isinstance(comment, dict) and marker in str(comment.get("body") or ""):
            return api(ctx, "PATCH", f"/repos/{repo}/issues/comments/{comment['id']}", {"body": body})
    return api(ctx, "POST", f"/repos/{repo}/issues/{number}/comments", {"body": body})


def open_pull_request(ctx: Context, *, head: str, base: str, title: str, body: str) -> dict:
    """Open a PR, or update the open one from the same branch."""
    repo = urllib.parse.quote(ctx.repo or "", safe="/")
    body = body if len(body) <= MAX_COMMENT else body[: MAX_COMMENT - 40] + "\n\n_(truncated)_\n"
    try:
        return api(ctx, "POST", f"/repos/{repo}/pulls", {"title": title, "head": head, "base": base, "body": body, "maintainer_can_modify": True})
    except ProviderError as exc:
        if exc.status != 422:
            raise
        owner = (ctx.owner or "").strip()
        existing = api(ctx, "GET", f"/repos/{repo}/pulls?state=open&head={urllib.parse.quote(f'{owner}:{head}')}")
        if isinstance(existing, list) and existing:
            number = existing[0]["number"]
            return api(ctx, "PATCH", f"/repos/{repo}/pulls/{number}", {"title": title, "body": body})
        raise


def default_branch(ctx: Context) -> str | None:
    try:
        data = api(ctx, "GET", f"/repos/{urllib.parse.quote(ctx.repo or '', safe='/')}")
    except (ProviderError, UsageError):
        return None
    return data.get("default_branch") if isinstance(data, dict) else None
