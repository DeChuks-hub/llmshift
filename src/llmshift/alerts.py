"""Alert delivery for `llmshift watch`: Slack, Discord, Microsoft Teams,
generic webhooks, GitHub issues, e-mail (SMTP) and files.

Destinations come from llmshift.yaml (watch.alerts) or --alert flags:
  slack:https://hooks.slack.com/...   (or plain "slack" + SLACK_WEBHOOK_URL)
  discord:<url>  teams:<url>  webhook:<url>  github  email:ops@example.com  file:alerts.json
Delivery problems are reported but never fail the watch run itself.
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import urllib.parse
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from .providers import http

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}
URL_ENV = {"slack": "SLACK_WEBHOOK_URL", "discord": "DISCORD_WEBHOOK_URL", "teams": "TEAMS_WEBHOOK_URL", "webhook": "LLMSHIFT_WEBHOOK_URL"}


@dataclass
class AlertMessage:
    title: str
    severity: str
    model: str
    lines: list[str] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    run_url: str | None = None
    issue_title: str | None = None  # GitHub issue to open or update (default: the model's drift issue)

    def text(self, markdown: bool = True) -> str:
        bullet = "•"
        body = [f"*{self.title}*" if markdown else self.title]
        body += [f"{bullet} {line}" for line in self.lines]
        if self.run_url:
            body.append(f"Details: {self.run_url}")
        return "\n".join(body)

    def payload(self) -> dict:
        return {
            "source": "llmshift",
            "title": self.title,
            "severity": self.severity,
            "model": self.model,
            "lines": self.lines,
            "alerts": self.alerts,
            "summary": self.summary,
            "run_url": self.run_url,
        }


def parse_destination(spec: Any) -> dict:
    """'slack:https://...' / 'github' / {'type': 'slack', 'url': ...} -> dict."""
    if isinstance(spec, dict):
        return dict(spec)
    s = str(spec).strip()
    kind, _, rest = s.partition(":")
    kind = kind.lower()
    if kind in ("http", "https"):
        return {"type": "webhook", "url": s}
    d: dict = {"type": kind}
    if rest:
        if kind == "email":
            d["to"] = rest
        elif kind == "file":
            d["path"] = rest
        else:
            d["url"] = rest
    return d


def run_url() -> str | None:
    server, repo, run_id = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return f"{server}/{repo}/actions/runs/{run_id}"
    return None


def _url(dest: dict) -> str:
    url = dest.get("url") or os.environ.get(dest.get("url_env") or URL_ENV.get(dest.get("type", ""), "") or "", "")
    if not url:
        env = dest.get("url_env") or URL_ENV.get(dest.get("type", ""), "")
        raise ValueError(f"no URL for the {dest.get('type')} alert (set 'url' or the {env} environment variable)")
    return url


def _post(url: str, body: Any, **kwargs: Any) -> Any:
    # A delivery whose answer timed out is not sent again: it may have arrived already.
    return http.post_json(url, body, timeout=30, max_retries=2, raw=True, idempotent=False, **kwargs)


def _slack_escape(text: str) -> str:
    """Slack reads <!channel>, <@U123> and <url|label> in message text: model output
    and file names must not ping anyone or turn into links."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _send_slack(dest: dict, msg: AlertMessage) -> None:
    _post(_url(dest), {"text": _slack_escape(msg.text(markdown=True))})


def _send_discord(dest: dict, msg: AlertMessage) -> None:
    text = msg.text(markdown=False)
    _post(_url(dest), {"content": text[:1990], "allowed_mentions": {"parse": []}})  # never @everyone


def _send_teams(dest: dict, msg: AlertMessage) -> None:
    card = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        {"type": "TextBlock", "text": msg.title, "weight": "Bolder", "size": "Medium", "wrap": True},
                        {"type": "TextBlock", "text": "\n\n".join("- " + line for line in msg.lines), "wrap": True},
                    ]
                    + ([{"type": "TextBlock", "text": f"[Details]({msg.run_url})", "wrap": True}] if msg.run_url else []),
                },
            }
        ],
    }
    _post(_url(dest), card)


def _send_webhook(dest: dict, msg: AlertMessage) -> None:
    headers = dict(dest.get("headers") or {})
    _post(_url(dest), msg.payload(), headers=headers)


def _github_target(dest: dict) -> tuple[str, str, str]:
    token = os.environ.get(dest.get("token_env") or "GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = dest.get("repo") or os.environ.get("GITHUB_REPOSITORY")
    api = (dest.get("api_url") or os.environ.get("GITHUB_API_URL") or "https://api.github.com").rstrip("/")
    if not token or not repo:
        raise ValueError("the github alert needs GITHUB_TOKEN and GITHUB_REPOSITORY (both are set inside GitHub Actions)")
    return token, repo, api


def _send_github(dest: dict, msg: AlertMessage) -> None:
    token, repo, api = _github_target(dest)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    label = str(dest.get("label", "llmshift"))
    title = msg.issue_title or f"[llmshift] {msg.model}: behaviour drift detected"
    body = "\n".join([f"**{msg.title}**", ""] + [f"- {line}" for line in msg.lines] + (["", f"Run: {msg.run_url}"] if msg.run_url else []))
    existing = None
    for page in range(1, 11):  # every open issue with the label, not just the first page
        issues = http.get_json(
            f"{api}/repos/{repo}/issues?state=open&labels={urllib.parse.quote(label, safe='')}&per_page=100&page={page}",
            headers=headers, timeout=30, max_retries=2,
        )
        existing = next((i for i in issues or [] if isinstance(i, dict) and i.get("title") == title), None)
        if existing is not None or not isinstance(issues, list) or len(issues) < 100:
            break
    if existing:
        _post(f"{api}/repos/{repo}/issues/{existing['number']}/comments", {"body": body}, headers=headers)
    else:
        _post(f"{api}/repos/{repo}/issues", {"title": title, "body": body, "labels": [label]}, headers=headers)


def _send_email(dest: dict, msg: AlertMessage) -> None:
    host = dest.get("host") or os.environ.get("LLMSHIFT_SMTP_HOST")
    to = dest.get("to") or os.environ.get("LLMSHIFT_ALERT_EMAIL")
    if not host or not to:
        raise ValueError("the email alert needs a recipient and LLMSHIFT_SMTP_HOST (plus _PORT, _USER, _PASSWORD, _FROM as needed)")
    port = int(dest.get("port") or os.environ.get("LLMSHIFT_SMTP_PORT") or 587)
    user = dest.get("user") or os.environ.get("LLMSHIFT_SMTP_USER")
    password = os.environ.get(dest.get("password_env") or "LLMSHIFT_SMTP_PASSWORD")
    sender = dest.get("from") or os.environ.get("LLMSHIFT_SMTP_FROM") or user or "llmshift@localhost"
    email = EmailMessage()
    email["Subject"] = f"[llmshift] {msg.title}"
    email["From"] = sender
    email["To"] = to
    email.set_content(msg.text(markdown=False))
    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as smtp:
            if user and password:
                smtp.login(user, password)
            smtp.send_message(email)
    else:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.ehlo()
            if smtp.has_extn("starttls"):
                smtp.starttls(context=context)
                smtp.ehlo()
            elif user and password:
                raise ValueError(
                    f"the mail server {host}:{port} offers no encryption (STARTTLS), so the password is not sent; "
                    "use port 465 (SMTP over TLS) or a server that supports STARTTLS"
                )
            if user and password:
                smtp.login(user, password)
            smtp.send_message(email)


def _send_file(dest: dict, msg: AlertMessage) -> None:
    path = Path(dest.get("path") or "llmshift-alert.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(msg.payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


SENDERS = {
    "slack": _send_slack,
    "discord": _send_discord,
    "teams": _send_teams,
    "webhook": _send_webhook,
    "github": _send_github,
    "email": _send_email,
    "file": _send_file,
}


def check_destinations(destinations: list[Any]) -> list[str]:
    """What would stop an alert from arriving, found before a run that costs money: an
    unknown type, or a missing URL, token, mail server or recipient."""
    problems = []
    for spec in destinations:
        try:
            dest = parse_destination(spec)
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"invalid alert destination ({exc.__class__.__name__})")
            continue
        kind = dest.get("type", "")
        try:
            if kind not in SENDERS:
                raise ValueError(f"unknown alert type {kind!r}; use one of {', '.join(sorted(SENDERS))}")
            if kind in URL_ENV and not _url(dest).strip().lower().startswith(("https://", "http://")):
                raise ValueError(f"the {kind} alert URL must start with https://")
            if kind == "github":
                _github_target(dest)
            if kind == "email" and not ((dest.get("host") or os.environ.get("LLMSHIFT_SMTP_HOST")) and (dest.get("to") or os.environ.get("LLMSHIFT_ALERT_EMAIL"))):
                raise ValueError("the email alert needs a recipient and LLMSHIFT_SMTP_HOST")
        except ValueError as exc:
            problems.append(str(exc))
    return problems


_SEVERITY_ALIASES = {"warn": "warning", "crit": "critical", "error": "critical", "information": "info"}


def _severity(value: Any, default: str) -> str:
    name = str(value or default).strip().lower()
    return _SEVERITY_ALIASES.get(name, name)


def deliver(destinations: list[Any], msg: AlertMessage, default_min_severity: str = "warning") -> list[tuple[str, str | None]]:
    """Send to every destination whose min_severity (default: `default_min_severity`) is met.
    Returns [(name, error|None)]."""
    results = []
    for spec in destinations:
        try:
            dest = parse_destination(spec)
        except Exception as exc:  # pragma: no cover - defensive
            results.append((str(spec), f"invalid destination: {exc}"))
            continue
        kind = dest.get("type", "")
        sender = SENDERS.get(kind)
        if sender is None:  # the name never repeats the destination itself: it can hold a secret URL
            results.append((str(kind) or "alert", f"unknown alert type {kind!r}; use one of {', '.join(sorted(SENDERS))}"))
            continue
        min_sev = _severity(dest.get("min_severity"), _severity(default_min_severity, "warning"))
        if SEVERITY_RANK.get(msg.severity, 0) < SEVERITY_RANK.get(min_sev, 1):
            continue
        try:
            sender(dest, msg)
            results.append((kind, None))
        except Exception as exc:
            results.append((kind, str(exc) or exc.__class__.__name__))
    return results
