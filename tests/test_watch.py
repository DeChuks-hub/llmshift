"""watch: history, drift detection, scheduling helpers and alert delivery."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from llmshift.alerts import AlertMessage, deliver, parse_destination
from llmshift.providers.mock import MockProvider
from llmshift.testset import TestCase
from llmshift.watch import load_history, mcnemar_p, parse_every, run_watch, seconds_until


def cases(n=12):
    return [
        TestCase(
            id=f"c{i}",
            input={"messages": [{"role": "user", "content": f"q{i}"}], "params": {}},
            baseline={"output": f"answer {i}"},
            checks=[{"type": "equals", "value": f"answer {i}"}],
        )
        for i in range(n)
    ]


def provider(broken: set, model="m-1", fingerprint="fp1", fail_all=False):
    from llmshift.providers.base import Response, ProviderError

    def fn(req):
        if fail_all:
            raise ProviderError("The model `old` has been deprecated", status=404, code="model_not_found")
        i = int(req["messages"][-1]["content"][1:])
        return Response(text="wrong" if i in broken else f"answer {i}", model=model, fingerprint=fingerprint)

    p = MockProvider("watched", fn=fn)
    return p


def test_mcnemar():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(0, 5) == 1.0
    assert mcnemar_p(6, 0) == pytest.approx(1 / 64)
    assert mcnemar_p(3, 3) > 0.5


def test_first_run_then_drift(tmp_path):
    hist = str(tmp_path / "watch")
    first = run_watch(cases(), provider(set()), history_dir=hist, show_progress=False)
    assert first.drifts == [] and first.summary["pass_rate"] == 1.0
    assert len(load_history(hist, "mock:watched")) == 1
    second = run_watch(cases(), provider({0, 1, 2, 3, 4}, model="m-2", fingerprint="fp2"), history_dir=hist, show_progress=False)
    titles = [d.title for d in second.drifts]
    assert any("Pass rate dropped 100% -> 58%" in t for t in titles)
    assert any("New facts failures" in t for t in titles)
    assert any("different model version" in t for t in titles)
    assert any("fingerprint changed" in t for t in titles)
    assert second.worst == "critical"
    assert len(load_history(hist, "mock:watched")) == 2


def test_small_noise_does_not_alert(tmp_path):
    hist = str(tmp_path / "w")
    run_watch(cases(40), provider(set()), history_dir=hist, show_progress=False)
    out = run_watch(cases(40), provider({7}), history_dir=hist, show_progress=False)
    assert not any("dropped" in d.title for d in out.drifts)  # 1 of 40: not significant


def test_model_that_is_gone_aborts_quickly(tmp_path):
    from llmshift.runner import RunAborted

    with pytest.raises(RunAborted):
        run_watch(cases(5), provider(set(), fail_all=True), history_dir=str(tmp_path), show_progress=False)


def test_error_rate_and_min_pass_rate(tmp_path):
    from llmshift.providers.base import Response

    def fn(req):
        i = int(req["messages"][-1]["content"][1:])
        if i < 6:
            raise RuntimeError("HTTP 404 model_not_found")
        return Response(text="wrong")

    out = run_watch(cases(10), MockProvider("x", fn=fn), history_dir=str(tmp_path), min_pass_rate=0.9, show_progress=False)
    titles = " | ".join(d.title + " " + d.detail for d in out.drifts)
    assert "Most requests to the model are failing" in titles and "retired or renamed" in titles
    assert "below the 90% minimum" in titles


def test_retirement_warning(tmp_path):
    from llmshift.providers.base import Response

    p = MockProvider("x", fn=lambda r: Response(text="answer 0"))
    p.name, p.model = "openai", "gpt-4o-2024-05-13"  # spec openai:gpt-4o-2024-05-13 retires 2026-10-23
    out = run_watch(cases(1), p, history_dir=str(tmp_path), show_progress=False, dry_run=True)
    assert any("retires in 2 days" in d.title for d in out.drifts)
    assert not (tmp_path / "openai_gpt-4o-2024-05-13").exists()  # dry run writes nothing


def test_alert_file_and_threshold(tmp_path):
    hist = str(tmp_path / "w")
    alert_file = tmp_path / "alert.json"
    run_watch(cases(), provider(set()), history_dir=hist, show_progress=False, alerts=[f"file:{alert_file}"])
    assert not alert_file.exists()  # no drift, no alert
    out = run_watch(cases(), provider({0, 1, 2, 3, 4}), history_dir=hist, show_progress=False, alerts=[f"file:{alert_file}", "carrier-pigeon:x"])
    payload = json.loads(alert_file.read_text(encoding="utf-8"))
    assert payload["severity"] == "critical" and payload["model"] == "mock:watched" and payload["alerts"]
    assert ("file", None) in out.deliveries and any(err and "unknown alert type" in err for _, err in out.deliveries)


def test_slack_teams_webhook_delivery(server):
    received = []
    server.route("POST", "/hook", lambda r: (received.append(r) or (200, "ok")))
    msg = AlertMessage(title="Drift", severity="warning", model="m", lines=["Pass rate dropped"])
    results = deliver([f"slack:{server.url}/hook", {"type": "teams", "url": server.url + "/hook"}, f"{server.url}/hook", {"type": "discord", "url": server.url + "/hook"}], msg)
    assert all(err is None for _, err in results)
    assert received[0]["json"]["text"].startswith("*Drift*")
    assert received[1]["json"]["attachments"][0]["contentType"] == "application/vnd.microsoft.card.adaptive"
    assert received[2]["json"]["title"] == "Drift"
    assert received[3]["json"]["content"].startswith("Drift")
    info = deliver([{"type": "slack", "url": server.url + "/hook", "min_severity": "critical"}], msg)
    assert info == []  # below the destination's minimum severity
    missing = deliver(["slack"], msg)
    assert "SLACK_WEBHOOK_URL" in missing[0][1]


def test_github_issue_alert(server, monkeypatch):
    issues = []
    server.route("GET", "/repos/acme/app/issues", lambda r: [{"number": 7, "title": "[llmshift] m: behaviour drift detected"}] if issues else [])
    server.route("POST", "/repos/acme/app/issues/7/comments", lambda r: (issues.append(("comment", r["json"])) or (201, {})))
    server.route("POST", "/repos/acme/app/issues", lambda r: (issues.append(("issue", r["json"])) or (201, {"number": 7})))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "acme/app")
    monkeypatch.setenv("GITHUB_API_URL", server.url)
    msg = AlertMessage(title="Drift", severity="critical", model="m", lines=["x"])
    assert deliver(["github"], msg) == [("github", None)]
    assert deliver(["github"], msg) == [("github", None)]
    assert issues[0][0] == "issue" and issues[0][1]["labels"] == ["llmshift"] and issues[1][0] == "comment"


def test_parse_destination_and_schedule():
    assert parse_destination("slack:https://hooks.slack.com/x") == {"type": "slack", "url": "https://hooks.slack.com/x"}
    assert parse_destination("email:ops@example.com") == {"type": "email", "to": "ops@example.com"}
    assert parse_destination("https://example.com/h") == {"type": "webhook", "url": "https://example.com/h"}
    assert parse_every("30m") == 1800 and parse_every("6h") == 21600 and parse_every("1d") == 86400 and parse_every("2") == 7200
    with pytest.raises(ValueError):
        parse_every("soon")
    now = dt.datetime(2026, 10, 21, 1, 0, 0)
    assert seconds_until("02:30", now) == 5400
    assert seconds_until("00:30", now) == 23.5 * 3600
    with pytest.raises(ValueError):
        seconds_until("25:00", now)
