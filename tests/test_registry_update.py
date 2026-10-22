"""The daily lifecycle-data update: reading the providers' deprecation tables and
merging them into models.json without ever losing curated data."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TODAY = "2026-10-22"


@pytest.fixture
def updater(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("update_registry", ROOT / "scripts" / "update_registry.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "update_registry", module)  # dataclasses look their module up there
    spec.loader.exec_module(module)
    models = tmp_path / "models.json"
    models.write_bytes((ROOT / "tests" / "data" / "models-2026-10-21.json").read_bytes())  # frozen: the shipped copy moves daily
    monkeypatch.setattr(module, "MODELS", models)
    monkeypatch.setattr(module, "STATE", tmp_path / "sources.json")
    return module


def _table(header, rows, heading=""):
    cells = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<h3>{heading}</h3><table><tr>{cells}</tr>{body}</table>"


OPENAI = "<html><body><main>" + _table(
    ["Shutdown date", "Model / system", "Recommended replacement"],
    [["January 14, 2027", "GPT-4o (gpt-4o-2024-08-06, gpt-4o-2024-11-20)", "gpt-6-luna"],
     ["March 3, 2027", "gpt-4.9-preview-2026", "<a>gpt-6-sol</a>or a newer model"],  # a new deprecation
     ["Jan 6, 2027", "tts-1", "gpt-realtime-2.1-mini"],
     ["Dec 1, 2026", "gpt-image-1.5", "gpt-image-2.5-sunburst or gpt-image-2.5-flare"],
     ["2024-10-28", "babbage-002 fine-tuning", "gpt-4o-mini"],  # an older fine-tuning row
     ["Oct 1, 2026", "gpt-5.4-cyber", "The most capable cyber model available to you."]],
    heading="2026-10-14: GPT-4o and GPT-4o mini",
) + _table(["Date", "Update"], [["June 3, 2026", "Deprecation announced for the Evals platform."]]) + "</main></body></html>"

ANTHROPIC = "<html><body>" + _table(
    ["API model name", "Current state", "Deprecated", "Tentative retirement date"],
    [["claude-fable-5-1", "Active", "N/A", "Not sooner than September 1, 2027"],
     ["claude-sonnet-4-5-20250929", "Deprecated", "September 30, 2026", "December 15, 2026"],  # date moved
     ["claude-3-5-haiku-20241022", "Retired", "October 28, 2025", "February 19, 2026"],
     ["claude-opus-4-8", "Active", "N/A", "Not sooner than May 1, 2027"],
     ["claude-haiku-4-5-20251001", "Active", "N/A", "Not sooner than October 15, 2026"]],
) + "</body></html>"


def test_tables_become_rows(updater):
    rows = updater.parse_page(OPENAI, "openai", "https://x/openai", ("model",), ("shutdown",), ("replacement", "substitute"))
    by = {r.id: r for r in rows}
    assert set(by) == {"gpt-4o-2024-08-06", "gpt-4o-2024-11-20", "gpt-4.9-preview-2026", "tts-1", "gpt-image-1.5", "babbage-002", "gpt-5.4-cyber"}
    assert by["gpt-4o-2024-08-06"].retires == "2027-01-14" and by["gpt-4o-2024-08-06"].deprecated == "2026-10-14"  # from the heading
    assert by["gpt-4.9-preview-2026"].replacement == "gpt-6-sol"  # "<a>gpt-6-sol</a>or" stays two words
    assert by["gpt-image-1.5"].replacement == "gpt-image-2.5-sunburst" and by["gpt-5.4-cyber"].replacement is None
    rows = updater.parse_page(ANTHROPIC, "anthropic", "https://x/anthropic", ("model",), ("retirement",), ("replacement",))
    by = {r.id: r for r in rows}
    assert by["claude-fable-5-1"].retires is None and by["claude-fable-5-1"].deprecated is None  # "Not sooner than" is no date
    assert by["claude-sonnet-4-5-20250929"].retires == "2026-12-15" and by["claude-sonnet-4-5-20250929"].deprecated == "2026-09-30"
    assert updater.parse_date("October 13, 2027 or later") is None and updater.parse_date("Feb 30, 2027") is None
    assert updater.model_ids("fine-tuning of gpt-3.5-turbo-0125 | gpt-3.5-turbo") == ["gpt-3.5-turbo-0125", "gpt-3.5-turbo"]


def test_merge_is_conservative(updater):
    data = json.loads(updater.MODELS.read_text(encoding="utf-8"))
    rows = updater.parse_page(OPENAI, "openai", "https://x/openai", ("model",), ("shutdown",), ("replacement", "substitute"))
    rows += updater.parse_page(ANTHROPIC, "anthropic", "https://x/anthropic", ("model",), ("retirement",), ("replacement",))
    rows.append(updater.Row("openai", "gpt-0-legacy-2023", "2024-01-01", None, None, "https://x/openai"))  # gone long ago
    by = lambda d: {(m["provider"], m["id"]): m for m in d["models"]}  # noqa: E731
    old = by(json.loads(json.dumps(data)))
    changes = updater.merge(data, rows, TODAY)
    new = by(data)
    assert new[("openai", "gpt-4.9-preview-2026")]["retires"] == "2027-03-03" and new[("openai", "gpt-4.9-preview-2026")]["status"] == "deprecated"
    assert ("openai", "gpt-0-legacy-2023") not in new  # past shutdowns are not added
    assert new[("anthropic", "claude-sonnet-4-5-20250929")]["retires"] == "2026-12-15"  # a moved date is followed
    assert new[("anthropic", "claude-3-5-haiku-20241022")] == old[("anthropic", "claude-3-5-haiku-20241022")]  # retired: untouched
    assert new[("anthropic", "claude-fable-5-1")] == old[("anthropic", "claude-fable-5-1")]  # "not sooner than": untouched
    assert new[("openai", "babbage-002")] == old[("openai", "babbage-002")]  # old fine-tuning row: history stays
    assert len(data["models"]) == len(old) + 1 and all(k in new for k in old)  # nothing removed
    assert len(changes) == 2 and any("added openai/gpt-4.9-preview-2026" in c for c in changes)


def test_layout_round_trips(updater):
    for path in (updater.MODELS, ROOT / "src" / "llmshift" / "data" / "models.json"):
        raw = path.read_bytes()
        assert updater.dump(json.loads(raw)) == raw, path  # automatic updates make one-line diffs


def test_the_shipped_data_is_valid_today(monkeypatch):
    # The other tests read a frozen copy; this one checks the data users get, on the real date.
    from llmshift import registry

    monkeypatch.delenv("LLMSHIFT_BUNDLED_REGISTRY")
    monkeypatch.delenv("LLMSHIFT_TODAY")
    assert registry.registry_problems(registry.bundled_registry(), min_models=600) == []


def test_main_writes_only_valid_reviewed_sized_updates(updater, tmp_path, capsys):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "openai.html").write_text(OPENAI, encoding="utf-8")
    (pages / "anthropic.html").write_text(ANTHROPIC, encoding="utf-8")
    (pages / "google.html").write_text("<html><p>moved</p></html>", encoding="utf-8")  # unreadable: left to the source watch
    before = updater.MODELS.read_bytes()
    assert updater.main(["--pages", str(pages), "--today", TODAY]) == 0 and updater.MODELS.read_bytes() == before  # dry run
    out = capsys.readouterr().out
    assert "2 change(s)" in out and "google: only 0 rows" in out
    assert updater.main(["--pages", str(pages), "--today", TODAY, "--max-changes", "1", "--write"]) == 3
    assert updater.MODELS.read_bytes() == before  # too many changes at once: nothing written
    summary = tmp_path / "summary.md"
    assert updater.main(["--pages", str(pages), "--today", TODAY, "--write", "--summary", str(summary)]) == 0
    data = json.loads(updater.MODELS.read_text(encoding="utf-8"))
    assert data["updated"] == TODAY and "added openai/gpt-4.9-preview-2026" in summary.read_text(encoding="utf-8")
    state = json.loads((tmp_path / "sources.json").read_text(encoding="utf-8"))
    assert set(state["pages"]) == {"https://developers.openai.com/api/docs/deprecations", "https://platform.claude.com/docs/en/about-claude/model-deprecations"}
    capsys.readouterr()
    assert updater.main(["--pages", str(pages), "--today", TODAY]) == 0 and "No changes" in capsys.readouterr().out  # idempotent


def test_an_added_field_keeps_the_usual_field_order(updater):
    data = {"updated": "2026-10-21", "models": [{"id": "tts-1", "provider": "openai", "status": "deprecated", "retires": "2027-01-06", "source": "s"}]}
    rows = [updater.Row("openai", "tts-1", "2027-01-06", "gpt-realtime-2.1-mini", None, "s")]
    assert updater.merge(data, rows, TODAY) == ["openai/tts-1: replacement none -> gpt-realtime-2.1-mini"]
    assert list(data["models"][0]) == ["id", "provider", "status", "retires", "replacement", "source"]


def test_a_newly_scheduled_date_also_keeps_the_field_order(updater):
    data = {"updated": "2026-10-21", "models": [{"id": "gemini-3.8-flash-tts", "provider": "google", "status": "active", "source": "s", "note": "rel 2026-09-22"}]}
    rows = [updater.Row("google", "gemini-3.8-flash-tts", "2027-09-22", None, None, "s")]
    assert updater.merge(data, rows, TODAY) == ["google/gemini-3.8-flash-tts: retires not scheduled -> 2027-09-22"]
    assert data["models"][0] == {"id": "gemini-3.8-flash-tts", "provider": "google", "status": "deprecated", "retires": "2027-09-22", "source": "s", "note": "rel 2026-09-22"}
    assert list(data["models"][0]) == ["id", "provider", "status", "retires", "source", "note"]
