"""llmshift pick: the savings finder (offline, simulated models)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmshift import cli
from llmshift.demo import create_project
from llmshift.errors import UsageError
from llmshift.pick import parse_volume, run_pick, split_models
from llmshift.pricing import money
from llmshift.providers import mock, resolve
from llmshift.testset import load_tests

LEGACY, FRUGAL = (5.0, 15.0), (0.15, 0.6)  # simulated prices, USD per 1M tokens (input, output)


@pytest.fixture
def demo(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli, "EXTENSIONS", ("pick",))
    assert cli.main(["snapshot", "logs"]) == 0
    return root


def pick(capsys, *argv):
    capsys.readouterr()
    code = cli.main(["pick", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def _cost(usage: dict, price: tuple) -> float:
    return (usage["input_tokens"] * price[0] + usage["output_tokens"] * price[1]) / 1e6


def test_recommends_the_cheapest_model_that_keeps_quality(demo, capsys):
    code, out, err = pick(capsys, "--models", "mock:nextgen,mock:frugal", "--volume", "100k", "--format", "json")
    data = json.loads(out)  # pure JSON on stdout; progress went to stderr
    assert code == 0 and data["recommended"] == "mock:frugal"
    assert "mock:frugal: pass rate 100%" in err
    cases = load_tests(".llmshift/tests.jsonl")
    # Today (baseline: mock:legacy in the demo config) priced on the usage recorded in the logs.
    today = sum(_cost(c.baseline["usage"], LEGACY) for c in cases) / len(cases)
    # mock:frugal answers like legacy and reports its own usage.
    frugal = sum(_cost(mock.frugal({"messages": c.messages, "params": c.params}).usage, FRUGAL) for c in cases) / len(cases)
    cand = {c["model"]: c for c in data["candidates"]}
    assert data["today"]["cost_per_1k_requests"] == pytest.approx(today * 1000, abs=2e-6)
    assert cand["mock:frugal"]["cost_per_1k_requests"] == pytest.approx(frugal * 1000, abs=2e-6)
    assert cand["mock:frugal"]["saving_pct"] == pytest.approx((today - frugal) / today, abs=1e-4)
    assert data["monthly_saving"] == pytest.approx((today - frugal) * 100_000, abs=1e-4)
    assert cand["mock:frugal"]["pass_rate"] == 1.0 and cand["mock:frugal"]["meets_bar"]
    assert cand["mock:nextgen"]["meets_bar"] is False and cand["mock:nextgen"]["top_failures"][0].startswith("Broken format")
    low, high = cand["mock:frugal"]["pass_rate_range"]
    assert 0.75 < low < 1.0 and high == 1.0
    assert [c["model"] for c in data["candidates"]] == ["mock:frugal", "mock:nextgen"]  # cheapest first
    saving = (today - frugal) * 100_000
    assert data["recommendation"] == (
        f"Switch to mock:frugal: same quality (all 18 cases unchanged), {(today - frugal) / today:.0%} cheaper - "
        f"about {money(saving)} a month saved at 100,000 requests a month."
    )
    assert data["today"]["cost_source"] == "recorded"
    # The demo logs record the usage the simulated model reports, so today's model costs the same when tested.
    legacy = sum(_cost(mock.legacy({"messages": c.messages, "params": c.params}).usage, LEGACY) for c in cases) / len(cases)
    assert today == pytest.approx(legacy, rel=1e-9)


def test_todays_model_among_the_candidates_is_measured_on_the_same_tests(demo, capsys):
    code, out, _ = pick(capsys, "--models", "mock:legacy,mock:frugal", "--format", "json")
    data = json.loads(out)
    cand = {c["model"]: c for c in data["candidates"]}
    assert code == 0 and data["recommended"] == "mock:frugal"
    assert data["today"]["cost_source"] == "tests" and data["today"]["unpriced_models"] == []
    assert data["today"]["cost_per_1k_requests"] == cand["mock:legacy"]["cost_per_1k_requests"]
    assert cand["mock:legacy"]["saving_pct"] == 0 and cand["mock:legacy"]["saving_per_1k_requests"] == 0
    legacy, frugal = cand["mock:legacy"]["cost_per_1k_requests"], cand["mock:frugal"]["cost_per_1k_requests"]
    assert cand["mock:frugal"]["saving_pct"] == pytest.approx((legacy - frugal) / legacy, abs=1e-4)
    code, out, _ = pick(capsys, "--models", "mock:legacy,mock:frugal")
    row = next(line for line in out.splitlines() if "mock:legacy" in line and "%" in line)
    assert "same" in row and "measured on these tests" in out and "today mock:legacy" not in out
    code, out, _ = pick(capsys, "--models", "mock:legacy", "--format", "json")
    assert code == 0 and json.loads(out)["recommendation"] == "Keep mock:legacy: no cheaper candidate keeps quality."


def test_the_same_model_written_two_ways():
    from llmshift.pick import same_model

    assert same_model("gpt-4o", "openai:gpt-4o") and same_model("MOCK:legacy", "mock:legacy")
    assert not same_model("gpt-4o", "gpt-4o-mini") and not same_model("openai:gpt-4o", "anthropic:gpt-4o")
    assert not same_model("", "") and same_model("gpt-4o-2024-05-13", "gpt-4o-2024-05-13")


def test_terminal_table_and_saving_per_1000_without_volume(demo, capsys):
    code, out, _ = pick(capsys, "--models", "mock:frugal", "--models", "mock:nextgen")
    assert code == 0
    assert "Pass rate" in out and "95% range" in out and "Cost / 1k req" in out and "vs today" in out
    assert "100% (18/18)" in out and "33% (6/18)" in out
    assert "saved per 1,000 requests (add --volume N for a monthly figure)" in out
    assert "Prices per 1M tokens" in out and "simulated" in out


def test_no_candidate_keeps_quality(demo, capsys):
    code, out, _ = pick(capsys, "--models", "mock:nextgen", "--format", "json")
    data = json.loads(out)
    assert code == 1 and data["recommended"] is None
    assert data["recommendation"].startswith("No candidate keeps quality: the best, mock:nextgen, passes 33% of cases (minimum 95%).")
    assert "Broken format (8 failed)" in data["recommendation"] and "llmshift fix --to mock:nextgen" in data["recommendation"]


def test_a_failing_candidate_does_not_stop_the_others(demo, capsys):
    code, out, err = pick(capsys, "--models", "openai:gpt-6-luna,mock:frugal")
    assert code == 0 and "Switch to mock:frugal" in out
    assert "openai:gpt-6-luna: could not run" in err and "OPENAI_API_KEY" in out
    code, out, _ = pick(capsys, "--models", "openai:gpt-6-luna,anthropic:claude-haiku-5-5")
    assert code == 2 and "None of the candidates could be tested" in out


def test_unknown_prices_and_price_overrides(demo, capsys, tmp_path):
    fn = tmp_path / "my_model.py"
    fn.write_text("from llmshift.providers.mock import legacy\n\ndef answer(request):\n    return legacy(request)\n", encoding="utf-8")
    spec = f"python:{fn}:answer"
    code, out, _ = pick(capsys, "--models", spec, "--format", "json")
    data = json.loads(out)
    assert code == 0 and data["recommended"] == spec
    assert "Its price is unknown" in data["recommendation"] and f"--price {spec}=INPUT/OUTPUT" in data["recommendation"]
    assert any(f"No price known for {spec}" in n for n in data["notes"])
    # With frugal in the race, the priced candidate wins and the unpriced one is listed after it.
    code, out, _ = pick(capsys, "--models", f"{spec},mock:frugal", "--format", "json")
    data = json.loads(out)
    assert data["recommended"] == "mock:frugal" and [c["model"] for c in data["candidates"]] == ["mock:frugal", spec]
    # A price set on the command line makes it comparable (and here, the cheapest).
    code, out, _ = pick(capsys, "--models", f"{spec},mock:frugal", "--price", f"{spec}=0.01/0.02", "--format", "json")
    data = json.loads(out)
    assert data["recommended"] == spec and data["candidates"][0]["price"]["source"] == "--price"
    code, _, err = pick(capsys, "--models", "mock:frugal", "--price", "mock:frugal=cheap")
    assert code == 2 and "--price expects MODEL=INPUT/OUTPUT" in err


def test_a_more_expensive_model_is_called_out(demo, capsys):
    code, out, _ = pick(capsys, "--models", "mock:legacy", "--baseline", "mock:frugal", "--volume", "1000", "--format", "json")
    data = json.loads(out)
    assert code == 0 and data["recommended"] == "mock:legacy"
    assert data["recommendation"].startswith("No candidate is cheaper at the same quality. The cheapest that passes, mock:legacy, costs")
    assert "more than today's model (about $" in data["recommendation"]
    assert data["candidates"][0]["saving_pct"] < 0


def test_today_is_the_recorded_model_when_no_baseline_is_set(demo, capsys):
    Path("llmshift.yaml").write_text("tests: .llmshift/tests.jsonl\nmin_pass_rate: 0.95\n", encoding="utf-8")
    code, out, _ = pick(capsys, "--models", "mock:frugal", "--format", "json")
    data = json.loads(out)
    assert code == 0
    # gpt-4o-2024-05-13 has a price; claude-3-5-haiku-20241022 (retired) is no longer listed.
    assert data["today"]["unpriced_models"] == ["claude-3-5-haiku-20241022"]
    assert [p["model"] for p in data["today"]["prices"]] == ["gpt-4o-2024-05-13"]
    assert any("14 of 18 cases" in n for n in data["notes"])
    assert data["candidates"][0]["saving_pct"] > 0.9


def test_settings_from_llmshift_yaml(demo, capsys):
    with open("llmshift.yaml", "a", encoding="utf-8") as fh:
        fh.write("pick:\n  models: mock:nextgen, mock:frugal\n  volume: 50k\npricing:\n  mock:frugal: {input: 0.0, output: 0.0}\n")
    code, out, _ = pick(capsys, "--format", "json")
    data = json.loads(out)
    assert code == 0 and data["volume_per_month"] == 50_000 and data["recommended"] == "mock:frugal"
    frugal = next(c for c in data["candidates"] if c["model"] == "mock:frugal")
    assert frugal["cost_per_1k_requests"] == 0 and frugal["price"]["source"] == "config" and frugal["saving_pct"] == 1.0
    Path("llmshift.yaml").write_text("tests: .llmshift/tests.jsonl\n", encoding="utf-8")
    code, _, err = pick(capsys)
    assert code == 2 and "which models should pick try?" in err


def test_reports_markdown_and_github_outputs(demo, capsys, monkeypatch):
    summary, outputs = demo / "summary.md", demo / "outputs.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    code, out, _ = pick(capsys, "--models", "mock:frugal,mock:nextgen", "--format", "markdown", "--json", "pick.json", "--markdown")
    assert code == 0 and out.startswith("## LLMShift pick")
    assert "| ✅ | `mock:frugal` | 100% (18/18) |" in out and "| ❌ | `mock:nextgen` |" in out
    assert json.loads(Path("pick.json").read_text(encoding="utf-8"))["recommended"] == "mock:frugal"
    assert list(Path(".llmshift/reports").glob("pick-*.md"))
    assert "## LLMShift pick" in summary.read_text(encoding="utf-8")
    assert "recommended=mock:frugal" in outputs.read_text(encoding="utf-8")


def test_library_api_with_provider_objects(demo):
    cases = load_tests(".llmshift/tests.jsonl")[:5]
    seen = []
    res = run_pick(cases, [resolve("mock:frugal"), resolve("mock:nextgen")], baseline="mock:legacy", min_pass_rate=0.9, volume="2k", progress=lambda c: seen.append(c.spec))
    assert sorted(seen) == ["mock:frugal", "mock:nextgen"] and res.recommended == "mock:frugal"
    assert res.to_dict()["kind"] == "pick" and res.monthly_saving > 0 and res.best.spec == "mock:frugal" and res.volume == 2000
    with pytest.raises(UsageError):
        run_pick(cases, [])


def test_volume_and_model_lists():
    assert [parse_volume(v) for v in ("100000", "100,000", "100k", "1.5m", 250, None)] == [100000, 100000, 100000, 1500000, 250, None]
    for bad in ("lots", "0", "-5", "1e9"):
        with pytest.raises(UsageError):
            parse_volume(bad)
    assert split_models(["a,b", " c ", "a"]) == ["a", "b", "c"] and split_models("x, y") == ["x", "y"] and split_models(None) == []
