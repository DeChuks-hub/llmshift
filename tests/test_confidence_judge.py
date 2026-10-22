"""Repeat runs and pass-rate ranges, the paired test against the old model, judge gating
and judge calibration."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from llmshift import cli
from llmshift.cache import ResponseCache
from llmshift.calibrate import calibration_note, cohen_kappa, load_labels, run_calibration
from llmshift.checks import CheckContext, Sample, run_checks
from llmshift.compare import run_compare
from llmshift.demo import create_project
from llmshift.providers.mock import MockProvider
from llmshift.stats import cases_needed, wilson_interval
from llmshift.testset import TestCase, load_tests


def _cases(n: int = 6) -> list[TestCase]:
    return [
        TestCase(id=f"c{i}", input={"messages": [{"role": "user", "content": f"label {i}"}], "params": {}}, baseline={"output": "yes"}, checks=[{"type": "equals", "value": "yes"}])
        for i in range(n)
    ]


def test_repeats_find_flaky_cases_and_give_a_range(tmp_path):
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def flaky(request):
        user = request["messages"][-1]["content"]
        with lock:
            calls[user] = calls.get(user, 0) + 1
            n = calls[user]
        if user == "label 0":  # fails every second run
            return "no" if n % 2 == 0 else "yes"
        return "yes"

    cache = ResponseCache(tmp_path / "cache")
    res = run_compare(_cases(), MockProvider("flaky", fn=flaky), repeat=3, cache=cache, show_progress=False)
    s = res.summary
    assert s["repeats"] == 3 and s["flaky"] == 1 and s["flaky_cases"] == ["c0"]
    c0 = next(c for c in res.cases if c.id == "c0")
    assert c0.flaky and c0.run_statuses == ["pass", "fail", "pass"] and c0.pass_fraction == pytest.approx(2 / 3, abs=1e-3)
    assert s["pass_rate"] == pytest.approx((5 + 2 / 3) / 6, abs=1e-3)
    low, high = s["pass_rate_range"]
    assert low < s["pass_rate"] < high <= 1.0
    assert any("flaky" in note for note in res.insights)
    assert all(v == 3 for v in calls.values())
    # The first run is shared with ordinary cached runs; repeats are cached separately.
    run_compare(_cases(), MockProvider("flaky", fn=flaky), repeat=3, cache=cache, show_progress=False)
    assert all(v == 3 for v in calls.values())


def test_single_run_summary_matches_previous_pass_rate_and_reports_sample_size():
    res = run_compare(_cases(4), "mock:echo", show_progress=False)  # echo answers "label i" -> all fail
    s = res.summary
    assert s["pass_rate"] == 0.0 and s["repeats"] == 1 and s["flaky"] == 0
    assert s["pass_rate_range"][0] == 0.0 and s["pass_rate_range"][1] > 0.3  # 4 cases say little
    assert s["cases_needed"] > 4 and any("pin it down" in note for note in res.insights)
    data = res.to_dict()["cases"][0]
    assert "runs" not in data and "flaky" not in data  # nothing extra for ordinary runs


@pytest.fixture
def demo_cases(tmp_path, monkeypatch):
    root = create_project(str(tmp_path / "demo"))
    monkeypatch.chdir(root)
    assert cli.main(["snapshot", "logs"]) == 0
    return load_tests(".llmshift/tests.jsonl")


@pytest.mark.parametrize("rate", [0.0, 0.2, 0.5, 0.8, 0.95, 1.0])
def test_cases_needed_is_the_smallest_count_with_a_10_point_range(rate):
    n = cases_needed(rate)
    low, high = wilson_interval(rate * n, n)
    assert high - low <= 0.1 + 1e-9  # the promised +/- 5 points
    low, high = wilson_interval(rate * (n - 1), n - 1)
    assert high - low > 0.1  # and no fewer cases would do
    assert cases_needed(1.0) >= 35  # 16 of 16 passing still only shows 81%-100%


def test_paired_comparison_against_the_live_old_model(demo_cases):
    worse = run_compare(demo_cases, "mock:nextgen", baseline="mock:legacy", show_progress=False)
    vs = worse.summary["vs_baseline"]
    assert vs["compared"] == 18 and vs["regressions"] >= 12 and vs["improvements"] == 0
    assert vs["significantly_worse"] and vs["p_value"] < 0.001
    assert any("significantly worse" in note for note in worse.insights)
    same = run_compare(demo_cases, "mock:frugal", baseline="mock:legacy", show_progress=False)
    assert same.summary["vs_baseline"]["regressions"] == 0 and not same.summary["vs_baseline"]["significantly_worse"]


# ------------------------------------------------------------------ judge


class CountingJudge(MockProvider):
    def __init__(self, verdict="pass"):
        self.verdict, self.calls = verdict, 0
        super().__init__("judge", fn=self._answer)

    def _answer(self, request):
        self.calls += 1
        return json.dumps({"verdict": self.verdict, "reason": "checked", "differences": []})


def _judged(output: str, baseline: str, extra: list | None = None) -> TestCase:
    return TestCase(id="j", input={"messages": [{"role": "user", "content": "q"}], "params": {}}, baseline={"output": baseline}, checks=(extra or []) + [{"type": "llm_judge"}])


def test_judge_is_only_asked_when_rules_cannot_decide():
    judge = CountingJudge()
    ctx = CheckContext(judge=judge)
    identical = run_checks(_judged("The refund is $40.", "The refund is $40."), Sample(text="The refund is  $40."), ctx)
    assert identical[-1].status == "pass" and "identical" in identical[-1].message and judge.calls == 0
    failing = run_checks(_judged('{"a": 1}', '{"a": 1}', [{"type": "json_valid"}]), Sample(text="not json"), ctx)
    assert failing[-1].status == "skip" and failing[-1].details["skip_reason"] == "decided_by_rules" and judge.calls == 0
    open_ended = run_checks(_judged("We will refund $40 within five days.", "A $40 refund arrives in 5 business days."), Sample(text="We will refund $40 within five days."), ctx)
    assert open_ended[-1].status == "pass" and judge.calls == 1
    assert ctx.judge_skipped == 2
    always = CheckContext(judge=judge, judge_mode="always")
    run_checks(_judged("The refund is $40.", "The refund is $40."), Sample(text="The refund is $40."), always)
    assert judge.calls == 2


def test_compare_reports_saved_judge_calls():
    judge = CountingJudge()
    cases = [_judged("same answer here", "same answer here"), _judged("a different answer", "the reference answer")]
    cases[1].id = "k"
    res = run_compare(cases, MockProvider("m", fn=lambda r: "same answer here"), judge=judge, show_progress=False)
    assert judge.calls == 1  # only the case whose answer differs from its reference
    assert "1 judge call(s) made; 1 skipped because the rule-based checks already decided." in res.insights


# ------------------------------------------------------------ calibration


def _labels(path: Path) -> None:
    rows = []
    for i, (cand, label) in enumerate([("good answer", "pass"), ("good reply", "pass"), ("bad answer", "fail"), ("bad reply", "fail"), ("good but wrong", "fail"), ("fine answer", "pass")]):
        rows.append({"id": f"l{i}", "messages": [{"role": "user", "content": "question"}], "reference": "reference answer", "candidate": cand, "label": label})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _judge_module(tmp_path: Path) -> str:
    mod = tmp_path / "judge_fn.py"
    mod.write_text(
        "import json\n\ndef judge(request):\n"
        "    text = request['messages'][-1]['content'].split('[New answer to evaluate]')[1].split('[Criteria]')[0]\n"
        "    return json.dumps({'verdict': 'pass' if 'good' in text else 'fail', 'reason': 'r'})\n",
        encoding="utf-8",
    )
    return f"python:{mod}:judge"


def test_calibration_measures_agreement(tmp_path):
    path = tmp_path / ".llmshift" / "judge-labels.jsonl"
    _labels(path)
    from llmshift.providers import resolve

    cal = run_calibration(load_labels(path), resolve(_judge_module(tmp_path)))
    assert (cal.n, cal.agree, cal.too_lenient, cal.too_strict) == (6, 4, 1, 1)
    assert cal.kappa == pytest.approx(cohen_kappa([("pass", "pass")] * 2 + [("fail", "fail")] * 2 + [("pass", "fail"), ("fail", "pass")]))
    assert cohen_kappa([("pass", "pass"), ("fail", "fail")]) == 1.0 and cohen_kappa([]) is None


def test_judge_commands_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    spec = _judge_module(tmp_path)
    assert cli.main(["judge", "calibrate", "--judge", spec]) == 2  # no labels yet
    _labels(tmp_path / ".llmshift" / "judge-labels.jsonl")
    capsys.readouterr()
    assert cli.main(["judge", "calibrate", "--judge", spec, "--min-agreement", "0.6"]) == 0
    out = capsys.readouterr().out
    assert "4 of 6 pairs (67%)" in out and "too lenient 1x" in out
    saved = json.loads((tmp_path / ".llmshift" / "judge-calibration.json").read_text(encoding="utf-8"))
    assert saved["judges"][spec]["agree"] == 4
    assert cli.main(["judge", "calibrate", "--judge", spec]) == 1  # below the default 0.8
    note = calibration_note(spec, {})
    assert "4 of 6 pairs (67%" in note and "care" in note
    assert "were checked with" in calibration_note("openai:other", {})
    assert "not been checked" in calibration_note(spec, {"cache_dir": str(tmp_path / "elsewhere" / "cache")})
    # Checking a second judge keeps the first one's result.
    (tmp_path / "strict_fn.py").write_text("import json\n\ndef judge(request):\n    return json.dumps({'verdict': 'fail', 'reason': 'r'})\n", encoding="utf-8")
    strict = f"python:{tmp_path / 'strict_fn.py'}:judge"
    cli.main(["judge", "calibrate", "--judge", strict, "--min-agreement", "0"])
    assert "4 of 6 pairs" in calibration_note(spec, {}) and "pairs" in calibration_note(strict, {})
    # The older one-judge file format is still read.
    (tmp_path / ".llmshift" / "judge-calibration.json").write_text(json.dumps({"judge": spec, "n": 6, "agree": 4, "rate": 0.67}), encoding="utf-8")
    assert "4 of 6 pairs" in calibration_note(spec, {})


def test_labelling_is_interactive_and_saves_as_it_goes(demo_cases, monkeypatch, capsys):
    assert cli.main(["compare", "--to", "mock:nextgen", "--json", "res.json", "--no-report"]) == 1
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["judge", "label", "--results", "res.json"]) == 2  # needs a terminal
    answers = iter(["maybe", "y", "n", "s", "q"])
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    capsys.readouterr()
    assert cli.main(["judge", "label", "--results", "res.json", "--count", "5"]) == 0
    labels = load_labels(Path(".llmshift/judge-labels.jsonl"))
    assert [r["label"] for r in labels] == ["pass", "fail"]
    assert all(r["messages"][0]["role"] == "system" for r in labels)  # full input from the tests, not a preview
    assert "saved 2 label(s)" in capsys.readouterr().out


def test_compare_quotes_the_judge_calibration(demo_cases, tmp_path, capsys):
    spec = _judge_module(tmp_path)
    _labels(Path(".llmshift/judge-labels.jsonl"))
    assert cli.main(["judge", "calibrate", "--judge", spec, "--min-agreement", "0.5"]) == 0
    capsys.readouterr()
    cli.main(["compare", "--to", "mock:nextgen", "--judge", spec, "--no-report", "--limit", "4"])
    assert "agrees with your labels on 4 of 6 pairs" in capsys.readouterr().out


def test_repeat_flag_validation(demo_cases, capsys):
    with pytest.raises(SystemExit) as info:  # refused before anything runs
        cli.main(["compare", "--to", "mock:legacy", "--repeat", "0", "--no-report"])
    assert info.value.code == 2 and "must be 1 or more" in capsys.readouterr().err
    assert cli.main(["compare", "--to", "mock:legacy", "--repeat", "2", "--no-report", "--limit", "3"]) == 0
    assert "x 2 runs" in capsys.readouterr().out


def test_a_judge_given_for_the_run_judges_free_text_cases(demo_cases):
    # The demo test set was made without --with-judge-checks: --judge must still judge
    # the prose answers (and never the JSON triage cases, which are checked exactly).
    assert not any(c.get("type") == "llm_judge" for case in demo_cases for c in case.checks)
    asked = []

    class Judge(MockProvider):
        def __init__(self):
            super().__init__("judge", fn=self._answer)

        def _answer(self, request):
            asked.append(request["messages"][-1]["content"])
            return json.dumps({"verdict": "pass", "reason": "same meaning", "differences": []})

    res = run_compare(demo_cases, "mock:nextgen", judge=Judge(), show_progress=False)
    prose = {c.id for c in res.cases if not c.baseline_output.lstrip().startswith("{")}
    judged = [c for c in res.cases if any(r.type == "llm_judge" for r in c.checks)]
    assert len(judged) == len(prose) == 10 and {c.id for c in judged} == prose
    assert asked  # nextgen rewords the support replies without breaking a rule: the judge decides those
    assert not any('"category"' in q.split("[New answer to evaluate]")[-1] for q in asked)  # never a JSON answer
    plain = run_compare(demo_cases, "mock:frugal", show_progress=False)
    assert not any(r.type == "llm_judge" for c in plain.cases for r in c.checks)  # no judge, no judge checks
