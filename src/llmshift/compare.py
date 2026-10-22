"""`llmshift compare`: run the test set on a new model and show exactly what changed.

Each case's output is checked against what the old model did (format, facts,
tone, tool calls, refusals...). With a live baseline model, checks that also
fail for the old model are treated as noise rather than regressions.
"""

from __future__ import annotations

import copy
import datetime as _dt
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .cache import ResponseCache
from .checks import CheckContext, CheckResult, Sample, run_checks
from .messages import request_preview
from .providers import resolve
from .providers.base import Provider, Response
from .runner import adaptations, run_cases
from .testset import TestCase, unique_ids
from .textutil import word_count


@dataclass
class CaseResult:
    id: str
    group: str
    status: str  # pass | warn | fail | error
    input_preview: str = ""
    baseline_output: str = ""
    candidate_output: str = ""
    candidate_tool_calls: list = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    latency_ms: float | None = None
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    model: str | None = None
    fingerprint: str | None = None
    unstable: list[str] = field(default_factory=list)
    baseline_live_output: str | None = None
    baseline_latency_ms: float | None = None
    baseline_model: str | None = None

    @property
    def failing(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status in ("fail", "error")]

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "group": self.group,
            "status": self.status,
            "input_preview": self.input_preview,
            "baseline_output": self.baseline_output,
            "candidate_output": self.candidate_output,
            "checks": [c.to_dict() for c in self.checks],
        }
        for key in ("candidate_tool_calls", "error", "latency_ms", "usage", "finish_reason", "model", "fingerprint", "unstable", "baseline_live_output", "baseline_latency_ms", "baseline_model"):
            value = getattr(self, key)
            if value not in (None, [], {}):
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CaseResult":
        r = cls(id=d["id"], group=d.get("group", ""), status=d.get("status", "error"))
        for key in ("input_preview", "baseline_output", "candidate_output", "candidate_tool_calls", "error", "latency_ms", "usage", "finish_reason", "model", "fingerprint", "unstable", "baseline_live_output", "baseline_latency_ms", "baseline_model"):
            if key in d:
                setattr(r, key, d[key])
        r.checks = [CheckResult.from_dict(c) for c in d.get("checks") or []]
        return r


@dataclass
class CompareResult:
    candidate: str
    baseline: str
    cases: list[CaseResult]
    summary: dict = field(default_factory=dict)
    adaptations: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    tests_path: str | None = None
    judge: str | None = None
    created_at: str = ""
    min_pass_rate: float | None = None
    label: str = "compare"

    def to_dict(self) -> dict:
        return {
            "llmshift_version": __version__,
            "kind": self.label,
            "created_at": self.created_at,
            "candidate": self.candidate,
            "baseline": self.baseline,
            "judge": self.judge,
            "tests": self.tests_path,
            "min_pass_rate": self.min_pass_rate,
            "summary": self.summary,
            "insights": self.insights,
            "adaptations": self.adaptations,
            "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompareResult":
        return cls(
            candidate=d.get("candidate", "?"),
            baseline=d.get("baseline", "recorded"),
            cases=[CaseResult.from_dict(c) for c in d.get("cases") or []],
            summary=d.get("summary") or {},
            adaptations=d.get("adaptations") or [],
            insights=d.get("insights") or [],
            tests_path=d.get("tests"),
            judge=d.get("judge"),
            created_at=d.get("created_at", ""),
            min_pass_rate=d.get("min_pass_rate"),
            label=d.get("kind", "compare"),
        )

    @property
    def passed(self) -> bool:
        threshold = self.min_pass_rate if self.min_pass_rate is not None else 0.0
        s = self.summary
        if s.get("evaluated", 0) == 0:
            return False
        return s.get("pass_rate", 0.0) + 1e-9 >= threshold


def case_status(results: list[CheckResult], error: str | None) -> str:
    """fail: an error-level check failed, or could not be evaluated (unverified is not a pass);
    warn: a warning-level check failed; pass otherwise. Info-level checks never count."""
    if error:
        return "error"
    if any(r.status in ("fail", "error") and r.severity == "error" for r in results):
        return "fail"
    if any(r.status in ("fail", "error") and r.severity == "warn" for r in results):
        return "warn"
    return "pass"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return round(values[idx], 1)


def failure_patterns(result: CheckResult) -> list[str]:
    """Short, countable descriptions of what went wrong in one check."""
    d = result.details or {}
    if result.type == "tone" and d.get("differences"):
        from .tone import EMOJI_RE

        return [EMOJI_RE.sub("", re.sub(r"\s*\(.*?\)", "", x["text"])).strip() for x in d["differences"]]
    if result.type == "structure" and d.get("changes"):
        return list(d["changes"])
    if result.type == "json_match" and (d.get("mismatches") or d.get("missing")):
        out = [f"{m['path']} changed" for m in d.get("mismatches") or []]
        return out + [f"{p} missing" for p in d.get("missing") or []]
    if result.type == "facts":
        out = []
        if d.get("changed"):
            out.append("numbers changed")
        if any(True for _ in d.get("missing") or []):
            out.append("facts missing")
        return out or ["facts differ"]
    if result.type == "json_schema" and d.get("errors"):
        return [re.sub(r"^\$[^:]*:\s*", "", e) for e in d["errors"][:3]]
    if result.type in ("equals", "one_of"):
        return ["label/answer changed" if result.type == "equals" else "answer outside the label set"]
    if result.type == "similarity":
        return ["low similarity to the reference answer"]
    if result.type == "length":
        return ["much longer" if "longer" in (result.message or "") else "much shorter"]
    msg = re.sub(r"\s*\(.*?\)", "", result.message or result.type)
    msg = re.sub(r'"[^"]*"', '"..."', msg)
    return [msg.strip()[:80] or result.type]


SKIP_REASONS = {
    "needs_judge": "needs a judge model (--judge)",
    "python_disabled": "python graders disabled (--allow-python)",
}


def summarize(cases: list[CaseResult], responses: dict[str, Response] | None = None) -> dict:
    counts = Counter(c.status for c in cases)
    evaluated = len(cases) - counts["error"]
    by_category: dict[str, dict] = defaultdict(lambda: {"fail": 0, "warn": 0, "patterns": Counter(), "examples": {}})
    by_check: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, Counter] = defaultdict(Counter)
    skipped: Counter = Counter()
    unstable = 0
    for c in cases:
        groups[c.group]["total"] += 1
        groups[c.group][c.status] += 1
        unstable += len(c.unstable)
        cats_fail, cats_warn = set(), set()
        for r in c.checks:
            by_check[r.type][r.status] += 1
            if r.status == "skip":
                reason = (r.details or {}).get("skip_reason")
                if reason in SKIP_REASONS:
                    skipped[SKIP_REASONS[reason]] += 1
            if r.status in ("fail", "error") and r.severity in ("error", "warn"):
                (cats_fail if r.severity == "error" else cats_warn).add(r.category)
                for p in failure_patterns(r):
                    by_category[r.category]["patterns"][p] += 1
                    by_category[r.category]["examples"].setdefault(p, r.message)
        for cat in cats_fail:
            by_category[cat]["fail"] += 1
        for cat in cats_warn - cats_fail:
            by_category[cat]["warn"] += 1
    if counts["error"]:
        by_category["errors"]["fail"] = counts["error"]
        for c in cases:
            if c.status == "error":
                by_category["errors"]["patterns"][re.sub(r"\s*\(.*?\)$", "", (c.error or "error"))[:90]] += 1
    latencies = [c.latency_ms for c in cases if c.latency_ms is not None and c.status != "error"]
    base_lat = [c.baseline_latency_ms for c in cases if c.baseline_latency_ms is not None]
    tokens_in = sum(int((c.usage or {}).get("input_tokens") or 0) for c in cases)
    tokens_out = sum(int((c.usage or {}).get("output_tokens") or 0) for c in cases)
    # Average lengths over the same cases on both sides (answered, with a baseline).
    both = [c for c in cases if c.status != "error" and c.baseline_output and c.candidate_output]
    cand_words = [word_count(c.candidate_output) for c in both]
    base_words = [word_count(c.baseline_output) for c in both]
    total = len(cases)
    return {
        "total": total,
        "evaluated": evaluated,
        "passed": counts["pass"],
        "warned": counts["warn"],
        "failed": counts["fail"],
        "errors": counts["error"],
        # Errors count against the pass rate: a case that could not be checked has not passed.
        "pass_rate": round((counts["pass"] + counts["warn"]) / total, 4) if total else 0.0,
        "unchanged_rate": round(counts["pass"] / total, 4) if total else 0.0,
        "by_category": {
            k: {
                "fail": v["fail"],
                "warn": v["warn"],
                "patterns": dict(v["patterns"].most_common(8)),
                "examples": {p: v["examples"].get(p, "") for p, _ in v["patterns"].most_common(8)},
            }
            for k, v in by_category.items()
        },
        "by_check": {k: dict(v) for k, v in sorted(by_check.items())},
        "groups": {k: dict(v) for k, v in sorted(groups.items())},
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
        "baseline_latency_ms": {"p50": _percentile(base_lat, 0.5), "p95": _percentile(base_lat, 0.95)} if base_lat else None,
        "tokens": {"input": tokens_in, "output": tokens_out},
        "truncated": sum(1 for c in cases if c.finish_reason == "length"),
        "avg_words": {"baseline": round(statistics.mean(base_words), 1) if base_words else None, "candidate": round(statistics.mean(cand_words), 1) if cand_words else None},
        "skipped_checks": dict(skipped),
        "unstable_checks": unstable,
    }


def build_insights(summary: dict, cases: list[CaseResult], adapt: list[str]) -> list[str]:
    out = []
    evaluated = summary.get("evaluated", 0)
    if summary.get("truncated"):
        out.append(
            f"{summary['truncated']} response(s) hit the output token limit (finish_reason=length). Reasoning models count "
            "thinking tokens toward max_tokens/max_completion_tokens - raise the limit for the new model."
        )
    fenced = sum(1 for c in cases for r in c.checks if r.type == "json_valid" and r.status == "fail" and (r.details or {}).get("fenced"))
    if fenced:
        out.append(f"JSON wrapped in markdown code fences in {fenced} case(s): parsers that call json.loads() on the raw text will break.")
    refusals = sum(1 for c in cases for r in c.checks if r.type == "refusal" and r.status == "fail")
    if refusals:
        out.append(f"The new model declined {refusals} request(s) that the old model answered.")
    aw = summary.get("avg_words") or {}
    if aw.get("baseline") and aw.get("candidate"):
        change = (aw["candidate"] - aw["baseline"]) / aw["baseline"]
        if abs(change) >= 0.3:
            out.append(f"Answers are {abs(change):.0%} {'longer' if change > 0 else 'shorter'} on average ({aw['baseline']:.0f} -> {aw['candidate']:.0f} words).")
    bl, cl = summary.get("baseline_latency_ms"), summary.get("latency_ms") or {}
    if bl and bl.get("p50") and cl.get("p50"):
        out.append(f"Median latency {bl['p50']:.0f} ms -> {cl['p50']:.0f} ms.")
    for note in adapt:
        out.append(f"Request adapted automatically: {note}.")
    for reason, n in (summary.get("skipped_checks") or {}).items():
        out.append(f"{n} check(s) skipped: {reason}.")
    if summary.get("unstable_checks"):
        out.append(f"{summary['unstable_checks']} check failure(s) ignored because the old model fails them too (noise).")
    if summary.get("errors") and evaluated == 0:
        out.append("Every request failed - check the model name and API key.")
    return out


def run_compare(
    cases: list[TestCase],
    candidate: Any,
    *,
    baseline: Any = None,
    judge: Any = None,
    embedder: Any = None,
    config: dict | None = None,
    concurrency: int = 4,
    cache: ResponseCache | None = None,
    allow_python: bool = False,
    severity: dict | None = None,
    patches: Any = None,
    overrides: dict | None = None,
    tests_path: str | None = None,
    min_pass_rate: float | None = None,
    show_progress: bool | None = None,
    label: str = "compare",
) -> CompareResult:
    config = config or {}
    cases = unique_ids(cases)
    cand: Provider = resolve(candidate, config) if isinstance(candidate, str) else candidate
    base: Provider | None = resolve(baseline, config) if isinstance(baseline, str) else baseline
    judge_p: Provider | None = resolve(judge, config) if isinstance(judge, str) else judge
    embed_p: Provider | None = resolve(embedder, config) if isinstance(embedder, str) else embedder

    responses = run_cases(cand, cases, concurrency=concurrency, cache=cache, patches=patches, overrides=overrides, label=cand.spec, show_progress=show_progress)
    base_responses: dict[str, Response] = {}
    if base is not None:
        base_responses = run_cases(base, cases, concurrency=concurrency, cache=cache, overrides=overrides, label=f"baseline {base.spec}", show_progress=show_progress)

    grader_models: dict[str, Provider | None] = {}
    lock = threading.Lock()

    def resolve_model(spec: str) -> Provider | None:
        with lock:
            if spec not in grader_models:
                try:
                    grader_models[spec] = resolve(spec, config)
                except Exception:
                    grader_models[spec] = None
            return grader_models[spec]

    ctx = CheckContext(
        judge=judge_p,
        embedder=embed_p,
        allow_python=allow_python,
        severity=dict(severity or {}),
        resolve_model=resolve_model,
        judge_cache=cache,
    )

    def evaluate(case: TestCase) -> CaseResult:
        resp = responses.get(case.id) or Response(error="no response", error_code="missing")
        live = base_responses.get(case.id)
        eval_case = case
        if not case.has_baseline and live is not None and live.ok:
            eval_case = copy.copy(case)
            eval_case.baseline = {"output": live.text, "tool_calls": live.tool_calls, "model": live.model}
        result = CaseResult(
            id=case.id,
            group=case.group or "default",
            status="error",
            input_preview=request_preview(case.messages),
            baseline_output=eval_case.baseline_output,
            candidate_output=resp.text,
            candidate_tool_calls=resp.tool_calls,
            error=resp.error,
            latency_ms=resp.latency_ms,
            usage=resp.usage,
            finish_reason=resp.finish_reason,
            model=resp.model,
            fingerprint=resp.fingerprint,
        )
        if live is not None:
            result.baseline_live_output = live.text if live.ok else None
            result.baseline_latency_ms = live.latency_ms if live.ok else None
            result.baseline_model = live.model
        if not resp.ok:
            result.status = "error"
            return result
        checks = run_checks(eval_case, Sample.from_response(resp), ctx)
        if live is not None and live.ok:
            base_checks = run_checks(eval_case, Sample.from_response(live), ctx)
            for i, (c, b) in enumerate(zip(checks, base_checks)):
                if c.status == "fail" and b.status == "fail":
                    result.unstable.append(c.type)
                    checks[i] = CheckResult(c.type, c.category, c.severity, "skip", f"also fails on the old model (noise): {c.message}", c.score, c.name, c.details)
        result.checks = checks
        result.status = case_status(checks, None)
        return result

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        results = list(pool.map(evaluate, cases))

    adapt = adaptations(cand) + (adaptations(base) if base else [])
    summary = summarize(results)
    recorded_models = sorted({c.baseline_model for c in cases if c.baseline_model})
    base_label = base.spec if base else ("recorded" + (f" ({', '.join(recorded_models[:3])})" if recorded_models else ""))
    res = CompareResult(
        candidate=cand.spec,
        baseline=base_label,
        cases=results,
        summary=summary,
        adaptations=adapt,
        tests_path=tests_path,
        judge=judge_p.spec if judge_p else None,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        min_pass_rate=min_pass_rate,
        label=label,
    )
    res.insights = build_insights(summary, results, adapt)
    if ctx.judge_calls:
        res.insights.append(f"{ctx.judge_calls} judge call(s) made.")
    return res
