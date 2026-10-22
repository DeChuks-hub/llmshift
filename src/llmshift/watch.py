"""`llmshift watch`: a scheduled check of the live model that alerts when its
behaviour drifts.

Each run executes the test set against the live model (never cached), stores a
summary in the history directory and compares it with the previous run and the
first (reference) run. Alerts fire for: requests failing (model removed?),
pass rate below the minimum, a statistically significant pass-rate drop
(paired McNemar test), new kinds of failures, outputs moving away from the
baseline, a different model version/fingerprint, and upcoming retirement.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .alerts import AlertMessage, SEVERITY_RANK, deliver, run_url
from .compare import CompareResult, run_compare
from .registry import Registry
from .testset import TestCase, unique_ids
from .textutil import rouge_l, stable_hash

KEEP_RUNS = 60
MIN_INTERVAL = 60.0  # seconds between scheduled runs


@dataclass
class Drift:
    severity: str  # info | warning | critical
    title: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"severity": self.severity, "title": self.title, "detail": self.detail}


@dataclass
class WatchOutcome:
    model: str
    run_id: str
    summary: dict
    drifts: list[Drift] = field(default_factory=list)
    previous: dict | None = None
    reference: dict | None = None
    deliveries: list[tuple[str, str | None]] = field(default_factory=list)
    compare: CompareResult | None = None
    history_file: str = ""

    @property
    def worst(self) -> str:
        if not self.drifts:
            return "ok"
        return max(self.drifts, key=lambda d: SEVERITY_RANK.get(d.severity, 0)).severity


def mcnemar_p(b: int, c: int) -> float:
    """One-sided exact McNemar p-value for b regressions vs c improvements."""
    n = b + c
    if n == 0 or b == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / (2**n)


def two_proportion_p(p1: float, n1: int, p2: float, n2: int) -> float:
    """One-sided p-value that pass rate p2 (of n2 cases) is lower than p1 (of n1)."""
    if not n1 or not n2:
        return 1.0
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0 if p1 > p2 else 1.0
    return 0.5 * math.erfc(((p1 - p2) / se) / math.sqrt(2))


def case_hash(case: TestCase) -> str:
    """Identifies a test's definition: a changed test is not compared with its old results."""
    return stable_hash({"input": case.input, "checks": case.checks, "baseline": case.baseline}, 12)


def tests_fingerprint(hashes: dict[str, str]) -> str:
    return stable_hash(sorted(hashes.items()), 16)


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def history_paths(history_dir: str | Path, model: str) -> tuple[Path, Path]:
    base = Path(history_dir) / _slug(model)
    return base / "history.jsonl", base / "runs"


def load_history(history_dir: str | Path, model: str) -> list[dict]:
    hist_file, _ = history_paths(history_dir, model)
    if not hist_file.exists():
        return []
    out = []
    for line in hist_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _load_run(runs_dir: Path, run_id: str) -> dict:
    path = runs_dir / f"{run_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def summarize_run(res: CompareResult, case_hashes: dict[str, str] | None = None) -> tuple[dict, dict]:
    """(summary, per-case data) for a watch run."""
    case_hashes = case_hashes or {}
    per_case = {}
    sims = []
    for c in res.cases:
        sim = None
        if c.status != "error" and c.baseline_output and c.candidate_output is not None:
            sim = round(rouge_l(c.candidate_output, c.baseline_output), 4)
            sims.append(sim)
        per_case[c.id] = {
            "case": case_hashes.get(c.id),
            "status": c.status,
            "hash": stable_hash(c.candidate_output or "", 12) if c.status != "error" else None,
            "similarity": sim,
            "failing": sorted({r.category for r in c.checks if r.status == "fail" and r.severity == "error"}),
            "model": c.model,
            "fingerprint": c.fingerprint,
            "error": c.error,
            "output": (c.candidate_output or "")[:4000],
        }
    s = res.summary
    cats = {k: v.get("fail", 0) for k, v in (s.get("by_category") or {}).items()}
    summary = {
        "total": s.get("total", 0),
        "evaluated": s.get("evaluated", 0),
        "passed": s.get("passed", 0),
        "warned": s.get("warned", 0),
        "failed": s.get("failed", 0),
        "errors": s.get("errors", 0),
        "pass_rate": s.get("pass_rate", 0.0),
        "error_rate": round(s.get("errors", 0) / s["total"], 4) if s.get("total") else 0.0,
        "similarity": round(statistics.mean(sims), 4) if sims else None,
        "failing_categories": {k: v for k, v in cats.items() if v},
        "models": dict(Counter(c.model for c in res.cases if c.model)),
        "fingerprints": dict(Counter(c.fingerprint for c in res.cases if c.fingerprint)),
        "latency_ms": s.get("latency_ms"),
    }
    return summary, per_case


def comparable_runs(history: list[dict], fingerprint: str | None) -> list[dict]:
    """Earlier runs of the same test set that produced results (an outage run is skipped)."""
    return [
        h for h in history
        if (fingerprint is None or h.get("tests") == fingerprint) and (h.get("summary") or {}).get("evaluated")
    ]


def detect_drift(
    summary: dict,
    per_case: dict,
    history: list[dict],
    prev_cases: dict,
    *,
    model: str,
    min_pass_rate: float | None,
    max_pass_rate_drop: float,
    max_similarity_drop: float,
    retirement_warn_days: int,
    registry: Registry | None = None,
    errors_sample: list[str] | None = None,
    fingerprint: str | None = None,
) -> list[Drift]:
    """`history` holds earlier runs, oldest first; `prev_cases` the per-case data of the
    latest comparable one. Runs of a different test set (`fingerprint`) are not compared."""
    drifts: list[Drift] = []
    total = summary.get("total", 0) or 1
    if summary.get("error_rate", 0) >= 0.5:
        sample = "; ".join((errors_sample or [])[:2])
        removed = any(re.search(r"404|not[_ ]found|does not exist|model_not_found|deprecated|decommission", e or "", re.I) for e in errors_sample or [])
        detail = f"{summary.get('errors')} of {total} requests failed. {sample}"
        if removed:
            detail += " The model may have been retired or renamed."
        drifts.append(Drift("critical", "Most requests to the model are failing", detail.strip()))
    elif summary.get("errors"):
        drifts.append(Drift("info", f"{summary['errors']} request(s) failed", "; ".join((errors_sample or [])[:2])))

    rate = summary.get("pass_rate", 0.0)
    if min_pass_rate is not None and summary.get("evaluated") and rate + 1e-9 < min_pass_rate:
        drifts.append(Drift("critical", f"Pass rate {rate:.0%} is below the {min_pass_rate:.0%} minimum", ""))

    runs = comparable_runs(history, fingerprint)
    previous = runs[-1] if runs else None
    reference = runs[0] if runs else None
    if history and not runs and summary.get("evaluated"):
        drifts.append(Drift("info", "The test set changed: this run is the new reference", "Earlier runs used different test cases, so they are not compared."))
    if not runs and min_pass_rate is None and summary.get("evaluated") and rate < 0.5:
        drifts.append(
            Drift(
                "warning",
                f"Only {rate:.0%} of cases match the recorded baseline",
                "This run becomes the reference for drift detection. Set watch.min_pass_rate to alert on an absolute level.",
            )
        )
    if previous and summary.get("evaluated"):
        # Compare case by case: only tests that ran (unchanged) in both runs count.
        pairs = []
        for cid, cur in per_case.items():
            old = prev_cases.get(cid)
            if not old or cur.get("status") == "error" or old.get("status") == "error":
                continue
            if old.get("case") and cur.get("case") and old["case"] != cur["case"]:
                continue
            pairs.append((old.get("status") in ("pass", "warn"), cur.get("status") in ("pass", "warn")))
        if pairs:
            b = sum(1 for was, now in pairs if was and not now)
            c = sum(1 for was, now in pairs if now and not was)
            before = sum(1 for was, _ in pairs if was) / len(pairs)
            after = sum(1 for _, now in pairs if now) / len(pairs)
            p = mcnemar_p(b, c)
            detail = f"{b} case(s) regressed, {c} improved (McNemar p={p:.3f})."
        elif not prev_cases:  # no per-case results kept for the previous run: compare the totals
            prev_summary = previous.get("summary", {})
            before, after = prev_summary.get("pass_rate", 0.0), rate
            p = two_proportion_p(before, prev_summary.get("evaluated", 0), after, summary.get("evaluated", 0))
            b = c = 0
            detail = f"p={p:.3f}"
        else:  # no test ran unchanged in both runs: nothing to compare
            before = after = rate
            p, b, c, detail = 1.0, 0, 0, ""
        drop = before - after
        if drop >= max_pass_rate_drop and p < 0.05:
            sev = "critical" if drop >= 2 * max_pass_rate_drop else "warning"
            drifts.append(Drift(sev, f"Pass rate dropped {before:.0%} -> {after:.0%} since the last run", detail))
        elif b >= 1 and p < 0.05:
            drifts.append(Drift("warning", f"{b} case(s) regressed since the last run", f"{c} improved; McNemar p={p:.3f}."))
        prev_cats = set((previous.get("summary", {}).get("failing_categories") or {}).keys())
        for cat, n in (summary.get("failing_categories") or {}).items():
            if cat not in prev_cats and cat != "errors":
                drifts.append(Drift("warning", f"New {cat} failures: {n} case(s)", "This kind of failure did not occur in the previous run."))
        changed = sum(1 for cid, cur in per_case.items() if prev_cases.get(cid, {}).get("hash") and cur.get("hash") and cur["hash"] != prev_cases[cid]["hash"])
        compared = sum(1 for cid, cur in per_case.items() if prev_cases.get(cid, {}).get("hash") and cur.get("hash"))
        if compared:
            summary["changed_outputs"] = round(changed / compared, 4)
    if reference and reference is not previous and summary.get("evaluated"):
        ref_summary = reference.get("summary", {})
        ref_rate = ref_summary.get("pass_rate", 0.0)
        p_ref = two_proportion_p(ref_rate, ref_summary.get("evaluated", 0), rate, summary.get("evaluated", 0))
        if ref_rate - rate >= max_pass_rate_drop and p_ref < 0.05 and not any("dropped" in d.title for d in drifts):
            drifts.append(Drift("warning", f"Pass rate is {ref_rate - rate:.0%} lower than the reference run ({reference.get('run_id')})", f"{ref_rate:.0%} -> {rate:.0%}: slow drift (p={p_ref:.3f})."))

    sims = [h.get("summary", {}).get("similarity") for h in runs[-7:] if h.get("summary", {}).get("similarity") is not None]
    if sims and summary.get("similarity") is not None:
        med = statistics.median(sims)
        if med - summary["similarity"] >= max_similarity_drop:
            drifts.append(Drift("warning", "Outputs moved away from the baseline", f"Mean similarity to the baseline fell from {med:.2f} to {summary['similarity']:.2f}."))

    last = next((h for h in reversed(history) if (h.get("summary") or {}).get("evaluated")), None)
    if last:  # model versions are compared whatever the test set
        previous = last
        old_models = set((previous.get("summary", {}).get("models") or {}).keys())
        new_models = set((summary.get("models") or {}).keys())
        if old_models and new_models and old_models != new_models:
            drifts.append(Drift("info", "The provider served a different model version", f"{', '.join(sorted(old_models))} -> {', '.join(sorted(new_models))}"))
        old_fp = set((previous.get("summary", {}).get("fingerprints") or {}).keys())
        new_fp = set((summary.get("fingerprints") or {}).keys())
        if old_fp and new_fp and not (old_fp & new_fp):
            drifts.append(Drift("info", "The backend fingerprint changed", f"{', '.join(sorted(old_fp))[:80]} -> {', '.join(sorted(new_fp))[:80]}. Behaviour can change with it."))

    if registry is not None:
        bare = model.split(":", 1)[1] if ":" in model else model
        lk = registry.lookup(bare, warn_days=retirement_warn_days)
        if lk.status == "retired":
            drifts.append(Drift("critical", f"{bare} is retired", lk.message))
        elif lk.status == "retiring" and lk.days_left is not None and lk.days_left <= retirement_warn_days:
            drifts.append(Drift("warning", f"{bare} retires in {lk.days_left} days", lk.message))
    return drifts


def run_watch(
    cases: list[TestCase],
    model: Any,
    *,
    history_dir: str = ".llmshift/watch",
    config: dict | None = None,
    judge: Any = None,
    concurrency: int = 4,
    allow_python: bool = False,
    severity: dict | None = None,
    min_pass_rate: float | None = None,
    max_pass_rate_drop: float = 0.05,
    max_similarity_drop: float = 0.10,
    retirement_warn_days: int = 30,
    alerts: list[Any] | None = None,
    alert_on: str = "warning",
    dry_run: bool = False,
    show_progress: bool | None = None,
) -> WatchOutcome:
    from .providers import resolve

    config = config or {}
    provider = resolve(model, config) if isinstance(model, str) else model
    cases = unique_ids(cases)
    hashes = {c.id: case_hash(c) for c in cases}
    fingerprint = tests_fingerprint(hashes)
    res = run_compare(cases, provider, judge=judge, config=config, concurrency=concurrency, cache=None, allow_python=allow_python, severity=severity, show_progress=show_progress, label="watch")
    summary, per_case = summarize_run(res, hashes)
    now = _dt.datetime.now(_dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    history = load_history(history_dir, provider.spec)
    hist_file, runs_dir = history_paths(history_dir, provider.spec)
    runs = comparable_runs(history, fingerprint)
    prev_cases = _load_run(runs_dir, runs[-1]["run_id"]) if runs else {}
    try:
        registry = Registry.load()
    except Exception:
        registry = None
    errors_sample = [c.error for c in res.cases if c.error][:3]
    drifts = detect_drift(
        summary,
        per_case,
        history,
        prev_cases,
        model=provider.spec,
        min_pass_rate=min_pass_rate,
        max_pass_rate_drop=max_pass_rate_drop,
        max_similarity_drop=max_similarity_drop,
        retirement_warn_days=retirement_warn_days,
        registry=registry,
        errors_sample=errors_sample,
        fingerprint=fingerprint,
    )
    outcome = WatchOutcome(
        model=provider.spec,
        run_id=run_id,
        summary=summary,
        drifts=drifts,
        previous=runs[-1] if runs else None,
        reference=runs[0] if runs else None,
        compare=res,
        history_file=str(hist_file),
    )
    if not dry_run:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{run_id}.json").write_text(json.dumps(per_case, ensure_ascii=False), encoding="utf-8")
        record = {
            "run_id": run_id,
            "created_at": now.isoformat(timespec="seconds"),
            "model": provider.spec,
            "tests": fingerprint,
            "summary": summary,
            "drift": [d.to_dict() for d in drifts],
        }
        prefix = ""
        try:  # a hand-edited or truncated history must not swallow this run's line
            if hist_file.stat().st_size:
                with hist_file.open("rb") as fh:
                    fh.seek(-1, os.SEEK_END)
                    prefix = "" if fh.read(1) == b"\n" else "\n"
        except OSError:
            pass
        with hist_file.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(prefix + json.dumps(record, ensure_ascii=False) + "\n")
        run_files = sorted(runs_dir.glob("*.json"))
        for old in run_files[:-KEEP_RUNS]:
            try:
                old.unlink()
            except OSError:
                pass
    actionable = [d for d in drifts if SEVERITY_RANK.get(d.severity, 0) >= SEVERITY_RANK.get(alert_on, 1)]
    if alerts and actionable:
        worst = max(actionable, key=lambda d: SEVERITY_RANK[d.severity]).severity
        lines = [f"{d.title}" + (f" - {d.detail}" if d.detail else "") for d in actionable]
        lines.append(f"Pass rate {summary['pass_rate']:.0%} ({summary['passed'] + summary['warned']}/{summary['evaluated']}), errors {summary['errors']}")
        msg = AlertMessage(
            title=f"LLMShift: {provider.spec} behaviour drift ({worst})",
            severity=worst,
            model=provider.spec,
            lines=lines,
            alerts=[d.to_dict() for d in actionable],
            summary=summary,
            run_url=run_url(),
        )
        outcome.deliveries = deliver(alerts, msg, default_min_severity=alert_on)
    return outcome


def parse_every(value: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", value or "")
    if not m:
        raise ValueError(f"cannot parse interval {value!r}; use e.g. 30m, 6h, 1d")
    seconds = float(m.group(1)) * {"": 3600, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    if seconds < MIN_INTERVAL:
        raise ValueError(f"interval {value!r} is too short; use at least 1m")
    return seconds


def seconds_until(at: str, now: _dt.datetime | None = None) -> float:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", at or "")
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"cannot parse time {at!r}; use HH:MM (24h, local time)")
    now = now or _dt.datetime.now()
    target = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
    if target <= now:
        target += _dt.timedelta(days=1)
    return (target - now).total_seconds()


def sleep_until_next(every: str | None, at: str | None) -> None:
    delay = seconds_until(at) if at else parse_every(every or "24h")
    end = time.time() + delay
    while True:
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 60))
