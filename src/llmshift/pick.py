"""`llmshift pick`: the savings finder. Runs your real-traffic tests on several
cheaper models at once and recommends the cheapest one that still passes.

Each candidate goes through the same checks as `llmshift compare`. Costs come from
the token usage each provider reports (estimated from the text when it reports
none) and the prices in pricing.py; today's cost comes from the usage recorded
with the baseline answers.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import console as con
from . import messages as msgs
from . import pricing
from .checks import CATEGORY_TITLES
from .errors import EXIT_FAIL, EXIT_OK, EXIT_USAGE, UsageError
from .pricing import Price, money
from .stats import pass_rate_interval
from .testset import TestCase, unique_ids

DEFAULT_MIN_PASS_RATE = 0.95


@dataclass
class CandidateResult:
    spec: str
    total: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    errors: int = 0
    pass_rate: float | None = None
    low: float | None = None  # 95% range of the pass rate
    high: float | None = None
    price: Price | None = None
    cost_per_1k: float | None = None  # USD per 1,000 requests
    estimated: bool = False  # token counts estimated from the text
    saving_per_1k: float | None = None  # vs today's model, same cases; negative = more expensive
    saving_pct: float | None = None
    latency_p50_ms: float | None = None
    top_failures: list[str] = field(default_factory=list)
    meets_bar: bool = False
    error: str | None = None
    seconds: float = 0.0
    compare: Any = None  # the CompareResult (not serialised)

    def to_dict(self) -> dict:
        d: dict = {
            "model": self.spec,
            "cases": self.total,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "pass_rate_range": [self.low, self.high] if self.low is not None else None,
            "meets_bar": self.meets_bar,
            "cost_per_1k_requests": _round(self.cost_per_1k),
            "cost_estimated": self.estimated,
            "saving_per_1k_requests": _round(self.saving_per_1k),
            "saving_pct": _round(self.saving_pct, 4),
            "latency_p50_ms": self.latency_p50_ms,
            "top_failures": self.top_failures,
            "price": _price_dict(self.price),
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class PickResult:
    candidates: list[CandidateResult]
    cases: int
    min_pass_rate: float
    baseline: str = "recorded"
    baseline_prices: list[dict] = field(default_factory=list)
    baseline_cost_per_1k: float | None = None
    baseline_estimated: bool = False
    baseline_unpriced: list[str] = field(default_factory=list)
    volume: int | None = None
    recommended: str | None = None
    recommendation: str = ""
    monthly_saving: float | None = None
    notes: list[str] = field(default_factory=list)
    tests_path: str | None = None
    created_at: str = ""

    @property
    def best(self) -> CandidateResult | None:
        return next((c for c in self.candidates if c.spec == self.recommended), None)

    def to_dict(self) -> dict:
        return {
            "kind": "pick",
            "created_at": self.created_at,
            "tests": self.tests_path,
            "cases": self.cases,
            "min_pass_rate": self.min_pass_rate,
            "volume_per_month": self.volume,
            "today": {
                "model": self.baseline,
                "cost_per_1k_requests": _round(self.baseline_cost_per_1k),
                "cost_estimated": self.baseline_estimated,
                "prices": self.baseline_prices,
                "unpriced_models": self.baseline_unpriced,
            },
            "recommended": self.recommended,
            "recommendation": self.recommendation,
            "monthly_saving": _round(self.monthly_saving),
            "candidates": [c.to_dict() for c in self.candidates],
            "notes": self.notes,
        }


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _price_dict(price: Price | None) -> dict | None:
    if price is None:
        return None
    return {"input_per_1m": price.input, "output_per_1m": price.output, "source": price.source}


def _prompt_text(case: TestCase) -> str:
    parts = [msgs.content_text(m.get("content")) for m in case.messages]
    for m in case.messages:
        if m.get("tool_calls"):
            parts.append(json.dumps(m["tool_calls"], ensure_ascii=False, default=str))
    tools = case.params.get("tools")
    if tools:
        parts.append(json.dumps(tools, ensure_ascii=False, default=str))
    return "\n".join(parts)


def _output_text(text: str | None, tool_calls: list | None) -> str:
    return (text or "") + (json.dumps(tool_calls, ensure_ascii=False, default=str) if tool_calls else "")


def parse_volume(value: Any) -> int | None:
    """Requests per month: 100000, "100,000", "100k" or "1.5m"."""
    if value is None or value == "":
        return None
    text = str(value).strip().lower().replace(",", "").replace("_", "")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", text)
    if not m:
        raise UsageError(f"--volume expects requests per month, e.g. 100000 or 100k (got {value!r})")
    number = float(m.group(1)) * {"": 1, "k": 1_000, "m": 1_000_000}[m.group(2)]
    if number < 1:
        raise UsageError(f"--volume must be at least 1 request a month (got {value!r})")
    return int(round(number))


def split_models(values: Any) -> list[str]:
    """Candidates from repeated and comma-separated --models (or pick.models in llmshift.yaml)."""
    if values is None:
        return []
    items = [values] if isinstance(values, str) else list(values)
    out: list[str] = []
    for item in items:
        for part in str(item).split(","):
            spec = part.strip()
            if spec and spec not in out:
                out.append(spec)
    return out


def _baseline_costs(cases: list[TestCase], baseline: str | None, overrides: dict) -> tuple[dict[str, float], bool, list[str], list[dict]]:
    """Today's cost per case: (case id -> USD, estimated, unpriced models, prices used)."""
    costs: dict[str, float] = {}
    estimated = False
    unpriced: list[str] = []
    used: dict[str, Price] = {}
    for case in cases:
        model = baseline or case.baseline_model
        if not model:
            continue
        price = pricing.price_for(model, overrides)
        if price is None:
            if model not in unpriced:
                unpriced.append(model)
            continue
        used[model] = price
        usage = (case.baseline or {}).get("usage") or {}
        inp, out, guessed = pricing.usage_tokens(usage, _prompt_text(case), _output_text(case.baseline_output, case.baseline_tool_calls))
        estimated = estimated or guessed
        costs[case.id] = price.cost(inp, out)
    prices = [{"model": m, **(_price_dict(p) or {})} for m, p in used.items()]
    return costs, estimated, unpriced, prices


def _evaluate_candidate(cand: CandidateResult, result: Any, cases: dict[str, TestCase], overrides: dict, threshold: float, base_costs: dict[str, float]) -> None:
    s = result.summary
    cand.compare = result
    cand.total = s.get("total", 0)
    cand.passed, cand.warned, cand.failed, cand.errors = s.get("passed", 0), s.get("warned", 0), s.get("failed", 0), s.get("errors", 0)
    cand.pass_rate = s.get("pass_rate", 0.0)
    _, cand.low, cand.high = pass_rate_interval([1.0 if c.status in ("pass", "warn") else 0.0 for c in result.cases])
    cand.meets_bar = cand.total > 0 and cand.pass_rate + 1e-9 >= threshold
    cand.latency_p50_ms = (s.get("latency_ms") or {}).get("p50")
    cats = sorted(
        ((k, v.get("fail", 0), v.get("warn", 0)) for k, v in (s.get("by_category") or {}).items() if v.get("fail") or v.get("warn")),
        key=lambda x: (-x[1], -x[2], x[0]),
    )
    cand.top_failures = [
        f"{CATEGORY_TITLES.get(k, k)} ({', '.join(p for p in (f'{f} failed' if f else '', f'{w} changed' if w else '') if p)})" for k, f, w in cats[:3]
    ]
    cand.price = pricing.price_for(cand.spec, overrides)
    if cand.price is None:
        return
    costs: dict[str, float] = {}
    for c in result.cases:
        if c.status == "error" or c.id not in cases:
            continue
        inp, out, guessed = pricing.usage_tokens(c.usage, _prompt_text(cases[c.id]), _output_text(c.candidate_output, c.candidate_tool_calls))
        cand.estimated = cand.estimated or guessed
        costs[c.id] = cand.price.cost(inp, out)
    if not costs:
        return
    cand.cost_per_1k = 1000 * sum(costs.values()) / len(costs)
    common = [cid for cid in costs if cid in base_costs]
    if common:
        base = sum(base_costs[cid] for cid in common) / len(common)
        mine = sum(costs[cid] for cid in common) / len(common)
        cand.saving_per_1k = 1000 * (base - mine)
        cand.saving_pct = (base - mine) / base if base > 0 else None


def _rank(c: CandidateResult) -> tuple:
    return (c.error is not None, c.cost_per_1k is None, c.cost_per_1k if c.cost_per_1k is not None else 0.0, -(c.pass_rate or 0.0), c.latency_p50_ms or 0.0)


def _quality(c: CandidateResult, threshold: float) -> str:
    if c.total and c.passed == c.total:
        return f"same quality (all {c.total} cases unchanged)"
    if c.total and c.passed + c.warned == c.total:
        return f"same quality (all {c.total} cases pass, {c.warned} with minor differences)"
    return f"passes your quality bar (pass rate {c.pass_rate:.0%}, minimum {threshold:.0%})"


def _recommend(res: PickResult) -> None:
    threshold = res.min_pass_rate
    tested = [c for c in res.candidates if c.error is None]
    passing = [c for c in tested if c.meets_bar]
    if not tested:
        res.recommendation = "None of the candidates could be tested: " + "; ".join(f"{c.spec}: {c.error}" for c in res.candidates)
        return
    if not passing:
        best = max(tested, key=lambda c: (c.pass_rate or 0.0, -(c.cost_per_1k or 0.0)))
        why = f" Main differences: {'; '.join(best.top_failures)}." if best.top_failures else ""
        res.recommendation = (
            f"No candidate keeps quality: the best, {best.spec}, passes {best.pass_rate:.0%} of cases "
            f"(minimum {threshold:.0%}).{why} Try `llmshift fix --to {best.spec}` to close the gaps with prompt changes."
        )
        return
    best = min(passing, key=_rank)
    res.recommended = best.spec
    quality = _quality(best, threshold)
    if best.cost_per_1k is None:
        res.recommendation = (
            f"Switch to {best.spec}: {quality}. Its price is unknown, so the saving can't be worked out: "
            f"set it with --price {best.spec}=INPUT/OUTPUT (USD per 1M tokens) or under pricing: in llmshift.yaml."
        )
        return
    if best.spec == res.baseline:
        res.recommendation = f"Keep {best.spec}: no cheaper candidate keeps quality."
        return
    if best.saving_per_1k is None:
        res.recommendation = (
            f"Switch to {best.spec}: {quality}, {money(best.cost_per_1k)} per 1,000 requests. "
            "Today's cost is unknown: pass --baseline MODEL (the model you use now) or set its price with --price."
        )
        return
    pct = best.saving_pct
    if best.saving_per_1k <= 0:
        more = f"{-pct:.0%} more" if pct is not None else "more"
        extra = f" (about {money(-best.saving_per_1k * res.volume / 1000)} more a month)" if res.volume else ""
        res.recommendation = (
            f"No candidate is cheaper at the same quality. The cheapest that passes, {best.spec}, costs {more} than today's model{extra}."
        )
        return
    saving = f"{pct:.0%} cheaper" if pct is not None else f"{money(best.saving_per_1k)} cheaper per 1,000 requests"
    if res.volume:
        res.monthly_saving = best.saving_per_1k * res.volume / 1000
        tail = f"about {money(res.monthly_saving)} a month saved at {res.volume:,} requests a month"
    else:
        tail = f"{money(best.saving_per_1k)} saved per 1,000 requests (add --volume N for a monthly figure)"
    res.recommendation = f"Switch to {best.spec}: {quality}, {saving} - {tail}."
    if best.low is not None and best.low + 1e-9 < threshold:
        res.notes.append(
            f"With {best.total} cases the true pass rate of {best.spec} could be as low as {best.low:.0%}; more test cases give a firmer answer."
        )


def run_pick(
    cases: list[TestCase],
    candidates: list[Any],
    *,
    baseline: str | None = None,
    config: dict | None = None,
    min_pass_rate: float | None = DEFAULT_MIN_PASS_RATE,
    volume: int | None = None,
    prices: dict[str, Price] | None = None,
    concurrency: int = 4,
    cache: Any = None,
    judge: Any = None,
    embedder: Any = None,
    allow_python: bool = False,
    severity: dict | None = None,
    tests_path: str | None = None,
    progress: Callable[[CandidateResult], None] | None = None,
) -> PickResult:
    """Run the test set on every candidate (in parallel) and recommend the cheapest that passes."""
    from .compare import run_compare
    from .providers import http
    from .providers.custom import kill_running

    config = config or {}
    cases = unique_ids(cases)
    if not cases:
        raise UsageError("no test cases to run")
    specs = [c if isinstance(c, str) else getattr(c, "spec", str(c)) for c in candidates]
    if not specs:
        raise UsageError("no candidate models: pass --models a,b,c or set pick.models in llmshift.yaml")
    try:
        overrides = {**pricing.overrides_from_config(config), **{k.lower(): v for k, v in (prices or {}).items()}}
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    threshold = DEFAULT_MIN_PASS_RATE if min_pass_rate is None else float(min_pass_rate)
    by_id = {c.id: c for c in cases}
    base_costs, base_estimated, unpriced, base_prices = _baseline_costs(cases, baseline, overrides)
    recorded = sorted({c.baseline_model for c in cases if c.baseline_model})
    res = PickResult(
        candidates=[],
        cases=len(cases),
        min_pass_rate=threshold,
        baseline=baseline or ("recorded" + (f" ({', '.join(recorded[:3])})" if recorded else "")),
        baseline_prices=base_prices,
        baseline_cost_per_1k=1000 * sum(base_costs.values()) / len(base_costs) if base_costs else None,
        baseline_estimated=base_estimated,
        baseline_unpriced=unpriced,
        volume=volume,
        tests_path=tests_path,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )
    if unpriced:
        covered = len(base_costs)
        res.notes.append(
            f"Today's price is unknown for {', '.join(unpriced)}; savings use the {covered} of {len(cases)} cases with a known price. "
            "Set it with --price MODEL=INPUT/OUTPUT or pricing: in llmshift.yaml."
        )

    def run_one(candidate: Any, spec: str) -> CandidateResult:
        cand = CandidateResult(spec=spec)
        start = time.perf_counter()
        try:
            result = run_compare(
                cases, candidate, judge=judge, embedder=embedder, config=config, concurrency=concurrency, cache=cache,
                allow_python=allow_python, severity=severity, min_pass_rate=threshold, show_progress=False, label="pick",
            )
            if result.summary.get("evaluated", 0) == 0:
                first = next((c.error for c in result.cases if c.error), "every request failed")
                cand.error = first
            else:
                _evaluate_candidate(cand, result, by_id, overrides, threshold, base_costs)
        except Exception as exc:  # one broken candidate must not stop the others
            message = str(exc) or exc.__class__.__name__
            cand.error = message[len(spec) + 2:] if message.startswith(f"{spec}: ") else message
        cand.seconds = time.perf_counter() - start
        return cand

    pool = ThreadPoolExecutor(max_workers=max(1, len(specs)))
    interrupted = False
    try:
        pending = {pool.submit(run_one, cand, spec) for cand, spec in zip(candidates, specs)}
        while pending:
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in done:
                cand = fut.result()
                res.candidates.append(cand)
                if progress is not None:
                    progress(cand)
    except KeyboardInterrupt:
        interrupted = True
        http.CANCEL.set()
        kill_running()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True)
    order = {s: i for i, s in enumerate(specs)}
    res.candidates.sort(key=lambda c: (_rank(c), order.get(c.spec, 0)))
    for c in res.candidates:
        if c.error is None and c.price is None:
            res.notes.append(f"No price known for {c.spec}: set it with --price {c.spec}=INPUT/OUTPUT (USD per 1M tokens) or pricing: in llmshift.yaml.")
    if any(c.estimated for c in res.candidates) or res.baseline_estimated:
        res.notes.append("~ marks costs worked out from estimated token counts (the provider or log did not report usage).")
    _recommend(res)
    return res


# ------------------------------------------------------------------ output


def _cost_cell(cost: float | None, estimated: bool) -> str:
    if cost is None:
        return "?"
    return ("~" if estimated else "") + money(cost)


def _vs_today(c: CandidateResult) -> str:
    if c.saving_pct is None:
        return "?" if c.cost_per_1k is not None else ""
    if abs(c.saving_pct) < 0.005:
        return "same"
    return f"{-c.saving_pct:+.0%}"


def _rate_cell(c: CandidateResult) -> str:
    return f"{c.pass_rate:.0%} ({c.passed + c.warned}/{c.total})"


def _range_cell(c: CandidateResult) -> str:
    if c.low is None or c.high is None:
        return ""
    return f"{c.low:.0%}-{c.high:.0%}"


def _latency_cell(c: CandidateResult) -> str:
    return f"{c.latency_p50_ms:.0f} ms" if c.latency_p50_ms is not None else ""


def _price_line(res: PickResult) -> str:
    bits = []
    for c in res.candidates:
        if c.price is not None:
            bits.append(f"{c.spec} {money(c.price.input)}/{money(c.price.output)} ({c.price.source})")
    for p in res.baseline_prices:
        bits.append(f"today {p['model']} {money(p['input_per_1m'])}/{money(p['output_per_1m'])} ({p['source']})")
    return "Prices per 1M tokens (input/output): " + "; ".join(bits) if bits else ""


def pick_terminal(res: PickResult) -> str:
    lines = [
        con.style("LLMShift pick", "bold") + f"  {res.cases} test case(s), {len(res.candidates)} candidate(s), quality bar {res.min_pass_rate:.0%}",
        f"Today: {res.baseline}  " + (f"{_cost_cell(res.baseline_cost_per_1k, res.baseline_estimated)} per 1,000 requests" if res.baseline_cost_per_1k is not None else "cost unknown"),
        "",
    ]
    rows = []
    for c in res.candidates:
        if c.error is not None:
            rows.append([con.style(con.sym("error"), "red"), c.spec, con.style("error", "red"), "", "", "", ""])
            continue
        icon = con.style(con.sym("pass"), "green") if c.meets_bar else con.style(con.sym("fail"), "red")
        name = con.style(c.spec, "bold") if c.spec == res.recommended else c.spec
        rows.append([icon, name, _rate_cell(c), _range_cell(c), _cost_cell(c.cost_per_1k, c.estimated), _vs_today(c), _latency_cell(c)])
    lines.append(con.table(rows, ["", "Model", "Pass rate", "95% range", "Cost / 1k req", "vs today", "p50 latency"], max_widths=[3, 44, 16, 10, 13, 9, 12]))
    for c in res.candidates:
        if c.error is not None:
            lines.append(con.style(f"  {c.spec}: {con.truncate(c.error, 160)}", "red"))
    lines.append("")
    if res.recommended:
        lines.append(con.style(res.recommendation, "green", "bold"))
    else:
        lines.append(con.style(res.recommendation, "red", "bold"))
    price_line = _price_line(res)
    if price_line or res.notes:
        lines.append("")
    if price_line:
        lines.append(con.style(price_line, "grey"))
    for note in res.notes:
        lines.append(con.style(f"Note: {note}", "grey"))
    return "\n".join(lines)


def _md(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def pick_markdown(res: PickResult) -> str:
    lines = [
        "## LLMShift pick",
        "",
        f"{res.cases} test case(s), quality bar {res.min_pass_rate:.0%}. Today: `{_md(res.baseline)}`, "
        + (f"{_cost_cell(res.baseline_cost_per_1k, res.baseline_estimated)} per 1,000 requests." if res.baseline_cost_per_1k is not None else "cost unknown."),
        "",
        f"**{_md(res.recommendation)}**",
        "",
        "| | Model | Pass rate | 95% range | Cost / 1k requests | vs today | p50 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in res.candidates:
        if c.error is not None:
            lines.append(f"| ✗ | `{_md(c.spec)}` | error: {_md(con.truncate(c.error, 120))} | | | | |")
            continue
        icon = "✅" if c.meets_bar else "❌"
        lines.append(f"| {icon} | `{_md(c.spec)}` | {_rate_cell(c)} | {_range_cell(c)} | {_cost_cell(c.cost_per_1k, c.estimated)} | {_vs_today(c)} | {_latency_cell(c)} |")
    price_line = _price_line(res)
    if price_line or res.notes:
        lines.append("")
    if price_line:
        lines.append(_md(price_line))
    for note in res.notes:
        lines.append(f"- {_md(note)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- cli


def register(sub: Any) -> None:
    from . import cli

    p = sub.add_parser(
        "pick",
        help="find the cheapest model that still passes your tests (savings finder)",
        description="Run the test set on several candidate models at once and recommend the cheapest one that keeps quality, "
        "with its pass rate, cost per 1,000 requests and speed.",
    )
    p.add_argument("--models", "--model", dest="models", action="append", metavar="MODEL[,MODEL...]", help="candidate models (repeatable or comma-separated); default pick.models in llmshift.yaml")
    p.add_argument("--baseline", metavar="MODEL", help="the model you use today, for today's cost (default: baseline in llmshift.yaml, else the recorded model)")
    p.add_argument("--min-pass-rate", type=float, help="quality bar a candidate must reach (default min_pass_rate in llmshift.yaml: 0.95)")
    p.add_argument("--volume", metavar="N", help="requests per month, for a monthly saving, e.g. 100000 or 100k (default pick.volume)")
    p.add_argument("--price", action="append", metavar="MODEL=IN/OUT", help="price in USD per 1M tokens, e.g. azure:prod=2.5/10 (repeatable; wins over pricing: in llmshift.yaml)")
    p.add_argument("--format", choices=["table", "json", "markdown"], default="table", help="output on the terminal")
    p.add_argument("--json", nargs="?", const="", metavar="FILE", help="also write the results as JSON (default: in the reports directory)")
    p.add_argument("--markdown", nargs="?", const="", metavar="FILE", help="also write a Markdown summary, e.g. for PR comments")
    cli._add_run_options(p)
    p.set_defaults(func=cmd_pick)


def cmd_pick(args: argparse.Namespace, cfg: dict) -> int:
    from . import cli

    section = cfg.get("pick") if isinstance(cfg.get("pick"), dict) else {}
    specs = split_models(args.models) or split_models(section.get("models"))
    if not specs:
        raise UsageError("which models should pick try? Pass --models a,b,c (e.g. --models openai:gpt-6-luna,anthropic:claude-haiku-5-5) or set pick.models in llmshift.yaml")
    prices: dict[str, Price] = {}
    for item in args.price or []:
        try:
            model, price = pricing.parse_price(item)
        except ValueError as exc:
            raise UsageError(str(exc)) from None
        prices[model.lower()] = price
    volume = parse_volume(args.volume if args.volume is not None else section.get("volume"))
    min_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.get("min_pass_rate")
    if min_rate is not None and not 0 <= float(min_rate) <= 1:
        raise UsageError(f"--min-pass-rate must be between 0 and 1 (got {min_rate})")
    cases, path = cli._load_cases(cfg, args)
    con.out(con.style(f"Trying {len(specs)} model(s) on {len(cases)} test case(s) at once...", "grey"), err=True)

    def progress(c: CandidateResult) -> None:
        if c.error is not None:
            con.out(con.style(f"  {c.spec}: could not run ({con.truncate(c.error, 120)})", "red"), err=True)
        else:
            cost = f", {_cost_cell(c.cost_per_1k, c.estimated)} per 1,000 requests" if c.cost_per_1k is not None else ""
            con.out(con.style(f"  {c.spec}: pass rate {c.pass_rate:.0%}{cost} ({c.seconds:.0f}s)", "grey"), err=True)

    res = run_pick(
        cases,
        specs,
        baseline=args.baseline or cfg.get("baseline"),
        config=cfg,
        min_pass_rate=float(min_rate) if min_rate is not None else None,
        volume=volume,
        prices=prices,
        concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=cli._cache(cfg, args),
        judge=args.judge or cfg.get("judge"),
        embedder=cfg.get("embedder"),
        allow_python=args.allow_python or bool(cfg.get("allow_python_checks")),
        severity=cli._severity(cfg, args),
        tests_path=path,
        progress=progress,
    )
    markdown = pick_markdown(res)
    if args.format == "json":
        con.out(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        con.out(markdown)
    else:
        con.out(pick_terminal(res))
    reports_dir = Path(cfg.get("reports_dir") or ".llmshift/reports")
    stamp = cli._timestamp()
    written = []
    for value, suffix, text in ((args.json, ".json", lambda: json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n"), (args.markdown, ".md", lambda: markdown)):
        if value is None:
            continue
        target = Path(value) if value else reports_dir / f"pick-{stamp}{suffix}"
        written.append(cli._rel(cli._write(target, text())))
    if written:
        con.out(con.style("Wrote " + ", ".join(written), "grey"), err=True)
    cli._step_summary(markdown)
    if res.recommended:
        cli._github_output("recommended", res.recommended)
    if not any(c.error is None for c in res.candidates):
        return EXIT_USAGE
    return EXIT_OK if res.recommended else EXIT_FAIL
