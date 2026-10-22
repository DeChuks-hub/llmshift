"""Shadow mode: send a share of live traffic to a new model, in the background.

    from llmshift.record import wrap
    client = wrap(OpenAI(), shadow="openai:gpt-5.6-sol", shadow_rate=0.05)

After a real call returns, a copy of the request goes to the new model on a
background thread. The app never waits for it and never sees its errors; when the
new model falls behind, copies are dropped rather than queued without limit. Each
pair (the real answer and the shadow answer) is appended to
logs/llmshift-shadow.jsonl, and `llmshift shadow report` checks the shadow answers
against the real ones with the same checks as `compare`.
"""

from __future__ import annotations

import atexit
import datetime as _dt
import json
import os
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import console as con
from .errors import EXIT_FAIL, EXIT_OK, UsageError
from .providers.base import Response
from .textutil import stable_hash

DEFAULT_PATH = "logs/llmshift-shadow.jsonl"


def _canonical(api: str, request: dict) -> dict | None:
    from . import logparse

    parser = {"chat": logparse.chat_request, "messages": logparse.anthropic_request, "responses": logparse.responses_request}.get(api)
    if parser is None:
        return None
    messages, params = parser(request)
    params.pop("stream", None)
    return {"messages": messages, "params": params} if messages else None


def _answer(api: str, response: Any) -> dict | None:
    from . import logparse
    from .record import to_dict

    data = to_dict(response)
    if not isinstance(data, dict):
        return None
    parser = {"chat": logparse.chat_response, "messages": logparse.anthropic_response, "responses": logparse.responses_response}.get(api)
    if parser is None:
        return None
    out = parser(data)
    return {"output": out.get("output") or "", "tool_calls": out.get("tool_calls") or [], "model": out.get("model"), "usage": out.get("usage") or {}}


class Shadow:
    """Copies sampled requests to `model` in the background; never raises into the app."""

    def __init__(
        self,
        model: Any,
        *,
        rate: float = 0.05,
        path: str | os.PathLike = DEFAULT_PATH,
        config: dict | None = None,
        max_workers: int = 2,
        max_pending: int = 100,
        redact_secrets: bool = True,
        seed: int | None = None,
    ):
        rate = float(rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"shadow rate must be between 0 and 1, got {rate}")
        self.model = model
        self.rate = rate
        self.path = Path(path)
        self.config = config
        self.max_pending = max(1, int(max_pending))
        self.stats = {"seen": 0, "sent": 0, "written": 0, "dropped": 0, "errors": 0}
        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._pending = 0
        self._provider: Any = None
        self._provider_error: str | None = None
        self._warned = False
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="llmshift-shadow")
        from .record import _path_lock
        from .redact import Redactor

        self._file_lock = _path_lock(self.path)
        self._redactor = Redactor(secrets=True, pii=False) if redact_secrets else None
        self._closed = False
        atexit.register(self.close)

    @property
    def spec(self) -> str:
        return self.model if isinstance(self.model, str) else getattr(self.model, "spec", repr(self.model))

    def observe(self, api: str, request: dict, response: Any) -> None:
        """Called by the recording proxy after a real call returned."""
        try:
            with self._lock:
                self.stats["seen"] += 1
                if self._closed or not (self.rate >= 1.0 or self._rng.random() < self.rate):
                    return
                if self._pending >= self.max_pending:
                    self.stats["dropped"] += 1
                    return
                self._pending += 1
            canonical, answer = _canonical(api, request), _answer(api, response)
            if canonical is None or answer is None:
                self._done(None)
                return
            self._pool.submit(self._run, api, canonical, answer)
        except Exception as exc:  # shadowing must never break the app
            self._done(exc)

    def _get_provider(self) -> Any:
        if self._provider is None and self._provider_error is None:
            from .providers import resolve

            try:
                self._provider = resolve(self.model, self.config or {}) if isinstance(self.model, str) else self.model
            except Exception as exc:
                self._provider_error = str(exc)
        if self._provider is None:
            raise RuntimeError(self._provider_error or "shadow model unavailable")
        return self._provider

    def _run(self, api: str, request: dict, answer: dict) -> None:
        shadow: dict = {"spec": self.spec}
        try:
            start = time.perf_counter()
            resp = self._get_provider().complete(request)
            shadow.update({
                "model": resp.model, "output": resp.text or "", "tool_calls": resp.tool_calls, "usage": resp.usage,
                "latency_ms": resp.latency_ms if resp.latency_ms is not None else round((time.perf_counter() - start) * 1000, 1),
            })
            if resp.error:
                shadow["error"] = resp.error
        except Exception as exc:
            shadow["error"] = f"{exc.__class__.__name__}: {exc}"[:500]
        record = {
            "llmshift_shadow": 1,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
            "api": api,
            "request": request,
            "baseline": answer,
            "shadow": shadow,
        }
        try:
            if self._redactor is not None:
                record = self._redactor.obj(record)
            data = memoryview((json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            with self._file_lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
                try:
                    while data:
                        data = data[os.write(fd, data):]
                finally:
                    os.close(fd)
            with self._lock:
                self.stats["sent"] += 1
                self.stats["written"] += 1
                if "error" in shadow:
                    self.stats["errors"] += 1
            self._done(None)
        except Exception as exc:
            self._done(exc)

    def _done(self, exc: BaseException | None) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if exc is not None:
                self.stats["errors"] += 1
        if exc is not None and not self._warned:
            self._warned = True
            warnings.warn(f"llmshift shadow: {exc}", RuntimeWarning, stacklevel=2)

    def flush(self, timeout: float = 30.0) -> bool:
        """Wait (up to `timeout` seconds) until the copies in flight are written."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if self._pending == 0:
                    return True
            time.sleep(0.02)
        return False

    def close(self, timeout: float = 5.0) -> None:
        """Stop taking copies; give the ones in flight a moment to finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.flush(timeout)
        self._pool.shutdown(wait=False)


# ------------------------------------------------------------------ report


class ReplayProvider:
    """Answers each request with the shadow answer recorded for it."""

    kind = "replay"

    def __init__(self, spec: str, answers: dict[str, list[Response]]):
        self.model = spec
        self.name = "shadow"
        self._answers = answers
        self._lock = threading.Lock()
        self.adaptations: list[str] = []

    @property
    def spec(self) -> str:
        return self.model

    def identity(self) -> dict:
        return {"kind": "replay", "model": self.model}

    def complete(self, request: dict) -> Response:
        key = stable_hash({"messages": request.get("messages"), "params": request.get("params") or {}}, 24)
        with self._lock:
            queue = self._answers.get(key) or []
            return queue.pop(0) if queue else Response(error="no shadow answer recorded for this request", error_code="missing")


def load_shadow_logs(paths: list[str]) -> list[dict]:
    from .logparse import iter_files

    records = []
    for path in iter_files(paths):
        for n, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("llmshift_shadow") and isinstance(rec.get("request"), dict):
                rec["_where"] = f"{path.as_posix()}:{n}"
                records.append(rec)
    return records


def build_cases(records: list[dict]) -> tuple[list, dict[str, list[Response]], str]:
    """Test cases (real answers as baselines, checks inferred as in snapshot) and the
    shadow answers to replay; plus the shadow model's name."""
    from . import messages as msgs
    from .infer import group_stats, infer_checks
    from .testset import TestCase, case_id_for

    groups: dict[str, list[dict]] = {}
    for rec in records:
        req = rec["request"]
        groups.setdefault(msgs.template_key(req.get("messages") or [], req.get("params") or {}), []).append(rec)
    stats = {g: group_stats([(r.get("baseline") or {}).get("output") or "" for r in recs]) for g, recs in groups.items()}
    cases, answers, specs, seen = [], {}, set(), {}
    for group, recs in groups.items():
        for rec in recs:
            req = {"messages": rec["request"].get("messages") or [], "params": rec["request"].get("params") or {}}
            base = rec.get("baseline") or {}
            shadow = rec.get("shadow") or {}
            specs.add(shadow.get("spec") or shadow.get("model") or "shadow")
            checks = infer_checks(req["messages"], req["params"], base.get("output") or "", base.get("tool_calls") or [], stats.get(group))
            cid = case_id_for(req)
            seen[cid] = seen.get(cid, 0) + 1
            if seen[cid] > 1:
                cid = f"{cid}-{seen[cid]}"
            baseline = {"output": base.get("output") or ""}
            for key in ("tool_calls", "model", "usage"):
                if base.get(key):
                    baseline[key] = base[key]
            cases.append(TestCase(id=cid, input=req, baseline=baseline, checks=checks, group=group, metadata={"source": rec.get("_where"), "timestamp": rec.get("ts")}))
            resp = Response(
                text=shadow.get("output") or "", tool_calls=shadow.get("tool_calls") or [], model=shadow.get("model"),
                usage=shadow.get("usage") or {}, latency_ms=shadow.get("latency_ms"),
                error=shadow.get("error"), error_code="shadow_error" if shadow.get("error") else None,
            )
            answers.setdefault(stable_hash(req, 24), []).append(resp)
    return cases, answers, ", ".join(sorted(specs)) or "shadow"


def cmd_shadow_report(args: Any, cfg: dict) -> int:
    from . import cli, report
    from .compare import run_compare

    paths = args.logs or [DEFAULT_PATH]
    try:
        records = load_shadow_logs(paths)
    except FileNotFoundError as exc:
        raise UsageError(str(exc)) from None
    if not records:
        raise UsageError(f"no shadow records in {', '.join(paths)}; wrap your client with llmshift.record.wrap(..., shadow=...) first")
    cases, answers, spec = build_cases(records)
    min_rate = args.min_pass_rate if args.min_pass_rate is not None else cfg.get("min_pass_rate")
    result = run_compare(
        cases, ReplayProvider(spec, answers), config=cfg, concurrency=args.concurrency or int(cfg.get("concurrency", 4)),
        cache=None, min_pass_rate=float(min_rate) if min_rate is not None else None, show_progress=False, label="shadow",
        judge=args.judge or cfg.get("judge"),
    )
    result.baseline = "live traffic"
    times = sorted(str(r.get("ts")) for r in records if r.get("ts"))
    if times:
        result.insights.insert(0, f"{len(records)} shadowed request(s) from {times[0][:16].replace('T', ' ')} to {times[-1][:16].replace('T', ' ')} UTC.")
    if args.format == "json":
        con.out(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        con.out(report.compare_markdown(result, title="LLMShift shadow report"))
    else:
        con.out(report.compare_terminal(result, show=cli._show(args.show), verbose=args.verbose))
    if args.html is not None:
        path = args.html or Path(cfg.get("reports_dir") or ".llmshift/reports") / f"shadow-{cli._timestamp()}.html"
        cli._write(path, report.compare_html(result))
        con.out(f"wrote {cli._rel(path)}", err=args.format == "json")
    cli._step_summary(report.compare_markdown(result, title="LLMShift shadow report"))
    if result.summary.get("evaluated", 0) == 0 or (result.min_pass_rate is not None and not result.passed):
        return EXIT_FAIL
    return EXIT_OK


def register(sub: Any) -> None:
    p = sub.add_parser("shadow", help="report on live traffic shadowed to a new model (see llmshift.record.wrap)")
    ssub = p.add_subparsers(dest="shadow_command", metavar="<report>")
    ssub.required = True
    rp = ssub.add_parser("report", help="check the shadow answers against the real ones")
    rp.add_argument("logs", nargs="*", help=f"shadow logs (default {DEFAULT_PATH})")
    rp.add_argument("--min-pass-rate", type=float, help="exit 1 below this pass rate (default from config)")
    rp.add_argument("--judge", metavar="MODEL", help="LLM judge for meaning checks")
    rp.add_argument("--format", choices=["table", "json", "markdown"], default="table")
    rp.add_argument("--html", nargs="?", const="", metavar="FILE", help="also write an HTML report")
    rp.add_argument("--show", help="how many changed cases to print (number or 'all')")
    rp.add_argument("--verbose", "-v", action="store_true")
    rp.add_argument("--concurrency", type=int)
    rp.set_defaults(func=cmd_shadow_report)
