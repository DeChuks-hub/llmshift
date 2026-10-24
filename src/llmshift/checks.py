"""Checks: assertions evaluated on every model output.

A check is a dict with a `type` and parameters, stored in the test file so it
is visible and editable. Each result carries a category (format, facts, tone,
semantics, tools, safety, custom), a severity (error = the case fails,
warn = the case is flagged as changed) and a human-readable message that says
exactly what changed.
"""

from __future__ import annotations

import json
import math
import re
import string
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from . import facts as factlib
from . import tone as tonelib
from . import userregex
from .formats import (
    STRUCTURE_NAMES,
    analyze_json,
    is_refusal,
    parse_json_lenient,
    structure_signature,
)
from .jsonschema_lite import describe as describe_schema
from .jsonschema_lite import get_path, validate
from .templating import render, render_text
from .testset import TestCase
from .textutil import (
    detect_language,
    json_loads,
    language_name,
    normalize,
    rouge_l,
    shorten,
    similarity,
    tokenize,
    vector_cosine,
    word_count,
)

DEFAULT_SEVERITY = {
    "agent": "error",
    "not_empty": "error",
    "json_valid": "error",
    "json_schema": "error",
    "json_match": "error",
    "json_equals": "error",
    "tool_calls": "error",
    "equals": "error",
    "not_equals": "error",
    "contains": "error",
    "contains_any": "error",
    "not_contains": "error",
    "starts_with": "error",
    "regex": "error",
    "one_of": "error",
    "fuzzy_match": "error",
    "facts": "error",
    "language": "error",
    "refusal": "error",
    "length": "warn",
    "structure": "warn",
    "tone": "warn",
    "similarity": "warn",
    "llm_judge": "error",
    "llm_label": "error",
    "llm_score": "error",
    "modelgraded": "error",
    "python": "error",
    "multi": "error",
}

CATEGORY = {
    "not_empty": "format",
    "json_valid": "format",
    "json_schema": "format",
    "structure": "format",
    "regex": "format",
    "one_of": "format",
    "starts_with": "facts",
    "language": "format",
    "json_match": "facts",
    "json_equals": "facts",
    "equals": "facts",
    "not_equals": "facts",
    "contains": "facts",
    "contains_any": "facts",
    "not_contains": "facts",
    "fuzzy_match": "facts",
    "facts": "facts",
    "tool_calls": "tools",
    "agent": "tools",
    "tone": "tone",
    "length": "tone",
    "similarity": "semantics",
    "llm_judge": "semantics",
    "llm_label": "custom",
    "llm_score": "custom",
    "modelgraded": "custom",
    "python": "custom",
    "multi": "custom",
    "refusal": "safety",
    "request": "request",  # added by compare: the new model rejects a setting the request sends
}

CATEGORY_TITLES = {
    "request": "Rejected request settings",
    "format": "Broken format",
    "facts": "Wrong or missing facts",
    "tone": "Different tone / style",
    "semantics": "Different meaning",
    "tools": "Tool-call changes",
    "safety": "Refusals",
    "custom": "Imported graders",
    "errors": "API errors",
    "unchecked": "Could not check",
}


@dataclass
class Sample:
    """A model output as the checks see it."""

    text: str = ""
    tool_calls: list = field(default_factory=list)
    model: str | None = None
    refusal: str | None = None
    finish_reason: str | None = None
    steps: list = field(default_factory=list)
    source: str | None = None  # who answered (in compare, the side and its provider): each has its own regex breaker

    @classmethod
    def from_response(cls, resp: Any, source: str | None = None) -> "Sample":
        return cls(
            text=getattr(resp, "text", "") or "",
            tool_calls=list(getattr(resp, "tool_calls", []) or []),
            model=getattr(resp, "model", None),
            refusal=getattr(resp, "refusal", None),
            finish_reason=getattr(resp, "finish_reason", None),
            steps=list(getattr(resp, "steps", []) or []),
            source=source,
        )

    @property
    def json(self) -> Any:
        if not hasattr(self, "_json_cache"):
            self._json_cache = parse_json_lenient(self.text)
        return self._json_cache

    def namespace(self) -> dict:
        tools = [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": tc.get("arguments") if isinstance(tc.get("arguments"), str) else json.dumps(tc.get("arguments"), ensure_ascii=False),
                },
            }
            for tc in self.tool_calls
        ]
        message = {"role": "assistant", "content": self.text}
        if tools:
            message["tool_calls"] = tools
        return {
            "output_text": self.text,
            "output_json": self.json,
            "output_tools": tools,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason or "stop"}],
            "model": self.model,
        }


@dataclass
class CheckContext:
    judge: Any = None  # Provider used for LLM-judged checks
    embedder: Any = None  # Provider used for cosine similarity
    allow_python: bool = False
    severity: dict = field(default_factory=dict)
    resolve_model: Callable[[str], Any] | None = None  # spec -> Provider (for graders naming their own model)
    judge_cache: Any = None
    judge_calls: int = 0
    judge_mode: str = "auto"  # auto: ask the judge only when rule-based checks cannot decide
    judge_skipped: int = 0
    judge_failed: int = 0  # judge calls that got no answer (a wrong key, an unknown model...)
    judge_error: str | None = None  # the first of those errors
    # Patterns stopped after userregex.TIMEOUT, per source of answers (compare's candidate and
    # live baseline apart, even when both report the same model name, or none): a pattern given
    # up on in one run on one source's answers is tried again for the other and in another run.
    regex_timeouts: dict = field(default_factory=dict)

    def regex_breaker(self, sample: "Sample") -> dict:
        return self.regex_timeouts.setdefault(sample.source or sample.model or "", {})


@dataclass
class CheckResult:
    type: str
    category: str
    severity: str
    status: str  # pass | fail | skip | error
    message: str = ""
    score: float | None = None
    name: str | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type, "category": self.category, "severity": self.severity, "status": self.status, "message": self.message}
        if self.name:
            d["name"] = self.name
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.details:
            d["details"] = self.details
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        return cls(
            type=d.get("type", "?"),
            category=d.get("category", "custom"),
            severity=d.get("severity", "error"),
            status=d.get("status", "skip"),
            message=d.get("message", ""),
            score=d.get("score"),
            name=d.get("name"),
            details=d.get("details") or {},
        )

    @property
    def failed(self) -> bool:
        return self.status in ("fail", "error")


class Outcome:
    """What a check function returns: status + message (+ score/details)."""

    __slots__ = ("status", "message", "score", "details")

    def __init__(self, status: str, message: str = "", score: float | None = None, details: dict | None = None):
        self.status = status
        self.message = message
        self.score = score
        self.details = details or {}


def PASS(message: str = "", score: float | None = 1.0, **details: Any) -> Outcome:
    return Outcome("pass", message, score, details)


def FAIL(message: str, score: float | None = 0.0, **details: Any) -> Outcome:
    return Outcome("fail", message, score, details)


def SKIP(message: str, reason: str | None = None) -> Outcome:
    return Outcome("skip", message, None, {"skip_reason": reason} if reason else None)


def _fmt(value: Any, limit: int = 60) -> str:
    if isinstance(value, str):
        return '"' + shorten(value, limit) + '"'
    return shorten(json.dumps(value, ensure_ascii=False), limit)


def _target(check: dict, sample: Sample, case: TestCase) -> str:
    target = check.get("target")
    if not target:
        return sample.text
    return render_text(target, {"item": case.vars, "sample": sample.namespace()})


def _as_text(v: Any) -> str:
    """How a JSON value is written in model output: true/false/null, not Python's True/None."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False)


_MISSING = object()


def _values(check: dict, case: TestCase, key: str = "value") -> list[str]:
    raw = check.get(key, _MISSING)
    if raw is _MISSING and key == "value":
        raw = check.get("values", _MISSING)
    if raw is _MISSING:
        return []
    raw = render(raw, {"item": case.vars})
    if isinstance(raw, list):
        return [_as_text(v) for v in raw]
    return [_as_text(raw)]


# --------------------------------------------------------------- format


def c_not_empty(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    if sample.text.strip() or sample.tool_calls:
        return PASS()
    why = f" (finish_reason={sample.finish_reason})" if sample.finish_reason and sample.finish_reason != "stop" else ""
    if sample.refusal:
        why = f" (refused: {shorten(sample.refusal, 80)})"
    return FAIL("empty response" + why)


def _nan_ok(case: TestCase) -> bool:
    """Whether this case's own baseline is JSON only with NaN or Infinity in it: test files
    from 0.1.0 inferred JSON checks from such answers, and the same answer must still pass."""
    base = case.baseline_output or ""
    if "NaN" not in base and "Infinity" not in base:
        return False
    return parse_json_lenient(base) is None and parse_json_lenient(base, allow_nan=True) is not None


def c_json_valid(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    text = _target(check, sample, case)
    info = analyze_json(text, allow_nan=_nan_ok(case))
    if info["valid"]:
        if isinstance(info["value"], (dict, list)) or check.get("allow_scalar"):
            return PASS()
        return FAIL(f"valid JSON but a bare {type(info['value']).__name__}, not an object/array: {_fmt(text.strip())}")
    if info["fenced"] and check.get("allow_fences"):
        return PASS("JSON inside a code fence (allowed)")
    return FAIL(info["error"] or "not valid JSON", fenced=info["fenced"], embedded=info["embedded"])


def c_json_schema(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    schema = check.get("schema") or {}
    nan_ok = _nan_ok(case)
    value = parse_json_lenient(_target(check, sample, case), allow_nan=nan_ok)
    if value is None:
        return FAIL(f"no JSON found; expected {describe_schema(schema)}")
    errors = validate(value, schema)
    if nan_ok:  # the baseline's own NaN or Infinity is no failure
        errors = [e for e in errors if not e.endswith("is not a JSON number")]
    if errors:
        extra = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        return FAIL("; ".join(errors[:3]) + extra, score=0.0, errors=errors[:20])
    details = {}
    props = schema.get("properties")
    if isinstance(value, dict) and isinstance(props, dict) and props:
        extra_keys = sorted(set(value) - set(props))
        if extra_keys:
            details["extra_keys"] = extra_keys
            return PASS(f"matches schema (new keys: {', '.join(extra_keys[:5])})", **details)
    return PASS()


def c_structure(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    expect = check.get("expect") or {}
    sig = structure_signature(_target(check, sample, case))
    diffs = []
    for feature, want in expect.items():
        if feature not in sig:
            continue
        if bool(sig[feature]) != bool(want):
            name = STRUCTURE_NAMES.get(feature, feature)
            diffs.append(("no longer uses " if want else "adds ") + name)
    if diffs:
        return FAIL("; ".join(diffs), score=1 - len(diffs) / max(1, len(expect)), changes=diffs)
    return PASS()


def c_language(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    expected = check.get("expected")
    code, conf = detect_language(_target(check, sample, case))
    if not expected:
        return SKIP("no expected language")
    if code is None:
        return SKIP("language could not be determined")
    if code != expected:
        return FAIL(f"answers in {language_name(code)} instead of {language_name(expected)}")
    return PASS()


def c_regex(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    pattern = check.get("pattern") or ""
    flags = 0
    for ch in str(check.get("flags", "")):
        flags |= {"i": re.I, "m": re.M, "s": re.S, "x": re.X}.get(ch, 0)
    try:
        ok = userregex.search(pattern, _target(check, sample, case), flags)
    except re.error as exc:
        return Outcome("error", f"invalid regex {pattern!r}: {exc}")
    except userregex.RegexStopped as exc:
        return Outcome("error", userregex.stopped_message(pattern, exc))
    if check.get("negate"):
        return FAIL(f"output matches forbidden pattern /{pattern}/") if ok else PASS()
    return PASS() if ok else FAIL(f"output does not match /{shorten(pattern, 60)}/")


def c_one_of(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    values = _values(check, case, "values")
    text = _target(check, sample, case).strip()
    cs = check.get("case_sensitive", True)
    pool = values if cs else [v.lower() for v in values]
    probe = text if cs else text.lower()
    if probe in pool:
        return PASS()
    return FAIL(f"{_fmt(text)} is not one of the expected labels {values[:12]}")


# ---------------------------------------------------------------- facts


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _equal(check: dict, sample: Sample, case: TestCase) -> tuple[str, list[str], bool]:
    """The answer, the values, and whether the answer equals one of them: stripped of spaces
    and newlines unless strip is false, as numbers with numeric, in any case with
    case_sensitive false. equals and not_equals both compare this way."""
    options = _values(check, case)
    text = _target(check, sample, case)
    if check.get("strip", True):
        text = text.strip()
        options = [o.strip() for o in options]
    if check.get("numeric"):
        try:
            got = float(re.sub(r"[,$€£%\s]", "", text))
            tolerance = float(check.get("tolerance", 1e-9))
            return text, options, any(abs(got - float(re.sub(r"[,$€£%\s]", "", o))) <= tolerance for o in options)
        except ValueError:
            return text, options, False
    cs = check.get("case_sensitive", True)
    probe = text if cs else text.lower()
    return text, options, any(probe == (o if cs else o.lower()) for o in options)


def c_equals(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    text, options, equal = _equal(check, sample, case)
    if equal:
        return PASS()
    if check.get("numeric"):
        return FAIL(f"expected {_fmt(options[0] if options else '')}, got {_fmt(text)}")
    hint = ""
    if any(tokenize(text) == tokenize(o) for o in options):
        hint = " (differs only in case/punctuation/spacing)"
    shown = options[0] if len(options) == 1 else options[:5]
    return FAIL(f"expected {_fmt(shown)}, got {_fmt(text)}{hint}")


def c_not_equals(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    """The opposite of equals with the same settings: "N/A\\n" equals the forbidden "N/A"."""
    text, _, equal = _equal(check, sample, case)
    if equal:
        return FAIL(f"output equals the forbidden value {_fmt(text)}")
    return PASS()


def c_contains(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    values = _values(check, case)
    text = _target(check, sample, case)
    cs = check.get("case_sensitive", True)
    hay = text if cs else text.lower()
    missing = [v for v in values if (v if cs else v.lower()) not in hay]
    if missing:
        return FAIL("missing " + ", ".join(_fmt(m, 40) for m in missing[:5]), score=1 - len(missing) / max(1, len(values)))
    return PASS()


def c_contains_any(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    values = _values(check, case)
    text = _target(check, sample, case)
    cs = not check.get("ignore_case", False) and check.get("case_sensitive", True)
    hay = text if cs else text.lower()
    if any((v if cs else v.lower()) in hay for v in values):
        return PASS()
    return FAIL("contains none of " + ", ".join(_fmt(v, 40) for v in values[:5]))


def c_not_contains(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    values = _values(check, case)
    text = _target(check, sample, case)
    cs = check.get("case_sensitive", False)
    hay = text if cs else text.lower()
    found = [v for v in values if (v if cs else v.lower()) in hay]
    if found:
        return FAIL("contains forbidden text " + ", ".join(_fmt(v, 40) for v in found[:5]))
    return PASS()


def c_starts_with(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    values = _values(check, case)
    text = _target(check, sample, case)
    if check.get("lstrip", True):
        text = text.lstrip()
    if any(text.startswith(v) for v in values):
        return PASS()
    return FAIL(f"does not start with {_fmt(values[0] if len(values) == 1 else values[:5])}; got {_fmt(text, 50)}")


def _oss_normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def c_fuzzy_match(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    """openai/evals FuzzyMatch semantics: normalised containment either way."""
    values = _values(check, case)
    got = _oss_normalize(_target(check, sample, case))
    for v in values:
        exp = _oss_normalize(v)
        if (got == "" or exp == "") and got == exp:
            return PASS()
        if got and exp and (got in exp or exp in got):
            return PASS()
    return FAIL(f"no fuzzy match with {_fmt(values[0] if len(values) == 1 else values[:5])}")


def _oss_json_equal(sampled: Any, correct: Any) -> bool:
    if sampled is None or correct is None:
        return False
    if isinstance(sampled, dict):
        if not isinstance(correct, dict):
            return False
        return all(_oss_json_equal(sampled.get(k), correct.get(k)) for k in set(sampled) | set(correct))
    if isinstance(sampled, list):
        if not isinstance(correct, list) or len(sampled) != len(correct):
            return False
        return all(_oss_json_equal(a, b) for a, b in zip(sampled, correct))
    return sampled == correct


def c_json_equals(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    """openai/evals JsonMatch semantics: the output must be exactly one of the expected JSON values."""
    raw = check.get("value", check.get("values"))
    expected = raw if isinstance(raw, list) and check.get("values") is not None else [raw]
    parsed_expected = []
    for e in expected:
        if isinstance(e, str):
            try:
                e = json_loads(e)
            except (ValueError, RecursionError):
                pass
        parsed_expected.append(e)
    text = _target(check, sample, case)
    try:
        got = json_loads(text) if not check.get("lenient") else parse_json_lenient(text)
    except (ValueError, RecursionError):
        return FAIL(f"not valid JSON: {_fmt(text, 50)}")
    if any(_oss_json_equal(got, e) for e in parsed_expected):
        return PASS()
    return FAIL(f"JSON differs: expected {_fmt(parsed_expected[0], 80)}, got {_fmt(got, 80)}")


def _norm_value(v: Any, case_sensitive: bool) -> str:
    """Canonical text of a JSON value for order-insensitive comparison (1 == 1.0, true != 1)."""
    if isinstance(v, bool) or v is None:
        return json.dumps(v)
    if isinstance(v, (int, float)):
        return "n:" + repr(round(float(v), 9))
    if isinstance(v, str):
        s = _norm_ws(v.strip())
        return "s:" + (s if case_sensitive else s.lower())
    if isinstance(v, list):
        return "l:[" + ",".join(sorted(_norm_value(x, case_sensitive) for x in v)) + "]"
    if isinstance(v, dict):
        return "d:{" + ",".join(f"{k}={_norm_value(v[k], case_sensitive)}" for k in sorted(v)) + "}"
    return "o:" + str(v)


def _scalar_equal(expected: Any, actual: Any, case_sensitive: bool = True, tol: float = 1e-9) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        e, a = float(expected), float(actual)
        if not (math.isfinite(e) and math.isfinite(a)):  # NaN equals NaN here; infinity equals itself
            return (math.isnan(e) and math.isnan(a)) or e == a
        return abs(e - a) <= tol * max(1.0, abs(e))
    if isinstance(expected, str) and isinstance(actual, str):
        a, b = _norm_ws(expected.strip()), _norm_ws(actual.strip())
        return a == b if case_sensitive else a.lower() == b.lower()
    if isinstance(expected, list) and isinstance(actual, list):
        return sorted(_norm_value(v, case_sensitive) for v in expected) == sorted(_norm_value(v, case_sensitive) for v in actual)
    return expected == actual


def c_json_match(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    fields = check.get("fields") or {}
    if not fields:
        return SKIP("no fields to compare")
    value = parse_json_lenient(_target(check, sample, case), allow_nan=_nan_ok(case))
    if value is None:
        return SKIP("no JSON output to compare (see the format checks)")
    cs = check.get("case_sensitive", True)
    mismatches, missing = [], []
    for path, expected in fields.items():
        found, actual = get_path(value, path)
        if not found:
            missing.append(path)
        elif not _scalar_equal(expected, actual, cs):
            mismatches.append({"path": path, "expected": expected, "actual": actual})
    bad = len(mismatches) + len(missing)
    if not bad:
        return PASS()
    parts = [f"{m['path']}: {_fmt(m['expected'], 30)} -> {_fmt(m['actual'], 30)}" for m in mismatches[:4]]
    parts += [f"{p}: missing" for p in missing[:3]]
    return FAIL("; ".join(parts), score=1 - bad / len(fields), mismatches=mismatches[:20], missing=missing[:20])


def c_facts(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    facts = check.get("facts") or []
    if not facts:
        return SKIP("no facts recorded")
    result = factlib.match(facts, _target(check, sample, case))
    min_hard = float(check.get("min_hard_recall", 1.0))
    min_ent = float(check.get("min_entity_recall", 0.5))
    changed = result["changed"]
    changed_from = {c["from"] for c in changed}
    missing = [m for m in result["missing"] if m.get("text") not in changed_from]
    hard_missing = [m for m in missing if m.get("kind") != "entity"]
    ent_missing = [m for m in missing if m.get("kind") == "entity"]
    ok = not changed and result["hard_recall"] >= min_hard - 1e-9 and result["entity_recall"] >= min_ent - 1e-9
    score = (result["hard_recall"] * 2 + result["entity_recall"]) / 3
    parts = [f"{_fmt(c['from'], 30)} -> {_fmt(c['to'], 30)}" for c in changed[:4]]
    if hard_missing:
        parts.append("missing " + ", ".join(_fmt(m["text"], 30) for m in hard_missing[:5]))
    if ent_missing and (not ok or len(ent_missing) <= 3):
        parts.append("no mention of " + ", ".join(_fmt(m["text"], 30) for m in ent_missing[:4]))
    details = {
        "changed": changed,
        "missing": [m.get("text") for m in missing][:20],
        "missing_entities": [m.get("text") for m in ent_missing][:20],
        "hard_recall": round(result["hard_recall"], 3),
        "entity_recall": round(result["entity_recall"], 3),
    }
    if ok:
        return PASS("; ".join(parts) if parts else "", score=score, **details)
    if not changed and result["hard_recall"] >= min_hard - 1e-9:
        # Only names went missing: a visible change, but not a wrong number/date/ID.
        details["soft"] = True
    return FAIL("; ".join(parts) or "facts differ", score=score, **details)


# ------------------------------------------------------------ tone/style


def c_length(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    n = word_count(_target(check, sample, case))
    lo, hi = check.get("min_words"), check.get("max_words")
    base = check.get("baseline_words")
    ref = f" (baseline: {base} words)" if base else ""
    if lo is not None and n < lo:
        return FAIL(f"much shorter: {n} words{ref}", score=n / max(1, lo), words=n)
    if hi is not None and n > hi:
        return FAIL(f"much longer: {n} words{ref}", score=hi / max(1, n), words=n)
    return PASS(words=n)


def c_tone(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    text = _target(check, sample, case)
    if word_count(text) < tonelib.MIN_WORDS:
        return SKIP("too short to judge tone")
    base = check.get("profile") or (tonelib.profile(case.baseline_output) if case.baseline_output else None)
    if not base:
        return SKIP("no baseline tone")
    score, diffs = tonelib.compare(base, tonelib.profile(text), float(check.get("tolerance", 1.0)))
    if diffs:
        return FAIL("; ".join(d["text"] for d in diffs), score=score, differences=diffs)
    return PASS(score=score)


def c_similarity(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    reference = check.get("reference")
    reference = render_text(reference, {"item": case.vars}) if reference is not None else case.baseline_output
    if not reference:
        return SKIP("no reference text")
    text = _target(check, sample, case)
    metric = check.get("metric", "rouge_l")
    threshold = float(check.get("threshold", check.get("pass_threshold", 0.5)))
    embedder = ctx.embedder
    if metric == "cosine" and embedder is None and check.get("embedding_model") and ctx.resolve_model is not None:
        embedder = ctx.resolve_model(check["embedding_model"])
    score: float | None = None
    metric_name = metric
    if metric == "cosine" and embedder is not None:
        try:
            vecs = embedder.embed([text or " ", reference])
            score = vector_cosine(vecs[0], vecs[1]) if len(vecs) == 2 else 0.0
            metric_name = "embedding cosine"
        except Exception as exc:
            metric_name = f"bag-of-words cosine (embeddings unavailable: {shorten(str(exc), 60)})"
    elif metric == "cosine":
        metric_name = "bag-of-words cosine (no --embedder set)"
    if score is None:
        score = similarity(text, reference, metric)
    if score >= threshold:
        return PASS(score=score)
    return FAIL(f"{metric_name} similarity {score:.3f} is below {threshold:g}", score=score, metric=metric)


def c_refusal(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    text = _target(check, sample, case)
    if sample.refusal:
        return FAIL(f"the model refused ({shorten(sample.refusal, 80)})")
    if is_refusal(text):
        return FAIL(f"the model declined: {_fmt(text, 90)}")
    return PASS()


# ----------------------------------------------------------------- tools


def _short_scalar(v: Any) -> bool:
    if isinstance(v, bool) or v is None or isinstance(v, int):
        return True
    if isinstance(v, float):
        return True
    return isinstance(v, str) and len(v) <= 40 and len(v.split()) <= 4


def c_tool_calls(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    expected = check.get("expected") or []
    actual = sample.tool_calls or []
    exp_names = Counter(c.get("name") for c in expected)
    act_names = Counter(c.get("name") for c in actual)
    if exp_names != act_names:
        if not actual:
            msg = f"no tool call; expected {', '.join(sorted(exp_names))}"
            if sample.text.strip():
                msg += f" (answered in text: {_fmt(sample.text, 50)})"
            return FAIL(msg)
        missing = list((exp_names - act_names).elements())
        extra = list((act_names - exp_names).elements())
        bits = []
        if missing:
            bits.append("missing call(s) to " + ", ".join(missing))
        if extra:
            bits.append("unexpected call(s) to " + ", ".join(extra))
        return FAIL("; ".join(bits), score=0.0)
    mode = check.get("match_args", "auto")
    problems: list[str] = []
    for name in sorted(exp_names):
        exps = [c for c in expected if c.get("name") == name]
        acts = [c for c in actual if c.get("name") == name]
        problems.extend(_pair_calls(name, exps, acts, mode))
    if problems:
        return FAIL("; ".join(problems[:4]), score=0.5, problems=problems[:20])
    return PASS()


def _call_problems(name: str, exp: dict, act: dict, mode: str) -> list[str]:
    args = act.get("arguments")
    if isinstance(args, str):
        return [f"{name}: arguments are not valid JSON ({_fmt(args, 40)})"]
    if mode == "none":
        return []
    problems: list[str] = []
    _compare_args(exp.get("arguments") if exp.get("arguments") is not None else {}, args if args is not None else {}, name, mode, problems)
    return problems


def _pair_calls(name: str, exps: list[dict], acts: list[dict], mode: str) -> list[str]:
    """Pair expected and actual calls of one tool by best argument agreement (order-free)."""
    import itertools

    if len(acts) <= 6:
        best: list[str] | None = None
        for perm in itertools.permutations(acts):
            found = [p for e, a in zip(exps, perm) for p in _call_problems(name, e, a, mode)]
            if best is None or len(found) < len(best):
                best = found
                if not best:
                    break
        return best or []
    out: list[str] = []
    pool = list(acts)
    for e in exps:
        if not pool:  # more expected calls than actual ones: the missing ones are reported apart
            break
        scored = sorted(range(len(pool)), key=lambda i: len(_call_problems(name, e, pool[i], mode)))
        out.extend(_call_problems(name, e, pool.pop(scored[0]), mode))
    return out


def _compare_args(exp: Any, act: Any, path: str, mode: str, problems: list[str]) -> None:
    if isinstance(exp, dict):
        if not isinstance(act, dict):
            problems.append(f"{path}: expected an object, got {_fmt(act, 30)}")
            return
        for key, value in exp.items():
            sub = f"{path}.{key}"
            if key not in act:
                problems.append(f"{sub}: missing argument")
            else:
                _compare_args(value, act[key], sub, mode, problems)
        return
    if mode == "keys":
        return
    if isinstance(exp, list):
        if not isinstance(act, list):
            problems.append(f"{path}: {_fmt(exp, 30)} -> {_fmt(act, 30)}")
        elif all(_short_scalar(v) for v in exp):
            if not _scalar_equal(exp, act, case_sensitive=False):
                problems.append(f"{path}: {_fmt(exp, 30)} -> {_fmt(act, 30)}")
        elif len(exp) != len(act):
            problems.append(f"{path}: {len(exp)} items -> {len(act)}")
        else:
            for i, (e, a) in enumerate(zip(exp, act)):
                _compare_args(e, a, f"{path}[{i}]", mode, problems)
        return
    if mode == "values" or _short_scalar(exp):
        if not _scalar_equal(exp, act, case_sensitive=False):
            problems.append(f"{path}: {_fmt(exp, 30)} -> {_fmt(act, 30)}")


def _agent_check(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> Outcome:
    from .agents import c_agent

    return c_agent(check, sample, case, ctx)


# ------------------------------------------------------------- dispatch

CHECKS: dict[str, Callable[[dict, Sample, TestCase, CheckContext], Outcome]] = {
    "agent": lambda check, sample, case, ctx: _agent_check(check, sample, case, ctx),
    "not_empty": c_not_empty,
    "json_valid": c_json_valid,
    "json_schema": c_json_schema,
    "json_match": c_json_match,
    "json_equals": c_json_equals,
    "structure": c_structure,
    "language": c_language,
    "regex": c_regex,
    "one_of": c_one_of,
    "equals": c_equals,
    "not_equals": c_not_equals,
    "contains": c_contains,
    "contains_any": c_contains_any,
    "includes": c_contains_any,
    "not_contains": c_not_contains,
    "starts_with": c_starts_with,
    "fuzzy_match": c_fuzzy_match,
    "facts": c_facts,
    "length": c_length,
    "tone": c_tone,
    "similarity": c_similarity,
    "refusal": c_refusal,
    "tool_calls": c_tool_calls,
}


def check_types() -> set[str]:
    """Every check type a test file or a severity setting can name."""
    if "llm_judge" not in CHECKS:
        _register_graders()
    return set(CHECKS) | set(CATEGORY)


def _register_graders() -> None:
    from . import graders

    CHECKS.update(
        {
            "llm_judge": graders.c_llm_judge,
            "llm_label": graders.c_llm_label,
            "llm_score": graders.c_llm_score,
            "modelgraded": graders.c_modelgraded,
            "python": graders.c_python,
            "multi": graders.c_multi,
        }
    )


SEVERITY_ALIASES = {
    "error": "error", "fail": "error", "failure": "error", "critical": "error", "high": "error",
    "warn": "warn", "warning": "warn", "changed": "warn", "medium": "warn",
    "info": "info", "ignore": "info", "off": "info", "low": "info", "none": "info",
}


def normalize_severity(value: Any) -> str | None:
    if value is False:  # YAML reads `off` and `no` as false
        return "info"
    if value is True or value in (None, ""):
        return None
    return SEVERITY_ALIASES.get(str(value).strip().lower())


def severity_of(check: dict, ctx: CheckContext) -> str:
    ctype = check.get("type", "")
    for candidate in (check.get("severity"), ctx.severity.get(ctype)):
        sev = normalize_severity(candidate)
        if sev:
            return sev
    return DEFAULT_SEVERITY.get(ctype, "error")


def evaluate(check: dict, sample: Sample, case: TestCase, ctx: CheckContext) -> CheckResult:
    if "llm_judge" not in CHECKS:
        _register_graders()
    ctype = check.get("type", "?")
    fn = CHECKS.get(ctype)
    severity = severity_of(check, ctx)
    category = check.get("category") or CATEGORY.get(ctype, "custom")
    if fn is None:
        return CheckResult(ctype, category, severity, "error", f"unknown check type {ctype!r}", name=check.get("name"))
    try:
        with userregex.breaker(ctx.regex_breaker(sample)):  # this run's, for this source's answers
            outcome = fn(check, sample, case, ctx)
    except Exception as exc:  # a broken check must never abort a run
        from .providers.base import ProviderError

        if isinstance(exc, ProviderError):  # the judge or grader model did not answer
            outcome = Outcome("error", f"the grading model failed: {exc}")
        else:
            outcome = Outcome("error", f"check crashed: {exc.__class__.__name__}: {exc}")
    explicit = bool(check.get("severity")) or ctype in (ctx.severity or {})
    if outcome.details.get("soft") and severity == "error" and not explicit:
        severity = "warn"  # e.g. only a name went missing; an explicit severity always wins
    return CheckResult(ctype, category, severity, outcome.status, outcome.message, outcome.score, check.get("name"), outcome.details)


# The open-ended equivalence judge; imported graders (llm_label, llm_score, modelgraded)
# are criteria in their own right and always run.
GATED_JUDGES = ("llm_judge",)
NEAR_IDENTICAL = 0.9
_SKIPPED_LOCK = threading.Lock()


def _judge_unneeded(check: dict, decided: list[CheckResult], case: TestCase, sample: Sample) -> tuple[str, str] | None:
    """(status, message) when the rule-based results already settle the case, else None."""
    if any(r.status in ("fail", "error") and r.severity == "error" for r in decided):
        return "skip", "not needed: the rule-based checks already failed this case"
    reference = check.get("reference", case.baseline_output)
    text = sample.text or ""
    if not reference or not text.strip():
        return None
    if normalize(text) == normalize(reference):
        return "pass", "identical to the reference answer (no judge call needed)"
    if not any(r.status in ("fail", "error") for r in decided):
        score = rouge_l(text, reference)
        if score >= NEAR_IDENTICAL:
            return "pass", f"near-identical to the reference (similarity {score:.2f}; no judge call needed)"
    return None


def run_checks(case: TestCase, sample: Sample, ctx: CheckContext) -> list[CheckResult]:
    results: list[CheckResult | None] = [None] * len(case.checks)
    gated = []
    for i, check in enumerate(case.checks):
        if check.get("type") in GATED_JUDGES and ctx.judge_mode == "auto" and not check.get("always"):
            gated.append(i)
        else:
            results[i] = evaluate(check, sample, case, ctx)
    for i in gated:
        check = case.checks[i]
        decided = _judge_unneeded(check, [r for r in results if r is not None], case, sample)
        if decided is None:
            results[i] = evaluate(check, sample, case, ctx)
            continue
        with _SKIPPED_LOCK:
            ctx.judge_skipped += 1
        status, message = decided
        details = {"skip_reason": "decided_by_rules"} if status == "skip" else {"judge": "not needed"}
        ctype = check["type"]
        results[i] = CheckResult(ctype, check.get("category") or CATEGORY.get(ctype, "semantics"), severity_of(check, ctx), status, message, None, check.get("name"), details)
    return [r for r in results if r is not None]
