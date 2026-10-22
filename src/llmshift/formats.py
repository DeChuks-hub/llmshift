"""Output format analysis: JSON detection, code fences, markdown structure,
refusals and output kinds."""

from __future__ import annotations

import json
import re
from typing import Any

from .textutil import json_loads, json_too_deep

_FENCE_RE = re.compile(r"^\s*(```|~~~)[ \t]*([\w+#.-]*)[^\n]*\n(.*?)\n?[ \t]*\1[ \t]*\s*$", re.S)


def strip_fences(text: str) -> tuple[str, bool, str]:
    """If the whole text is one fenced code block, return (inner, True, lang)."""
    m = _FENCE_RE.match(text or "")
    if m:
        return m.group(3), True, (m.group(2) or "").lower()
    return text or "", False, ""


def find_embedded_json(text: str, max_attempts: int = 50) -> tuple[Any, int, int] | None:
    """Find the largest JSON object/array embedded in text. Returns (value, start, end).

    The largest span wins, so "As noted in [1], here it is: {...}" yields the object.
    """
    decoder = json.JSONDecoder()
    best: tuple[Any, int, int] | None = None
    attempts = 0
    pos_limit = len(text or "")
    for m in re.finditer(r"[\[{]", text or ""):
        if best is not None and m.start() < best[2]:
            continue  # inside the value already found
        attempts += 1
        if attempts > max_attempts:
            break
        if json_too_deep(text, m.start()):
            continue
        try:
            value, end = decoder.raw_decode(text, m.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(value, (dict, list)) and (value or end - m.start() > 2):
            if best is None or (end - m.start()) > (best[2] - best[1]):
                best = (value, m.start(), end)
            if end >= pos_limit:
                break
    return best


def _loads(text: str) -> Any:
    try:
        return json_loads(text)
    except RecursionError:
        raise ValueError("JSON nesting is too deep") from None


def analyze_json(text: str) -> dict:
    """Describe whether `text` is JSON and how it is wrapped.

    Returns a dict with keys: valid (exact JSON), value (parsed value or None),
    fenced (inside a ``` block), embedded (surrounded by prose), error (str|None),
    prefix/suffix (the extra text around embedded JSON).
    """
    raw = (text or "").strip()
    res: dict = {"valid": False, "value": None, "fenced": False, "embedded": False, "error": None, "prefix": "", "suffix": ""}
    if not raw:
        res["error"] = "empty output"
        return res
    try:
        res["value"] = _loads(raw)
        res["valid"] = True
        return res
    except ValueError as exc:
        first_error = exc
    inner, fenced, lang = strip_fences(raw)
    if fenced:
        try:
            res["value"] = _loads(inner)
            res["fenced"] = True
            label = f"```{lang}" if lang else "```"
            res["error"] = f"JSON is wrapped in a markdown code fence ({label} ... ```)"
            return res
        except ValueError:
            pass
    found = find_embedded_json(raw)
    if found is not None:
        value, start, end = found
        res["value"] = value
        res["embedded"] = True
        res["prefix"] = raw[:start].strip()
        res["suffix"] = raw[end:].strip()
        extra = res["prefix"] or res["suffix"]
        snippet = " ".join(extra.split())[:60]
        res["error"] = f'JSON is surrounded by extra text ("{snippet}")'
        return res
    msg = getattr(first_error, "msg", str(first_error))
    line = getattr(first_error, "lineno", None)
    col = getattr(first_error, "colno", None)
    where = f" at line {line} column {col}" if line is not None else ""
    res["error"] = f"not valid JSON ({msg}{where})"
    return res


def parse_json_lenient(text: str) -> Any:
    """Parse JSON from text, tolerating fences and surrounding prose. None if absent."""
    info = analyze_json(text)
    return info["value"]


# --------------------------------------------------------------- markdown


_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_BULLET_RE = re.compile(r"^[-*+•]\s+\S")
_NUMBERED_RE = re.compile(r"^\d{1,3}[.)]\s+\S")
_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


def md_structure(text: str) -> dict:
    """Count markdown structural elements outside code blocks."""
    st = {"headings": 0, "bullets": 0, "numbered": 0, "table_rows": 0, "tables": 0, "code_blocks": 0, "bold": 0, "lines": 0, "paragraphs": 0}
    in_code = False
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("```") or s.startswith("~~~"):
            if not in_code:
                st["code_blocks"] += 1
            in_code = not in_code
            continue
        if in_code or not s:
            continue
        st["lines"] += 1
        if _HEADING_RE.match(s):
            st["headings"] += 1
        elif _BULLET_RE.match(s):
            st["bullets"] += 1
        elif _NUMBERED_RE.match(s):
            st["numbered"] += 1
        elif _TABLE_SEP_RE.match(s):
            st["tables"] += 1
        elif s.startswith("|") and s.count("|") >= 2:
            st["table_rows"] += 1
        if re.search(r"\*\*[^*\n]+\*\*|__[^_\n]+__", s):
            st["bold"] += 1
    st["paragraphs"] = len([p for p in re.split(r"\n\s*\n", text or "") if p.strip()])
    return st


def structure_signature(text: str) -> dict:
    """Boolean structural features used by the `structure` check."""
    st = md_structure(text)
    return {
        "headings": st["headings"] > 0,
        "list": (st["bullets"] + st["numbered"]) >= 2,
        "numbered_list": st["numbered"] >= 2,
        "table": st["tables"] > 0,
        "code_block": st["code_blocks"] > 0,
        "bold": st["bold"] > 0,
    }


STRUCTURE_NAMES = {
    "headings": "markdown headings",
    "list": "a bulleted/numbered list",
    "numbered_list": "a numbered list",
    "table": "a table",
    "code_block": "a fenced code block",
    "bold": "bold text",
}


# --------------------------------------------------------------- refusals

# A refusal declines the request itself ("I can't help with that."), not a specific
# thing a policy rules out ("I can't provide a refund after 30 days, but ...").
_GENERIC_OBJECT = (
    r"(?:that|this|it|them|those|these|anything like (?:that|this)|that request|this request|your request|"
    r"your question|this question|that question|the request|such (?:a )?requests?|requests? like (?:this|that)|"
    r"that information|this information|that kind of (?:request|content|thing))"
)
_CANNOT = (
    r"(?:can(?:no|['’])t|cannot|won['’]t(?: be able to)?|will not(?: be able to)?|am unable to|['’]m unable to|"
    r"are unable to|am not able to|['’]m not able to|are not able to|am not allowed to|['’]m not allowed to|"
    r"am not permitted to|['’]m not permitted to)"
)
_REFUSAL_RE = re.compile(
    r"(?i)(?:"
    rf"\b(?:i|we)\s*{_CANNOT}\s+(?:help|assist)(?:\s+you)?(?:\s+with\s+{_GENERIC_OBJECT})?\s*(?:[.!,;]|$)|"
    rf"\b(?:i|we)\s*{_CANNOT}\s+(?:comply|fulfil+|proceed|continue|engage)(?:\s+(?:with|in))?(?:\s+{_GENERIC_OBJECT})?\s*(?:[.!,;]|$)|"
    rf"\b(?:i|we)\s*{_CANNOT}\s+(?:do|provide|share|create|generate|write|produce|answer|discuss|give|support)\s+{_GENERIC_OBJECT}\s*(?:[.!,;]|$)|"
    r"\b(?:i|we)\s+(?:must|have to|need to)\s+(?:respectfully\s+)?(?:decline|refuse)\b|"
    r"\bas an ai(?: language model| assistant)?,?\s+i\s+(?:can(?:no|['’])t|cannot|don['’]t|do not|am not able|am unable)\b"
    r")"
)


def is_refusal(text: str, *, window: int = 220) -> bool:
    """Heuristic: does the response open by declining the request itself?"""
    head = (text or "").strip()[:window]
    return bool(head) and bool(_REFUSAL_RE.search(head))


# ------------------------------------------------------------ output kinds

_NUMBER_ONLY_RE = re.compile(r"^[-+]?[$€£]?\d[\d,]*(\.\d+)?%?$")


def output_kind(text: str) -> str:
    """Classify an output: empty, json, number, label, code, markdown or prose."""
    s = (text or "").strip()
    if not s:
        return "empty"
    info = analyze_json(s)
    if info["value"] is not None and isinstance(info["value"], (dict, list)) and (info["valid"] or info["fenced"]):
        return "json"
    if _NUMBER_ONLY_RE.match(s):
        return "number"
    from .textutil import word_count  # CJK characters count as words

    n_words = word_count(s)
    ends_sentence = s.endswith((".", "!", "?", "。", "！", "？"))
    if "\n" not in s and len(s) <= 40 and ((n_words <= 4 and not ends_sentence) or n_words <= 2):
        return "label"
    inner, fenced, _ = strip_fences(s)
    if fenced:
        return "code"
    sig = structure_signature(s)
    if sig["headings"] or sig["list"] or sig["table"] or sig["code_block"]:
        return "markdown"
    return "prose"
