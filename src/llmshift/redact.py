"""Redaction of secrets and personal data before logs become committed test files.

Placeholders are consistent within one Redactor: the same e-mail address always
becomes <EMAIL_1>, so a model that echoes the input still matches the redacted
baseline output.
"""

from __future__ import annotations

import re
import threading
from typing import Any

SECRET_PATTERNS = [
    ("SECRET", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("SECRET", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("SECRET", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SECRET", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("SECRET", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("SECRET", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("SECRET", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("SECRET", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("SECRET", re.compile(r"(?i)\b(?:bearer)\s+[A-Za-z0-9_\-.=]{20,}")),
    ("SECRET", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
]

PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w-])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}(?![\w-])")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


_IMAGE_DATA_URL_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,[A-Za-z0-9+/=\s]*")


def _identity(label: str, value: str) -> str:
    """The same address or number written differently gets the same placeholder."""
    if label == "EMAIL":
        return value.lower()
    if label in ("PHONE", "CARD", "SSN"):
        return re.sub(r"\D", "", value)
    if label == "IBAN":
        return re.sub(r"\s", "", value).upper()
    return value


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


class Redactor:
    def __init__(self, *, secrets: bool = True, pii: bool = False):
        self.patterns = (SECRET_PATTERNS if secrets else []) + (PII_PATTERNS if pii else [])
        self.mapping: dict[tuple[str, str], str] = {}
        self.counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def _placeholder(self, label: str, value: str) -> str:
        key = (label, _identity(label, value))
        with self._lock:
            if key not in self.mapping:
                self.counts[label] = self.counts.get(label, 0) + 1
                self.mapping[key] = f"<{label}_{self.counts[label]}>"
            return self.mapping[key]

    def text(self, text: str) -> str:
        if not text or not self.patterns:
            return text
        if _IMAGE_DATA_URL_RE.fullmatch(text):  # an inline image: nothing readable to redact
            return text
        for label, pattern in self.patterns:
            def repl(m: re.Match, label: str = label) -> str:
                value = m.group(0)
                if label == "CARD":
                    digits = re.sub(r"\D", "", value)
                    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                        return value
                return self._placeholder(label, value)

            text = pattern.sub(repl, text)
        return text

    def obj(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.obj(v) for v in value]
        if isinstance(value, dict):
            return {k: (v if k in ("role", "type", "id", "tool_call_id", "name") and isinstance(v, str) and len(v) < 64 else self.obj(v)) for k, v in value.items()}
        return value

    @property
    def total(self) -> int:
        return sum(self.counts.values())
