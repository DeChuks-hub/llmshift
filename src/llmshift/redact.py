"""Redaction of secrets and personal data before logs become committed test files.

Placeholders are consistent within one Redactor: the same e-mail address always
becomes <EMAIL_1>, so a model that echoes the input still matches the redacted
baseline output.
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

# (pattern, what it must contain): a substring test is far cheaper than a regex that tries
# every position of a long text (a document or an image in a request).
_SECRETS = [
    (r"\bsk-ant-[A-Za-z0-9_\-]{20,}", ("sk-ant-",)),  # Anthropic
    (r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}", ("sk-",)),  # OpenAI (and DeepSeek)
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", ("AKIA", "ASIA")),  # AWS, long-lived and temporary
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", ("ghp_", "gho_", "ghu_", "ghs_", "ghr_")),
    (r"\bgithub_pat_[A-Za-z0-9_]{40,}\b", ("github_pat_",)),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", ("xox",)),  # Slack
    (r"\bAIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])", ("AIza",)),  # Google (a key may end in "-")
    (r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", ("eyJ",)),  # JWTs
    (r"(?i)\b(?:bearer)\s+[A-Za-z0-9_\-.=]{20,}", ()),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----", ("PRIVATE KEY",)),
    (r"\bxai-[A-Za-z0-9]{20,}", ("xai-",)),  # xAI
    (r"\bgsk_[A-Za-z0-9]{20,}", ("gsk_",)),  # Groq
    (r"\bpplx-[A-Za-z0-9]{20,}", ("pplx-",)),  # Perplexity
    (r"\bfw_[A-Za-z0-9]{20,}", ("fw_",)),  # Fireworks
    (r"\bhf_[A-Za-z0-9]{20,}", ("hf_",)),  # Hugging Face
    (r"\br8_[A-Za-z0-9]{20,}", ("r8_",)),  # Replicate
    (r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}", ("_live_", "_test_")),  # Stripe
]
SECRET_PATTERNS = [("SECRET", re.compile(p)) for p, _ in _SECRETS]
_MARKERS = {p: markers for p, markers in _SECRETS if markers}

PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]{1,64}@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b\d(?:[ .-]?\d){12,18}\b")),  # ends on a digit: the space after the number stays
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b", re.I)),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w-])\+\d{1,3}(?:[ .-]?\(?\d{1,5}\)?){2,5}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w.-])0\d(?:[ .-]\d{2}){4}(?![\w-])")),  # 01 23 45 67 89
    ("PHONE", re.compile(r"(?<![\w.-])0\d{2,4}[ .-]\d{3,4}[ .-]?\d{3,4}(?![\w-])")),  # 020 7946 0958, 030 12345678
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]

# Environment variables whose values are credentials: their exact values are redacted
# too, which catches keys with no recognisable shape (Azure, Mistral, Cohere...).
_SECRET_ENV_RE = re.compile(r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|AD_TOKEN)$", re.I)


def _env_secrets() -> list[str]:
    values = {v.strip() for k, v in os.environ.items() if _SECRET_ENV_RE.search(k) and v and len(v.strip()) >= 16}
    return sorted(values, key=len, reverse=True)


_IMAGE_DATA_URL_RE = re.compile(r"data:[\w.+-]+/[\w.+-]+;base64,")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/=\s]+")
_DIGIT_RE = re.compile(r"\d")


def _may_match(pattern: re.Pattern, text: str, lowered: list) -> bool:
    markers = _MARKERS.get(pattern.pattern)
    if markers is not None:
        return any(m in text for m in markers)
    if "bearer" in pattern.pattern:
        if not lowered:
            lowered.append(text.lower())
        return "bearer" in lowered[0]
    if "@" in pattern.pattern:  # e-mail
        return "@" in text
    return _DIGIT_RE.search(text) is not None  # cards, numbers, IBANs and IPs all hold digits


_PLAIN_KEYS = ("role", "type", "id", "tool_call_id", "name")  # short values here are never secrets


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


def _iban_ok(value: str) -> bool:
    """The IBAN check digits (ISO 13616, mod 97): product codes and the like fail it."""
    text = re.sub(r"\s", "", value).upper()
    digits = "".join(str(int(c, 36)) for c in text[4:] + text[:4])
    return int(digits) % 97 == 1


def _plausible(label: str, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if label == "CARD":
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if label == "IBAN":
        return _iban_ok(value)
    if label == "PHONE":
        return len(digits) >= 7  # "+10 20 30" is not a phone number
    return True


class Redactor:
    def __init__(self, *, secrets: bool = True, pii: bool = False):
        self.patterns = (SECRET_PATTERNS if secrets else []) + (PII_PATTERNS if pii else [])
        self.env_secrets = _env_secrets() if secrets else []
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
        if _IMAGE_DATA_URL_RE.match(text, 0, 128):  # an inline image: nothing readable to redact
            return text
        if len(text) > 4096 and _BASE64_RE.fullmatch(text):  # raw base64 (an image or file): the same
            return text
        for secret in self.env_secrets:  # the live value of a key variable, whatever its shape
            if secret in text:
                text = text.replace(secret, self._placeholder("SECRET", secret))
        lowered: list = []
        for label, pattern in self.patterns:
            if not _may_match(pattern, text, lowered):
                continue

            def repl(m: re.Match, label: str = label) -> str:
                value = m.group(0)
                return self._placeholder(label, value) if _plausible(label, value) else value

            redacted = pattern.sub(repl, text)
            if redacted is not text:
                text, lowered = redacted, []
        return text

    def obj(self, value: Any) -> Any:
        """A copy of JSON-like data with every string redacted. Iterative: data nested
        deeper than Python's recursion limit (a tool's arguments, say) is handled too."""
        root: list = [None]
        stack: list[tuple[Any, Any, Any]] = [(root, 0, value)]
        while stack:
            parent, key, item = stack.pop()
            if isinstance(item, str):
                parent[key] = self.text(item)
            elif isinstance(item, list):
                parent[key] = copy = [None] * len(item)
                stack.extend((copy, i, v) for i, v in enumerate(item))
            elif isinstance(item, dict):
                parent[key] = out = {}
                for k, v in item.items():
                    if k in _PLAIN_KEYS and isinstance(v, str) and len(v) < 64:
                        out[k] = v
                    else:
                        out[k] = None  # keeps the key order; filled in below
                        stack.append((out, k, v))
            else:
                parent[key] = item
        return root[0]

    @property
    def total(self) -> int:
        return sum(self.counts.values())
