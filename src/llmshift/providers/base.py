"""Provider interface and the normalised Response returned by every model call."""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class ProviderError(Exception):
    """A failed model call. `retryable` errors may succeed if retried later."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        param: str | None = None,
        retryable: bool = False,
        body: Any = None,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.param = param
        self.retryable = retryable
        self.body = body
        self.retry_after = retry_after  # seconds the server asked us to wait, if it said

    @property
    def fatal(self) -> bool:
        """Errors that will fail identically for every request (auth, unknown model...)."""
        if self.status in (401, 403):
            return True
        if self.status == 404 or self.code in ("model_not_found", "not_found_error", "NOT_FOUND"):
            return True
        return self.code in (
            "insufficient_quota", "invalid_api_key", "authentication_error", "permission_error", "missing_api_key", "ssl_error",
            "invalid_header", "invalid_url", "invalid_config", "missing_endpoint",
        )

    def __str__(self) -> str:
        bits = []
        if self.status:
            bits.append(f"HTTP {self.status}")
        if self.code and str(self.code) != str(self.status):  # GitHub repeats the status as "status": "403"
            bits.append(str(self.code))
        prefix = f"[{', '.join(bits)}] " if bits else ""
        return prefix + self.message


_INVISIBLE = "\ufeff\u200b\u200c\u200d\u2060"


def clean_key(value: Any) -> str | None:
    """An API key without the whitespace, BOM or zero-width space that a CRLF .env file
    or a copy-paste leaves around it; None when nothing is left."""
    if value is None:
        return None
    text = str(value).strip().strip(_INVISIBLE).strip()
    return text or None


def base_url_from(options: dict, env_name: str | None, default: str, spec: str) -> str:
    """The base URL from the provider's settings, else the environment, else the public
    API. A base_url that is set but empty (an unset ${VAR} in llmshift.yaml) is an error:
    falling back would send the requests, and the key, to the public API instead."""
    if "base_url" in options:
        value, where = str(options["base_url"] or "").strip(), "base_url in llmshift.yaml"
        if not value:
            raise ProviderError(
                f"{spec}: base_url is empty in llmshift.yaml (an unset environment variable?); set it, or remove it to use {default}",
                code="invalid_config",
            )
    else:
        env = (os.environ.get(env_name) or "").strip() if env_name else ""
        value, where = env or default, f"{env_name} in the environment" if env else "the default URL"
    if not value.lower().startswith(("https://", "http://")):  # localhost:8000/v1 would fail on every request
        from .http import display_url

        raise ProviderError(f"{spec}: {where} must start with https:// or http://, got {display_url(value)!r}", code="invalid_config")
    return value.rstrip("/")


def option_number(options: dict, key: str, default: float, spec: str, *, minimum: float, integer: bool = False) -> Any:
    """A numeric provider setting, checked where it is read: `timeout: abc` or `timeout: 0`
    in llmshift.yaml is reported as such, not as every request failing oddly."""
    value = options.get(key)
    if value is None or value == "":
        return int(default) if integer else float(default)
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(str(value).strip())
        if not math.isfinite(number) or number < minimum or (integer and not number.is_integer()):
            raise ValueError
    except (TypeError, ValueError):
        what = "a whole number" if integer else "a number"
        raise ProviderError(f"{spec}: {key} must be {what} of at least {minimum:g}, got {value!r}", code="invalid_config") from None
    return int(number) if integer else number


def option_mapping(options: dict, key: str, spec: str, *, text_values: bool = False) -> dict:
    """A provider setting that must be a mapping (headers, extra_body, env)."""
    value = options.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProviderError(f"{spec}: {key} must be a mapping of names to values, got {type(value).__name__}", code="invalid_config")
    if not text_values:
        return dict(value)
    out = {}
    for name, item in value.items():
        if isinstance(item, (dict, list)):
            raise ProviderError(f"{spec}: {key}.{name} must be a single value, not a {type(item).__name__}", code="invalid_config")
        if item is not None:
            out[str(name)] = str(item).lower() if isinstance(item, bool) else str(item)
    return out


@dataclass
class Response:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    model: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency_ms: float | None = None
    fingerprint: str | None = None
    refusal: str | None = None
    error: str | None = None
    error_code: str | None = None
    cached: bool = False
    steps: list[dict] = field(default_factory=list)  # agent replays: one entry per model call

    def __post_init__(self) -> None:
        # Whatever a provider, a cache file or a custom function hands over, the fields get
        # the types the rest of llmshift relies on: one odd value must not crash a report
        # after every (paid) call has been made.
        self.text = self.text if isinstance(self.text, str) else ("" if self.text is None else str(self.text))
        self.tool_calls = [c for c in self.tool_calls if isinstance(c, dict)] if isinstance(self.tool_calls, list) else []
        for name in ("model", "finish_reason", "fingerprint", "refusal", "error", "error_code"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                setattr(self, name, str(value))
        self.usage = clean_usage(self.usage)
        lat = self.latency_ms
        self.latency_ms = float(lat) if isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat >= 0 else None
        self.steps = [s for s in self.steps if isinstance(s, dict)] if isinstance(self.steps, list) else []
        self.cached = self.cached is True

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {}, False) or k == "text"}

    @classmethod
    def from_dict(cls, data: dict) -> "Response":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @classmethod
    def failure(cls, exc: BaseException) -> "Response":
        code = getattr(exc, "code", None)
        return cls(error=str(exc) or exc.__class__.__name__, error_code=str(code) if code else exc.__class__.__name__)


class Provider:
    """Base class. Subclasses implement `_complete(request) -> Response`."""

    kind = "base"
    supports_embeddings = False

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        self.model = model
        self.name = name or self.kind
        self.options = dict(options or {})
        self.timeout = option_number(self.options, "timeout", 120, self.spec, minimum=0.1)
        self.max_retries = option_number(self.options, "max_retries", 4, self.spec, minimum=0, integer=True)

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.model}"

    def identity(self) -> dict:
        """What makes two providers produce the same outputs (used for caching)."""
        return {"kind": self.kind, "model": self.model, "base_url": getattr(self, "base_url", None)}

    def complete(self, request: dict) -> Response:
        start = time.perf_counter()
        try:
            response = self._complete(request)
        except (AttributeError, TypeError, KeyError, IndexError) as exc:
            # A reply of an unexpected shape (a string where an object belongs, an empty
            # list...): a clear provider error, never a crash with a Python type name.
            raise ProviderError(f"{self.spec} sent a reply llmshift cannot read ({exc.__class__.__name__}: {exc})", code="invalid_response") from None
        if response.latency_ms is None:
            response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return response

    def _complete(self, request: dict) -> Response:  # pragma: no cover - abstract
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderError(f"{self.spec} does not support embeddings")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.spec}>"


def params_of(request: dict) -> dict:
    return dict(request.get("params") or {})


def clean_usage(raw: Any) -> dict:
    """Token counts as whole numbers of zero or more; anything else a provider or a cache
    file holds (NaN, infinity, "12.5", a nested object, -1) is left out."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            out[key] = int(value)
    return out
