"""Provider interface and the normalised Response returned by every model call."""

from __future__ import annotations

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
            "invalid_header", "invalid_url", "invalid_config",
        )


_INVISIBLE = "﻿​‌‍⁠"


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
        value = str(options["base_url"] or "").strip()
        if not value:
            raise ProviderError(
                f"{spec}: base_url is empty in llmshift.yaml (an unset environment variable?); set it, or remove it to use {default}",
                code="invalid_config",
            )
        return value.rstrip("/")
    env = (os.environ.get(env_name) or "").strip() if env_name else ""
    return (env or default).rstrip("/")

    def __str__(self) -> str:
        bits = []
        if self.status:
            bits.append(f"HTTP {self.status}")
        if self.code and str(self.code) != str(self.status):  # GitHub repeats the status as "status": "403"
            bits.append(str(self.code))
        prefix = f"[{', '.join(bits)}] " if bits else ""
        return prefix + self.message


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
        self.timeout = float(self.options.get("timeout", 120))
        self.max_retries = int(self.options.get("max_retries", 4))

    @property
    def spec(self) -> str:
        return f"{self.name}:{self.model}"

    def identity(self) -> dict:
        """What makes two providers produce the same outputs (used for caching)."""
        return {"kind": self.kind, "model": self.model, "base_url": getattr(self, "base_url", None)}

    def complete(self, request: dict) -> Response:
        start = time.perf_counter()
        response = self._complete(request)
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
