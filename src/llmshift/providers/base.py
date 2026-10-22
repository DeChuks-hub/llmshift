"""Provider interface and the normalised Response returned by every model call."""

from __future__ import annotations

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
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.param = param
        self.retryable = retryable
        self.body = body

    @property
    def fatal(self) -> bool:
        """Errors that will fail identically for every request (auth, unknown model...)."""
        if self.status in (401, 403):
            return True
        if self.status == 404 or self.code in ("model_not_found", "not_found_error", "NOT_FOUND"):
            return True
        return self.code in ("insufficient_quota", "invalid_api_key", "authentication_error", "permission_error", "missing_api_key")

    def __str__(self) -> str:
        bits = []
        if self.status:
            bits.append(f"HTTP {self.status}")
        if self.code:
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
