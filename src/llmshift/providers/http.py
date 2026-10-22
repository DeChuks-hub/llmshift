"""Minimal JSON-over-HTTP client (urllib) with retries, backoff and error parsing.

Redirects are never followed: provider APIs do not redirect, and following one
would re-send the API key to whatever host the redirect names.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import math
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from ..textutil import clean_surrogates, has_surrogate_escape
from .base import ProviderError

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
NON_RETRYABLE_CODES = ("insufficient_quota", "billing_hard_limit_reached", "invalid_api_key")
MAX_DELAY = 60.0

MAX_RESPONSE_BYTES = 64 * 1024 * 1024  # no model answer is this large; a runaway body is cut off

# Set by the runner on Ctrl+C: requests stop retrying and new ones are not sent.
CANCEL = threading.Event()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None  # urllib then raises HTTPError for the 3xx response


_OPENER = urllib.request.build_opener(_NoRedirect)


def _user_agent() -> str:
    from .. import __version__

    return f"llmshift/{__version__} (+python-urllib)"


def _parse_error(status: int | None, raw: bytes, url: str, headers: Any = None) -> ProviderError:
    text = raw.decode("utf-8", errors="replace") if raw else ""
    message, code, param, body = text.strip()[:500] or f"HTTP {status}", None, None, None
    try:
        body = json.loads(text) if text.strip() else None
    except ValueError:
        body = None
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            message = str(err.get("message") or err.get("msg") or message)
            code = err.get("code")
            if isinstance(code, int) or code is None:  # Gemini: {"code": 400, "status": "INVALID_ARGUMENT"}
                code = err.get("type") or err.get("status") or code
            param = err.get("param")
            for detail in err.get("details") or []:
                if isinstance(detail, dict) and detail.get("reason") == "API_KEY_INVALID":
                    code = "invalid_api_key"
        elif isinstance(err, str):
            message = err
        if body.get("type") == "error" and isinstance(body.get("error"), dict):
            code = body["error"].get("type") or code
    if status is not None and 300 <= status < 400:
        location = (headers.get("Location") if headers is not None else None) or "another URL"
        return ProviderError(
            f"the server redirected to {location}; not followed so credentials are never sent to another host (check base_url)",
            status=status,
            code="redirect",
        )
    if status == 404 and not code:
        code = "not_found"
    retryable = (status in RETRY_STATUSES) and str(code) not in NON_RETRYABLE_CODES
    host = urllib.parse.urlsplit(url).netloc
    return ProviderError(f"{message} ({host})", status=status, code=str(code) if code is not None else None, param=param, retryable=retryable, body=body)


def _retry_delay(headers: Any, attempt: int) -> float:
    if headers is not None:
        ms = headers.get("retry-after-ms")
        if ms:
            try:
                return min(MAX_DELAY, max(0.0, float(ms) / 1000.0))
            except ValueError:
                pass
        ra = headers.get("retry-after")
        if ra:
            try:
                return min(MAX_DELAY, max(0.0, float(ra)))
            except ValueError:
                try:
                    parsed = email.utils.parsedate_to_datetime(ra)
                    return min(MAX_DELAY, max(0.0, parsed.timestamp() - time.time()))
                except (TypeError, ValueError, IndexError, OverflowError):
                    pass  # unusual value such as "30s": fall back to exponential backoff
    return min(MAX_DELAY, (0.5 * 2**attempt) + random.uniform(0, 0.5))


def _sleep(seconds: float, sleep: Callable[[float], None]) -> None:
    end = time.time() + seconds
    while not CANCEL.is_set():
        remaining = end - time.time()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.25))


def _clean_floats(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_floats(v) for v in value]
    return value


def _encode(body: Any) -> bytes:
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8", "replace")
    except ValueError:  # NaN/Infinity are not valid JSON: send null instead
        return json.dumps(_clean_floats(body), ensure_ascii=False, allow_nan=False).encode("utf-8", "replace")


def request(
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    body: Any = None,
    timeout: float = 120,
    max_retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
    raw: bool = False,
) -> Any:
    """Send a request and return parsed JSON (or bytes when raw=True)."""
    data = _encode(body) if body is not None else None
    hdrs = {"User-Agent": _user_agent(), "Accept": "*/*" if raw else "application/json"}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update({k: v for k, v in (headers or {}).items() if v is not None})
    attempt = 0
    while True:
        if CANCEL.is_set():
            raise ProviderError("cancelled", code="cancelled")
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
        try:
            with _OPENER.open(req, timeout=timeout) as resp:
                payload = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ProviderError(f"response from {url} is larger than {MAX_RESPONSE_BYTES // (1024 * 1024)} MB", code="invalid_response")
            if raw:
                return payload
            text = payload.decode("utf-8", errors="replace")
            if not text.strip():
                return {}
            try:
                data = json.loads(text)
                return clean_surrogates(data) if has_surrogate_escape(text) else data
            except RecursionError:
                raise ProviderError(f"response from {url} is nested too deeply to read", code="invalid_response") from None
            except ValueError:
                raise ProviderError(f"invalid JSON response from {url}: {text[:200]!r}", code="invalid_response", retryable=True) from None
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read()
            except Exception:
                payload = b""
            err = _parse_error(exc.code, payload, url, exc.headers)
            if err.retryable and attempt < max_retries:
                _sleep(_retry_delay(exc.headers, attempt), sleep)
                attempt += 1
                continue
            raise err from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if attempt < max_retries:
                _sleep(_retry_delay(None, attempt), sleep)
                attempt += 1
                continue
            host = urllib.parse.urlsplit(url).netloc or url
            raise ProviderError(f"cannot reach {host}: {reason}", code="connection_error", retryable=True) from None


def post_json(url: str, body: Any, **kwargs: Any) -> Any:
    return request("POST", url, body=body, **kwargs)


def get_json(url: str, **kwargs: Any) -> Any:
    return request("GET", url, **kwargs)


def require_object(data: Any, what: str, *keys: str) -> dict:
    """A provider reply must be a JSON object with at least one of `keys`."""
    if not isinstance(data, dict) or (keys and not any(k in data for k in keys)):
        shown = json.dumps(data)[:120] if data not in ({}, None) else "an empty body"
        raise ProviderError(f"unexpected {what} response: {shown}", code="invalid_response", retryable=True)
    return data
