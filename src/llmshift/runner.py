"""Run test cases against a model concurrently, with caching and fail-fast on
errors that would repeat for every request (bad key, unknown model)."""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable

from . import messages as msgs
from .cache import ResponseCache
from .console import Progress
from .providers import http
from .providers.base import Provider, ProviderError, Response
from .providers.custom import kill_running
from .testset import TestCase, group_key


class RunAborted(RuntimeError):
    """Every request failed with the same fatal error."""


# Failures that will repeat for every request while nothing has succeeded yet.
_REPEATING = ("connection_error", "redirect")


def build_request(case: TestCase, patch: dict | None = None, overrides: dict | None = None) -> dict:
    messages = msgs.apply_patch(case.messages, patch)
    params = dict(case.params)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        params[key] = value
    return {"messages": messages, "params": params}


def run_cases(
    provider: Provider,
    cases: list[TestCase],
    *,
    concurrency: int = 4,
    cache: ResponseCache | None = None,
    patches: dict | Callable[[TestCase], dict | None] | None = None,
    overrides: dict | None = None,
    label: str | None = None,
    salt: str = "",
    show_progress: bool | None = None,
) -> dict[str, Response]:
    """Return {case.id: Response}. Errors become Response(error=...)."""
    results: dict[str, Response] = {}
    if not cases:
        return results
    http.CANCEL.clear()  # a previous run may have been interrupted
    lock = threading.Lock()
    state: dict[str, Any] = {"ok": 0, "fatal": 0, "first_fatal": None}
    abort = threading.Event()
    progress = Progress(len(cases), label or provider.spec, enabled=show_progress)
    fatal_limit = min(3, len(cases))

    def work(case: TestCase) -> tuple[str, Response]:
        if abort.is_set():
            return case.id, Response(error="skipped: earlier requests failed with a fatal error", error_code="aborted")
        if callable(patches):
            patch = patches(case)
        elif isinstance(patches, dict):
            patch = patches.get(group_key(case))
        else:
            patch = None
        request = build_request(case, patch, overrides)
        key = None
        if cache is not None and cache.enabled:
            try:
                key = cache.key(provider.identity(), request, salt)
                hit = cache.get(key)
            except Exception:  # a broken cache entry must never fail the run
                key, hit = None, None
            if hit is not None:
                with lock:
                    state["ok"] += 1
                return case.id, hit
        try:
            resp = provider.complete(request)
        except ProviderError as exc:
            resp = Response.failure(exc)
            if exc.fatal or exc.code in _REPEATING:
                with lock:
                    state["fatal"] += 1
                    state["first_fatal"] = state["first_fatal"] or exc
                    if state["ok"] == 0 and state["fatal"] >= fatal_limit:
                        abort.set()
            return case.id, resp
        except Exception as exc:  # e.g. a bug in a user-supplied python provider
            return case.id, Response.failure(exc)
        with lock:
            state["ok"] += 1
        if key is not None:
            try:
                cache.put(key, resp)
            except Exception:
                pass
        return case.id, resp

    pool = ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    interrupted = False
    try:
        pending = {pool.submit(work, c) for c in cases}
        while pending:
            # A timed wait keeps Ctrl+C responsive; an untimed one blocks it on Windows.
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            for fut in done:
                cid, resp = fut.result()
                results[cid] = resp
                progress.advance(resp.ok)
    except KeyboardInterrupt:
        interrupted = True
        abort.set()
        http.CANCEL.set()  # in-flight requests stop retrying
        kill_running()
        pool.shutdown(wait=False, cancel_futures=True)
        progress.close()
        raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True)
    progress.close()
    if state["ok"] == 0 and state["first_fatal"] is not None:
        raise RunAborted(f"{provider.spec}: {state['first_fatal']}")
    return results


def adaptations(provider: Provider) -> list[str]:
    return list(getattr(provider, "adaptations", []) or [])
