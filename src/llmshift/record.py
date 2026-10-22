"""Record real LLM traffic from your app so `llmshift snapshot` can turn it into tests.

    from openai import OpenAI
    from llmshift.record import wrap

    client = wrap(OpenAI(), path="logs/llm.jsonl")   # also works with AsyncOpenAI, Anthropic, AsyncAnthropic
    client.chat.completions.create(...)               # behaves exactly as before

Agents: run each task inside `agent_run` so its calls (and the tool results they
carry) become one multi-step test case:

    from llmshift.record import agent_run
    with agent_run("refund-flow"):
        my_agent.run(task)

Or log calls yourself (any SDK or raw HTTP):

    from llmshift.record import log
    log(request={"model": ..., "messages": [...]}, response=resp, path="logs/llm.jsonl")

Every call appends one JSON line. Recording never raises into your app: if
writing fails, a single warning is emitted and the call result is returned
unchanged. Secrets (API keys, tokens) are redacted before writing.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import inspect
import json
import os
import random
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from . import fileio
from .redact import Redactor

_ENDPOINTS = {
    ("chat", "completions", "create"): "chat",
    ("chat", "completions", "parse"): "chat",
    ("responses", "create"): "responses",
    ("responses", "parse"): "responses",
    ("messages", "create"): "messages",
    ("messages", "parse"): "messages",
}
# SDK stream helpers (`with client.messages.stream(...) as s:`): recorded when the block ends.
_STREAM_HELPERS = {
    ("chat", "completions", "stream"): "chat",
    ("responses", "stream"): "responses",
    ("messages", "stream"): "messages",
}
# Attribute paths that lead to an endpoint; everything else is returned untouched.
_ROUTES = {key[:i] for key in (*_ENDPOINTS, *_STREAM_HELPERS) for i in range(1, len(key))}
_ROUTES |= {("beta",)} | {("beta",) + route for route in _ROUTES}
_FORMAT_ARGS = ("response_format", "text_format", "output_format")  # may be a pydantic class
_CLIENT_FACTORIES = ("with_options", "copy")  # methods that return a configured copy of the client

_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _path_key(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(path))


def _path_lock(path: str | os.PathLike) -> threading.Lock:
    """One lock per log file, shared by every Recorder writing to it."""
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(_path_key(path), threading.Lock())


def _dump(model_dump: Any) -> Any:
    try:
        return model_dump(warnings=False)  # pydantic v2: no serializer warnings in the app's output
    except TypeError:
        return model_dump()


def to_dict(obj: Any) -> Any:
    """Best-effort conversion of SDK objects (pydantic models etc.) to plain data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    for method in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                return to_dict(_dump(fn) if method == "model_dump" else fn())
            except Exception:
                continue
    if hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


_AGENT_RUN: contextvars.ContextVar = contextvars.ContextVar("llmshift_agent_run", default=None)


@contextlib.contextmanager
def agent_run(name: str | None = None, run_id: str | None = None) -> Any:
    """Link every call recorded inside the block (also in asyncio tasks started there)
    into one agent run. Yields the run id."""
    state = {"id": str(run_id or uuid.uuid4().hex[:16]), "name": name, "step": 0, "lock": threading.Lock()}
    token = _AGENT_RUN.set(state)
    try:
        yield state["id"]
    finally:
        _AGENT_RUN.reset(token)


class Recorder:
    def __init__(self, path: str | os.PathLike = "logs/llmshift.jsonl", *, sample_rate: float = 1.0, redact_secrets: bool = True, metadata: dict | None = None):
        self.path = Path(path)
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.redactor = Redactor(secrets=True, pii=False) if redact_secrets else None
        self.metadata = dict(metadata or {})
        self._lock = _path_lock(self.path)
        self._warned = False

    def sampled(self) -> bool:
        return self.sample_rate >= 1.0 or random.random() < self.sample_rate

    def log(self, request: dict, response: Any = None, *, provider: str | None = None, api: str = "chat", latency_ms: float | None = None, error: str | None = None, metadata: dict | None = None) -> None:
        try:
            req = to_dict(request) or {}
            req = {k: v for k, v in req.items() if k not in ("extra_headers", "extra_query", "timeout", "api_key")}
            rec: dict = {
                "llmshift_record": 1,
                "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
                "provider": provider,
                "api": api,
                "request": req,
                "response": to_dict(response),
            }
            if latency_ms is not None:
                rec["latency_ms"] = round(latency_ms, 1)
            if error:
                rec["error"] = error
            meta = {**self.metadata, **(metadata or {})}
            if meta:
                rec["metadata"] = meta
            run = _AGENT_RUN.get()
            if run is not None:
                with run["lock"]:
                    run["step"] += 1
                    step = run["step"]
                rec["agent_run"] = {"id": run["id"], "step": step, **({"name": run["name"]} if run["name"] else {})}
            if self.redactor is not None:
                rec = self.redactor.obj(rec)
            data = memoryview((json.dumps(rec, ensure_ascii=False, default=str) + "\n").encode("utf-8", "replace"))
            with self._lock:
                # One locked append per record: lines stay whole even when several
                # processes (web workers, jobs) log to the same file.
                fileio.append(self.path, bytes(data))
        except Exception as exc:  # recording must never break the app
            if not self._warned:
                self._warned = True
                warnings.warn(f"llmshift: could not record an LLM call ({exc})", RuntimeWarning, stacklevel=2)


_recorders: dict[str, Recorder] = {}
_recorders_lock = threading.Lock()


def _release_after_fork() -> None:
    # Only the forking thread survives a fork; a lock another thread held at that moment
    # would stay locked in the child forever.
    for lock in (_PATH_LOCKS_GUARD, _recorders_lock, *_PATH_LOCKS.values()):
        if lock.locked():
            lock.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_release_after_fork)


def log(request: dict, response: Any = None, *, path: str | os.PathLike | None = None, **kwargs: Any) -> None:
    """Append one request/response pair to the log file."""
    if path is None:
        path = os.environ.get("LLMSHIFT_LOG", "logs/llmshift.jsonl")
    key = _path_key(path)
    with _recorders_lock:
        recorder = _recorders.get(key)
        if recorder is None:
            recorder = _recorders[key] = Recorder(path)
    recorder.log(request, response, **kwargs)


def _get(obj: Any, *path: Any) -> Any:
    for key in path:
        if obj is None:
            return None
        if isinstance(key, int):
            try:
                obj = obj[key]
            except (IndexError, KeyError, TypeError):
                return None
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj


def _arguments(text: str) -> Any:
    try:
        return json.loads(text) if text.strip() else {}
    except ValueError:
        return {"_raw": text}


class _StreamAccumulator:
    """Rebuilds a complete response from the streamed chunks of the major SDKs:
    text, tool calls (with arguments split across chunks), finish reason and usage."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.model: str | None = None
        self.finish: str | None = None
        self.usage: dict = {}
        self.calls: dict[tuple, dict] = {}
        self.final: dict | None = None  # Responses API: the whole response arrives at the end

    def add(self, chunk: Any) -> None:
        try:
            self._add(chunk)
        except Exception:
            pass

    def _usage(self, usage: Any) -> None:
        usage = to_dict(usage)
        if isinstance(usage, dict):
            self.usage.update({k: v for k, v in usage.items() if v is not None})

    def _add(self, chunk: Any) -> None:
        self.model = self.model or _get(chunk, "model") or _get(chunk, "message", "model") or _get(chunk, "modelVersion")
        ctype = _get(chunk, "type")
        if ctype is None and _get(chunk, "candidates") is not None:  # Gemini
            cand = _get(chunk, "candidates", 0)
            for part in _get(cand, "content", "parts") or []:
                text = _get(part, "text")
                if isinstance(text, str) and not _get(part, "thought"):
                    self.text.append(text)
                call = _get(part, "functionCall") or _get(part, "function_call")
                if _get(call, "name"):
                    args = to_dict(_get(call, "args"))
                    self.calls[("gemini", len(self.calls))] = {"id": _get(call, "id"), "name": _get(call, "name"), "arguments": json.dumps(args or {})}
            self.finish = _get(cand, "finishReason") or _get(cand, "finish_reason") or self.finish
            if (_get(chunk, "usageMetadata") or _get(chunk, "usage_metadata")) is not None:
                self._usage(_get(chunk, "usageMetadata") or _get(chunk, "usage_metadata"))
        elif ctype is None:  # OpenAI chat completion chunks
            choice = _get(chunk, "choices", 0)
            delta = _get(choice, "delta")
            content = _get(delta, "content")
            if isinstance(content, str):
                self.text.append(content)
            for tc in _get(delta, "tool_calls") or []:
                index = _get(tc, "index")
                slot = self.calls.setdefault(("chat", index if isinstance(index, int) else len(self.calls)), {"id": None, "name": "", "arguments": ""})
                slot["id"] = slot["id"] or _get(tc, "id")
                slot["name"] = slot["name"] or _get(tc, "function", "name") or ""
                args = _get(tc, "function", "arguments")
                if isinstance(args, str):
                    slot["arguments"] += args
            fin = _get(choice, "finish_reason")
            if fin:
                self.finish = fin
            if _get(chunk, "usage") is not None:  # sent last with stream_options.include_usage
                self._usage(_get(chunk, "usage"))
        elif ctype == "message_start":  # Anthropic
            self._usage(_get(chunk, "message", "usage"))
        elif ctype == "content_block_start":
            block = _get(chunk, "content_block")
            if _get(block, "type") == "tool_use":
                self.calls[("msg", _get(chunk, "index"))] = {"id": _get(block, "id"), "name": _get(block, "name") or "", "arguments": ""}
        elif ctype == "content_block_delta":
            dtype = _get(chunk, "delta", "type")
            if dtype == "text_delta":
                self.text.append(_get(chunk, "delta", "text") or "")
            elif dtype == "input_json_delta":
                slot = self.calls.get(("msg", _get(chunk, "index")))
                if slot is not None:
                    slot["arguments"] += _get(chunk, "delta", "partial_json") or ""
        elif ctype == "message_delta":
            self.finish = _get(chunk, "delta", "stop_reason") or self.finish
            self._usage(_get(chunk, "usage"))
        elif ctype == "response.output_text.delta":  # OpenAI Responses
            self.text.append(_get(chunk, "delta") or "")
        elif ctype in ("response.completed", "response.incomplete", "response.failed"):
            final = to_dict(_get(chunk, "response"))
            if isinstance(final, dict):
                self.final = final

    def response(self, api: str) -> dict:
        if api == "responses" and self.final:
            return self.final
        text = "".join(self.text)
        calls = [self.calls[k] for k in sorted(self.calls, key=lambda k: (k[0], k[1] if isinstance(k[1], int) else 0))]
        out: dict
        if api == "gemini":
            parts: list[dict] = [{"text": text}] if text or not calls else []
            for c in calls:
                fc = {"name": c["name"], "args": _arguments(c["arguments"])}
                if c["id"]:
                    fc["id"] = c["id"]
                parts.append({"functionCall": fc})
            out = {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": self.finish}], "modelVersion": self.model}
            if self.usage:
                out["usageMetadata"] = dict(self.usage)
            return out
        if api == "messages":
            content: list[dict] = [{"type": "text", "text": text}] if text or not calls else []
            content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": _arguments(c["arguments"])} for c in calls]
            out = {"type": "message", "role": "assistant", "model": self.model, "content": content, "stop_reason": self.finish}
        elif api == "responses":
            out = {"object": "response", "model": self.model, "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}
        else:
            message: dict = {"role": "assistant", "content": text if text or not calls else None}
            if calls:
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for c in calls
                ]
            out = {"object": "chat.completion", "model": self.model, "choices": [{"index": 0, "message": message, "finish_reason": self.finish}]}
        if self.usage:
            out["usage"] = dict(self.usage)
        return out


class _StreamProxy:
    """Passes a stream through unchanged and records the whole response once it ends."""

    def __init__(self, stream: Any, done: Any):
        self._stream = stream
        self._done = done
        self._acc = _StreamAccumulator()
        self._finished = False
        self._it: Any = None
        self._ait: Any = None

    def _finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._done(self._acc)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            self._it = iter(self._stream)
        try:
            chunk = next(self._it)
        except BaseException:  # StopIteration, or the connection failed mid-stream
            self._finish()
            raise
        self._acc.add(chunk)
        return chunk

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if self._ait is None:
            self._ait = self._stream.__aiter__()
        try:
            chunk = await self._ait.__anext__()
        except BaseException:
            self._finish()
            raise
        self._acc.add(chunk)
        return chunk

    def close(self) -> Any:
        self._finish()
        closer = getattr(self._stream, "close", None)
        return closer() if callable(closer) else None

    async def aclose(self) -> Any:
        self._finish()
        closer = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                return await result
            return result
        return None

    def __enter__(self) -> Any:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        self._finish()
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*exc)
        return None

    async def __aenter__(self) -> Any:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        self._finish()
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return None

    def __del__(self) -> None:  # a stream abandoned part-way is still recorded
        try:
            self._finish()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _json_schema(cls: Any, converter: str | None = None) -> Any:
    """What the SDK sends for a pydantic class given as a response format: its own
    conversion when the SDK provides one, else the class's JSON schema."""
    if converter:
        module, _, name = converter.rpartition(".")
        try:
            return getattr(__import__(module, fromlist=[name]), name)(cls)
        except Exception:
            pass
    try:
        return cls.model_json_schema()
    except Exception:
        return None


def _wire_request(api: str, kwargs: dict) -> dict:
    """The request as sent on the wire: pydantic classes passed as `response_format`
    (chat), `text_format` (responses) or `output_format` (messages) become the JSON
    schema the SDK derives from them, so recorded cases replay with the same format."""
    if not any(isinstance(kwargs.get(k), type) for k in _FORMAT_ARGS):
        return kwargs
    try:
        return _wire_formats(api, kwargs)
    except Exception:  # recording never breaks the app
        return kwargs


def _wire_formats(api: str, kwargs: dict) -> dict:
    out = dict(kwargs)
    cls = out.pop("text_format", None) if api == "responses" else out.get("output_format") if api == "messages" else out.get("response_format")
    if not isinstance(cls, type):
        return out
    name = getattr(cls, "__name__", "response")
    if api == "chat":
        param = _json_schema(cls, "openai.lib._parsing._completions.type_to_response_format_param")
        if isinstance(param, dict) and param.get("type") == "json_schema":
            out["response_format"] = param
        elif isinstance(param, dict):
            out["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "schema": param, "strict": True}}
    elif api == "responses":
        param = _json_schema(cls, "openai.lib._parsing._responses.type_to_text_format_param")
        if isinstance(param, dict) and param.get("type") != "json_schema":
            param = {"type": "json_schema", "name": name, "schema": param, "strict": True}
        if isinstance(param, dict):
            out["text"] = {**(out.get("text") or {}), "format": param}
    elif api == "messages":
        out.pop("output_format", None)
        schema = _json_schema(cls)
        if isinstance(schema, dict):
            out["output_config"] = {**(out.get("output_config") or {}), "format": {"type": "json_schema", "schema": schema}}
    return out


def _final_snapshot(stream: Any) -> Any:
    """The answer an SDK stream helper has accumulated so far (reads nothing more)."""
    readers = (
        lambda s: s.current_message_snapshot,  # Anthropic MessageStream
        lambda s: s.current_completion_snapshot,  # OpenAI ChatCompletionStream
        lambda s: s._state._completed_response,  # OpenAI ResponseStream (set once completed)
    )
    for read in readers:
        try:
            snapshot = read(stream)
        except Exception:  # nothing received yet, or another SDK version
            continue
        if snapshot is not None:
            return snapshot
    return None


class _HelperStream:
    """What an SDK stream helper returns (a context manager that makes the request on
    entering). Passed through unchanged; the final answer is recorded when the block ends."""

    def __init__(self, manager: Any, done: Any):
        self._manager = manager
        self._done = done
        self._stream: Any = None
        self._start = time.perf_counter()

    def _record(self, exc: BaseException | None) -> None:
        try:
            latency = (time.perf_counter() - self._start) * 1000
            if exc is not None:
                self._done(None, f"{exc.__class__.__name__}: {exc}"[:500], latency)
            else:
                self._done(_final_snapshot(self._stream), None, latency)
        except Exception:
            pass  # recording never breaks the app

    def __enter__(self) -> Any:
        self._start = time.perf_counter()
        try:
            self._stream = self._manager.__enter__()
        except Exception as exc:
            self._record(exc)
            raise
        return self._stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._manager.__exit__(*exc)
        finally:
            self._record(exc[1])

    async def __aenter__(self) -> Any:
        self._start = time.perf_counter()
        try:
            self._stream = await self._manager.__aenter__()
        except Exception as exc:
            self._record(exc)
            raise
        return self._stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._manager.__aexit__(*exc)
        finally:
            self._record(exc[1])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)


class _Proxy:
    def __init__(self, target: Any, recorder: Recorder | None, provider: str | None, parts: tuple = (), shadow: Any = None):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_recorder", recorder)
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_parts", parts)
        object.__setattr__(self, "_shadow", shadow)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        parts = self._parts + (name,)
        api = _ENDPOINTS.get(parts[-3:]) or _ENDPOINTS.get(parts[-2:])
        if api and callable(attr):
            return self._wrap_call(attr, api)
        helper = _STREAM_HELPERS.get(parts[-3:]) or _STREAM_HELPERS.get(parts[-2:])
        if helper and callable(attr):
            return self._wrap_helper(attr, helper)
        if not self._parts and name in _CLIENT_FACTORIES and callable(attr):
            return self._wrap_factory(attr)
        if parts in _ROUTES:
            return _Proxy(attr, self._recorder, self._provider, parts, self._shadow)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __enter__(self) -> Any:
        self._target.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._target.__exit__(*exc)

    async def __aenter__(self) -> Any:
        await self._target.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._target.__aexit__(*exc)

    def _wrap_factory(self, fn: Any) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def call(*args: Any, **kwargs: Any) -> Any:
            return _Proxy(fn(*args, **kwargs), recorder, provider, shadow=shadow)

        return call

    def _wrap_helper(self, fn: Any, api: str) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def call(*args: Any, **kwargs: Any) -> Any:
            logging = recorder is not None and recorder.sampled()
            manager = fn(*args, **kwargs)
            if not logging and shadow is None:
                return manager
            request = _wire_request(api, kwargs)

            def done(snapshot: Any, error: str | None, latency_ms: float) -> None:
                if snapshot is None and error is None:
                    return
                if logging:
                    recorder.log(request, snapshot, provider=provider, api=api, latency_ms=latency_ms, error=error)
                if shadow is not None and error is None:
                    try:
                        shadow.observe(api, request, snapshot)
                    except Exception:
                        pass  # shadowing never breaks the app

            return _HelperStream(manager, done)

        return call

    def _wrap_call(self, fn: Any, api: str) -> Any:
        recorder, provider, shadow = self._recorder, self._provider, self._shadow

        def elapsed(start: float) -> float:
            return (time.perf_counter() - start) * 1000

        def observe(kwargs: dict, response: Any) -> None:
            if shadow is not None:
                try:
                    shadow.observe(api, kwargs, response)
                except Exception:
                    pass  # shadowing never breaks the app

        def failed(kwargs: dict, start: float, exc: BaseException, logging: bool) -> None:
            if logging:
                recorder.log(kwargs, None, provider=provider, api=api, error=f"{exc.__class__.__name__}: {exc}"[:500], latency_ms=elapsed(start))

        def finished(kwargs: dict, start: float, result: Any, logging: bool) -> Any:
            if kwargs.get("stream"):

                def done(acc: _StreamAccumulator) -> None:
                    response = acc.response(api)
                    if logging:
                        recorder.log(kwargs, response, provider=provider, api=api, latency_ms=elapsed(start))
                    observe(kwargs, response)

                return _StreamProxy(result, done)
            if logging:
                recorder.log(kwargs, result, provider=provider, api=api, latency_ms=elapsed(start))
            observe(kwargs, result)
            return result

        async def finish_async(awaitable: Any, kwargs: dict, start: float, logging: bool) -> Any:
            try:
                result = await awaitable
            except Exception as exc:
                failed(kwargs, start, exc, logging)
                raise
            return finished(kwargs, start, result, logging)

        def sampled() -> bool:
            return recorder is not None and recorder.sampled()

        if inspect.iscoroutinefunction(fn):

            async def async_call(*args: Any, **kwargs: Any) -> Any:
                logging = sampled()
                if not logging and shadow is None:
                    return await fn(*args, **kwargs)
                return await finish_async(fn(*args, **kwargs), _wire_request(api, kwargs), time.perf_counter(), logging)

            return async_call

        def call(*args: Any, **kwargs: Any) -> Any:
            logging = sampled()
            if not logging and shadow is None:
                return fn(*args, **kwargs)
            request = _wire_request(api, kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                failed(request, start, exc, logging)
                raise
            if inspect.isawaitable(result):
                # Async SDK methods are often plain functions that return a coroutine.
                return finish_async(result, request, start, logging)
            return finished(request, start, result, logging)

        return call

    def __repr__(self) -> str:
        return f"<llmshift recording proxy for {self._target!r}>"


def wrap(
    client: Any,
    path: str | os.PathLike = "logs/llmshift.jsonl",
    *,
    sample_rate: float = 1.0,
    recorder: Recorder | None = None,
    metadata: dict | None = None,
    record: bool = True,
    shadow: Any = None,
    shadow_rate: float = 0.05,
    shadow_path: str | os.PathLike | None = None,
) -> Any:
    """Return a proxy of an OpenAI/Anthropic client that records every call. With
    `shadow` (a model spec or a `llmshift.shadow.Shadow`), a share of the calls is also
    copied to that model in the background (shadow mode); `record=False` keeps only that."""
    module = type(client).__module__.split(".")[0]
    provider = {"openai": "openai", "anthropic": "anthropic"}.get(module, module)
    if shadow is not None and not hasattr(shadow, "observe"):
        from .shadow import DEFAULT_PATH, Shadow

        shadow = Shadow(shadow, rate=shadow_rate, path=shadow_path or DEFAULT_PATH)
    rec = (recorder or Recorder(path, sample_rate=sample_rate, metadata=metadata)) if record else None
    return _Proxy(client, rec, provider, shadow=shadow)
