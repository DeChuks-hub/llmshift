"""Bring-your-own providers.

* `python:my_app.llm:answer` calls `answer(request)` in your code, so you can test
  your real pipeline (prompt building, post-processing) instead of a raw model.
  A file path works too: `python:scripts/call_model.py:run`. `async def` works.
* A `command` provider (configured in llmshift.yaml) runs a program, writes the
  request as JSON to its stdin and reads the answer (text or JSON) from stdout.

The function or program receives {"model": ..., "messages": [...], "params": {...}}
and may return a string, {"text": ..., "tool_calls": [...]}, or a raw OpenAI /
Anthropic / Gemini response object.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..textutil import stable_hash
from . import http
from .base import Provider, ProviderError, Response
from .mock import coerce_response

_RUNNING: set = set()  # live command processes, killed on Ctrl+C
_RUNNING_LOCK = threading.Lock()


def resolve_awaitable(result: Any) -> Any:
    """Run `async def` results to completion (the runner calls providers from worker threads)."""
    if not inspect.isawaitable(result):
        return result

    async def wait() -> Any:
        return await result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(wait())
    # Called from inside a running event loop (e.g. a notebook): use a separate thread.
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = asyncio.run(wait())
        except BaseException as exc:  # re-raised in the calling thread
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _file_digest(path: Any) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except (OSError, TypeError, ValueError):
        return None


class PythonProvider(Provider):
    kind = "python"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "python", options=options)
        target = model
        if ":" not in target:
            raise ProviderError(f"python provider needs module:function, got {target!r}", code="invalid_spec")
        module_ref, func_name = target.rsplit(":", 1)
        try:
            if module_ref.endswith(".py") or "/" in module_ref or "\\" in module_ref:
                path = Path(module_ref).resolve()
                if not path.is_file():
                    raise ProviderError(f"file not found: {path}", code="invalid_spec")
                module_name = f"llmshift_user_{path.stem}_{stable_hash(str(path), 8)}"
                module = sys.modules.get(module_name)
                if module is None:
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    if spec is None or spec.loader is None:
                        raise ProviderError(f"cannot import {path}", code="invalid_spec")
                    module = importlib.util.module_from_spec(spec)
                    # Registered first, as a normal import would, so dataclasses, pickle and
                    # typing.get_type_hints can find the module while it runs.
                    sys.modules[module_name] = module
                    try:
                        spec.loader.exec_module(module)
                    except BaseException:
                        sys.modules.pop(module_name, None)
                        raise
            else:
                cwd = os.getcwd()
                if cwd not in sys.path:
                    sys.path.insert(0, cwd)
                module = importlib.import_module(module_ref)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"cannot import {module_ref!r}: {exc}", code="invalid_spec") from None
        fn = getattr(module, func_name, None)
        if not callable(fn):
            raise ProviderError(f"{module_ref!r} has no callable {func_name!r}", code="invalid_spec")
        self.fn = fn
        # Cached answers are only reused while the code that produced them is unchanged.
        self._code = _file_digest(getattr(module, "__file__", None))

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("timeout", "max_retries")}
        return {"kind": self.kind, "model": self.model, "code": self._code, "options": stable_hash(shaping, 16)}

    def _complete(self, request: dict) -> Response:
        payload = copy.deepcopy(request)
        payload.setdefault("model", self.options.get("target_model"))
        return coerce_response(resolve_awaitable(self.fn(payload)), default_model=self.model)


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a command and everything it started (a shell's children would otherwise linger)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, timeout=15)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _reap(proc: subprocess.Popen) -> None:
    try:
        proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass  # a grandchild still holds the pipes; it has been sent a kill already


def kill_running() -> None:
    """Stop every command still running (used when the user presses Ctrl+C)."""
    with _RUNNING_LOCK:
        procs = list(_RUNNING)
    for proc in procs:
        _kill_tree(proc)


class CommandProvider(Provider):
    kind = "command"

    def __init__(self, model: str, *, name: str | None = None, options: dict | None = None):
        super().__init__(model, name=name or "command", options=options)
        self.command = self.options.get("command")
        if not self.command:
            raise ProviderError(f"provider {self.name!r} has type 'command' but no 'command' setting", code="invalid_spec")
        if not isinstance(self.command, (str, list)) or (isinstance(self.command, list) and not all(isinstance(c, str) for c in self.command)):
            raise ProviderError(f"provider {self.name!r}: 'command' must be a string or a list of strings", code="invalid_spec")

    def identity(self) -> dict:
        shaping = {k: v for k, v in self.options.items() if k not in ("timeout", "max_retries")}
        return {"kind": self.kind, "model": self.model, "command": self.command, "options": stable_hash(shaping, 16)}

    def _complete(self, request: dict) -> Response:
        # ASCII JSON reads correctly whatever encoding the program assumes for stdin.
        payload = json.dumps({"model": self.model, **request}, ensure_ascii=True, default=str)
        env = dict(os.environ)
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        env.update({str(k): str(v) for k, v in (self.options.get("env") or {}).items()})
        # Its own process group on POSIX, so a timeout can kill a shell and its children.
        kwargs: dict = {"start_new_session": True} if os.name != "nt" else {}
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=isinstance(self.command, str),
                cwd=self.options.get("cwd") or None,
                env=env,
                **kwargs,
            )
        except OSError as exc:
            raise ProviderError(f"cannot run command: {exc}", code="command_error") from None
        with _RUNNING_LOCK:
            _RUNNING.add(proc)
        deadline = time.monotonic() + self.timeout
        sent: str | None = payload
        try:
            while True:
                try:
                    stdout, stderr = proc.communicate(sent, timeout=max(0.05, min(0.5, deadline - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    sent = None  # communicate() has already written the input
                    if http.CANCEL.is_set():
                        _kill_tree(proc)
                        _reap(proc)
                        raise ProviderError("cancelled", code="cancelled") from None
                    if time.monotonic() >= deadline:
                        _kill_tree(proc)
                        _reap(proc)
                        raise ProviderError(f"command timed out after {self.timeout:.0f}s", code="timeout", retryable=True) from None
                except BrokenPipeError:
                    sent = None  # the program exited without reading its input
        finally:
            with _RUNNING_LOCK:
                _RUNNING.discard(proc)
        if proc.returncode != 0:
            raise ProviderError(f"command exited with {proc.returncode}: {(stderr or '').strip()[:300]}", code="command_error")
        text = (stdout or "").lstrip("﻿")
        out = text.strip()
        try:
            data = json.loads(out)
        except ValueError:
            return Response(text=text.rstrip("\n"), model=self.model, finish_reason="stop")
        if isinstance(data, (dict, str)):
            return coerce_response(data, default_model=self.model)
        return Response(text=out, model=self.model, finish_reason="stop")
