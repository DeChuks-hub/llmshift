"""Regular expressions from the user (a regex check, a JSON schema's pattern), searched
where one that backtracks without end cannot stop the run.

Python's re cannot be interrupted: ^(\\w+\\s?)*$ takes minutes on a 141-character answer it
does not match, and \\d+x seconds on a long run of digits. So a search runs in a worker
process that is killed after TIMEOUT seconds.
"""

from __future__ import annotations

import atexit
import json
import re
import subprocess
import sys
import threading
import time

TIMEOUT = 2.0  # seconds one search may take

# Says "ready", then answers each [pattern, flags, text] line with a [matched, error] line.
_WORKER = r"""
import json, re, sys
sys.stdout.write("ready\n")
sys.stdout.flush()
for line in sys.stdin:
    try:
        pattern, flags, text = json.loads(line)
        reply = [re.search(pattern, text, flags) is not None, None]
    except Exception as exc:
        reply = [None, type(exc).__name__ + ": " + str(exc)]
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
"""

# Without a repeat or an alternative, a pattern cannot backtrack: it is searched here.
_MAY_BACKTRACK_RE = re.compile(r"[*+?{|]")


class RegexStopped(Exception):
    """A search that could not finish: it ran over TIMEOUT seconds, or broke the worker."""


_lock = threading.Lock()
_worker: subprocess.Popen | None = None
_no_worker = False  # this Python cannot start one: searches run here, as before


def _start() -> subprocess.Popen | None:
    if getattr(sys, "frozen", False) or not sys.executable:
        return None  # sys.executable is the frozen app itself, or nothing at all
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, ValueError):
        return None
    timer = threading.Timer(30, proc.kill)  # a slow start is fine, no start at all is not
    timer.start()
    try:
        ready = proc.stdout.readline()
    except (OSError, ValueError):
        ready = ""
    finally:
        timer.cancel()
    if ready.strip() != "ready":
        _stop(proc)
        return None
    return proc


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    for pipe in (proc.stdin, proc.stdout):
        try:
            pipe.close()
        except Exception:
            pass


def search(pattern: str, text: str, flags: int = 0) -> bool:
    """Whether re.search(pattern, text, flags) finds a match, searched in the worker.
    Raises re.error for an invalid pattern and RegexStopped for a search that ran too long."""
    global _worker, _no_worker
    compiled = re.compile(pattern, flags)  # an invalid pattern fails here, at once
    if not _MAY_BACKTRACK_RE.search(pattern):
        return compiled.search(text) is not None
    request = json.dumps([pattern, flags, text]) + "\n"
    with _lock:
        if _worker is None and not _no_worker:
            _worker = _start()
            _no_worker = _worker is None
        proc = _worker
        if proc is None:
            return compiled.search(text) is not None
        limit = TIMEOUT + len(request) / 20_000_000  # sending a very long answer takes time too
        timer = threading.Timer(limit, proc.kill)
        start = time.monotonic()
        timer.start()
        try:
            proc.stdin.write(request)
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError):
            line = ""
        finally:
            timer.cancel()
        if not line:  # killed by the timer, or by what the search did
            _worker = None
            _stop(proc)
            took = time.monotonic() - start
            raise RegexStopped(f"stopped after {limit:.3g} s" if took >= limit * 0.9 else "the search stopped its worker process")
    matched, error = json.loads(line)
    if error is not None:
        raise RegexStopped(error)
    return bool(matched)


def stopped_message(pattern: str, exc: RegexStopped) -> str:
    """Why a pattern could not be checked, and how to fix it."""
    shown = pattern if len(pattern) <= 60 else pattern[:57] + "..."
    return (
        f"regex /{shown}/ could not be checked ({exc}): it backtracks without end on this text. "
        "A repeat inside a repeat, such as (\\w+\\s?)*, is the usual cause: write it so each "
        "character can match only one way, such as \\w+(?:\\s\\w+)*"
    )


atexit.register(lambda: _stop(_worker))
