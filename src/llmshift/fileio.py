"""Appending whole lines to a file that several processes may write at once.

O_APPEND alone is not enough on Windows, where the C runtime seeks to the end and
then writes, so two processes can overwrite each other's lines. Every append here
holds an exclusive OS lock: flock on Linux and macOS, and on Windows a lock on a
byte far past the end of the file, so the lock never blocks reading or writing the
data itself.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

LOCK_TIMEOUT = 15.0  # seconds to wait for other writers
_LOCK_OFFSET = 1 << 40

if os.name == "nt":
    import msvcrt

    def _lock(fd: int) -> None:
        deadline = time.monotonic() + LOCK_TIMEOUT
        while True:
            os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.002)

    def _unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def append(path: str | os.PathLike, data: bytes) -> None:
    """Append `data` (complete lines) without interleaving with other writers."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        _lock(fd)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def ends_with_newline(path: str | os.PathLike) -> bool:
    """True for a missing or empty file, or one whose last byte is a newline."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                return True
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"
    except OSError:
        return True
