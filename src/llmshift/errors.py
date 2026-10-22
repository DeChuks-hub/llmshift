"""Errors and exit codes shared by the command-line modules."""

from __future__ import annotations

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130


class UsageError(Exception):
    """A problem with the command line or the project setup; shown without a traceback (exit 2)."""
