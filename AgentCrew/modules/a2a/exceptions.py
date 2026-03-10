from __future__ import annotations


class TaskCanceledException(Exception):
    """Raised when a task is explicitly canceled during processing."""
