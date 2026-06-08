# -*- coding: utf-8 -*-
"""
Timing utilities for Flow-Py.

Records are collected process-locally and flushed to a per-tile CSV at the
end of each tile calculation.  main.py merges all tile CSVs into one master
timings.csv so results are ready for pandas analysis.

Typical usage
-------------
# In the pool initializer (once per worker process):
    timers.set_run_context(version="1.0", git_hash="abc1234")

# In flow_core.calculation(), around specific blocks:
    with timed("tile IO"):
        ...

# On functions:
    @timeit()
    def my_func(...): ...

    @timeit("friendly name")
    def my_func(...): ...

# At the end of calculation(), flush collected records:
    timers.flush_to_csv(path)
"""

import csv
import functools
import os
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Process-local state
# ---------------------------------------------------------------------------

_records: list[dict] = []   # accumulated per process, flushed after each tile
_context: dict = {}         # version, git_hash, tile_i, tile_j, worker – set at runtime


def set_run_context(**kwargs) -> None:
    """Update the shared context that is stamped on every timing record."""
    _context.update(kwargs)


def _record(function: str, duration_ms: float) -> None:
    _records.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_context,
        "function": function,
        "duration_ms": round(duration_ms, 3),
    })


def flush_to_csv(path: str) -> None:
    """Write all pending records to *path* (append) and clear the local list."""
    if not _records:
        return
    fieldnames = list(_records[0].keys())
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(_records)
    _records.clear()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def timeit(name: str | None = None):
    """Decorator that records the wall-clock duration of a function call."""
    def decorator(fn):
        label = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _record(label, (time.perf_counter() - t0) * 1000)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class timed:
    """Context manager that records the wall-clock duration of a block."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc):
        _record(self.label, (time.perf_counter() - self._t0) * 1000)
