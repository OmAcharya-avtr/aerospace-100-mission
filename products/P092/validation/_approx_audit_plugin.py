"""A pytest plugin that makes a bare ``rel=`` call to ``pytest.approx`` an error.

``pytest.approx(expected, rel=r)`` passes if EITHER the relative or the absolute
bound is met, and the absolute default is ``1e-12``. For any quantity of order
1e-5 that silently replaces every relative tolerance tighter than 2e-8, and the
assertion no longer tests what it says. The audit is to re-run the suite with
the default forced off and report the COUNT of assertions that depended on it.

Collected by ``validate_approx_audit.py``, which runs pytest in a subprocess
with this plugin loaded. The plugin records every call site rather than failing
the run, so the audit reports a number rather than a stack trace.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import pytest

_ORIGINAL = pytest.approx
_RECORD: list[dict] = []
_OUTPUT = os.environ.get("DEADLINERISK_APPROX_AUDIT", "")


def _caller() -> str:
    for frame in reversed(traceback.extract_stack()):
        if "/tests/" in frame.filename.replace(os.sep, "/"):
            return f"{Path(frame.filename).name}:{frame.lineno}"
    return "unknown"


def _audited(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002 -- pytest's name
    """``pytest.approx`` with every bare ``rel=`` recorded and ``abs`` forced to 0."""
    if rel is not None and abs is None:
        _RECORD.append({"site": _caller(), "rel": rel, "forced_abs": 0.0})
        return _ORIGINAL(expected, rel=rel, abs=0.0, nan_ok=nan_ok)
    return _ORIGINAL(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def pytest_configure(config) -> None:  # noqa: ARG001
    pytest.approx = _audited


def pytest_unconfigure(config) -> None:  # noqa: ARG001
    pytest.approx = _ORIGINAL
    if _OUTPUT:
        Path(_OUTPUT).write_text(json.dumps(_RECORD, indent=2) + "\n")
