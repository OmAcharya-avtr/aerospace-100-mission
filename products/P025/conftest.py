"""Ensure ``src/`` is importable, including for tests that spawn subprocesses."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = str((Path(__file__).parent / "src").resolve())

if SRC not in sys.path:
    sys.path.insert(0, SRC)

_existing = os.environ.get("PYTHONPATH", "")
if SRC not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = SRC if not _existing else SRC + os.pathsep + _existing
