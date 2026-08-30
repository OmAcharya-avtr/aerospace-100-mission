"""Make ``python -m pytest tests/ -q`` work from this directory.

The package lives under ``src/`` and is not installed in the mission's build
environment. ``[tool.pytest.ini_options] pythonpath`` puts it on the path for the
test process, but NOT for child processes, and this product's CLI tests invoke
``python -m beamtwin`` in a subprocess. Exporting PYTHONPATH here covers both.

Without this, the CLI tests fail with ModuleNotFoundError while the rest of the
suite passes — which is exactly what the release gate caught on 2026-08-29.
"""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).parent / "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

_existing = os.environ.get("PYTHONPATH", "")
if SRC not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = (SRC + os.pathsep + _existing) if _existing else SRC
