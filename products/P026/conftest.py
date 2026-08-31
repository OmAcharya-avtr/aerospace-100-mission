"""Make ``src/`` importable for tests and for any subprocess a test spawns.

``[tool.pytest.ini_options] pythonpath = ["src"]`` covers the test process
itself.  A subprocess started by a test (the CLI tests use
``python -m wahbakit``) inherits the environment, not ``sys.path``, so
``PYTHONPATH`` is exported here as well.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = str((Path(__file__).parent / "src").resolve())

if SRC not in sys.path:
    sys.path.insert(0, SRC)

_existing = os.environ.get("PYTHONPATH", "")
if SRC not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = SRC + (os.pathsep + _existing if _existing else "")
