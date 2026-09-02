"""Make ``src/`` importable for tests and for any subprocess a test spawns.

``[tool.pytest.ini_options] pythonpath = ["src"]`` covers the in-process case; the
environment variable below covers the CLI tests, which run ``python -m momentummgr`` in a
child process that does not inherit pytest's sys.path manipulation.
"""

from __future__ import annotations

import os
import pathlib
import sys

SRC = str(pathlib.Path(__file__).parent / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = SRC + (os.pathsep + existing if existing else "")
