"""Make ``src/`` importable for tests, including any that spawn a subprocess.

``[tool.pytest.ini_options] pythonpath = ["src"]`` covers in-process imports,
but a test that runs ``python -m slewforge`` in a subprocess gets a fresh
interpreter that never sees it, so PYTHONPATH is exported here as well. The
same export is what lets ``slewforge.dataset`` label problems in worker
processes.
"""

import os
import sys
from pathlib import Path

SRC = str(Path(__file__).parent / "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = SRC + (os.pathsep + existing if existing else "")
