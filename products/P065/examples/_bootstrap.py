"""Make ``src/`` importable when an example is run from a cold clone.

``pip install -e .`` makes this unnecessary. It is here so that
``python examples/<name>.py`` works immediately after ``git clone``, which is
what the README's first-run block does.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CASES = ROOT / "examples" / "cases"
FIXTURES = ROOT / "fixtures"
SCREENSHOTS = ROOT / "screenshots"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
