"""Put ``src/`` on ``sys.path`` and name the output directories.

Imported first by every example so ``python examples/<script>.py`` works from a
cold clone with no editable install.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
CASES = os.path.join(REPO_ROOT, "examples", "cases")
SCREENSHOTS = os.path.join(REPO_ROOT, "screenshots")
VALIDATION = os.path.join(REPO_ROOT, "validation")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.makedirs(SCREENSHOTS, exist_ok=True)
