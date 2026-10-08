"""Put ``src/`` on ``sys.path`` so the validation scripts run from a cold clone.

Importing this module is the first statement of every script in this directory.
It exists so that ``python validation/<script>.py`` works without an editable
install, which keeps the committed raw output reproducible by anyone.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
EXAMPLES = os.path.join(REPO_ROOT, "examples", "cases")
SCREENSHOTS = os.path.join(REPO_ROOT, "screenshots")

if SRC not in sys.path:
    sys.path.insert(0, SRC)
