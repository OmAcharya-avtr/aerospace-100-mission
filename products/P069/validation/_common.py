"""Shared paths for the validation scripts.

Every script in this directory is run from the product root as

    python validation/<name>.py > validation/<name>_output.txt

and the committed ``_output.txt`` files are the raw output of exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def rule(title: str) -> str:
    """A section header used by every script, so the outputs read alike."""
    return f"\n{title}\n" + "-" * len(title)
