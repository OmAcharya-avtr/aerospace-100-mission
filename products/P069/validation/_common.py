"""Shared paths for the validation scripts.

Every script in this directory is run from the product root as

    python validation/<name>.py > validation/<name>_output.txt

and the committed ``_output.txt`` files are the raw output of exactly that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# A validation script must produce the same result from ANY working directory.
# The scripts here quote README snippets verbatim, and those snippets use paths
# relative to the product root because that is how a reader runs them. Anchoring
# the working directory on __file__ makes that documented assumption true instead
# of merely assumed: the release gate runs these scripts with cwd=validation/,
# which is how three products in Batch 07 failed the gate on 2026-10-16 after
# passing every check that happened to run them from the product root.
os.chdir(ROOT)


def rule(title: str) -> str:
    """A section header used by every script, so the outputs read alike."""
    return f"\n{title}\n" + "-" * len(title)
