"""Path bootstrap and output tee shared by the validation scripts and examples.

The package is importable from ``src`` without an install step, matching the
``pythonpath = ["src"]`` setting in ``pyproject.toml`` that lets
``python -m pytest tests/ -q`` run from a cold clone.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    """Insert ``<repo>/src`` at the front of ``sys.path`` and return the repo root."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


class Tee:
    """Collect report lines, print them, and write them to a file at the end.

    Every validation script in this directory writes its raw output beside
    itself as ``<script stem>_output.txt`` so that every number in README.md and
    validation/VALIDATION.md can be traced to a file produced by a script that
    was actually run.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self) -> Path:
        self.path.write_text("\n".join(self.lines) + "\n")
        return self.path
