"""Shared helpers for the validation scripts (path setup and output tee)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class Tee:
    """Write to stdout and to ``<script>_output.txt`` at the same time."""

    def __init__(self, script_path: str) -> None:
        stem = Path(script_path).stem
        self.path = Path(script_path).resolve().parent / f"{stem}_output.txt"
        self._fh = open(self.path, "w", encoding="utf-8")

    def __enter__(self) -> Tee:
        return self

    def __exit__(self, *exc) -> None:
        self._fh.close()

    def __call__(self, line: str = "") -> None:
        print(line)
        self._fh.write(line + "\n")
        self._fh.flush()
