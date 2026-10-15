"""Shared helpers for the validation scripts.

Layout convention, binding for this portfolio from Batch 07: each script
writes ``validation/<script-name>_output.txt`` beside itself, and every number
quoted in ``validation/VALIDATION.md`` comes from one of those files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CASES = ROOT / "examples" / "cases"
FIXTURES = ROOT / "fixtures"
VALIDATION = ROOT / "validation"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class Emitter:
    """Collect report lines, print them, and write them to the output file."""

    def __init__(self, script: str) -> None:
        self.path = VALIDATION / f"{Path(script).stem}_output.txt"
        self.lines: list[str] = []
        self.failures: list[str] = []

    def __call__(self, text: str = "") -> None:
        """Record and print one line."""
        self.lines.append(text)
        print(text)

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        """Record a pass or fail verdict. A failure is reported, never hidden."""
        verdict = "PASS" if passed else "FAIL"
        if not passed:
            self.failures.append(label)
        suffix = f"  {detail}" if detail else ""
        self(f"  [{verdict}] {label}{suffix}")
        return passed

    def finish(self) -> int:
        """Write the output file and return a process exit code."""
        self()
        if self.failures:
            self(f"RESULT: {len(self.failures)} CHECK(S) FAILED")
            for label in self.failures:
                self(f"  failed: {label}")
        else:
            self("RESULT: every check in this script passed")
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return 1 if self.failures else 0
