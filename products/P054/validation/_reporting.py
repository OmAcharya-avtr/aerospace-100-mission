"""Shared output helper for the validation scripts.

Each validation script writes its raw output to ``validation/<name>.txt`` and
to standard output at the same time, so the committed evidence file and what
the operator saw cannot diverge.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class Report:
    """Tee lines to a file in validation/ and to stdout."""

    def __init__(self, name: str) -> None:
        self.path = HERE / f"{name}.txt"
        self._handle = self.path.open("w", encoding="utf-8")
        self.failures: list[str] = []
        self.line(f"# {name}")
        self.line(f"python {platform.python_version()} on {platform.platform()}")
        self.line(
            f"os.cpu_count()={os.cpu_count()} "
            f"len(os.sched_getaffinity(0))={len(os.sched_getaffinity(0))}"
        )
        import numpy
        import scipy
        import sklearn

        self.line(
            f"numpy {numpy.__version__} scipy {scipy.__version__} "
            f"scikit-learn {sklearn.__version__}"
        )
        self.line("")

    def line(self, text: str = "") -> None:
        self._handle.write(text + "\n")
        sys.stdout.write(text + "\n")

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        verdict = "PASS" if passed else "FAIL"
        self.line(f"[{verdict}] {label}{(' ' + detail) if detail else ''}")
        if not passed:
            self.failures.append(label)
        return passed

    def finish(self) -> int:
        self.line("")
        if self.failures:
            self.line(f"{len(self.failures)} CHECK(S) FAILED:")
            for failure in self.failures:
                self.line(f"  - {failure}")
        else:
            self.line("all checks passed")
        self._handle.close()
        return 1 if self.failures else 0
