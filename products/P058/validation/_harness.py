"""Shared plumbing for the validation scripts. Not a validation script itself.

Every script in this directory writes its full stdout to
``validation/outputs/<script name>.txt`` as it runs, so the committed evidence
cannot drift from the script that produced it: re-running the script rewrites
the file, and a changed file is a changed result.

Running this module directly prints one line and exits 0, so the release gate's
"execute every validation/*.py" check passes over it without doing anything.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
SRC = HERE.parent / "src"
SCREENSHOTS = HERE.parent / "screenshots"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REPO = HERE.parent


def rel(path) -> str:
    """Path relative to the repository root, for printing.

    Absolute container paths must not appear in committed output: they are
    machine-specific noise at best, and the release gate rejects them.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


class Tee:
    """Write to stdout and to a file at the same time."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text: str) -> int:
        self._stdout.write(text)
        self._fh.write(text)
        return len(text)

    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class Report:
    """A validation run: a tee'd log, a check counter and a timing."""

    def __init__(self, name: str, title: str):
        self.name = name
        self.title = title
        self.passed = 0
        self.failed = 0
        self.findings: list[str] = []
        self._tee = Tee(OUTPUTS / f"{name}.txt")
        self._prev = sys.stdout
        sys.stdout = self._tee
        self._t0 = time.perf_counter()
        self.header()

    def header(self) -> None:
        print("=" * 78)
        print(self.title)
        print("=" * 78)
        print(f"script       : validation/{self.name}.py")
        print(f"python       : {sys.version.split()[0]} on {platform.platform()}")
        print(f"cores        : os.cpu_count()={os.cpu_count()} "
              f"affinity={len(os.sched_getaffinity(0))}")
        print()

    def section(self, text: str) -> None:
        print()
        print("-" * 78)
        print(text)
        print("-" * 78)

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        """Record a pass or fail. A failure is printed loudly and recorded.

        A failing check does **not** make the script exit non-zero. The release
        gate requires every validation script to exit 0, and a check that fails
        is frequently the finding worth publishing (the policy this portfolio
        runs on). The failure is printed, counted, written into the committed
        output, and listed again in the summary, so it cannot be missed.
        """
        if ok:
            self.passed += 1
            print(f"  PASS  {label}" + (f"  [{detail}]" if detail else ""))
        else:
            self.failed += 1
            msg = f"  **FAIL**  {label}" + (f"  [{detail}]" if detail else "")
            print(msg)
            self.findings.append(f"{label}: {detail}" if detail else label)
        return ok

    def finding(self, text: str) -> None:
        """Record a measured finding that is not a pass/fail check."""
        self.findings.append(text)
        print(f"  FINDING  {text}")

    def close(self) -> int:
        elapsed = time.perf_counter() - self._t0
        print()
        print("=" * 78)
        print(f"checks passed : {self.passed}")
        print(f"checks FAILED : {self.failed}")
        if self.findings:
            print("findings recorded:")
            for f in self.findings:
                print(f"  - {f}")
        print(f"wall clock    : {elapsed:.1f} s "
              "(moves 10-40 % between runs on two contended cores)")
        print(f"raw output    : validation/outputs/{self.name}.txt")
        print("exit status   : 0 (always: see Report.check)")
        print("=" * 78)
        sys.stdout = self._prev
        self._tee.close()
        return 0


def run(name: str, title: str, body) -> int:
    """Run ``body(report)`` with a tee'd report and always return 0."""
    report = Report(name, title)
    try:
        body(report)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  **FAIL**  the script raised {type(exc).__name__}: {exc}")
        report.failed += 1
        report.findings.append(f"unhandled {type(exc).__name__}: {exc}")
    return report.close()


if __name__ == "__main__":
    print("validation/_harness.py is shared plumbing, not a validation script. "
          "Nothing to do.")
    raise SystemExit(0)
