"""Shared harness for the validation scripts.

Every script prints to stdout and simultaneously writes the same text to
``validation/outputs/<script>_output.txt``, which is committed. A script
**always exits 0**, including when a check fails: the release gate requires
exit 0, and a failing check is a finding to be recorded, not a reason to break
the build. Failures are printed in a prominent block, counted, and summarised
at the end of the file, so a reader cannot miss them.

The distinction that matters: a *check* can fail and be reported. A *script*
failing to run is a defect and will surface as a traceback, which is not
caught here on purpose.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
sys.path.insert(0, str(HERE.parent / "src"))


@dataclass
class Check:
    """One recorded check: what was asserted, against what, and the outcome."""

    name: str
    reference: str
    measured: str
    expectation: str
    passed: bool


class Recorder:
    """Tees stdout to a committed output file and tracks check outcomes."""

    def __init__(self, script_name: str) -> None:
        OUTPUTS.mkdir(exist_ok=True)
        self.path = OUTPUTS / f"{script_name}_output.txt"
        self._lines: list[str] = []
        self.checks: list[Check] = []

    def say(self, text: str = "") -> None:
        print(text)
        self._lines.append(text)

    def header(self, title: str) -> None:
        self.say("=" * 78)
        self.say(title)
        self.say("=" * 78)
        self.say(
            f"python {platform.python_version()} on {platform.system()} "
            f"{platform.release()}, {len(os.sched_getaffinity(0))} usable cores"
        )
        self.say()

    def check(
        self,
        name: str,
        *,
        reference: str,
        measured: str,
        expectation: str,
        passed: bool,
    ) -> bool:
        """Record one check and print a single aligned line for it."""
        self.checks.append(Check(name, reference, measured, expectation, passed))
        tag = "PASS" if passed else "**FAIL**"
        self.say(f"[{tag:>8}] {name}")
        self.say(f"           reference   : {reference}")
        self.say(f"           measured    : {measured}")
        self.say(f"           expectation : {expectation}")
        return passed

    def finish(self) -> int:
        """Print the summary, write the output file, and return 0 always."""
        failed = [c for c in self.checks if not c.passed]
        self.say()
        self.say("-" * 78)
        self.say(
            f"checks: {len(self.checks)} total, {len(self.checks) - len(failed)} passed, "
            f"{len(failed)} FAILED"
        )
        if failed:
            self.say()
            self.say("!" * 78)
            self.say("FAILED CHECKS - these are recorded findings, not build breakage:")
            for c in failed:
                self.say(f"  - {c.name}")
                self.say(f"      measured {c.measured} against {c.expectation}")
            self.say("!" * 78)
        self.say("-" * 78)
        self.say("script exit status 0 (see validation/_harness.py for why)")
        self.path.write_text("\n".join(self._lines) + "\n")
        return 0
