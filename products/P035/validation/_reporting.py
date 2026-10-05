"""Shared output helper for the validation scripts.

Each validation script prints its report to stdout and writes the identical text
to ``validation/<name>_output.txt``, so the committed raw output and what a
reader sees when they rerun the script cannot drift apart.

Two kinds of result are distinguished, and the distinction is load-bearing.

:meth:`Report.check`
    A property this package must satisfy.  A failure is a defect here, is
    printed as ``FAILED``, and makes the script exit non-zero.

:meth:`Report.characterise`
    A measurement of how far a *published approximation* departs from a
    higher-accuracy reference, against a band stated before the measurement.
    Exceeding the band is a fact about that approximation's validity range, not
    a defect in this package, so it is printed as ``FAILED``, listed separately
    in the summary, and carried into ``VALIDATION.md`` and the README -- but it
    does not change the exit status, because the package is not free to fix
    someone else's closed form.  No band is ever widened to turn one of these
    into a pass; the measured number is reported as it came out.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy
import scipy
import sklearn


class Report:
    """Collects report lines, prints them, and writes them to a text file."""

    def __init__(self, name: str, title: str) -> None:
        self.name = name
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.characterisation_failures: list[str] = []
        self._t0 = time.perf_counter()
        self.line("=" * 78)
        self.line(title)
        self.line("=" * 78)
        self.line(f"script          : validation/{name}.py")
        self.line(f"python          : {platform.python_version()} on {platform.platform()}")
        self.line(
            f"numpy / scipy / scikit-learn : {numpy.__version__} / "
            f"{scipy.__version__} / {sklearn.__version__}"
        )
        self.line("")

    def line(self, text: str = "") -> None:
        """Append one line to the report."""
        self.lines.append(text)
        print(text, flush=True)

    def section(self, title: str) -> None:
        """Append a section heading."""
        self.line("")
        self.line("-" * 78)
        self.line(title)
        self.line("-" * 78)

    def check(self, label: str, passed: bool, detail: str = "") -> bool:
        """Record a named check.  A failure is printed as FAILED and remembered."""
        verdict = "PASS" if passed else "FAILED"
        self.line(f"[{verdict}] {label}" + (f"  {detail}" if detail else ""))
        if not passed:
            self.failures.append(label)
        return passed

    def characterise(self, label: str, within_band: bool, detail: str = "") -> bool:
        """Record a characterisation of a published approximation's error.

        Printed as ``FAILED`` when the stated band is exceeded and listed in the
        summary, but the exit status is unaffected -- see the module docstring.
        """
        verdict = "PASS" if within_band else "FAILED"
        self.line(f"[{verdict}] (characterisation) {label}" + (f"  {detail}" if detail else ""))
        if not within_band:
            self.characterisation_failures.append(label)
        return within_band

    def finish(self) -> int:
        """Write the output file and return a process exit status."""
        elapsed = time.perf_counter() - self._t0
        self.line("")
        self.line("-" * 78)
        self.line(f"elapsed: {elapsed:.1f} s on this machine")
        if self.failures:
            self.line(f"FAILED CHECKS ({len(self.failures)}):")
            for name in self.failures:
                self.line(f"  - {name}")
        else:
            self.line("all checks passed")
        if self.characterisation_failures:
            self.line("")
            self.line(
                f"OUT-OF-BAND CHARACTERISATIONS ({len(self.characterisation_failures)}) -- "
                f"measured error of a published approximation, outside the band stated"
            )
            self.line(
                "above.  These are properties of that approximation's validity range, not "
                "defects"
            )
            self.line(
                "in this package, so they do not change the exit status.  They are carried "
                "into"
            )
            self.line("VALIDATION.md and the README as measured:")
            for name in self.characterisation_failures:
                self.line(f"  - {name}")
        self.line("-" * 78)
        out = Path(__file__).resolve().parent / f"{self.name}_output.txt"
        out.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"wrote {out}", flush=True)
        return 1 if self.failures else 0
