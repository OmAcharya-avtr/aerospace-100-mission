"""Audit: how many assertions depended on pytest.approx's default absolute tolerance.

The defect class
----------------

``pytest.approx(expected, rel=r)`` carries a **default absolute tolerance of
1e-12** and passes if **either** bound is met. So

    ``assert computed == pytest.approx(expected, rel=1e-15)``

on a quantity of order 1e-13 is not a 1e-15 relative check at all: any value
within 1e-12 of the expected one passes, which for a quantity of 1e-13 is
every value in the neighbourhood including zero.

This is acute for this package rather than theoretical. Reliability
complements and small failure probabilities here are routinely of order 1e-5
and go down to 1e-18, so **any relative tolerance tighter than about 2e-8 on an
unreliability figure of order 1e-5 is silently replaced by the absolute
default**, and tighter still below that.

What this script does
---------------------

It re-runs the whole test suite twice:

1. **as committed**, which should pass;
2. with ``pytest.approx`` monkey-patched so that any call passing ``rel=``
   without an explicit ``abs=`` raises, which would catch any assertion that
   relied on the default.

Run 2 does not merely tighten the tolerance -- it makes the omission itself an
error, so the audit answers "is there anywhere in this suite that could be
relying on the default" rather than "does the suite still pass when the default
is removed", which a passing assertion could satisfy by accident.

The patch is applied through a ``conftest`` plugin written into a temporary
directory, so nothing in the repository is modified and the committed suite is
unchanged. The temporary directory is created with ``tempfile`` and its path is
never printed.

The suite's own convention
--------------------------

``tests/conftest.py`` provides :func:`rel` and :func:`abs_`, which pass
``abs=0.0`` and ``rel=0.0`` respectively, and every comparison in the suite goes
through one of them. This script is the evidence that the convention is
actually followed rather than merely documented.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from _bootstrap import ROOT, Tee, Timer, bootstrap, header, relative

bootstrap()

PLUGIN = '''
"""Injected plugin: make pytest.approx reject a rel= without an explicit abs=."""

import pytest

_original = pytest.approx


def _strict_approx(expected, rel=None, abs=None, nan_ok=False):  # noqa: A002
    if rel is not None and abs is None:
        raise AssertionError(
            "pytest.approx called with rel= and no explicit abs=. The default "
            "absolute tolerance of 1e-12 would apply and the relative tolerance "
            "would be unenforced for any quantity below about 1e-10. Use the "
            "rel() helper from tests/conftest.py, which passes abs=0.0."
        )
    return _original(expected, rel=rel, abs=abs, nan_ok=nan_ok)


def pytest_configure(config):
    pytest.approx = _strict_approx
'''


def _junit_counts(path: Path) -> dict[str, int]:
    """Parse a junit XML file into counts, cross-checking the literal element count."""
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:  # pragma: no cover - malformed XML
        raise ValueError("junit XML has no testsuite element")
    declared = {
        "tests": int(suite.get("tests", "0")),
        "failures": int(suite.get("failures", "0")),
        "errors": int(suite.get("errors", "0")),
        "skipped": int(suite.get("skipped", "0")),
    }
    declared["testcase_elements"] = len(suite.findall(".//testcase"))
    return declared


def _count_source_calls(tee: Tee) -> tuple[int, int, int]:
    """Count approx-style calls in the committed test sources.

    ``tests/conftest.py`` is excluded: it is where the two helpers are defined,
    so it necessarily contains the only legitimate direct ``pytest.approx``
    calls in the repository, and its docstring quotes the defective form in
    prose. Lines containing a reStructuredText literal marker are skipped for
    the same reason -- the scan must not count the documentation of the defect
    as an instance of it.
    """
    helper_calls = 0
    direct_with_abs = 0
    direct_without_abs = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text()
        helper_calls += len(re.findall(r"\b(?:rel|abs_)\(", text))
        for line in text.splitlines():
            if "``" in line or line.lstrip().startswith("#"):
                continue
            for match in re.finditer(r"pytest\.approx\(([^\n]*)", line):
                body = match.group(1)
                if "rel=" in body and "abs=" not in body:
                    direct_without_abs += 1
                    tee(f"    {relative(path)}: pytest.approx({body[:70]}")
                elif "rel=" in body or "abs=" in body:
                    direct_with_abs += 1
    return helper_calls, direct_with_abs, direct_without_abs


def main() -> int:
    tee = Tee("audit_approx_tolerances_output.txt")
    header(tee, "pytest.approx default-absolute-tolerance audit")
    tee("")
    tee("  pytest.approx(expected, rel=r) carries a DEFAULT ABSOLUTE TOLERANCE of")
    tee("  1e-12 and passes on EITHER bound. For this package that matters: the")
    tee("  unreliability figures here are routinely of order 1e-5 and go down to")
    tee("  1e-18, so a relative tolerance tighter than about 2e-8 on a 1e-5")
    tee("  quantity is silently replaced by the absolute default.")

    timer = Timer()

    tee.rule("static count of approx-style calls in the committed suite")
    tee("")
    tee("  direct pytest.approx calls with rel= and no abs=, if any:")
    helper_calls, direct_with_abs, direct_without_abs = _count_source_calls(tee)
    if direct_without_abs == 0:
        tee("    none")
    tee("")
    tee(f"  calls through the rel()/abs_() helpers : {helper_calls}")
    tee(f"  direct pytest.approx with explicit abs= : {direct_with_abs}")
    tee(f"  direct pytest.approx with rel= and no abs= : {direct_without_abs}")
    tee("")
    tee("  tests/conftest.py is excluded from this scan: it defines the two")
    tee("  helpers and is therefore the only file that legitimately calls")
    tee("  pytest.approx directly.")
    tee("")
    tee("  The helpers in tests/conftest.py pass abs=0.0 for a relative check and")
    tee("  rel=0.0 for an absolute one, so a call through them cannot inherit the")
    tee("  default. A direct call with rel= and no abs= can, and is the thing")
    tee("  this audit is looking for.")

    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        plugin = workspace / "strict_approx_plugin.py"
        plugin.write_text(PLUGIN)
        committed_xml = workspace / "committed.xml"
        strict_xml = workspace / "strict.xml"

        tee.rule("run 1: the suite as committed")
        with timer:
            first = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "-q",
                    f"--junit-xml={committed_xml}",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": "src",
                    "MPLBACKEND": "Agg",
                    "HOME": scratch,
                },
                timeout=1500,
            )
        committed = _junit_counts(committed_xml)
        tee("")
        tee(f"  exit code                          : {first.returncode}")
        for key in ("tests", "failures", "errors", "skipped", "testcase_elements"):
            tee(f"  {key:<34s} : {committed[key]}")
        tee.require(
            "the committed suite passes",
            first.returncode == 0 and committed["failures"] == 0 and committed["errors"] == 0,
            f"exit {first.returncode}",
        )
        tee.check(
            "the junit testsuite attributes agree with the literal testcase count",
            committed["tests"] == committed["testcase_elements"],
            f"{committed['tests']} declared against "
            f"{committed['testcase_elements']} testcase elements",
        )

        tee.rule("run 2: pytest.approx rejects a rel= without an explicit abs=")
        tee("")
        tee("  The injected plugin raises on any pytest.approx call that passes")
        tee("  rel= without abs=, so an assertion relying on the default fails")
        tee("  loudly rather than quietly keeping a tolerance it does not have.")
        with timer:
            second = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/",
                    "-q",
                    "-p",
                    "strict_approx_plugin",
                    f"--junit-xml={strict_xml}",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": f"src:{scratch}",
                    "MPLBACKEND": "Agg",
                    "HOME": scratch,
                },
                timeout=1500,
            )
        strict = _junit_counts(strict_xml)
        tee("")
        tee(f"  exit code                          : {second.returncode}")
        for key in ("tests", "failures", "errors", "skipped", "testcase_elements"):
            tee(f"  {key:<34s} : {strict[key]}")
        affected = strict["failures"] + strict["errors"]
        tee("")
        tee(f"  assertions that depended on the default absolute tolerance : {affected}")
        if affected:
            tee("")
            tee("  the failing tests:")
            root = ET.parse(strict_xml).getroot()
            suite = root if root.tag == "testsuite" else root.find("testsuite")
            assert suite is not None
            for case in suite.findall(".//testcase"):
                if case.find("failure") is not None or case.find("error") is not None:
                    tee(f"    {case.get('classname')}::{case.get('name')}")
        tee.check(
            "no assertion in the suite depends on pytest.approx's default absolute tolerance",
            affected == 0,
            f"{affected} of {strict['tests']} affected",
        )
        tee.check(
            "both runs collected the same number of tests",
            committed["tests"] == strict["tests"],
            f"{committed['tests']} against {strict['tests']}",
        )

    tee.rule("the answer, stated plainly")
    tee("")
    tee(f"  tests in the suite                 : {committed['tests']}")
    tee(f"  assertions relying on the 1e-12 default : {affected}")
    tee("")
    tee("  The suite uses two helpers for every comparison:")
    tee("    rel(expected, tolerance)  -> pytest.approx(expected, rel=t, abs=0.0)")
    tee("    abs_(expected, tolerance) -> pytest.approx(expected, rel=0.0, abs=t)")
    tee("")
    tee("  so the relative checks are relative and the absolute checks are")
    tee("  absolute, and neither can be replaced by the other without the test")
    tee("  text changing.")

    tee.rule("timing, declared unstable")
    tee("")
    tee("  | quantity | value | stability |")
    tee("  |---|---|---|")
    tee(f"  | two full suite runs | {timer.elapsed_s:.2f} s | UNSTABLE, shared hardware |")

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
