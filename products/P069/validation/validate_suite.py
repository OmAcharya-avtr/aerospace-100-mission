"""Run the test suite and count the tests from the junit XML, never from stdout.

    python validation/validate_suite.py > validation/validate_suite_output.txt

A pytest configuration that collects zero tests prints success and exits 0.
That is how one product in this portfolio shipped with a recorded 251 passing
tests and a configuration that collected none. This script therefore reads the
``testsuite`` element's attributes AND independently counts the ``testcase``
elements, and fails if the two disagree or if either is zero.

The junit file is written to the product root as ``junit.xml``, which this
product's ``.gitignore`` excludes with an ANCHORED ``/junit.xml`` pattern.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET

from _common import ROOT, rule


def main() -> int:
    """Run pytest, parse the junit XML two ways and compare."""
    out = sys.stdout
    junit = ROOT / "junit.xml"
    argv = [sys.executable, "-m", "pytest", "tests/", "-q", f"--junitxml={junit}"]
    out.write("test suite, counted from junit XML\n")
    out.write("=" * 78 + "\n")
    out.write("command: python -m pytest tests/ -q --junitxml=junit.xml\n")
    result = subprocess.run(
        argv, cwd=ROOT, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    out.write(rule("pytest stdout, last line, NOT the source of the counts") + "\n")
    tail = [line for line in result.stdout.strip().split("\n") if line.strip()]
    out.write((tail[-1] if tail else "(no output)") + "\n")
    out.write(f"exit status: {result.returncode}\n")

    out.write(rule("junit XML, the source of the counts") + "\n")
    if not junit.is_file():
        out.write("FAIL  no junit.xml was written\n")
        return 1
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    attrs = {k: int(suite.get(k, "0")) for k in ("tests", "failures", "errors", "skipped")}
    elements = len(suite.findall(".//testcase"))
    for key, value in attrs.items():
        out.write(f"  testsuite @{key:<10}{value}\n")
    out.write(f"  testcase elements  {elements}\n")
    out.write(f"  wall clock         {float(suite.get('time', '0')):.2f} s\n")

    failures = 0
    checks = (
        ("testsuite @tests equals the testcase element count", attrs["tests"] == elements),
        ("the suite collected more than zero tests", elements > 0),
        ("no failures", attrs["failures"] == 0),
        ("no errors", attrs["errors"] == 0),
        ("no skips", attrs["skipped"] == 0),
        ("pytest exited 0", result.returncode == 0),
    )
    out.write(rule("verdict") + "\n")
    for label, ok in checks:
        failures += 0 if ok else 1
        out.write(f"  {'PASS' if ok else 'FAIL'}  {label}\n")
    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
