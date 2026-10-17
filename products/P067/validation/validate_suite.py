"""Run the test suite and count the tests from the junit XML, never from stdout.

    python validation/validate_suite.py > validation/validate_suite_output.txt

A pytest configuration that collects nothing prints success and exits 0. The
only honest count comes from the junit XML, and it is cross-checked here
against a count of ``testcase`` elements, because the ``tests`` attribute and
the element count are produced by different code paths in pytest.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from _common import ROOT, rule


def main() -> int:
    """Run pytest, parse the junit XML and print both counts."""
    out = sys.stdout
    rule(out, "acqsearch test suite, counted from junit XML")
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", f"--junitxml={junit}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        out.write("command           python -m pytest tests/ -q --junitxml=<tmp>/junit.xml\n")
        # Relative to the repository root: committed output must not embed the
        # build machine's filesystem layout (release-gate check 8).
        out.write(f"working directory {ROOT.relative_to(ROOT.parent.parent)}\n")
        out.write(f"exit code         {completed.returncode}\n")
        stdout_tail = [line for line in completed.stdout.splitlines() if line.strip()]
        out.write(f"stdout last line  {stdout_tail[-1] if stdout_tail else '(empty)'}\n\n")
        if not junit.exists():
            out.write("junit XML was not written; the count cannot be trusted\n")
            return 1
        root = ET.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        attributes = dict(suite.attrib)
        elements = len(suite.findall(".//testcase"))
    out.write("junit testsuite attributes\n")
    for key in ("tests", "failures", "errors", "skipped", "time"):
        out.write(f"  {key:<10}{attributes.get(key, '(absent)')}\n")
    out.write(f"\ntestcase elements counted independently: {elements}\n")
    agree = int(attributes.get("tests", -1)) == elements
    out.write(f"attribute and element count agree:       {agree}\n")
    clean = (
        agree
        and int(attributes.get("failures", 1)) == 0
        and int(attributes.get("errors", 1)) == 0
        and completed.returncode == 0
        and elements > 0
    )
    rule(out)
    out.write(f"suite clean: {clean}\n")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
