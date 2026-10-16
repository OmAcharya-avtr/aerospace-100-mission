"""Record the environment and count the test suite from junit XML.

The count is parsed from the ``testsuite`` element AND cross-checked against
a count of ``testcase`` elements. A pytest configuration that collects zero
tests prints success and exits 0; that defect shipped elsewhere in this
portfolio, and reading the summary line rather than the XML is how it
happened.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from _support import ROOT, Emitter


def main() -> int:
    """Run the suite, parse the junit XML, and report."""
    out = Emitter(__file__)
    out("budgetroll validation: environment and test count")
    out("=" * 72)
    out(f"python      {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    for module in ("numpy", "yaml", "pytest", "hypothesis", "matplotlib"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "unknown")
        except ImportError:
            version = "NOT INSTALLED"
        out(f"{module:<12s}{version}")
    out()

    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        start = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", f"--junitxml={junit}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        wall = time.monotonic() - start
        out(f"pytest exit status {completed.returncode}, wall clock {wall:.2f} s")
        if not junit.exists():
            out("no junit XML was written; the run cannot be counted")
            out.check("junit XML written", False)
            return out.finish()
        tree = ET.parse(junit)
        root = tree.getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        attrs = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
        cases = len(list(suite.iter("testcase")))
        out()
        out("counted from the junit XML, not from pytest's summary line:")
        for key, value in attrs.items():
            out(f"  testsuite/@{key:<9s} {value}")
        out(f"  testcase elements  {cases}")
        out()
        out.check(
            "testsuite/@tests matches the testcase element count",
            attrs["tests"] == cases,
            f"{attrs['tests']} against {cases}",
        )
        out.check("the suite collected tests", attrs["tests"] > 0, f"{attrs['tests']} tests")
        out.check("no failures", attrs["failures"] == 0, str(attrs["failures"]))
        out.check("no errors", attrs["errors"] == 0, str(attrs["errors"]))
        out.check("no skips", attrs["skipped"] == 0, str(attrs["skipped"]))
        out.check("pytest exited 0", completed.returncode == 0, str(completed.returncode))
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
