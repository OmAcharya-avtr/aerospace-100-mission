"""Test count read from junit XML, which is the only count worth reporting.

A pytest configuration that collects nothing exits 0 and prints a success line.
That happened to another product in this portfolio and a fabricated test count
was carried forward for three weeks. The count below is parsed from the junit
XML element attributes, and the script records a failure if the suite collected
nothing at all.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from _harness import SRC, run

#: The junit XML is written to a temporary directory, NOT into validation/outputs/.
#: It is a single multi-kilobyte line whose test identifiers trip a secret
#: scanner's high-entropy heuristic sixteen times, and a false positive in the
#: release gate blocks sibling products. The committed evidence is this script's
#: own output, which quotes the XML's testsuite element verbatim so the source of
#: every count is visible, and the XML itself regenerates in seconds.
JUNIT = Path(tempfile.gettempdir()) / "telemdrift_junit_report.xml"


def body(report) -> None:
    import os

    repo = SRC.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["COLUMNS"] = "100"

    report.section("1. Running the suite with junit XML output")
    print(f"  $ python -m pytest tests/ -q --junitxml=$TMPDIR/{JUNIT.name}")
    JUNIT.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header",
         "-p", "no:cacheprovider", f"--junitxml={JUNIT}"],
        capture_output=True, text=True, env=env, cwd=repo, timeout=3600,
    )
    for line in res.stdout.rstrip("\n").splitlines()[-8:]:
        print(f"    {line}")
    print(f"    -> exit status {res.returncode}")

    report.section("2. Counts parsed from the junit XML")
    if not Path(JUNIT).exists():
        report.check("junit XML was written", False, "file missing")
        return
    raw = Path(JUNIT).read_text(encoding="utf-8")
    head = raw[: raw.index(">", raw.index("<testsuite ")) + 1]
    print("  The junit XML's own testsuite element, verbatim:")
    print(f"    {head[head.index('<testsuite '):]}")
    print()
    root = ET.parse(JUNIT).getroot()
    suites = root.findall("testsuite") or [root]
    total = failures = errors = skipped = 0
    time_s = 0.0
    for s in suites:
        total += int(s.get("tests", 0))
        failures += int(s.get("failures", 0))
        errors += int(s.get("errors", 0))
        skipped += int(s.get("skipped", 0))
        time_s += float(s.get("time", 0.0))
    passed = total - failures - errors - skipped
    print(f"  collected : {total}")
    print(f"  passed    : {passed}")
    print(f"  failed    : {failures}")
    print(f"  errored   : {errors}")
    print(f"  skipped   : {skipped}")
    print(f"  suite time: {time_s:.1f} s "
          "(wall clock moves 10-40 % between runs on two contended cores)")
    print()
    print("  By test module:")
    counts: dict[str, int] = {}
    for case in root.iter("testcase"):
        parts = [q for q in (case.get("classname") or "").split(".") if q]
        mod = next((q for q in parts if q.startswith("test_")), None)
        counts[mod or "unknown"] = counts.get(mod or "unknown", 0) + 1
    for mod in sorted(counts):
        print(f"    {mod:28s} {counts[mod]:4d}")

    report.check("the suite collected more than zero tests", total > 0, f"{total}")
    report.check("no test failed", failures == 0, f"{failures} failures")
    report.check("no test errored", errors == 0, f"{errors} errors")
    report.check("no test was skipped or xfailed to hide a defect",
                 skipped == 0, f"{skipped} skipped")
    report.check("pytest exited 0", res.returncode == 0, f"exit {res.returncode}")
    print()
    print(f"  The README badge must read {passed} passing. Any other number in any")
    print("  document in this repository is wrong.")
    report.finding(
        f"junit XML: {total} collected, {passed} passed, {failures} failed, "
        f"{errors} errored, {skipped} skipped. This is the authoritative count."
    )


if __name__ == "__main__":
    raise SystemExit(run("validate_tests",
                         "telemdrift 0.1.0 - test count from junit XML",
                         body))
