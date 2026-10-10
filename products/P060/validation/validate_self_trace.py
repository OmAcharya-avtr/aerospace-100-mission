"""Run this package's own suite, count it from junit XML, and audit itself.

Two things happen here and both matter.

1. The test count is read from the junit XML, never from pytest's stdout
   summary line. A suite that collects nothing prints a success line and
   exits 0, which is how a sibling product in this portfolio shipped with an
   empty suite. The count below is `tests`, `failures`, `errors` and
   `skipped` taken from the `<testsuite>` element, cross-checked against the
   outcomes counted from the individual `<testcase>` elements by this
   package's own parser.

2. The package audits its own `docs/REQUIREMENTS.md` against that report.
   This is the worked example on a real repository rather than on a fixture.

Exits 0. The audit's own exit status is captured and printed; it is not this
script's exit status.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import audit, parse_junit_xml, parse_requirements_file, render_text  # noqa: E402

FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<54s} got {got!r}   expected {expected!r}")


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    with tempfile.TemporaryDirectory() as scratch:
        xml_path = Path(scratch) / "junit.xml"
        print("1. running the suite")
        print(f"   python -m pytest tests/ -q --junitxml={xml_path.name}")
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider",
             f"--junitxml={xml_path}"],
            cwd=ROOT, capture_output=True, text=True, env=env, timeout=1800, check=False,
        )
        summary = [ln for ln in completed.stdout.splitlines() if ln.strip()][-1:]
        print(f"   pytest exit status      : {completed.returncode}")
        print(f"   pytest stdout last line : {summary[0] if summary else '(none)'}")
        print("   (that line is NOT the number reported below)")

        print()
        print("2. counts read from the junit XML <testsuite> element")
        root = ET.parse(xml_path).getroot()
        suite = root if root.tag == "testsuite" else root[0]
        tests = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped_attr = int(suite.get("skipped", 0))
        print(f"   tests    = {tests}")
        print(f"   failures = {failures}")
        print(f"   errors   = {errors}")
        print(f"   skipped  = {skipped_attr}   (pytest counts xfailed cases here too)")
        passed = tests - failures - errors - skipped_attr
        print(f"   passed   = tests - failures - errors - skipped = {passed}")
        check("suite collected more than zero tests", tests > 0, True)
        check("zero failures", failures, 0)
        check("zero errors", errors, 0)

        print()
        print("3. the same report through this package's own parser")
        report = parse_junit_xml(xml_path, relative_to=ROOT)
        counts = report.outcome_counts
        print(f"   parsed <testcase> elements : {len(report.cases)}")
        print(f"   passed                     : {counts['passed']}")
        print(f"   failed                     : {counts['failed']}")
        print(f"   errored                    : {counts['errored']}")
        print(f"   skipped                    : {counts['skipped']}")
        print(f"   xfailed                    : {counts['xfailed']}")
        check("case count matches the header", len(report.cases), tests)
        check("failed matches the header", counts["failed"], failures)
        check("errored matches the header", counts["errored"], errors)
        check("skipped + xfailed matches the header",
              counts["skipped"] + counts["xfailed"], skipped_attr)
        check("passed matches the header arithmetic", counts["passed"], passed)

        print()
        print("4. auditing this package's own requirements against that report")
        document = parse_requirements_file(ROOT / "docs" / "REQUIREMENTS.md",
                                           relative_to=ROOT)
        result = audit(document, report, test_root=ROOT / "tests")
        print()
        print(render_text(result))
        print()
        check("declared requirements", result.matrix.n_requirements, 26)
        check("findings", len(result.findings), 0)
        check("audit exit status", result.exit_code, 0)
        check("nominal coverage numerator/denominator",
              (result.coverage[0].numerator, result.coverage[0].denominator), (26, 26))
        check("passing coverage numerator/denominator",
              (result.coverage[2].numerator, result.coverage[2].denominator), (26, 26))
        print(f"   claiming test cases: {sum(1 for c in report.cases if c.claims)}"
              f" of {len(report.cases)}")

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
