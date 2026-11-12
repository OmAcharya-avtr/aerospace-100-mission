"""Re-run the property tests at a deeper example count and report the number.

The Hypothesis example database is disabled for this repository: a falsifying
example found by one run is lost on a fresh clone, so every counterexample found
while building was pinned as an explicit regression test with its literal inputs
in ``tests/test_regressions.py``. What the database cannot give, a deeper run
can, so this script runs the property suite at a declared higher example count
in a subprocess and reports what it found, including nothing.

The count is a number someone executed, not a claim: it comes from the junit XML
of the run below.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import subprocess  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

#: Examples per property for the deep run. 2000 is about ten times the suite's
#: default and completes in well under a minute on two cores, inside the release
#: gate's 600 s per-script budget. Reduced under THERMALBUDGET_FAST, which is set
#: only by the working-directory probe.
EXAMPLES = 150 if _bootstrap.FAST else 2000


def main() -> int:
    out = _bootstrap.Tee("validate_property_hunt_output.txt")
    out.rule("PROPERTY HUNT -- the property suite at a deeper example count")
    out("INPUT ORIGIN: generated inputs from Hypothesis strategies over this")
    out("repository's own estimators. No number here is a measurement of any board.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(f"examples per property: {EXAMPLES}")
    out("example database: DISABLED, so no result here depends on a cache a fresh clone")
    out("would not have. Counterexamples found during the build are pinned with their")
    out("literal inputs in tests/test_regressions.py.")

    scratch = Path(tempfile.mkdtemp(prefix="thermalbudget-hunt-"))
    junit = scratch / "junit.xml"
    environment = _bootstrap.subprocess_environment()
    environment["THERMALBUDGET_HYPOTHESIS_EXAMPLES"] = str(EXAMPLES)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_properties.py",
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
            f"--junit-xml={junit}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=900,
        check=False,
    )
    if not junit.exists():
        out.check("the deep run produced a junit XML", False, "pytest did not run")
        out(completed.stdout[-2000:])
        out(f"raw output written to {out.save()}")
        return 1
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    declared = int(suite.get("tests", "0"))
    counted = len(suite.findall(".//testcase"))
    failures = int(suite.get("failures", "0"))
    errors = int(suite.get("errors", "0"))
    duration = float(suite.get("time", "0"))
    out("")
    out(
        f"deep run: {declared} properties declared, {counted} testcase elements counted, "
        f"{failures} failed, {errors} errored, {duration:.1f} s"
    )
    out(
        f"total generated cases attempted: about {declared * EXAMPLES} "
        f"({declared} properties x {EXAMPLES} examples)"
    )
    out.check(
        "the junit XML agrees with itself and the property suite is not empty",
        declared > 0 and declared == counted,
        f"declared {declared}, counted {counted}",
    )
    out.check(
        f"no property is falsified at {EXAMPLES} examples each",
        failures == 0 and errors == 0,
        f"{failures} failure(s), {errors} error(s)",
    )
    if failures or errors:
        out("")
        out("FALSIFYING OUTPUT (verbatim, so the counterexample can be pinned):")
        out(completed.stdout[-4000:])

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
