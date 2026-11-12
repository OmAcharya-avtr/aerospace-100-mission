"""How many assertions in this suite depend on ``pytest.approx``'s hidden 1e-12.

``pytest.approx(expected, rel=r)`` passes if EITHER bound is met and the absolute
default is ``1e-12``. The audit re-runs the whole suite in a subprocess with a
plugin that substitutes ``abs=0.0`` into every bare ``rel=`` call, and reports:

- how many ``rel=``-only call sites were exercised at all;
- how many tests fail once the default is forced off, which is the number of
  assertions that were resting on it rather than on the bound they declare.

A non-zero count is not automatically a defect -- a residual that is genuinely
zero needs an absolute bound -- but it is always something the author must have
decided deliberately, and this file is where that decision is recorded.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()


def _count_sources() -> tuple[int, int]:
    """Static count of ``rel=`` call sites, and of those that also pass ``abs=``."""
    rel_only = 0
    both = 0
    for path in sorted((ROOT / "tests").rglob("*.py")):
        for match in re.finditer(r"approx\((?:[^()]|\([^()]*\))*\)", path.read_text()):
            call = match.group(0)
            if "rel=" not in call:
                continue
            if "abs=" in call:
                both += 1
            else:
                rel_only += 1
    return rel_only, both


def main() -> int:
    out = _bootstrap.Tee("validate_approx_audit_output.txt")
    out.rule("APPROX AUDIT -- assertions resting on the hidden 1e-12 absolute default")
    out("INPUT ORIGIN: this repository's own test suite.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    rel_only, both = _count_sources()
    out("")
    out(f"static scan of tests/: {rel_only + both} approx() call sites pass rel=")
    out(f"  of those, {both} also pass an explicit abs=")
    out(f"  of those, {rel_only} pass rel= alone")
    out.check(
        "every rel= call site in the suite also passes an explicit abs=",
        rel_only == 0,
        f"{rel_only} bare rel= site(s) found by static scan",
    )

    scratch = Path(tempfile.mkdtemp(prefix="thermalbudget-approx-"))
    record = scratch / "record.json"
    junit = scratch / "junit.xml"
    environment = _bootstrap.subprocess_environment()
    environment["THERMALBUDGET_APPROX_AUDIT"] = str(record)
    environment["PYTHONPATH"] = (
        str(ROOT / "validation") + ":" + environment.get("PYTHONPATH", "")
    )
    out("")
    out("re-running the whole suite with abs=0.0 substituted into every bare rel= call")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--tb=no",
            "-p",
            "no:cacheprovider",
            "-p",
            "_approx_audit_plugin",
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
        out.check(
            "the audited run produced a junit XML",
            False,
            "pytest did not run; the audit cannot report a number",
        )
        out(completed.stdout[-2000:])
        out(f"raw output written to {out.save()}")
        return 1
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    declared = int(suite.get("tests", "0"))
    counted = len(suite.findall(".//testcase"))
    failures = int(suite.get("failures", "0"))
    errors = int(suite.get("errors", "0"))
    out("")
    out(f"audited run: {declared} tests declared, {counted} testcase elements counted, "
        f"{failures} failed, {errors} errored")
    out.check(
        "the junit XML agrees with itself and the suite is not empty",
        declared > 0 and declared == counted,
        f"declared {declared}, counted {counted}",
    )
    out.check(
        "no test depends on the hidden absolute default",
        failures == 0 and errors == 0,
        f"{failures} failure(s) and {errors} error(s) with abs=0.0 forced",
    )
    if record.exists():
        exercised = json.loads(record.read_text())
        out(f"call sites exercised with a bare rel= at run time: {len(exercised)}")
        for entry in exercised[:20]:
            out(f"  {entry['site']}  rel={entry['rel']}")
    else:
        out("no bare rel= call site was exercised at run time")
    out.check(
        "the audit itself ran, so the count is measured rather than assumed",
        completed.returncode in (0, 1),
        f"pytest exit {completed.returncode}",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
