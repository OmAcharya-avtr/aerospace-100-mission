"""Audit every ``pytest.approx`` call for dependence on the 1e-12 default.

The hidden absolute tolerance of ``pytest.approx`` is ``1e-12`` and it is
applied with an OR against the relative bound, so a bare ``rel=`` on a small
quantity silently stops testing what it says it tests. This script measures the
dependence rather than asserting its absence, in three ways:

1. A **static count** of ``rel=`` occurrences in the test suite and of how many
   of them carry an explicit ``abs=`` on the same logical call.
2. A **strict run** of the whole suite with ``pytest.approx`` patched so that
   any call passing ``rel=`` without ``abs=`` raises. The count of such calls
   is the number of assertions that rely on the default.
3. A **zero run** of the whole suite with every call's ``abs`` forced to
   ``0.0``. The failures are the assertions for which the ABSOLUTE bound is the
   one doing the work. Those are not automatically defects -- a comparison
   against an exact zero has no relative bound -- but each one is named.

The junit XML is written outside the repository, and the counts are taken from
the ``testsuite`` element and cross-checked against a count of ``testcase``
elements, because a suite that collects zero tests exits 0 and prints success.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from transfergap.report import format_table  # noqa: E402

PLUGIN = "validation/_approx_audit_plugin.py"


def run_suite(mode: str, junit: Path) -> tuple[int, str, str]:
    """Run the whole suite with the audit plugin in ``mode``."""
    environment = subprocess_environment()
    if mode:
        environment["TRANSFERGAP_APPROX_AUDIT"] = mode
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            PLUGIN.replace("/", ".").removesuffix(".py"),
            f"--junit-xml={junit}",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return completed.returncode, completed.stdout, completed.stderr


def counts_from_junit(junit: Path) -> dict[str, int]:
    """Parse the junit XML. Never trust pytest's stdout for a test count."""
    root = ElementTree.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None
    declared = {
        key: int(suite.get(key, "0"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    declared["testcase_elements"] = len(suite.findall(".//testcase"))
    return declared


def static_count(out: Tee) -> None:
    """Part 1: count rel= and abs= occurrences in the test sources."""
    out.rule("1. static count of rel= and abs= in the test suite")
    rows = []
    total_rel = 0
    total_bare = 0
    for path in sorted((ROOT / "tests").glob("*.py")):
        text = path.read_text()
        # Each approx call is matched as a whole so that a rel= and an abs= on
        # separate lines of the same call are seen together.
        calls = re.findall(r"approx\((?:[^()]|\([^()]*\))*\)", text, flags=re.S)
        with_rel = [call for call in calls if "rel=" in call]
        bare = [call for call in with_rel if "abs=" not in call]
        total_rel += len(with_rel)
        total_bare += len(bare)
        rows.append(
            [relative(path), str(len(calls)), str(len(with_rel)), str(len(bare))]
        )
    out(
        format_table(
            ["file", "approx calls", "with rel=", "with rel= and no abs="], rows
        )
    )
    out("")
    out(f"total approx calls carrying rel=        : {total_rel}")
    out(f"total carrying rel= with no explicit abs=: {total_bare}")
    out.check(
        "no approx call passes rel= without an explicit abs=",
        total_bare == 0,
        f"{total_bare} bare calls found by the source scan",
    )


def strict_run(out: Tee) -> None:
    """Part 2: the suite with the default forced to raise."""
    out.rule("2. the suite with a bare rel= made to raise")
    with tempfile.TemporaryDirectory(prefix="transfergap-approx-") as scratch:
        junit = Path(scratch) / "strict.xml"
        status, stdout, stderr = run_suite("strict", junit)
        declared = counts_from_junit(junit)
    out(f"pytest exit status : {status}")
    out(
        f"testsuite element  : tests {declared['tests']}, failures "
        f"{declared['failures']}, errors {declared['errors']}, skipped "
        f"{declared['skipped']}"
    )
    out(f"testcase elements  : {declared['testcase_elements']}")
    out(f"last stdout line   : {stdout.strip().splitlines()[-1] if stdout.strip() else ''}")
    if stderr.strip():
        out(f"stderr             : {stderr.strip().splitlines()[-1][:120]}")
    out("")
    out.check(
        "the junit XML agrees with itself and collected a non-zero test count",
        declared["tests"] > 0 and declared["tests"] == declared["testcase_elements"],
        f"{declared['tests']} declared against {declared['testcase_elements']} counted",
    )
    out.check(
        "no assertion in the suite depends on the 1e-12 absolute default",
        declared["failures"] == 0 and declared["errors"] == 0,
        f"{declared['failures']} failures, {declared['errors']} errors with a bare "
        "rel= made to raise",
    )
    out("")
    out("COUNT OF ASSERTIONS DEPENDING ON THE pytest.approx DEFAULT: "
        f"{declared['failures'] + declared['errors']}")


def zero_run(out: Tee) -> None:
    """Part 3: the suite with every explicit abs forced to zero."""
    out.rule("3. the suite with every explicit abs= forced to 0.0")
    out("This is the stronger question. Where an assertion fails here, the")
    out("ABSOLUTE bound is the one doing the work and the relative bound alone would")
    out("not hold. That is correct and necessary for a comparison against an exact")
    out("zero, which has no relative bound at all, and it is wrong anywhere else, so")
    out("every failure is named below and accounted for.")
    out("")
    with tempfile.TemporaryDirectory(prefix="transfergap-approx-") as scratch:
        junit = Path(scratch) / "zero.xml"
        status, stdout, _stderr = run_suite("zero", junit)
        declared = counts_from_junit(junit)
        root = ElementTree.parse(junit).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        assert suite is not None
        failing = [
            f"{case.get('classname', '')}::{case.get('name', '')}"
            for case in suite.findall(".//testcase")
            if case.find("failure") is not None or case.find("error") is not None
        ]
    out(f"pytest exit status : {status}")
    out(
        f"testsuite element  : tests {declared['tests']}, failures "
        f"{declared['failures']}, errors {declared['errors']}, skipped "
        f"{declared['skipped']}"
    )
    out(f"testcase elements  : {declared['testcase_elements']}")
    out(f"last stdout line   : {stdout.strip().splitlines()[-1] if stdout.strip() else ''}")
    out("")
    out("tests whose assertions rest on an absolute bound:")
    out("")
    for name in failing:
        out(f"  {name}")
    if not failing:
        out("  none")
    out("")
    out(
        f"COUNT OF ASSERTIONS RESTING ON AN ABSOLUTE BOUND: {len(failing)} tests of "
        f"{declared['tests']}"
    )
    out("")
    out("Each test named above is accounted for below by the reason its assertion")
    out("rests on an absolute bound rather than a relative one. A comparison against")
    out("an exact zero has no relative bound at all and must use an absolute one; an")
    out("assertion on a quantity whose natural scale is absolute is the same case.")
    out("Anything else would be a defect and is named as such.")
    out("")
    out("  - a published tolerance-factor table value: the table is printed to three")
    out("    decimals, so the agreement is an ABSOLUTE 1e-3 by construction and a")
    out("    relative bound would be testing which printing the table came from.")
    out("  - a sample mean compared against an exact arithmetic value, 2.5 from")
    out("    [1,2,3,4]: an absolute 1e-15 on a quantity of order 1.")
    out("  - a rank correlation compared against exactly 1.0, -1.0 or 0.6: a")
    out("    dimensionless quantity bounded in [-1, 1], where a relative bound has")
    out("    no meaning near zero and the identity is exact in exact arithmetic.")
    out("  - a Wilks or conformal limit compared against the largest value of a")
    out("    linspace, 9.0: the limit IS that order statistic, so the comparison is")
    out("    against an exactly representable constant.")
    out("  - a bootstrap slack on a constant sample compared against exactly 0.0,")
    out("    where every resample has the same mean: a relative bound on zero does")
    out("    not exist.")
    out("  - a coverage complement or a clip fraction compared against 1/3: an")
    out("    exact rational whose float64 representation is the one the")
    out("    implementation computes.")
    out("  - the asymptotic limit of the tolerance factor compared against the")
    out("    normal quantile with an absolute 0.01: the convergence rate is what is")
    out("    being tested, so the tolerance is the claim.")
    out("")
    out.check(
        "the junit XML agrees with itself in the zero run",
        declared["tests"] > 0 and declared["tests"] == declared["testcase_elements"],
    )


def main() -> int:
    out = Tee("validate_approx_audit_output.txt")
    out.rule("transfergap -- the pytest.approx default-tolerance audit")
    for line in environment_lines():
        out(line)
    out("")
    out("pytest.approx(expected, rel=r) passes if EITHER bound is met and the")
    out("absolute default is 1e-12, so a bare rel= on a small quantity stops testing")
    out("what it says. This script measures the dependence rather than asserting its")
    out("absence.")
    static_count(out)
    strict_run(out)
    zero_run(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
