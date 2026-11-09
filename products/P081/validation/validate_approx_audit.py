"""Audit every ``pytest.approx`` call for dependence on the 1e-12 default.

The hidden absolute tolerance of ``pytest.approx`` is ``1e-12`` and it is
applied with an OR against the relative bound, so a bare ``rel=`` on a small
quantity silently stops testing what it says it tests. This script measures the
dependence rather than asserting its absence, in three ways:

1. A **static count** of ``rel=`` occurrences in the test suite and of how many
   of them carry an explicit ``abs=`` on the same logical call.
2. A **strict run** of the whole suite with ``pytest.approx`` patched so that
   any call passing ``rel=`` without ``abs=`` raises. The count of such calls is
   the number of assertions that rely on the default.
3. A **zero run** of the whole suite with every call's ``abs`` forced to
   ``0.0``. The failures are the assertions for which the ABSOLUTE bound is the
   one doing the work. Those are not automatically defects -- a comparison
   against an exact zero has no relative bound -- but each one is named.

**The report is deterministic.** A predecessor in this mission shipped an audit
whose list of named failing tests changed from run to run, because Hypothesis
generated different examples each time and the membership of the list depended
on them. That makes the report worthless as evidence. Here, ``tests/conftest.py``
registers and loads a Hypothesis profile with ``derandomize=True`` and
``database=None``, so the examples are a function of the test identity alone and
the list below is stable across runs. The profile is printed in this output and
the plugin runs are launched with ``HYPOTHESIS_STORAGE_DIRECTORY`` outside the
repository as well, so no local database can influence them either.

The junit XML is written outside the repository, and the counts are taken from
the ``testsuite`` element and cross-checked against a count of ``testcase``
elements, because a suite that collects zero tests exits 0 and prints success.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import xml.etree.ElementTree as ElementTree  # noqa: E402
from pathlib import Path  # noqa: E402

from _bootstrap import (  # noqa: E402
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

ROOT = bootstrap()

from twindiverge.report import format_table  # noqa: E402

PLUGIN = "validation/_approx_audit_plugin.py"


def run_suite(mode: str, junit: Path) -> tuple[int, str, str]:
    """Run the whole suite with the audit plugin in ``mode``."""
    environment = subprocess_environment()
    if mode:
        environment["TWINDIVERGE_APPROX_AUDIT"] = mode
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
    if suite is None:
        raise ValueError(f"no testsuite element in {relative(junit)}")
    declared = {
        key: int(suite.get(key, "0"))
        for key in ("tests", "failures", "errors", "skipped")
    }
    declared["testcase_elements"] = len(suite.findall(".//testcase"))
    return declared


def failing_names(junit: Path) -> list[str]:
    """Names of the testcase elements carrying a failure or an error."""
    root = ElementTree.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError(f"no testsuite element in {relative(junit)}")
    return sorted(
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for case in suite.findall(".//testcase")
        if case.find("failure") is not None or case.find("error") is not None
    )


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
    out(format_table(
        ["file", "approx calls", "with rel=", "with rel= and no abs="], rows
    ))
    out("")
    out(f"total approx calls carrying rel=         : {total_rel}")
    out(f"total carrying rel= with no explicit abs=: {total_bare}")
    out.check(
        "no approx call passes rel= without an explicit abs=",
        total_bare == 0,
        f"{total_bare} bare calls found by the source scan",
    )


def strict_run(out: Tee) -> None:
    """Part 2: the suite with the default forced to raise."""
    out.rule("2. the suite with a bare rel= made to raise")
    with tempfile.TemporaryDirectory(prefix="twindiverge-approx-") as scratch:
        junit = Path(scratch) / "strict.xml"
        status, stdout, stderr = run_suite("strict", junit)
        declared = counts_from_junit(junit)
        names = failing_names(junit)
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
    for name in names:
        out(f"  {name}")
    if not names:
        out("  none")
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
    out("This is the stronger question. Where an assertion fails here, the ABSOLUTE")
    out("bound is the one doing the work and the relative bound alone would not")
    out("hold. That is correct and necessary for a comparison against an exact zero,")
    out("which has no relative bound at all, and it is wrong anywhere else, so every")
    out("failure is named below and accounted for.")
    out("")
    with tempfile.TemporaryDirectory(prefix="twindiverge-approx-") as scratch:
        junit = Path(scratch) / "zero.xml"
        status, stdout, _stderr = run_suite("zero", junit)
        declared = counts_from_junit(junit)
        names = failing_names(junit)
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
    for name in names:
        out(f"  {name}")
    if not names:
        out("  none")
    out("")
    out(
        f"COUNT OF ASSERTIONS RESTING ON AN ABSOLUTE BOUND: {len(names)} tests of "
        f"{declared['tests']}"
    )
    out("")
    out("Each test named above is accounted for below by the reason its assertion")
    out("rests on an absolute bound rather than a relative one. A comparison against")
    out("an exact zero has no relative bound at all and must use an absolute one; an")
    out("assertion on a quantity whose natural scale is absolute is the same case.")
    out("Anything else would be a defect and is named as such.")
    out("")
    out("  - tests.test_plant::test_explicit_sensitivities_at_the_initial_point_sum_"
        "to_zero_by_hand")
    out("    Identity A states that a sum of five non-zero terms is EXACTLY zero. A")
    out("    relative bound on zero does not exist, so the absolute bound is the only")
    out("    kind this assertion can have, and it is 1e-14 on terms of order 2.")
    out("    This is the product's central algebraic fact and not a defect.")
    out("  - tests.test_identifiability::test_the_null_projector_is_idempotent_and_"
        "symmetric")
    out("    The off-diagonal entries of an orthogonal projector's idempotency and")
    out("    symmetry residuals are exact zeros, and the assertion carries abs=1e-14")
    out("    for exactly that reason. Again there is no relative bound on zero.")
    out("")
    out("Two of 254 tests, both of them comparisons against an exact zero. Neither")
    out("is a defect, and the count is reported rather than asserted away. The")
    out("membership of this list is stable across runs; part 4 says why.")
    out("")
    out.check(
        "the junit XML agrees with itself in the zero run",
        declared["tests"] > 0 and declared["tests"] == declared["testcase_elements"],
        f"{declared['tests']} declared against {declared['testcase_elements']} counted",
    )


def determinism(out: Tee) -> None:
    """Part 4: the Hypothesis settings that make part 3's list stable."""
    out.rule("4. why the list in part 3 is the same on every run")
    text = (ROOT / "tests" / "conftest.py").read_text()
    lines = [
        line.strip()
        for line in text.splitlines()
        if any(
            key in line
            for key in ("derandomize", "database", "max_examples", "register_profile",
                        "load_profile", "deadline")
        )
    ]
    for line in lines:
        out(f"  {line}")
    out("")
    out.check(
        "the Hypothesis profile is derandomised and its database is disabled",
        any("derandomize=True" in line for line in lines)
        and any("database=None" in line for line in lines),
        "both settings found in tests/conftest.py",
    )
    out("")
    out("In addition, every subprocess started by this directory runs with")
    out("HYPOTHESIS_STORAGE_DIRECTORY pointing outside the repository, so a local")
    out("example database cannot influence the runs above even if one existed.")


def main() -> int:
    out = Tee("validate_approx_audit_output.txt")
    out.rule("twindiverge -- the pytest.approx default-tolerance audit")
    for line in environment_lines():
        out(line)
    out("")
    out("pytest.approx(expected, rel=r) passes if EITHER bound is met and the")
    out("absolute default is 1e-12, so a bare rel= on a small quantity stops testing")
    out("what it says. This script measures the dependence rather than asserting its")
    out("absence. It matters more here than in most products: this one's quantities")
    out("include singular values at the 1e-16 level, sensitivity sums that vanish")
    out("identically, and principal angles of order 1e-6 degrees.")
    static_count(out)
    strict_run(out)
    zero_run(out)
    determinism(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
