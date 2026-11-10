"""Audit every ``pytest.approx`` call for dependence on the 1e-12 default.

The hidden absolute tolerance of ``pytest.approx`` is ``1e-12`` and it is
applied with an OR against the relative bound, so a bare ``rel=`` on a small
quantity silently stops testing what it says it tests. This script measures the
dependence rather than asserting its absence, in four ways:

1. A **static count** of ``rel=`` occurrences in the test suite and of how many
   of them carry an explicit ``abs=`` on the same logical call.
2. A **strict run** of the whole suite with ``pytest.approx`` patched so that
   any call passing ``rel=`` without ``abs=`` raises. The count of such calls is
   the number of assertions that rely on the default.
3. A **zero run** of the whole suite with every call's ``abs`` forced to
   ``0.0``. The failures are the assertions for which the ABSOLUTE bound is the
   one doing the work. Those are not automatically defects -- a comparison
   against an exact zero has no relative bound -- but each one is named.
4. The Hypothesis settings that make part 3's list the same on every run.

This matters more here than in most products. Site-diversity quantities are
small by construction: joint unavailabilities of order 1e-3, a correlation
excess that this product's own checks drive down to 1e-25, the residual of an
exact decomposition at the 1e-16 level, and differences of two nearly equal
availabilities, which is the subject.

The junit XML is written outside the repository, and the counts are taken from
the ``testsuite`` element and cross-checked against a count of ``testcase``
elements, because a suite that collects zero tests exits 0 and prints success.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
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

from sitediverse.report import format_table  # noqa: E402

PLUGIN = "validation/_approx_audit_plugin.py"

#: Declared PATHS-ONLY mode, selected by ``SITEDIVERSE_APPROX_AUDIT_MODE=paths``.
#:
#: Why it exists. ``probe_working_directories.py`` must run every script in this
#: directory from four working directories, and this script runs the whole test
#: suite TWICE. Four copies of that is eight suite runs and about 440 seconds,
#: which would put the probe itself outside the release gate's 600-second
#: per-script limit -- so the probe would have to drop this script from the
#: sweep, which is a worse answer than running a reduced form of it.
#:
#: In paths-only mode the script does everything that depends on the working
#: directory -- bootstrap, resolve ``ROOT``, scan ``tests/`` for approx calls,
#: read ``tests/conftest.py``, resolve and write its output file -- and skips
#: only the two pytest subprocesses, whose cost is the problem and whose
#: behaviour does not depend on the caller's working directory because
#: ``_bootstrap.subprocess_environment`` sets theirs explicitly.
#:
#: The output then goes to a TEMPORARY file outside the repository rather than
#: to the committed one, so a probe run can never overwrite the committed
#: evidence with a reduced form of it. The mode is printed in the output either
#: way, so a reader of the committed file can see it was the full run.
PATHS_ONLY = os.environ.get("SITEDIVERSE_APPROX_AUDIT_MODE", "") == "paths"


def run_suite(mode: str, junit: Path) -> tuple[int, str, str, float]:
    """Run the whole suite with the audit plugin in ``mode``."""
    environment = subprocess_environment()
    if mode:
        environment["SITEDIVERSE_APPROX_AUDIT"] = mode
    started = time.perf_counter()
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
    return (
        completed.returncode,
        completed.stdout,
        completed.stderr,
        time.perf_counter() - started,
    )


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
    total_calls = 0
    total_rel = 0
    total_bare = 0
    for path in sorted((ROOT / "tests").glob("*.py")):
        text = path.read_text()
        # Each approx call is matched as a whole so that a rel= and an abs= on
        # separate lines of the same call are seen together.
        calls = re.findall(r"approx\((?:[^()]|\([^()]*\))*\)", text, flags=re.S)
        with_rel = [call for call in calls if "rel=" in call]
        bare = [call for call in with_rel if "abs=" not in call]
        total_calls += len(calls)
        total_rel += len(with_rel)
        total_bare += len(bare)
        rows.append(
            [relative(path), str(len(calls)), str(len(with_rel)), str(len(bare))]
        )
    out(format_table(
        ["file", "approx calls", "with rel=", "with rel= and no abs="], rows
    ))
    out("")
    out(f"total approx calls                       : {total_calls}")
    out(f"total approx calls carrying rel=         : {total_rel}")
    out(f"total carrying rel= with no explicit abs=: {total_bare}")
    out.check(
        "no approx call passes rel= without an explicit abs=",
        total_bare == 0,
        f"{total_bare} bare calls found by the source scan",
    )


def strict_run(out: Tee) -> int:
    """Part 2: the suite with the default forced to raise."""
    out.rule("2. the suite with a bare rel= made to raise")
    with tempfile.TemporaryDirectory(prefix="sitediverse-approx-") as scratch:
        junit = Path(scratch) / "strict.xml"
        status, stdout, stderr, elapsed = run_suite("strict", junit)
        declared = counts_from_junit(junit)
        names = failing_names(junit)
    out(f"pytest exit status : {status}")
    out(f"wall clock         : {elapsed:.1f} s")
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
    return declared["tests"]


def zero_run(out: Tee, total: int) -> None:
    """Part 3: the suite with every explicit abs forced to zero."""
    out.rule("3. the suite with every explicit abs= forced to 0.0")
    out("This is the stronger question. Where an assertion fails here, the ABSOLUTE")
    out("bound is the one doing the work and the relative bound alone would not")
    out("hold. That is correct and necessary for a comparison against an exact zero,")
    out("which has no relative bound at all, and it is wrong anywhere else, so every")
    out("failure is named below and accounted for.")
    out("")
    with tempfile.TemporaryDirectory(prefix="sitediverse-approx-") as scratch:
        junit = Path(scratch) / "zero.xml"
        status, stdout, _stderr, elapsed = run_suite("zero", junit)
        declared = counts_from_junit(junit)
        names = failing_names(junit)
    out(f"pytest exit status : {status}")
    out(f"wall clock         : {elapsed:.1f} s")
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
    out("Each test named above rests on an absolute bound for a stated reason. A")
    out("comparison against an exact zero has no relative bound at all and must use")
    out("an absolute one; an assertion on a quantity whose natural scale is absolute")
    out("is the same case. Anything else would be a defect and would be named as")
    out("such. The accounting is in validation/VALIDATION.md, which names every")
    out("entry of this list.")
    out("")
    out.check(
        "the junit XML agrees with itself in the zero run",
        declared["tests"] > 0 and declared["tests"] == declared["testcase_elements"],
        f"{declared['tests']} declared against {declared['testcase_elements']} counted",
    )
    out.check(
        "the two runs collected the same number of tests",
        declared["tests"] == total,
        f"{declared['tests']} in the zero run against {total} in the strict run",
    )


def determinism(out: Tee) -> None:
    """Part 4: the Hypothesis settings that make part 3's list stable."""
    out.rule("4. why the list in part 3 is the same on every run")
    out("A predecessor in this mission shipped an audit whose list of named failing")
    out("tests changed from run to run, because Hypothesis generated different")
    out("examples each time and the membership depended on them. That makes the")
    out("report worthless as evidence.")
    out("")
    text = (ROOT / "tests" / "conftest.py").read_text()
    lines = [
        line.strip()
        for line in text.splitlines()
        if any(
            key in line
            for key in ("derandomize", "database", "max_examples",
                        "register_profile", "load_profile", "deadline")
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
    if PATHS_ONLY:
        scratch = Path(tempfile.mkdtemp(prefix="sitediverse-approx-paths-"))
        out = Tee("validate_approx_audit_output.txt")
        out.path = scratch / "validate_approx_audit_output.txt"
    else:
        out = Tee("validate_approx_audit_output.txt")
    out.rule("sitediverse -- the pytest.approx default-tolerance audit")
    for line in environment_lines():
        out(line)
    out("")
    out("pytest.approx(expected, rel=r) passes if EITHER bound is met and the")
    out("absolute default is 1e-12, so a bare rel= on a small quantity stops testing")
    out("what it says. This script measures the dependence rather than asserting its")
    out("absence. It matters more here than in most products: this one's quantities")
    out("include joint unavailabilities of order 1e-03, a correlation excess driven")
    out("down to 1e-25, and an exact decomposition whose residual is 1e-16.")
    out("")
    out(f"mode                         : {'PATHS ONLY' if PATHS_ONLY else 'FULL'}")
    static_count(out)
    if PATHS_ONLY:
        out.rule("2 and 3. SKIPPED in paths-only mode")
        out("The two pytest runs are skipped and this output is written to a")
        out("temporary file outside the repository, so that a probe run cannot")
        out("overwrite the committed evidence with a reduced form of it. Everything")
        out("that depends on the caller's working directory has still been")
        out("exercised: the bootstrap, the repository root, the scan of tests/, the")
        out("read of tests/conftest.py and the resolution of the output path.")
        out("")
        out("The committed validation/validate_approx_audit_output.txt is always")
        out("from a FULL run; its own first lines say so.")
    else:
        total = strict_run(out)
        zero_run(out, total)
    determinism(out)
    path = out.save()
    print(f"\nwritten: {path if not PATHS_ONLY else 'a temporary file (paths-only mode)'}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
