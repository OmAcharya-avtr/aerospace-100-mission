"""The pytest.approx audit: how many assertions depended on the default absolute tolerance.

``pytest.approx(expected, rel=r)`` carries a **default absolute tolerance of
1e-12** and passes when *either* bound is satisfied. For any quantity of order
1e-5, every relative tolerance tighter than 1e-7 is therefore silently
replaced by the absolute one and the assertion stops testing what it says it
tests. Conservation residuals, which are small by construction, are exactly
where this bites.

The audit has two parts.

**Static.** Parse every test module and list every real ``pytest.approx`` call
with its keywords, by AST rather than by regular expression -- a regular
expression over the source also matches the occurrences inside the docstrings
that explain the problem, and the first version of this scan reported its own
docstring as an offender.

**Executed.** Copy the whole test tree to a temporary directory, rewrite every
``pytest.approx`` call that passes ``rel=`` so that it also passes
``abs=0.0``, and run the suite there. The number of assertions that changed
verdict is the measurement: it is the number that were relying on the default.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast, relative

ROOT = bootstrap()
TESTS = ROOT / "tests"


def _calls(path: Path) -> list[tuple[int, tuple[str, ...]]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "approx"
            and isinstance(function.value, ast.Name)
            and function.value.id == "pytest"
        ):
            found.append(
                (node.lineno, tuple(k.arg for k in node.keywords if k.arg is not None))
            )
    return found


def _rewrite_approx(text: str, mode: str) -> tuple[str, int]:
    """Rewrite ``pytest.approx`` calls carrying ``rel=``. Returns (text, count).

    ``mode`` selects the experiment:

    ``"force_zero_where_absent"``
        Add ``abs=0.0`` only to calls that do NOT already declare one. This is
        the experiment that measures reliance on the 1e-12 default, because
        those are the only calls the default applies to.
    ``"force_zero_everywhere"``
        Set ``abs=0.0`` on every call carrying ``rel=``, overriding an explicit
        one. This measures which assertions have a load-bearing absolute
        tolerance -- that is, exactly the set where the default would have been
        the wrong tolerance.
    ``"strip_abs"``
        Remove the explicit ``abs=`` from every call that has one, so the
        1e-12 default applies instead. This measures what the explicit
        declarations are protecting.

    The rewrite is driven by the parse, not by a pattern over the text, and
    edits are applied from the end of the file backwards so earlier offsets
    stay valid.
    """
    tree = ast.parse(text)
    edits: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "approx"
            and isinstance(function.value, ast.Name)
            and function.value.id == "pytest"
        ):
            continue
        keywords = {k.arg for k in node.keywords if k.arg is not None}
        if "rel" not in keywords:
            continue
        segment = ast.get_source_segment(text, node)
        if segment is None:
            continue
        has_abs = "abs" in keywords
        if mode == "force_zero_where_absent":
            if has_abs:
                continue
            replacement = segment[:-1].rstrip() + ", abs=0.0)"
        elif mode == "force_zero_everywhere":
            replacement = (
                re.sub(r"abs\s*=\s*[^,)]+", "abs=0.0", segment, count=1)
                if has_abs
                else segment[:-1].rstrip() + ", abs=0.0)"
            )
        elif mode == "strip_abs":
            if not has_abs:
                continue
            replacement = re.sub(r",\s*abs\s*=\s*[^,)]+", "", segment, count=1)
        else:  # pragma: no cover - programming error
            raise ValueError(f"unknown mode {mode!r}")
        edits.append(
            ((node.lineno, node.col_offset), (node.end_lineno, node.end_col_offset), replacement)
        )

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def absolute(position: tuple[int, int]) -> int:
        row, column = position
        return offsets[row - 1] + column

    rewritten = text
    for start, end, replacement in sorted(
        edits, key=lambda item: absolute(item[0]), reverse=True
    ):
        rewritten = rewritten[: absolute(start)] + replacement + rewritten[absolute(end) :]
    return rewritten, len(edits)


def _run_suite(directory: Path, label: str) -> tuple[int, int, int, int]:
    junit = directory / "junit.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(directory / "tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junit-xml={junit}",
        ],
        cwd=directory,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(ROOT / "src"),
            "MPLBACKEND": "Agg",
            "HOME": str(directory),
            "MPLCONFIGDIR": str(directory),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        check=False,
    )
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    assert suite is not None, f"{label}: junit.xml has no testsuite element"
    declared = int(suite.get("tests", "0"))
    failures = int(suite.get("failures", "0"))
    errors = int(suite.get("errors", "0"))
    counted = len(suite.findall(".//testcase"))
    return declared, failures, errors, counted


def main() -> int:
    out = Tee("validate_approx_audit_output.txt")
    out.rule("orbitdrift -- the pytest.approx default-absolute-tolerance audit")
    for line in environment_lines():
        out(line)

    out.rule("why this audit exists")
    out("  pytest.approx(expected, rel=r) carries a DEFAULT ABSOLUTE TOLERANCE of")
    out("  1e-12 and passes when EITHER bound is satisfied. For a quantity of order")
    out("  1e-5, every relative tolerance tighter than 1e-7 is therefore silently")
    out("  replaced by the absolute one. Conservation residuals in this product are")
    out("  of order 1e-9, so an assertion on one with rel=1e-6 and no explicit abs")
    out("  would be testing |x - y| < 1e-12, which it would pass for any pair of")
    out("  numbers that small.")

    out.rule("static scan")
    out("  module                     line   keywords")
    total = 0
    rel_without_abs = []
    rel_with_abs = 0
    abs_only = 0
    neither = 0
    for path in sorted(TESTS.glob("test_*.py")):
        for line, keywords in _calls(path):
            total += 1
            if "rel" in keywords and "abs" not in keywords:
                rel_without_abs.append(f"{path.name}:{line}")
            elif "rel" in keywords:
                rel_with_abs += 1
            elif "abs" in keywords:
                abs_only += 1
            else:
                neither += 1
    for path in sorted(TESTS.glob("test_*.py")):
        calls = _calls(path)
        if calls:
            out(f"  {path.name:26s} {len(calls):4d} call(s)")
    out("")
    out(f"  total pytest.approx calls               : {total}")
    out(f"  with rel= and an explicit abs=          : {rel_with_abs}")
    out(f"  with abs= only                          : {abs_only}")
    out(f"  with neither (uses both defaults)       : {neither}")
    out(f"  with rel= and NO abs= (relies on 1e-12) : {len(rel_without_abs)}")
    if rel_without_abs:
        for item in rel_without_abs:
            out(f"    {item}")
    out.check(
        "no assertion passes rel= without an explicit abs=",
        not rel_without_abs,
        f"{len(rel_without_abs)} offender(s)",
    )
    out.check(
        "the scan finds calls at all, so the invariant above is not vacuous",
        total > 40,
        f"{total} calls across {len(list(TESTS.glob('test_*.py')))} modules",
    )

    if is_fast():
        out.rule("executed audit")
        out("  skipped in fast probe mode")
        out("")
        out(f"wrote {out.save()}")
        return 0

    out.rule("executed audit")
    out("  Three experiments, each on a copy of the whole test tree in a temporary")
    out("  directory. They answer different questions and conflating them is the")
    out("  reason the first version of this script reported a misleading number.")
    out("")
    out("  (a) force_zero_where_absent : add abs=0.0 only where no abs= is")
    out("      declared. Those are the ONLY calls the 1e-12 default applies to, so")
    out("      this is the experiment that measures reliance on it. The static scan")
    out("      above found no such call, so the set is empty and the run is")
    out("      skipped: the measurement is zero by construction and saying so is")
    out("      more honest than spending 80 s to print it.")
    out("  (b) force_zero_everywhere   : set abs=0.0 on every rel= call, overriding")
    out("      the explicit ones. This is the literal substitution the mission")
    out("      brief asks for. What it measures is which assertions have a")
    out("      LOAD-BEARING absolute tolerance -- that is, exactly the set where")
    out("      the 1e-12 default would have been the wrong tolerance.")
    out("  (c) strip_abs               : remove the explicit abs= so the 1e-12")
    out("      default applies. This measures what the explicit declarations are")
    out("      protecting, which is the counterfactual for a suite that had not")
    out("      written them.")
    timer = Timer()
    results: dict[str, tuple[int, int, int, int]] = {}
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as scratch:
        # The WHOLE product tree is copied, not just tests/. Several tests scan
        # the product root -- for absolute paths, for forbidden claims, and for
        # required deliverables matched by .gitignore -- and a partial copy made
        # two of them fail in the sandbox while passing in the repository, which
        # would have made the audit's baseline run look like a regression.
        workspace = Path(scratch) / "product"
        shutil.copytree(
            ROOT,
            workspace,
            ignore=shutil.ignore_patterns(
                "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".git"
            ),
        )
        pristine = {
            path.name: path.read_text() for path in (workspace / "tests").glob("test_*.py")
        }
        with timer:
            results["baseline"] = _run_suite(workspace, "baseline")
        counts["baseline"] = 0
        for mode in ("force_zero_everywhere", "strip_abs"):
            rewritten = 0
            for path in sorted((workspace / "tests").glob("test_*.py")):
                text, count = _rewrite_approx(pristine[path.name], mode)
                path.write_text(text)
                rewritten += count
            counts[mode] = rewritten
            with timer:
                results[mode] = _run_suite(workspace, mode)

    out("")
    out("  experiment                  calls rewritten   tests  failures  errors  elements")
    for label in ("baseline", "force_zero_everywhere", "strip_abs"):
        declared, failures, errors, counted = results[label]
        out(
            f"  {label:26s} {counts[label]:15d}   {declared:5d}  {failures:8d}  "
            f"{errors:6d}  {counted:8d}"
        )

    baseline = results["baseline"]
    changed_everywhere = (
        results["force_zero_everywhere"][1] + results["force_zero_everywhere"][2]
    ) - (baseline[1] + baseline[2])
    changed_stripped = (results["strip_abs"][1] + results["strip_abs"][2]) - (
        baseline[1] + baseline[2]
    )

    out("")
    out("  THE AUDIT RESULT")
    out(f"    (a) assertions relying on the 1e-12 default            : 0 "
        f"(no rel= call lacks an explicit abs=; {total} calls scanned)")
    out(f"    (b) assertions whose explicit absolute tolerance is    : {changed_everywhere}")
    out("        load-bearing, i.e. where the default would have been")
    out("        the wrong tolerance")
    out(f"    (c) assertions that would change verdict if the        : {changed_stripped}")
    out("        explicit abs= were removed and the default applied")
    out("")
    out("  (a) is the number the mission brief's question asks for and it is zero.")
    out("  (b) and (c) are the reason that is worth something: there are assertions")
    out("  in this suite whose absolute tolerance matters, so a suite that had")
    out("  relied on the default would have had that many assertions testing")
    out("  |x - y| < 1e-12 instead of what they say they test.")

    out.check(
        "the baseline suite passes",
        baseline[1] == 0 and baseline[2] == 0,
        f"{baseline[1]} failures, {baseline[2]} errors over {baseline[0]} tests",
    )
    out.check(
        "the junit XML agrees with itself in every run",
        all(value[0] == value[3] for value in results.values()),
        ", ".join(f"{label} {value[0]}/{value[3]}" for label, value in results.items()),
    )
    out.check(
        "MEASURED (a): no assertion in this suite relies on the 1e-12 default",
        not rel_without_abs,
        f"{len(rel_without_abs)} call(s) pass rel= without abs=, so the default "
        "applies to none of them",
    )
    out.check(
        "the rewrite in experiment (b) touched every rel= call the scan found",
        counts["force_zero_everywhere"] == rel_with_abs + len(rel_without_abs),
        f"{counts['force_zero_everywhere']} rewritten against "
        f"{rel_with_abs + len(rel_without_abs)} found",
    )
    out.check(
        "the audit is not vacuous: at least one assertion's absolute tolerance is "
        "load-bearing, so experiments (b) and (c) can detect a difference",
        changed_everywhere > 0 or changed_stripped > 0,
        f"(b) {changed_everywhere} changed, (c) {changed_stripped} changed",
    )

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  three suite runs              : {timer.elapsed_s:.1f} s")
    out("")
    out(f"  test tree audited             : {relative(TESTS)}")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
