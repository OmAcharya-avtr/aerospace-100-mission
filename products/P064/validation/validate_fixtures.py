"""Cold-clone check: every shipped fixture, run through `git check-ignore -v`.

This product bundles a sample project whose files are named exactly like
scratch artifacts, which is the shape of the defect that cost P060 its
committed `junit.xml` on 2026-10-10: a pattern with no leading slash matches at
every depth, so a product-level `junit.xml` excluded
`products/P060/fixtures/sample_project/junit.xml` while the uncommitted working
tree still had the file on disk and every local check passed.

This script enumerates the files the bundled sample project and the committed
fixtures need, asks git whether each one is ignored, and prints the matching
pattern and its source file when it is. A fixture that is ignored is a FAIL.

A trap found while writing this script, recorded because it is the same shape
as the defect it is guarding against: `git check-ignore -v` exits 0 whenever
ANY pattern matches the path, INCLUDING a negation. Run on this product's
committed fixture it exits 0 and prints
`!products/*/fixtures/**/junit.xml`, so an exit-status-only check reports the
required fixture as ignored. The verdict used below is therefore "a pattern
matched and it is not a negation", cross-checked independently against
`git ls-files --others --exclude-standard`, which lists exactly the untracked
files git would add.

It also probes two paths that this product deliberately does NOT ship, so the
record states why: `.coverage` (nothing here reads coverage data, and the file
is a coverage.py SQLite database whose schema is not a public interface) and a
`__pycache__` entry (ignored by the repository root at every depth, and a file
inside an excluded directory cannot be re-included by a later negation, so such
a fixture is unshippable by construction rather than by choice).

    python validation/validate_fixtures.py > validation/validate_fixtures_output.txt
"""

from __future__ import annotations

import subprocess
import sys

from _common import ROOT

# Files the bundled sample project and the fixture history need on a cold clone.
REQUIRED = [
    "fixtures/sample_project/junit.xml",
    "fixtures/sample_project/pytest.ini",
    "fixtures/sample_project/sampleproj/__init__.py",
    "fixtures/sample_project/sampleproj/filters.py",
    "fixtures/sample_project/sampleproj/integrate.py",
    "fixtures/sample_project/sampleproj/kinematics.py",
    "fixtures/sample_project/sampleproj/orbit.py",
    "fixtures/sample_project/sampleproj/quat.py",
    "fixtures/sample_project/sampleproj/stats.py",
    "fixtures/sample_project/sampleproj/vecops.py",
    "fixtures/sample_project/tests/test_filters.py",
    "fixtures/sample_project/tests/test_integrate.py",
    "fixtures/sample_project/tests/test_kinematics.py",
    "fixtures/sample_project/tests/test_orbit.py",
    "fixtures/sample_project/tests/test_quat.py",
    "fixtures/sample_project/tests/test_stats.py",
    "fixtures/sample_project/tests/test_vecops.py",
    "fixtures/faults/faults.json",
    "fixtures/faults/fault_matrix.json",
    "fixtures/history/history.json",
    "fixtures/history/generation_parameters.json",
    "fixtures/regenerate_fixtures.py",
]

# Paths that MUST be ignored: per-run artifacts at the product root.
MUST_BE_IGNORED = [
    "junit.xml",
    "junit_self.xml",
    ".coverage",
    "src/regressselect/__pycache__/metrics.cpython-313.pyc",
    "fixtures/sample_project/__pycache__/anything.pyc",
    ".pytest_cache/CACHEDIR.TAG",
    ".hypothesis/unicode_data/x",
]

# Paths this product chose not to ship, probed so the reason is on the record.
NOT_SHIPPED = [
    "fixtures/sample_project/.coverage",
    "fixtures/sample_project/__pycache__/vecops.cpython-313.pyc",
]


def check_ignore(rel: str) -> tuple[bool, str]:
    """Return (ignored, the `git check-ignore -v` line or an empty string).

    `ignored` is True only when a pattern matched AND that pattern is not a
    negation. git exits 0 for a matching negation too, so the exit status on
    its own is not the verdict.
    """
    proc = subprocess.run(
        ["git", "check-ignore", "-v", "--no-index", rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 1:
        return False, ""
    if proc.returncode != 0:
        raise RuntimeError(f"git check-ignore failed for {rel}: {proc.stderr.strip()}")
    line = proc.stdout.strip()
    pattern = line.split("\t", 1)[0].split(":")[-1] if line else ""
    return not pattern.startswith("!"), line


def git_would_add() -> set[str]:
    """Paths under the product root git would add: untracked and not ignored."""
    proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(proc.stdout.split())


def main() -> int:
    out = sys.stdout
    failures = 0
    out.write("Cold-clone fixture check: git check-ignore -v on every shipped file\n")
    out.write("=" * 96 + "\n")
    toplevel = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out.write(f"repository root   {toplevel}\n")
    out.write(f"product root      {ROOT.relative_to(toplevel)}\n")
    out.write("command           git check-ignore -v --no-index <path>, run in the product root\n")
    out.write("verdict           a pattern matched AND it is not a negation\n")
    out.write("cross-check       membership of git ls-files --others --exclude-standard\n\n")
    addable = git_would_add()
    out.write(
        f"git would add {len(addable)} untracked, non-ignored paths "
        "under the product root\n\n"
    )

    out.write("REQUIRED fixtures: every one of these must NOT be ignored\n")
    out.write("-" * 96 + "\n")
    for rel in REQUIRED:
        exists = (ROOT / rel).is_file()
        ignored, detail = check_ignore(rel)
        would_add = rel in addable
        ok = exists and not ignored and would_add
        failures += 0 if ok else 1
        status = "PASS" if ok else "FAIL"
        out.write(
            f"{status}  {'on disk' if exists else 'MISSING':<8}  "
            f"{'git would add' if would_add else 'GIT WOULD SKIP':<14}  {rel}\n"
        )
        if detail:
            out.write(f"      matched: {detail}\n")

    out.write("\nMUST BE IGNORED: per-run artifacts\n")
    out.write("-" * 96 + "\n")
    for rel in MUST_BE_IGNORED:
        ignored, detail = check_ignore(rel)
        not_addable = rel not in addable
        ok = ignored and not_addable
        failures += 0 if ok else 1
        out.write(f"{'PASS' if ok else 'FAIL'}  {rel}\n")
        out.write(f"      {detail if detail else 'NOT IGNORED'}\n")

    out.write("\nNOT SHIPPED: probed so the decision is recorded, not asserted\n")
    out.write("-" * 96 + "\n")
    for rel in NOT_SHIPPED:
        ignored, detail = check_ignore(rel)
        on_disk = (ROOT / rel).exists()
        out.write(f"      {rel}\n")
        out.write(f"      on disk: {on_disk}   ignored: {ignored}\n")
        out.write(f"      {detail if detail else 'no pattern matches'}\n")

    out.write(
        "\nThe junit.xml case, stated explicitly\n"
        "-------------------------------------\n"
        "The repository root .gitignore carries `**/junit.xml`, which matches at every\n"
        "depth, and `!products/*/fixtures/**/junit.xml`, which re-includes a committed\n"
        "fixture. This product's fixture sits at\n"
        "products/P064/fixtures/sample_project/junit.xml, which matches that negation,\n"
        "and the measurement above confirms it is not ignored. This product's own\n"
        ".gitignore anchors its pattern as `/junit.xml`, so it applies only to\n"
        "products/P064/junit.xml -- the per-run artifact -- and cannot reach into\n"
        "fixtures/ at all.\n"
        "\n"
        "A `__pycache__` fixture is unshippable, not merely unwanted: the repository\n"
        "root ignores `__pycache__/` at every depth, and git cannot re-include a file\n"
        "whose parent directory is excluded. The probe above records that. The sample\n"
        "project therefore needs no bytecode fixture, and the mutation harness copies\n"
        "the project with __pycache__ excluded before every run.\n"
    )

    out.write(
        f"\nchecks run {len(REQUIRED) + len(MUST_BE_IGNORED)}, failures {failures}\n"
    )
    out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
