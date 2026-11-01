"""Every command line entry point, in a subprocess, and the tree afterwards.

Each subprocess is given a temporary ``HOME`` and ``MPLCONFIGDIR`` outside the
repository and ``PYTHONDONTWRITEBYTECODE=1``, because a validation script in an
earlier product of this mission set ``HOME`` to the product root and matplotlib
wrote ``.config/matplotlib`` into the repository, which the release gate fails
twice over: once as a tracked build artifact and once as an absolute path
inside one.

The last section is the absolute check that the tree carries no cache
directory, which the test suite deliberately does not make because the test
runner itself may create one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from pointmodel.report import format_table  # noqa: E402

COMMANDS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("version", ("--version",), ("pointmodel 0.1.0",)),
    ("terms", ("terms",), ("DECLARED POINTING MODEL TERMS", "NPAE", "rho_hat(E)")),
    (
        "identify full_sky",
        ("identify", "--geometry", "full_sky", "--observations", "200"),
        ("WHAT THIS GEOMETRY CANNOT DETERMINE", "weakly determined"),
    ),
    (
        "identify narrow_el",
        ("identify", "--geometry", "narrow_el", "--observations", "200"),
        ("not separable", "terms refused:", "NOT IDENTIFIABLE"),
    ),
    (
        "compare",
        ("compare", "--observations", "200"),
        ("RANK VERDICT BY DECLARED GEOMETRY", "single_az", "two_az"),
    ),
    (
        "decompose full_sky",
        ("decompose", "--observations", "200"),
        ("POINTING ERROR BUDGET", "TRUTH, for reference", "not a measurement"),
    ),
    (
        "decompose narrow_el",
        ("decompose", "--geometry", "narrow_el", "--observations", "200", "--no-robust"),
        ("REFUSED per-term split for:",),
    ),
    (
        "outliers",
        ("outliers", "--observations", "200", "--outlier-fraction", "0.06"),
        ("ROBUST FIT (huber)", "ROBUST FIT (ransac)", "flagged             : YES"),
    ),
    (
        "refraction",
        ("refraction", "--steps", "6"),
        ("REFRACTION CROSS-CHECK", "No refraction package", "optical 550 nm"),
    ),
)

FAILING: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("no command", (), 2),
    ("unknown command", ("predict",), 2),
    ("unknown geometry", ("identify", "--geometry", "equator"), 2),
    ("negative sigma", ("decompose", "--observations", "100", "--sigma", "-1"), 2),
    (
        "out-of-range threshold",
        ("identify", "--observations", "100", "--null-threshold", "5"),
        2,
    ),
)

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")


def run(arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pointmodel`` with a polluting-proof environment."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "pointmodel", *arguments],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=subprocess_environment(),
    )


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("pointmodel -- the command line, in subprocesses")
    for line in environment_lines():
        out(line)
    out("")
    out("Each subprocess receives HOME and MPLCONFIGDIR in a temporary directory")
    out("outside the repository and PYTHONDONTWRITEBYTECODE=1.")

    out.rule("1. commands that must succeed")
    rows = []
    for label, arguments, expected in COMMANDS:
        completed = run(arguments)
        missing = [fragment for fragment in expected if fragment not in completed.stdout]
        rows.append(
            [
                label,
                " ".join(arguments) or "(none)",
                str(completed.returncode),
                str(len(completed.stdout.splitlines())),
                "ok" if not missing else f"MISSING {missing}",
            ]
        )
        out.check(
            f"`python -m pointmodel {' '.join(arguments)}` exits zero",
            completed.returncode == 0,
            completed.stderr.strip()[-200:],
        )
        out.check(
            f"`{label}` prints every expected fragment",
            not missing,
            f"missing: {missing}" if missing else "",
        )
    out("")
    out(format_table(["command", "arguments", "exit", "stdout lines", "fragments"], rows))

    out.rule("2. commands that must fail, with the status they must fail with")
    rows = []
    for label, arguments, expected_status in FAILING:
        completed = run(arguments)
        rows.append(
            [
                label,
                " ".join(arguments) or "(none)",
                str(completed.returncode),
                str(expected_status),
                "ok" if completed.returncode == expected_status else "WRONG",
            ]
        )
        out.control(
            f"`{label}` is rejected",
            completed.returncode == expected_status,
            must_fire=True,
            detail=f"exit {completed.returncode}, expected {expected_status}; "
            f"{completed.stderr.strip().splitlines()[-1][:80] if completed.stderr.strip() else ''}",
        )
    out("")
    out(format_table(["case", "arguments", "exit", "expected", "verdict"], rows))

    out.rule("3. every cache directory in the tree is covered by .gitignore")
    out("A cache directory is not itself a defect: the test runner and the")
    out("interpreter create them and .gitignore excludes them. What IS a defect is")
    out("one that .gitignore does not cover, because it then reaches a commit and")
    out("the release gate fails a tracked build artifact.")
    out("")
    out("Coverage is read from .gitignore the way git reads it. A pattern with no")
    out("slash matches at ANY depth; a pattern with a leading slash matches only at")
    out("the repository root. That distinction is the P060 defect of this mission,")
    out("where a bare junit.xml in a product .gitignore silently excluded a required")
    out("committed fixture several directories down.")
    out("")
    gitignore = (ROOT / ".gitignore").read_text().splitlines()
    patterns = [line.strip() for line in gitignore if line.strip() and not line.startswith("#")]
    floating = {pattern.rstrip("/") for pattern in patterns if "/" not in pattern.rstrip("/")}
    anchored = {
        pattern.lstrip("/").rstrip("/")
        for pattern in patterns
        if pattern.startswith("/")
    }
    rows = []
    uncovered: list[str] = []
    for name in CACHE_NAMES:
        matches = sorted(relative(path) for path in ROOT.rglob(name))
        for match in matches:
            covered = name in floating or (match == name and name in anchored)
            if not covered:
                uncovered.append(match)
        rows.append(
            [
                name,
                "floating" if name in floating else ("anchored" if name in anchored else "NONE"),
                str(len(matches)),
                ", ".join(matches[:3]) if matches else "-",
            ]
        )
    out(format_table(["pattern", "gitignore form", "found", "paths"], rows))
    out("")
    out(f"uncovered : {', '.join(uncovered) if uncovered else 'none'}")
    out("")
    out.check(
        "every cache directory present in the tree is covered by .gitignore",
        not uncovered,
        f"{len(uncovered)} uncovered",
    )

    out.rule("3b. this script's own subprocesses create no cache directory")
    out("The absolute question -- is the tree clean right now -- cannot be asked")
    out("here, because whatever ran before this script may have left a cache")
    out("directory of its own and that is not this script's business. What IS this")
    out("script's business is that the subprocesses it starts add nothing, and that")
    out("is measured by a snapshot taken before and after.")
    out("")
    before = {name: set(ROOT.rglob(name)) for name in CACHE_NAMES}
    run(("terms",))
    run(("compare", "--observations", "60"))
    appeared = sorted(
        relative(path)
        for name in CACHE_NAMES
        for path in set(ROOT.rglob(name)) - before[name]
    )
    out(f"appeared after two subprocesses : {', '.join(appeared) if appeared else 'none'}")
    out("")
    out.check(
        "no cache directory appears while this script runs its subprocesses",
        not appeared,
        f"{len(appeared)} appeared",
    )

    out.rule("4. no committed file carries an absolute path")
    out("Release-gate check 8 fails a batch on an absolute home-directory path in any")
    out("TRACKED file, so the scan skips anything inside a directory .gitignore")
    out("excludes: a Hypothesis example database or a pytest cache is full of")
    out("absolute paths and is not committed. The patterns are assembled from os.sep")
    out("so that this check does not contain the string it looks for, and this")
    out("script's own output file is excluded because it is rewritten at the end of")
    out("this run. Three consecutive sessions of this mission introduced that defect")
    out("in the act of writing about it.")
    out("")
    import os

    needle = os.sep + "home" + os.sep
    alternative = os.sep + "root" + os.sep
    excluded_directories = set(CACHE_NAMES) | {"build", "dist", ".venv", "venv", "htmlcov"}
    offenders = []
    skipped = 0
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if excluded_directories & set(path.relative_to(ROOT).parts[:-1]):
            skipped += 1
            continue
        if path.suffix.lower() in {".png", ".pyc"}:
            continue
        if path.name == Path(out.path).name:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        if needle in text or alternative in text:
            offenders.append(relative(path))
    out(f"text files scanned : {scanned}")
    out(f"skipped as ignored : {skipped}")
    out(f"offenders          : {', '.join(offenders) if offenders else 'none'}")
    out("")
    out.check(
        "no scanned text file contains a home-directory absolute path",
        not offenders,
        f"{len(offenders)} offenders",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
