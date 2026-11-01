"""The command line, run as a subprocess, and the tree checked for pollution.

Two things are verified here that nothing else can verify in process.

**The CLI as a user runs it.** Every subcommand is executed with
``python -m transfergap`` in a subprocess, its exit status is checked and its
output is searched for the fragments the README quotes.

**The tree is clean.** A build can pollute its own repository through a
subprocess environment it sets: an earlier product of this mission set ``HOME``
to the product root and matplotlib wrote ``.config/matplotlib`` into the
repository, which the release gate fails twice over, once as a tracked build
artifact and once as an absolute path inside one. Every subprocess started
anywhere in this directory gets its ``HOME`` and ``MPLCONFIGDIR`` in a
temporary directory outside the repository and ``PYTHONDONTWRITEBYTECODE=1``,
from ``_bootstrap.subprocess_environment``. This script asserts the result:
no ``__pycache__``, ``.pytest_cache``, ``.ruff_cache``, ``.hypothesis`` or
``.config`` anywhere in the tree.

It also scans every tracked-looking file for an absolute path, because
committed output that quotes the build machine's filesystem layout is not
reproducible for a reader with a cold clone. The scanner assembles its
patterns from ``os.sep`` and deletes its own previous output before scanning,
because a scanner that does neither reports itself -- which three consecutive
sessions of this mission managed in the act of writing about it.
"""

from __future__ import annotations

import os
import subprocess
import sys

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from transfergap.report import format_table  # noqa: E402

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")

COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--version",), ("transfergap 0.1.0",)),
    (("bounds", "--n", "60", "--seed", "202"),
     ("hoeffding", "empirical_bernstein", "wilks_tolerance", "split_conformal",
      "dist-free")),
    (("bounds", "--n", "15", "--seed", "101"),
     ("[not available] wilks_tolerance_upper", "[not available] split_conformal_upper")),
    (("plan", "--n", "15", "--target-slack", "2.0"), ("MORE", "existence", "plug-in")),
    (("exchangeability", "--n", "120", "--campaign", "swept", "--permutations", "399"),
     ("trend", "half_split", "REJECTED")),
    (("learned", "--n", "60", "--seed", "202"),
     ("achieved coverage", "This model is not certified for operational flight use.")),
    (("truth", "--draws", "50000"), ("MC standard error", "clip fraction at R")),
)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("transfergap -- the command line, and the tree it must not pollute")
    for line in environment_lines():
        out(line)
    out("")
    environment = subprocess_environment()
    out("Every subprocess below runs with HOME and MPLCONFIGDIR in a temporary")
    out("directory outside the repository and PYTHONDONTWRITEBYTECODE=1.")

    out.rule("1. every subcommand, as a user runs it")
    rows = []
    for argv, fragments in COMMANDS:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "transfergap", *argv],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        missing = [f for f in fragments if f not in completed.stdout]
        rows.append([
            " ".join(argv)[:52],
            str(completed.returncode),
            str(len(completed.stdout.splitlines())),
            "ok" if completed.returncode == 0 and not missing else "FAILED",
        ])
        out.check(
            f"python -m transfergap {' '.join(argv)}",
            completed.returncode == 0 and not missing,
            f"exit {completed.returncode}"
            + (f"; missing fragments {missing}" if missing else ""),
        )
    out("")
    out(format_table(["command", "exit", "stdout lines", "verdict"], rows))

    out.rule("2. a bad argument must exit non-zero and say why")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "transfergap", "bounds", "--confidence", "1.5"],
        cwd=str(ROOT), capture_output=True, text=True, check=False, env=environment,
    )
    out(f"  exit status {completed.returncode}")
    stderr_tail = (
        completed.stderr.strip().splitlines()[-1][:100] if completed.stderr.strip() else ""
    )
    out(f"  stderr      {stderr_tail}")
    out.check(
        "an out-of-range confidence exits 2 with an actionable message",
        completed.returncode == 2 and "strictly inside" in completed.stderr,
        f"exit {completed.returncode}",
    )

    out.rule("3. the tree carries no cache or build directory")
    found = []
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in CACHE_NAMES:
            found.append(relative(path))
    out(f"  directories searched for : {list(CACHE_NAMES)}")
    out(f"  found                    : {found if found else 'none'}")
    out.check("no cache or config directory exists anywhere in the tree", not found,
              f"{len(found)} found")

    pyc = [relative(p) for p in ROOT.rglob("*.pyc")]
    out(f"  .pyc files               : {pyc if pyc else 'none'}")
    out.check("no compiled bytecode file exists anywhere in the tree", not pyc,
              f"{len(pyc)} found")

    out.rule("4. no committed file carries an absolute path")
    out("The patterns are assembled from os.sep, and this script's own output file is")
    out("deleted before the scan so that the scanner cannot report itself.")
    out("")
    separator = os.sep
    needles = (
        f"{separator}home{separator}",
        f"{separator}Users{separator}",
        f"{separator}root{separator}",
        f"{separator}tmp{separator}",
    )
    own_output = ROOT / "validation" / "validate_cli_output.txt"
    if own_output.exists():
        own_output.unlink()
    offenders = []
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix in {".png", ".pyc"}:
            continue
        if any(part in CACHE_NAMES or part == ".git" for part in path.parts):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        for needle in needles:
            if needle in text:
                # The needle is reported by NAME and not as the literal
                # pattern, because this file is committed and a committed file
                # that contains the literal pattern is itself what a
                # repository-wide absolute-path scanner is looking for.
                offenders.append(f"{relative(path)} contains the {needle.strip(separator)} prefix")
    out(f"  text files scanned : {scanned}")
    out(f"  needles            : {len(needles)} directory prefixes assembled from "
        f"os.sep -- {', '.join(needle.strip(separator) for needle in needles)}")
    out("    (named rather than printed as literals: this output file is committed,")
    out("     and a committed file containing the literal pattern is exactly what a")
    out("     repository-wide absolute-path scanner is looking for)")
    out(f"  offenders          : {offenders if offenders else 'none'}")
    out.check("no text file in the tree contains an absolute host path", not offenders,
              f"{len(offenders)} offenders in {scanned} files")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
