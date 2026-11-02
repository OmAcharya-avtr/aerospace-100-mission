"""The command line, run as a subprocess, and the tree checked for pollution.

Two things are verified here that nothing else can verify in process.

**The CLI as a user runs it.** Every subcommand is executed with
``python -m hilverdict`` in a subprocess, its exit status is checked and its
output is searched for the fragments the README quotes.

**The tree is clean.** A build can pollute its own repository through a
subprocess environment it sets: an earlier product of this mission set ``HOME``
to the product root and matplotlib wrote ``.config/matplotlib`` into the
repository, which the release gate fails twice over, once as a tracked build
artifact and once as an absolute path inside one. Every subprocess started
anywhere in this directory gets its ``HOME`` and ``MPLCONFIGDIR`` in a
temporary directory outside the repository and ``PYTHONDONTWRITEBYTECODE=1``,
from ``_bootstrap.subprocess_environment``. This script asserts the
result in the only two forms it owns: that running it ADDS no such directory
and no bytecode file, and that git TRACKS none. It deliberately does not
assert that none exists, because the release gate lints from the product root
before re-executing this script and leaves ``.ruff_cache`` behind, so that
form of the claim fails on the harness's artifact and not on the product's --
the fifth instance of the self-reporting-scanner class in this mission.

It also scans every tracked-looking file for an absolute path, because
committed output that quotes the build machine's filesystem layout is not
reproducible for a reader with a cold clone. The scanner assembles its
patterns from ``os.sep`` and deletes its own previous output before scanning,
because a scanner that does neither reports itself -- which three consecutive
sessions of this mission managed in the act of writing about it.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that no __pycache__ directory and no .pyc file
# exists in the repository, so an interpreter that caches its own imports
# here makes that check report itself. It is set in each entry script,
# before _bootstrap is imported, and not inside _bootstrap, because CPython
# writes a module's cache BEFORE executing that module: _bootstrap cannot
# set this on its own behalf. The imports below therefore follow a statement
# and carry the same E402 waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import os  # noqa: E402
import subprocess  # noqa: E402

from _bootstrap import (  # noqa: E402
    Tee,
    bootstrap,
    environment_lines,
    relative,
    subprocess_environment,
)

ROOT = bootstrap()

from hilverdict.report import format_table  # noqa: E402

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")

COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--version",), ("hilverdict 0.1.0",)),
    (("--help",), ("hardware-in-the-loop", "certify", "resimulate", "clock",
                   "verdict", "threshold", "learned")),
    (("threshold", "--steps", "200"),
     ("certified overrun [ms]", "gamma_sup, eq (12)", "metric anisotropy",
      "That is a proof about the")),
    (("certify", "--seed", "902", "--profile", "loaded", "--amplitude-deg", "8.0"),
     ("DISMISSED", "UNDECIDED", "certified overrun", "dismissal rate",
      "is a theorem about the declared linearisation")),
    (("resimulate", "--seed", "902", "--profile", "loaded", "--amplitude-deg", "8.0"),
     ("peak divergence", "CONSEQUENTIAL", "plant evaluations spent")),
    (("verdict", "--seed", "902", "--profile", "loaded", "--amplitude-deg", "8.0"),
     ("RUN VERDICT: NOT_TIMING_FAITHFUL", "not a certification",
      "cascade / direct overruns")),
    (("clock", "--seed", "902", "--drift-ppm", "5000"),
     ("CLOCK VERDICT: CLOCK_DRIFTING", "theil-sen", "5000")),
    (("clock", "--seed", "902", "--resets", "0.4"),
     ("CLOCK VERDICT: CLOCK_RESET", "makes the run undecidable")),
    (("verdict", "--seed", "902", "--resets", "0.4"),
     ("RUN VERDICT: UNDECIDABLE", "reset")),
    (("certify", "--seed", "902", "--amplitude-deg", "14.0"),
     ("REFUSED", "saturat")),
    (("learned", "--train-runs", "9", "--test-runs", "6", "--steps", "120"),
     ("analytic certificate", "recall",
      "This model is not certified for operational flight use.")),
)


def cache_artifacts() -> list[str]:
    """Every cache directory and bytecode file currently in the tree."""
    found = [relative(path) for path in ROOT.rglob("*")
             if path.is_dir() and path.name in CACHE_NAMES]
    found += [relative(path) for path in ROOT.rglob("*.pyc")]
    return sorted(found)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    # Snapshotted before this script starts anything. What a HARNESS left in
    # the tree before calling this script is not this product's doing and is
    # measured, not failed; what this script adds is this product's doing and
    # is failed. See section 3.
    artifacts_at_entry = cache_artifacts()
    out.rule("hilverdict -- the command line, and the tree it must not pollute")
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
            [sys.executable, "-m", "hilverdict", *argv],
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
            f"python -m hilverdict {' '.join(argv)}",
            completed.returncode == 0 and not missing,
            f"exit {completed.returncode}"
            + (f"; missing fragments {missing}" if missing else ""),
        )
    out("")
    out(format_table(["command", "exit", "stdout lines", "verdict"], rows))

    out.rule("2. a bad argument must exit non-zero and say why")
    rows = []
    cases = (
        (("certify", "--tolerance", "0"), "strictly positive"),
        (("certify", "--steps", "1"), "--steps must be at least 2"),
        (("certify", "--profile", "turbo"), "invalid choice"),
        (("learned", "--target-recall", "1.5"), "--target-recall must lie"),
        (("invent",), "invalid choice"),
    )
    for argv, fragment in cases:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "hilverdict", *argv],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
            env=environment,
        )
        tail = (
            completed.stderr.strip().splitlines()[-1][:90]
            if completed.stderr.strip()
            else ""
        )
        rows.append([" ".join(argv)[:34], str(completed.returncode), tail])
        out.check(
            f"`{' '.join(argv)}` exits 2 with an actionable message",
            completed.returncode == 2 and fragment in completed.stderr,
            f"exit {completed.returncode}",
        )
    out("")
    out(format_table(["argv", "exit", "last stderr line"], rows))

    out.rule("3. this script pollutes nothing, and nothing polluting is tracked")
    out("Two distinct claims, because they have two distinct owners.")
    out("")
    out("An earlier revision of this script asserted that NO cache directory and no")
    out("bytecode file existed anywhere in the tree at the moment it ran. That claim")
    out("is not this product's to make, and it is unsatisfiable under the release")
    out("gate: the gate runs `ruff check src/ tests/` with its working directory set")
    out("to the product root as check 2, which writes `.ruff_cache` into the tree,")
    out("and then re-executes this script as check 6, which failed on the directory")
    out("the gate itself had just created. No change to this product could have")
    out("passed. Measured 2026-11-02 by running the two steps in the gate's own")
    out("order. The gate already prevents the identical pytest side effect with")
    out("`-p no:cacheprovider`; the ruff case is the same omission and is reported")
    out("to the owner rather than patched here, because this product does not edit")
    out("the gate that judges it.")
    out("")
    out("The two claims below are what this product can be held to, and both are")
    out("stricter than the old one in the direction that matters: the old check")
    out("said nothing at all about what is TRACKED, which is the only thing a")
    out("reader with a cold clone ever sees.")

    out("")
    out("3a. this script adds no cache directory and no bytecode file")
    artifacts_now = cache_artifacts()
    inherited = [a for a in artifacts_now if a in artifacts_at_entry]
    added = [a for a in artifacts_now if a not in artifacts_at_entry]
    def summarise(names: list[str]) -> str:
        """Count first, then a few names. The full list can run to thirty lines."""
        if not names:
            return "none"
        shown = ", ".join(names[:4])
        extra = f", and {len(names) - 4} more" if len(names) > 4 else ""
        return f"{len(names)} -- {shown}{extra}"

    out(f"  names treated as cache   : {list(CACHE_NAMES)}, plus every .pyc")
    out(f"  present at entry         : {summarise(artifacts_at_entry)}")
    out("    (inherited from whatever invoked this script; not this product's doing)")
    out(f"  still present, inherited : {summarise(inherited)}")
    out(f"  ADDED while this ran     : {summarise(added)}")
    out.check("this script's own execution adds no cache directory or bytecode file",
              not added, f"{len(added)} added")

    out("")
    out("3b. git tracks no cache directory, bytecode file or build artifact")
    out("Authoritative, because .gitignore can be edited and a tracked file stays")
    out("tracked regardless of it. A cold clone carries exactly what git tracks.")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "ls-files"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
        env=subprocess_environment(),
    )
    if completed.returncode != 0:
        out("  git ls-files unavailable here; the claim below cannot be measured and")
        out("  is reported as a FAILED check rather than quietly skipped.")
        out.check("git tracks no cache directory, bytecode file or build artifact",
                  False, f"git ls-files exit {completed.returncode}")
    else:
        tracked = [line for line in completed.stdout.splitlines() if line.strip()]
        offenders = sorted(
            name for name in tracked
            if name.endswith((".pyc", ".pyo"))
            or any(part in CACHE_NAMES for part in name.split("/"))
            or name.split("/")[0] in {"build", "dist"}
            or name.endswith(".egg-info")
        )
        out(f"  tracked files examined   : {len(tracked)}")
        out(f"  offenders                : {offenders or 'none'}")
        out.check("git tracks no cache directory, bytecode file or build artifact",
                  not offenders, f"{len(offenders)} offenders in {len(tracked)} tracked files")

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
