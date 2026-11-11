"""The command line, run as a subprocess, and the tree checked for pollution.

Two things are verified here that nothing else can verify in process.

**The CLI as a user runs it.** Every subcommand is executed with
``python -m boardprobe`` in a subprocess, its exit status is checked and its
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

from boardprobe import format_table  # noqa: E402

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")

COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--version",), ("boardprobe 0.1.0",)),
    (("channels",),
     ("channels declared: 14", "channels verified on hardware: 0 of 14",
      "UNVERIFIED ON HARDWARE", "monotonic_clock", "perf_cpu_cycles")),
    (("provenance",),
     ("meet is the greatest lower bound", "REFUSED, not taken as the identity",
      "No figure in this repository was measured", "Level 3, hardware-pending")),
    (("attest",),
     ("declared target : NVIDIA Jetson Orin Nano", "matches target  : NO",
      "MEASURED OFF-TARGET")),
    (("sample", "--backend", "simulated", "--seed", "3"),
     ("backend provenance: simulated", "SYNTHETIC", "die_temp_cpu")),
    (("sample", "--backend", "live"),
     ("monotonic_clock", "channels refused", "MEASURED OFF-TARGET")),
    (("replay", "--samples", "6"),
     ("content provenance  : recorded", "replay provenance   : recorded",
      "REPLAYED", "SYNTHESISED IN A CONTAINER")),
    (("benchmark", "--repeats", "24", "--overhead-repeats", "60"),
     ("SYNTHETIC", "overhead inside the bracket", "one clock read (independent)",
      "NOT of an")),
    (("benchmark", "--repeats", "24", "--environment", "none",
      "--overhead-repeats", "60"),
     ("MEASURED OFF-TARGET", "perturbation of the reference payload")),
    (("dry-run",),
     ("device accesses against the real root : 0",
      "side-effect free                      : yes",
      "MEASURED on simulated inputs refused")),
)


def cache_artifacts() -> list[str]:
    """Every cache directory and bytecode file currently in the tree."""
    found = [relative(path) for path in ROOT.rglob("*")
             if path.is_dir() and path.name in CACHE_NAMES]
    found += [relative(path) for path in ROOT.rglob("*.pyc")]
    return sorted(found)


def condense(names: list[str], limit: int = 6) -> str:
    """A short rendering of a possibly long artifact list.

    The full list of inherited bytecode files runs to dozens of entries and
    says nothing a count and a sample do not. What matters is the COUNT and
    the directories involved, so that is what is printed.
    """
    if not names:
        return "none"
    directories = sorted({name.rsplit("/", 1)[0] if "/" in name else name
                          for name in names})
    shown = ", ".join(directories[:limit])
    more = "" if len(directories) <= limit else f", +{len(directories) - limit} more"
    return f"{len(names)} entries under {shown}{more}"


def main() -> int:
    out = Tee("validate_cli_output.txt")
    # Snapshotted before this script starts anything. What a HARNESS left in
    # the tree before calling this script is not this product's doing and is
    # measured, not failed; what this script adds is this product's doing and
    # is failed. See section 3.
    artifacts_at_entry = cache_artifacts()
    out.rule("boardprobe -- the command line, and the tree it must not pollute")
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
            [sys.executable, "-m", "boardprobe", *argv],
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
            f"python -m boardprobe {' '.join(argv)}",
            completed.returncode == 0 and not missing,
            f"exit {completed.returncode}"
            + (f"; missing fragments {missing}" if missing else ""),
        )
    out("")
    out(format_table(["command", "exit", "stdout lines", "verdict"], rows))

    out.rule("2. a bad argument must exit non-zero and say why")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "boardprobe", "sample", "--backend", "psychic"],
        cwd=str(ROOT), capture_output=True, text=True, check=False, env=environment,
    )
    out(f"  exit status {completed.returncode}")
    stderr_tail = (
        completed.stderr.strip().splitlines()[-1][:100] if completed.stderr.strip() else ""
    )
    out(f"  stderr      {stderr_tail}")
    out.check(
        "an unknown backend exits 2 and lists the three that exist",
        completed.returncode == 2 and "simulated" in completed.stderr,
        f"exit {completed.returncode}",
    )
    out("")
    out("And the board-absent benchmark, which must exit 1 rather than traceback:")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "boardprobe", "benchmark", "--repeats", "4",
         "--environment", "live"],
        cwd=str(ROOT), capture_output=True, text=True, check=False, env=environment,
    )
    out(f"  exit status {completed.returncode}")
    tail = (
        completed.stderr.strip().splitlines()[-1][:140]
        if completed.stderr.strip()
        else ""
    )
    out(f"  stderr      {tail}")
    out.check(
        "a live environment backend on a board-less host exits 1 with the "
        "recovery pointer and no traceback",
        completed.returncode == 1
        and "RECOVERY.md" in completed.stderr
        and "Traceback" not in completed.stderr,
        f"exit {completed.returncode}",
    )

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
    out(f"  names treated as cache   : {list(CACHE_NAMES)}, plus every .pyc")
    out(f"  present at entry         : {condense(artifacts_at_entry)}")
    out("    (inherited from whatever invoked this script; not this product's doing)")
    out(f"  still present, inherited : {condense(inherited)}")
    out(f"  ADDED while this ran     : {added or 'none'}")
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
    tracked = [line for line in completed.stdout.splitlines() if line.strip()] \
        if completed.returncode == 0 else []
    if completed.returncode != 0 or not tracked:
        out("  STRUCTURALLY UNMEASURABLE BY A BUILD AGENT, and reported as such rather")
        out("  than passed on an empty set. This product's working tree is UNCOMMITTED")
        out("  while it is being built: the mission's coordinator holds every commit")
        out("  under ADR-010, so `git ls-files` inside this directory returns nothing")
        out("  and a check over nothing would report a vacuous PASS -- which is worse")
        out("  than reporting that it could not be measured, because it reads as")
        out("  evidence.")
        out(f"  git ls-files exit status : {completed.returncode}")
        out(f"  tracked files listed     : {len(tracked)}")
        out("  What CAN be measured here is what .gitignore would exclude, which is")
        out("  the next best thing and is measured below. The claim itself is")
        out("  re-measurable, and must be re-measured, by whoever commits this tree;")
        out("  release-gate check 9 enforces it independently over the whole")
        out("  repository.")
        out("")
        out("  what .gitignore excludes, per shipped fixture and data file:")
        for candidate in sorted((ROOT / "fixtures").glob("*")):
            ignore = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", "check-ignore", "-v", "--no-index", relative(candidate)],
                cwd=str(ROOT), capture_output=True, text=True, check=False,
                env=subprocess_environment(),
            )
            verdict = "IGNORED" if ignore.returncode == 0 else "not ignored"
            out(f"    {relative(candidate):<52s} {verdict}")
            out.check(
                f"the shipped fixture {candidate.name} is not excluded by .gitignore",
                ignore.returncode != 0,
                verdict,
            )
    else:
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
