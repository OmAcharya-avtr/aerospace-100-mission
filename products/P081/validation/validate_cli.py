"""The command line, run as a subprocess, and the tree checked for pollution.

Three things are verified here that nothing else can verify in process.

**The CLI as a user runs it.** Every subcommand is executed with
``python -m twindiverge`` in a subprocess, its exit status is checked and its
output is searched for the fragments README.md quotes.

**The tree is clean.** A build can pollute its own repository through a
subprocess environment it sets: an earlier product of this mission set ``HOME``
to the product root and matplotlib wrote ``.config/matplotlib`` into the
repository, which the release gate fails twice over, once as a tracked build
artifact and once as an absolute path inside one. Every subprocess started
anywhere in this directory gets its ``HOME`` and ``MPLCONFIGDIR`` in a temporary
directory outside the repository and ``PYTHONDONTWRITEBYTECODE=1``, from
``_bootstrap.subprocess_environment``. This script asserts the result in the
only two forms it owns: that running it ADDS no such directory and no bytecode
file, and that git TRACKS none. It deliberately does NOT assert that none
exists, because the release gate lints from the product root before re-executing
this script and leaves ``.ruff_cache`` behind, so that form of the claim fails
on the harness's artifact and not on the product's.

**No committed file carries an absolute path.** The scanner assembles its
patterns from ``os.sep`` and deletes its own previous output before scanning,
because a scanner that does neither reports itself.
"""

from __future__ import annotations

# See validate_plant.py for why this statement is here and not in _bootstrap.
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

from twindiverge.report import format_table  # noqa: E402

CACHE_NAMES = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".config")

COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--version",), ("twindiverge 0.1.0",)),
    (("--help",), ("identify", "nullspace", "attribute", "profile", "crb", "sweep",
                   "learned", "telemetry")),
    (("telemetry",), ("rate_full", "rate_torque_full", "rate_coast",
                      "DECLARED, not measured")),
    (("identify", "--telemetry", "rate_full"),
     ("numerical rank           : 4 of 5", "REFUSED", "pinv diagonal (NOT a bound)",
      "identifiable combinations", "refusal is an answer, not an error")),
    (("identify", "--telemetry", "rate_torque_full"),
     ("numerical rank           : 5 of 5", "IDENTIFIABLE")),
    (("identify", "--telemetry", "rate_torque_short"), ("WEAK",)),
    (("nullspace", "--telemetry", "rate_full"),
     ("derived dimension  : 1", "computed dimension : 1", "worst principal angle")),
    (("nullspace", "--telemetry", "rate_coast"),
     ("derived dimension  : 2", "computed dimension : 2")),
    (("nullspace", "--telemetry", "rate_torque_full"),
     ("both subspaces are empty",)),
    (("attribute", "--telemetry", "rate_full", "--seed", "20000", "--kind", "single"),
     ("baseline 1, the gate", "baseline 2, attribution", "REFUSED",
      "NOT UNIQUE", "what this product delivers")),
    (("attribute", "--telemetry", "rate_full", "--seed", "20000",
      "--kind", "equivalent"),
     ("true label                         : MULTI", "REFUSED")),
    (("attribute", "--telemetry", "rate_torque_full", "--seed", "20000"),
     ("no structural degeneracy",)),
    (("profile", "--telemetry", "rate_short", "--seed", "20000", "--points", "5"),
     ("FLAT", "OVERRIDES", "profiles overriding covariance")),
    (("crb", "--telemetry", "rate_torque_short", "--realisations", "5"),
     ("measured sd", "Neither computation uses")),
    (("sweep",), ("The rank and the refusal list do not move",)),
    (("learned", "--telemetry", "rate_short", "--train-runs", "40",
      "--test-runs", "10", "--null-runs", "10"),
     ("population equivalent", "computation budget",
      "This model is not certified for operational flight use.")),
)

BAD_ARGUMENTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("identify", "--telemetry", "invented"), "unknown telemetry set"),
    (("profile", "--points", "4"), "--points must be an odd integer"),
    (("crb", "--realisations", "2"), "--realisations must be at least 4"),
    (("learned", "--train-runs", "4"), "--train-runs must be at least 10"),
    (("attribute", "--kind", "invented"), "invalid choice"),
    (("invent",), "invalid choice"),
)


def cache_artifacts() -> list[str]:
    """Every cache directory and bytecode file currently in the tree."""
    found = [
        relative(path)
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name in CACHE_NAMES
    ]
    found += [relative(path) for path in ROOT.rglob("*.pyc")]
    return sorted(found)


def summarise(names: list[str]) -> str:
    """Count first, then a few names. The full list can run to thirty lines."""
    if not names:
        return "none"
    shown = ", ".join(names[:4])
    extra = f", and {len(names) - 4} more" if len(names) > 4 else ""
    return f"{len(names)} -- {shown}{extra}"


def main() -> int:
    out = Tee("validate_cli_output.txt")
    # Snapshotted before this script starts anything. What a HARNESS left in the
    # tree before calling this script is not this product's doing and is
    # measured, not failed; what this script adds is this product's doing and is
    # failed.
    artifacts_at_entry = cache_artifacts()
    out.rule("twindiverge -- the command line, and the tree it must not pollute")
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
            [sys.executable, "-m", "twindiverge", *argv],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        missing = [fragment for fragment in fragments if fragment not in completed.stdout]
        rows.append([
            " ".join(argv)[:58],
            str(completed.returncode),
            str(len(completed.stdout.splitlines())),
            "ok" if completed.returncode == 0 and not missing else "FAILED",
        ])
        out.check(
            f"python -m twindiverge {' '.join(argv)}",
            completed.returncode == 0 and not missing,
            f"exit {completed.returncode}"
            + (f"; missing fragments {missing}" if missing else ""),
        )
    out("")
    out(format_table(["command", "exit", "stdout lines", "verdict"], rows))

    out.rule("2. a bad argument must exit non-zero and say why")
    rows = []
    for argv, fragment in BAD_ARGUMENTS:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "twindiverge", *argv],
            cwd=str(ROOT), capture_output=True, text=True, check=False,
            env=environment,
        )
        combined = completed.stderr + completed.stdout
        tail = (
            combined.strip().splitlines()[-1][:88] if combined.strip() else ""
        )
        rows.append([" ".join(argv)[:40], str(completed.returncode), tail])
        out.check(
            f"`{' '.join(argv)}` exits non-zero with an actionable message",
            completed.returncode != 0 and fragment in combined,
            f"exit {completed.returncode}; looked for {fragment!r}",
        )
    out("")
    out(format_table(["argv", "exit", "last line of output"], rows))

    out.rule("3. this script pollutes nothing, and nothing polluting is tracked")
    out("Two distinct claims, because they have two distinct owners.")
    out("")
    out("An earlier revision of this check, in another product of this mission,")
    out("asserted that NO cache directory and no bytecode file existed anywhere in")
    out("the tree at the moment it ran. That claim is not a product's to make and it")
    out("is unsatisfiable under the release gate: the gate runs `ruff check src/")
    out("tests/` with its working directory set to the product root as check 2,")
    out("which writes `.ruff_cache` into the tree, and then re-executes the")
    out("validation scripts as check 6. No change to the product could have passed.")
    out("")
    out("The two claims below are what this product can be held to, and both are")
    out("stricter than the old one in the direction that matters: the old check said")
    out("nothing at all about what is TRACKED, which is the only thing a reader with")
    out("a cold clone ever sees.")
    out("")
    out("3a. this script adds no cache directory and no bytecode file")
    artifacts_now = cache_artifacts()
    inherited = [name for name in artifacts_now if name in artifacts_at_entry]
    added = [name for name in artifacts_now if name not in artifacts_at_entry]
    out(f"  names treated as cache   : {list(CACHE_NAMES)}, plus every .pyc")
    out(f"  present at entry         : {summarise(artifacts_at_entry)}")
    out("    (inherited from whatever invoked this script; not this product's doing)")
    out(f"  still present, inherited : {summarise(inherited)}")
    out(f"  ADDED while this ran     : {summarise(added)}")
    out.check(
        "this script's own execution adds no cache directory or bytecode file",
        not added,
        f"{len(added)} added",
    )

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
        out.check(
            "git tracks no cache directory, bytecode file or build artifact",
            False,
            f"git ls-files exit {completed.returncode}",
        )
    else:
        tracked = [line for line in completed.stdout.splitlines() if line.strip()]
        offenders = sorted(
            name
            for name in tracked
            if name.endswith((".pyc", ".pyo"))
            or any(part in CACHE_NAMES for part in name.split("/"))
            or name.split("/")[0] in {"build", "dist"}
            or name.endswith(".egg-info")
        )
        out(f"  tracked files examined   : {len(tracked)}")
        out(f"  offenders                : {offenders or 'none'}")
        if not tracked:
            out("  NOTE: zero files were examined, so this check is VACUOUS in this")
            out("  environment and is recorded as such rather than as a pass.")
        out.check(
            "git tracks no cache directory, bytecode file or build artifact",
            not offenders,
            f"{len(offenders)} offenders in {len(tracked)} tracked files"
            + (" -- VACUOUS: zero files examined" if not tracked else ""),
        )

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
                # The needle is reported by NAME and not as the literal pattern,
                # because this file is committed and a committed file containing
                # the literal pattern is itself what a repository-wide
                # absolute-path scanner is looking for.
                offenders.append(
                    f"{relative(path)} contains the {needle.strip(separator)} prefix"
                )
    out(f"  text files scanned : {scanned}")
    out(f"  needles            : {len(needles)} directory prefixes assembled from "
        f"os.sep -- {', '.join(needle.strip(separator) for needle in needles)}")
    out("    (named rather than printed as literals: this output file is committed,")
    out("     and a committed file containing the literal pattern is exactly what a")
    out("     repository-wide absolute-path scanner is looking for)")
    out(f"  offenders          : {offenders if offenders else 'none'}")
    out.check(
        "no text file in the tree contains an absolute host path",
        not offenders,
        f"{len(offenders)} offenders in {scanned} files",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
