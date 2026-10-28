"""Every command line entry point, executed in a subprocess, and the tree checked after.

The subprocess environment is deliberate. ``HOME`` and ``MPLCONFIGDIR`` are
pointed at a temporary directory outside the repository and
``PYTHONDONTWRITEBYTECODE`` is set, because a validation script in an earlier
product of this mission set ``HOME`` to the product root and matplotlib wrote
``.config/matplotlib`` into the repository. The release gate fails that twice
over: once as a tracked build artifact and once as an absolute path inside one.
The last section of this script asserts that the repository contains no such
directory after every command has run.
"""

from __future__ import annotations

import subprocess
import sys

from _bootstrap import Tee, Timer, bootstrap, environment_lines, subprocess_environment

ROOT = bootstrap()

from telemgap.report import format_table  # noqa: E402

#: Each command, with a fragment its output must contain.
COMMANDS: tuple[tuple[list[str], str], ...] = (
    (["--version"], "telemgap 0.1.0"),
    (["info"], "WHAT THIS RECORD CANNOT DETERMINE"),
    (["check", "--samples", "1200"], "INTEGRITY REPORT (mode=strengthened)"),
    (["check", "--samples", "1200", "--mode", "specified"], "mode=specified"),
    (["policy", "--samples", "1500"], "realised reconstruction error"),
    (
        ["policy", "--samples", "1500", "--no-resolvability-gate"],
        "GAP POLICY",
    ),
    (["refuse", "--outage-frames", "21600"], "REFUSAL CHECK PASSED"),
    (
        ["determinability", "--samples", "1200", "--withhold", "none"],
        "withheld: none",
    ),
    (
        ["determinability", "--samples", "1200", "--withhold", "validity"],
        "no per-frame error-control verdict is present",
    ),
    (
        ["determinability", "--samples", "1200", "--withhold", "raw"],
        "the raw words of equation (1) were not retained",
    ),
    (["benchmark", "--samples", "2000"], "baseline versus learned"),
)

#: Directories whose presence in the repository is a defect whatever
#: ``.gitignore`` says, because nothing in this product should ever create
#: them. ``.config`` is the one an earlier product of this mission actually
#: shipped, from a subprocess whose ``HOME`` pointed at the product root.
FORBIDDEN = (".config",)

#: Directories local tooling legitimately creates. Their presence is not a
#: defect, but every one of them must be covered by ``.gitignore``, or a cold
#: clone would carry a build artifact. The patterns are read from the committed
#: ``.gitignore`` rather than assumed.
TOOLING = ("__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis")


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("telemgap -- every command line entry point, in a subprocess")
    for line in environment_lines():
        out(line)
    out("")
    out("HOME and MPLCONFIGDIR are pointed outside the repository and")
    out("PYTHONDONTWRITEBYTECODE is set for every subprocess; see _bootstrap.py.")

    environment = subprocess_environment()
    timer = Timer()
    rows = []
    for arguments, fragment in COMMANDS:
        with timer:
            completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [sys.executable, "-m", "telemgap", *arguments],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
        found = fragment in completed.stdout
        rows.append(
            [
                " ".join(arguments),
                str(completed.returncode),
                str(len(completed.stdout.splitlines())),
                "yes" if found else "NO",
                "ok" if completed.returncode == 0 and found else "FAILED",
            ]
        )
        out.check(
            f"telemgap {' '.join(arguments)}",
            completed.returncode == 0 and found,
            f"exit {completed.returncode}, fragment {fragment!r} "
            f"{'found' if found else 'NOT FOUND'}"
            + (f", stderr: {completed.stderr.strip()[:200]}" if completed.stderr else ""),
        )
    out("")
    out(format_table(["command", "exit", "stdout lines", "fragment", "verdict"], rows))

    out.rule("an unknown command must fail rather than do something")
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "telemgap", "nonsense"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    out.check(
        "an unknown subcommand exits non-zero",
        completed.returncode != 0,
        f"exit {completed.returncode}",
    )
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "telemgap"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    out.check(
        "no subcommand exits non-zero",
        completed.returncode != 0,
        f"exit {completed.returncode}",
    )

    out.rule("the repository tree after every command has run")
    # A .gitignore pattern with a leading slash is anchored to the repository
    # root and covers only a root-level occurrence; one without a slash matches
    # at any depth. Both forms are parsed, because reading the file and
    # stripping every slash would silently treat an anchored pattern as a
    # floating one -- and the opposite mistake, a floating pattern where an
    # anchored one was meant, is what excluded a required committed fixture
    # from a cold clone in an earlier product of this mission.
    anchored: set[str] = set()
    floating: set[str] = set()
    for line in (ROOT / ".gitignore").read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("/"):
            anchored.add(text.strip("/"))
        else:
            floating.add(text.rstrip("/"))
    forbidden_found: list[str] = []
    uncovered: list[str] = []
    rows = []
    for name in FORBIDDEN + TOOLING:
        matches = sorted(
            str(path.relative_to(ROOT)) for path in ROOT.rglob(name) if ".venv" not in str(path)
        )
        nested = [match for match in matches if "/" in match]
        covered = name in floating or (name in anchored and not nested)
        if name in FORBIDDEN:
            forbidden_found.extend(matches)
        elif matches and not covered:
            uncovered.extend(matches)
        rows.append(
            [
                name,
                "forbidden" if name in FORBIDDEN else "tooling",
                str(len(matches)),
                "yes" if covered else "NO",
                str(matches[:3]) if matches else "-",
            ]
        )
    out(
        format_table(
            ["directory", "class", "occurrences", "in .gitignore", "first matches"], rows
        )
    )
    out("")
    out.check(
        "no forbidden artifact directory was written into the repository",
        not forbidden_found,
        f"{len(forbidden_found)} found: {forbidden_found[:6]}",
    )
    out.check(
        "every tooling directory present in the tree is covered by .gitignore",
        not uncovered,
        f"{len(uncovered)} uncovered: {uncovered[:6]}",
    )

    out.rule("command cost -- UNSTABLE, two cores shared with other work")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["commands run", str(timer.calls)],
                ["total s", f"{timer.elapsed_s:.2f}"],
                ["mean ms", f"{timer.mean_ms:.1f}"],
            ],
        )
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
