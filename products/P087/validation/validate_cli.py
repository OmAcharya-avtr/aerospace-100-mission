"""Every command line invocation quoted in README.md, run as a subprocess.

Each subprocess is given a temporary ``HOME`` and ``MPLCONFIGDIR`` outside the
repository and ``PYTHONDONTWRITEBYTECODE=1``, because a validation script in an
earlier product of this mission set ``HOME`` to the product root and matplotlib
wrote ``.config/matplotlib`` into the repository.

``python -m refractpath references`` exits 1 by design: one of the six external
reference checks does not pass, and the command says so. This script asserts
that exit status rather than treating it as a failure.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

from _bootstrap import Tee, bootstrap, environment_lines, relative, subprocess_environment

ROOT = bootstrap()

from refractpath import format_table  # noqa: E402

#: Command, expected exit status, and a fragment that must appear in stdout.
COMMANDS: tuple[tuple[tuple[str, ...], int, str], ...] = (
    (("--version",), 0, "refractpath"),
    (("info",), 0, "Not flight-qualified"),
    (("refract", "--elevation", "45"), 0, "VERDICT: USABLE"),
    (("refract", "--elevation", "80", "--tolerance", "1"), 0, "VERDICT:"),
    (("refract", "--elevation", "5", "--tolerance", "10"), 0, "VERDICT: REFUSE"),
    (
        (
            "refract",
            "--elevation",
            "30",
            "--pressure",
            "780",
            "--temperature",
            "-8",
            "--humidity",
            "0.35",
            "--wavelength",
            "0.85",
            "--observer-height",
            "2200",
        ),
        0,
        "profile envelope",
    ),
    (("curve", "--points", "13"), 0, "spread"),
    (("curve", "--low", "0", "--high", "3", "--points", "7"), 0, "undefined"),
    (("floor", "--tolerance", "10"), 0, "measured validity floor"),
    (("floor", "--tolerance", "1"), 0, "tolerance crossings on grid"),
    (("dispersion", "--points", "8"), 0, "rel spread"),
    (("references",), 1, "bennett_accuracy_claim"),
    (("refract", "--humidity", "43"), 2, ""),
    (("refract", "--elevation", "100"), 2, ""),
    (("floor", "--tolerance", "0"), 2, ""),
)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("refractpath -- command line, every invocation run as a subprocess")
    for line in environment_lines():
        out(line)
    out("")
    out("Subprocess environment: HOME and MPLCONFIGDIR point at a temporary directory")
    out("outside this repository and PYTHONDONTWRITEBYTECODE is set, so a subprocess")
    out("cannot write a cache into the tree.")

    #: Cache directories present BEFORE any subprocess runs. The property this
    #: script tests is that spawning a subprocess adds none, which is the
    #: defect an earlier product of this mission shipped. Directories already
    #: present are left alone: every pattern below is matched by .gitignore, so
    #: none of them is a committed file, and a run of pytest or ruff in the
    #: working tree legitimately creates them.
    CACHE_PATTERNS = (
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".config",
    )

    def cache_directories() -> set[str]:
        return {
            relative(path)
            for pattern in CACHE_PATTERNS
            for path in ROOT.rglob(pattern)
        }

    before = cache_directories()
    environment = subprocess_environment()
    rows = []
    for argv, expected_status, fragment in COMMANDS:
        completed = subprocess.run(  # fixed argv, no shell
            [sys.executable, "-m", "refractpath", *argv],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        status_ok = completed.returncode == expected_status
        fragment_ok = fragment in completed.stdout if fragment else True
        rows.append(
            [
                " ".join(argv)[:58],
                str(expected_status),
                str(completed.returncode),
                "yes" if status_ok else "NO",
                "yes" if fragment_ok else "NO",
                f"{len(completed.stdout)}",
            ]
        )
        out.check(
            f"python -m refractpath {' '.join(argv)}",
            status_ok and fragment_ok,
            f"exit {completed.returncode} (expected {expected_status})"
            + (f", stdout contains {fragment!r}: {fragment_ok}" if fragment else "")
            + (
                f", stderr: {completed.stderr.strip()[:160]}"
                if completed.returncode not in (0, 1, 2)
                else ""
            ),
        )

    out.rule("summary")
    out(
        format_table(
            ["command", "expected", "actual", "status ok", "fragment ok", "stdout bytes"],
            rows,
        )
    )
    out("")
    out("The `references` row expects exit 1. One of the six external reference checks")
    out("fails and the command reports it; an exit 0 there would mean the failure had")
    out("been hidden.")

    out.rule("the subprocess environment cannot pollute this repository")
    # Two things are checked here and they are not the same strength.
    #
    # The DETERMINISTIC check is on the environment itself: HOME and
    # MPLCONFIGDIR must point outside the repository and
    # PYTHONDONTWRITEBYTECODE must be set. That is the property P074 of this
    # mission got wrong, it is what the subprocesses above actually ran with,
    # and it cannot be disturbed by anything else on the machine.
    #
    # The differential over cache directories is reported but is NOT a check,
    # and the reason is worth stating: an unrelated `pytest` or `ruff` run in
    # the same working tree creates `.pytest_cache` and `.ruff_cache` at any
    # moment, and probe_working_directories.py runs scripts two at a time, so a
    # tree-wide differential has a race in it that has nothing to do with the
    # defect it is looking for. Measured during this build: this check failed
    # once in 90 probe runs, caused by a `ruff check` the build agent ran in
    # another shell while the probe was in flight.
    home = pathlib.Path(environment["HOME"]).resolve()
    mplconfig = pathlib.Path(environment["MPLCONFIGDIR"]).resolve()
    out(f"HOME outside the repository                  : {ROOT not in home.parents}")
    out(f"MPLCONFIGDIR outside the repository          : {ROOT not in mplconfig.parents}")
    out(f"PYTHONDONTWRITEBYTECODE                      : "
        f"{environment.get('PYTHONDONTWRITEBYTECODE')!r}")
    out(f"MPLBACKEND                                   : "
        f"{environment.get('MPLBACKEND')!r}")
    out.check(
        "the subprocess environment points HOME and MPLCONFIGDIR outside the "
        "repository and disables bytecode writing",
        ROOT not in home.parents
        and ROOT not in mplconfig.parents
        and environment.get("PYTHONDONTWRITEBYTECODE") == "1",
        "this is the exact property an earlier product of this mission got wrong, "
        "and it is deterministic",
    )
    stray_config = sorted(relative(path) for path in ROOT.rglob(".config"))
    out.check(
        "no .config directory exists anywhere inside the repository",
        not stray_config,
        "none found"
        if not stray_config
        else f"found: {', '.join(stray_config)}; matplotlib writes .config/matplotlib "
        "under HOME, so one here would mean HOME had been pointed at the tree",
    )
    after = cache_directories()
    created = sorted(after - before)
    out("")
    out(f"cache directories created during this run (informational, not a check) : "
        f"{len(created)}")
    out(f"which : {', '.join(created) or 'none'}")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
