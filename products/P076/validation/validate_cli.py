"""Run every command-line invocation quoted in README.md and record its output.

The README quotes the output of ``python -m pointahead info``, of
``python -m pointahead scenario leo_crossing`` and of two case-file
invocations. This script executes each of them as a subprocess, records the
exit status and the full output, and checks the error paths as well as the
success paths -- a CLI that exits zero on a missing file is worse than one that
crashes.

The subprocess environment carries ``PYTHONPATH`` pointing at ``src``, so this
works from a cold clone with nothing installed, which is how a reader will
first run it.
"""

from __future__ import annotations

import subprocess
import sys

from _bootstrap import ROOT, Tee, bootstrap, header, relative

bootstrap()

INVOCATIONS = (
    (["--version"], 0),
    (["info"], 0),
    (["scenario", "leo_crossing"], 0),
    (["scenario", "leo_trailing"], 0),
    (["scenario", "geo_relay_to_leo"], 0),
    (["budget", "examples/cases/leo_crossing.yaml"], 0),
    (["compare", "examples/cases/geo_relay_to_leo.yaml"], 0),
    (["lighttime", "examples/cases/leo_trailing.yaml"], 0),
)

FAILURES = (
    (["budget", "examples/cases/absent.yaml"], "case file not found"),
    (["scenario", "mars_relay"], "invalid choice"),
    ([], "the following arguments are required"),
)


def run(arguments: list[str]) -> subprocess.CompletedProcess:
    """Invoke the CLI as a subprocess from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "pointahead", *arguments],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env={"PYTHONPATH": str(relative(ROOT / "src")), "PATH": "/usr/bin:/bin"},
        check=False,
        timeout=600,
    )


def main() -> int:
    tee = Tee("validate_cli_output.txt")
    header(tee, "command-line interface: every invocation quoted in README.md")
    tee("")
    tee("  PYTHONPATH is set to the repository-relative src directory and the")
    tee("  working directory is the repository root, so these are the commands a")
    tee("  reader runs from a cold clone with nothing installed.")

    for arguments, expected in INVOCATIONS:
        tee.rule(f"python -m pointahead {' '.join(arguments)}")
        result = run(arguments)
        tee(f"  exit status                        : {result.returncode}")
        tee("")
        for line in (result.stdout or "").splitlines():
            tee(f"  | {line}")
        if result.stderr:
            tee("")
            tee("  stderr:")
            for line in result.stderr.splitlines():
                tee(f"  | {line}")
        tee("")
        tee.check(
            f"'{' '.join(arguments) or '(no arguments)'}' exits {expected}",
            result.returncode == expected,
            f"got {result.returncode}",
        )

    tee.rule("error paths")
    for arguments, fragment in FAILURES:
        result = run(arguments)
        tee("")
        tee(f"  python -m pointahead {' '.join(arguments) or '(no arguments)'}")
        tee(f"    exit status                      : {result.returncode}")
        tee(f"    stderr                           : "
            f"{(result.stderr or '').strip().splitlines()[-1] if result.stderr else '(empty)'}")
        tee.check(
            f"'{' '.join(arguments) or '(no arguments)'}' exits non-zero",
            result.returncode != 0,
            f"got {result.returncode}",
        )
        tee.check(
            f"'{' '.join(arguments) or '(no arguments)'}' names the problem",
            fragment in (result.stderr or ""),
            f"looking for {fragment!r}",
        )
        tee.check(
            f"'{' '.join(arguments) or '(no arguments)'}' prints no traceback",
            "Traceback" not in (result.stderr or ""),
        )

    tee.summarise_failures()
    tee("")
    tee(f"  written: {tee.save()}")
    return 1 if tee.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
