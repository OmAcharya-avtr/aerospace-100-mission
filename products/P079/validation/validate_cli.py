"""Check 11 -- every command-line entry point, run and captured.

Each command quoted in README.md is executed here through
:func:`timetransfer.cli.main`, its exit status is checked, and a fragment of
its output is printed, so the README's quoted output cannot drift from what the
code prints.

The commands are run in-process rather than as subprocesses so that the
captured output is exactly what a reader will see, without a shell quoting
layer in between.

Run: ``python validation/validate_cli.py``
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

from _bootstrap import Tee, bootstrap, environment_lines, relative, scaled

ROOT = bootstrap()

from timetransfer.cli import main as cli_main  # noqa: E402

SCRATCH = Path("fixtures") / "_scratch_cli_recording.json"

_N = str(scaled(600, minimum=40))
_N_SMALL = str(scaled(300, minimum=40))
_N_REC = str(scaled(200, minimum=40))

COMMANDS: tuple[tuple[str, list[str], int, int], ...] = (
    ("version", ["version"], 0, 2),
    ("info", ["info"], 0, 14),
    ("simulate", ["simulate", "--n", _N, "--asymmetry", "8e-4",
                  "--asymmetry-bound", "1.6e-3"], 0, 40),
    ("allan", ["allan", "--n", _N, "--noise-kind", "white_fm"], 0, 10),
    ("parity", ["parity", "--n", _N_SMALL, "--asymmetry", "8e-4", "--loss", "0.05"], 0, 6),
    ("preflight", ["preflight", "--n", "60"], 0, 12),
    ("record", ["record", "--n", _N_REC, "--loss", "0.04", str(SCRATCH)], 0, 2),
    ("dryrun", ["dryrun", str(SCRATCH)], 0, 14),
    ("bench", ["bench", "--n", _N_SMALL, "--repeats", "3", "--note",
               "validation run on a shared two-core container"], 0, 40),
    ("missing recording", ["dryrun", "fixtures/does-not-exist.json"], 2, 1),
    ("link too lossy", ["simulate", "--n", "10", "--loss", "0.999"], 2, 1),
)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out("timetransfer validation 11 -- the command-line interface")
    out("=" * 78)
    for line in environment_lines():
        out(line)
    failures = 0

    for label, argv, expected_status, show in COMMANDS:
        out.rule(f"$ python -m timetransfer {' '.join(argv)}")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = int(cli_main(argv))
        text = stdout.getvalue().splitlines()
        err = stderr.getvalue().splitlines()
        for line in text[:show]:
            out(line)
        if len(text) > show:
            out(f"... ({len(text) - show} further lines)")
        for line in err[:3]:
            out(f"stderr: {line}")
        out("")
        out(f"exit status : {status} (expected {expected_status})")
        if status != expected_status:
            failures += 1
            out(f"FAIL: {label}")

    SCRATCH.unlink(missing_ok=True)
    out.rule("cleanup")
    out(f"scratch recording removed : {not SCRATCH.exists()} ({relative(SCRATCH)})")

    out.rule("summary")
    out(f"commands run  : {len(COMMANDS)}")
    out(f"hard failures : {failures}")
    out(f"saved to      : {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
