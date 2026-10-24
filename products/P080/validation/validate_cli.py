"""Every CLI invocation quoted in README.md, run here with its exit status recorded.

A command quoted in a README that nobody runs is a claim. Each command below
is executed through the package's own ``main`` entry point, its exit status is
checked against what the README says it should be, and its first lines are
captured into this file so the quoted output cannot drift from the code.

Two exit statuses are part of the interface and are asserted rather than
assumed: ``certify`` exits non-zero when the bound does not hold, and
``negative-control`` exits non-zero when the injected violation is *not*
caught. Both are usable as checks in a shell script.
"""

from __future__ import annotations

import contextlib
import io

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from boundcompress.__main__ import main as cli_main  # noqa: E402

COMMANDS: list[tuple[list[str], int]] = [
    (["--version"], 0),
    (["info"], 0),
    (["certify", "--signal", "battery_voltage", "--bound", "1e-3"], 0),
    (["certify", "--signal", "battery_voltage", "--bound", "1e-3",
      "--predictor", "previous"], 0),
    (["certify", "--signal", "panel_temperature", "--mode", "rel",
      "--bound", "1e-5"], 0),
    (["certify", "--signal", "wheel_rate", "--mode", "range_rel",
      "--bound", "1e-4"], 0),
    (["certify", "--signal", "gyro_rate", "--samples", "1024", "--bound", "1e-6",
      "--predictor", "learned", "--learned-kind", "linear"], 0),
    (["sweep", "--signal", "battery_voltage", "--points", "3"], 0),
    (["negative-control", "--signal", "battery_voltage", "--bound", "1e-3"], 0),
    (["compare", "--samples", "1024", "--points", "3"], 0),
]


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("boundcompress -- every CLI command quoted in the README, executed")
    for line in environment_lines():
        out(line)

    for argv, expected in COMMANDS:
        buffer = io.StringIO()
        status: int | None = None
        try:
            with contextlib.redirect_stdout(buffer):
                status = cli_main(argv)
        except SystemExit as exit_error:  # --version raises SystemExit(0)
            status = int(exit_error.code or 0)
        text = buffer.getvalue()
        out.rule(f"python -m boundcompress {' '.join(argv)}")
        for line in text.splitlines()[:26]:
            out(line)
        remaining = max(0, len(text.splitlines()) - 26)
        if remaining:
            out(f"... {remaining} further line(s) not captured")
        out("")
        out(f"exit status: {status} (expected {expected})")
        out.check(f"exit status of: {' '.join(argv)}", status == expected,
                  f"got {status}")

    out.rule("exit statuses that signal a failure")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = cli_main(
            ["negative-control", "--signal", "magnetometer", "--bound", "0.01",
             "--index", "0"]
        )
    out("negative-control exits 0 when it CATCHES the injected violation, so a")
    out("non-zero status from that subcommand means the certificate missed it.")
    out(f"negative-control --index 0 exit status: {status}")
    out.check("negative-control catches a violation at index 0", status == 0)

    path = out.save()
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
