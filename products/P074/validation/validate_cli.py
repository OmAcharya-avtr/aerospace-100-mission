"""Every command-line invocation quoted in README.md, executed.

Each command is run as a subprocess with only ``PYTHONPATH=src`` set, which is
how the release gate runs them, and its exit status and a fragment of its
output are recorded. The point is that no command or output fragment quoted in
the documentation can drift from what the code actually prints.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

from _bootstrap import Tee, Timer, bootstrap, environment_lines, is_fast

ROOT = bootstrap()

_SPAN = "0.4" if is_fast() else "1.0"
_STEPS = "60" if is_fast() else "120"
_REPORT = "2" if is_fast() else "8"
"""Probe mode shortens the span but must still leave at least three reported ages.

A free-exponent growth fit needs three points at two or more distinct ages. A
first version of this probe used a span of 0.25 days with reports every eight
hours, which left one age, and the CLI correctly refused to fit -- so the probe
failed on a span it had chosen rather than on a defect.
"""

COMMANDS: tuple[tuple[str, tuple[str, ...], int, tuple[str, ...]], ...] = (
    (
        "version",
        ("version",),
        0,
        ("orbitdrift 0.1.0", "astropy      not installed", "not flight-qualified"),
    ),
    (
        "vectors",
        ("vectors",),
        0,
        ("33 cases, 666 rows compared, 1 rows skipped", "worst position disagreement"),
    ),
    (
        "atmosphere",
        ("atmosphere",),
        0,
        ("Table 8-4", "This model is static", "3.725000e-12"),
    ),
    (
        "drift",
        ("drift", "--span-days", _SPAN, "--steps-per-rev", _STEPS, "--report-every-hours", _REPORT),
        0,
        (
            "catalogue 25544",
            "model divergence",
            "lower bound on larger model error",
            "analytic growth forms fitted",
        ),
    ),
    (
        "stale",
        (
            "stale",
            "--span-days",
            _SPAN,
            "--steps-per-rev",
            _STEPS,
            "--report-every-hours",
            _REPORT,
            "--tolerance-km",
            "20",
            "--age-days",
            "3",
        ),
        0,
        ("median horizon", "conservative horizon", "no measured backing", "verdict"),
    ),
    (
        "stale with a refused purpose",
        (
            "stale",
            "--span-days",
            _SPAN,
            "--steps-per-rev",
            _STEPS,
            "--purpose",
            "covariance for conjunction screening",
        ),
        3,
        (),
    ),
    ("help", ("--help",), 0, ("No output is a covariance", "P071 ConjProb")),
)


def main() -> int:
    out = Tee("validate_cli_output.txt")
    out.rule("orbitdrift -- every documented command, executed")
    for line in environment_lines():
        out(line)
    out("")
    out("  Each command runs as a subprocess with PYTHONPATH=src and nothing else,")
    out("  which is how the release gate invokes it.")

    commands = (
        tuple(item for item in COMMANDS if item[0] in ("version", "drift", "stale", "help"))
        if is_fast()
        else COMMANDS
    )
    # HOME and MPLCONFIGDIR point at a temporary directory OUTSIDE the product.
    # An earlier version set HOME to the product root, so matplotlib wrote a
    # .config/matplotlib cache into the repository on every run -- a directory
    # carrying an absolute container path that must never be committed. Byte
    # code is suppressed for the same reason.
    scratch = tempfile.TemporaryDirectory()
    timer = Timer()
    for label, arguments, expected_status, fragments in commands:
        command = [sys.executable, "-m", "orbitdrift", *arguments]
        with timer:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": "src",
                    "MPLBACKEND": "Agg",
                    "HOME": scratch.name,
                    "MPLCONFIGDIR": scratch.name,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                capture_output=True,
                text=True,
                check=False,
            )
        out.rule(f"python -m orbitdrift {' '.join(arguments)}")
        out(f"  exit status                   : {completed.returncode} "
            f"(expected {expected_status})")
        out.check(
            f"{label}: exit status is {expected_status}",
            completed.returncode == expected_status,
            f"got {completed.returncode}; stderr: {completed.stderr.strip()[:120]}",
        )
        text = completed.stdout + completed.stderr
        for fragment in fragments:
            out.check(f"{label}: output contains {fragment!r}", fragment in text, "")
        if label == "stale with a refused purpose":
            out.check(
                f"{label}: the refusal names P071",
                "P071" in completed.stderr,
                completed.stderr.strip()[:120],
            )
        shown = [line for line in completed.stdout.splitlines() if line.strip()]
        for line in shown[: 6 if is_fast() else 24]:
            out(f"    {line}")
        if len(shown) > 24 and not is_fast():
            out(f"    ... {len(shown)} lines in all")

    scratch.cleanup()

    out.rule("timings (UNSTABLE: wall clock on a shared two-core container)")
    out(f"  {timer.calls} invocations           : {timer.elapsed_s:.1f} s total")

    out("")
    out(f"wrote {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    sys.exit(main())
