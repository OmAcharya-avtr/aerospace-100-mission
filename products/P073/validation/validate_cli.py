"""Every command quoted in README.md, run, with its real output captured.

The build guide makes this the last step before hand-off: every command, number
and output fragment quoted in README.md or VALIDATION.md is re-executed after
the final code change and the quoted text is replaced with what the run actually
printed. Two products of this mission caught their own wrong README numbers this
way, in consecutive batches.

Each command is run as a subprocess with ``PYTHONPATH=src`` and nothing else
from the ambient environment, which is also how the release gate runs them.
"""

from __future__ import annotations

import subprocess
import sys

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

FAILURES: list[str] = []

COMMANDS: tuple[tuple[tuple[str, ...], int, int], ...] = (
    (("--version",), 0, 4),
    (("info",), 0, 40),
    (("backends",), 0, 40),
    (("windows", "--seed", "7", "--horizon-h", "12", "--limit", "8"), 0, 40),
    (("schedule", "--seed", "7", "--horizon-h", "12", "--method", "greedy"), 0, 14),
    (("schedule", "--seed", "7", "--horizon-h", "12", "--method", "exact"), 0, 14),
    (("fairness", "--seed", "7", "--horizon-h", "12", "--method", "greedy"), 0, 40),
    (("fairness", "--seed", "7", "--horizon-h", "12", "--method", "exact"), 0, 40),
    (("contention", "--seed", "7", "--horizon-h", "12", "--limit", "8"), 0, 40),
    (("compare", "--seed", "7", "--horizon-h", "12"), 0, 60),
    (("schedule", "--seed", "7", "--horizon-h", "0.0003"), 2, 10),
)


def run(arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m contactsched <arguments>`` from the repository root."""
    return subprocess.run(
        [sys.executable, "-m", "contactsched", *arguments],
        cwd=ROOT,
        env={
            "PYTHONPATH": "src",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "MPLBACKEND": "Agg",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> None:
    out = Tee("validate_cli_output.txt")
    out.rule("contactsched validate_cli")
    for line in environment_lines():
        out(line)
    out("")
    out("Every command below is run from the repository root with PYTHONPATH=src and")
    out("nothing else inherited from the environment.")

    for arguments, expected_status, limit in COMMANDS:
        printed = " ".join(arguments)
        out.rule(f"python -m contactsched {printed}")
        result = run(arguments)
        out(f"exit status: {result.returncode} (expected {expected_status})")
        if result.returncode != expected_status:
            FAILURES.append(
                f"`{printed}` exited {result.returncode}, expected {expected_status}"
            )
        out("")
        lines = (result.stdout + result.stderr).splitlines()
        for line in lines[:limit]:
            out(f"  {line}")
        if len(lines) > limit:
            out(f"  ... {len(lines) - limit} further lines not shown")
        # The patterns are assembled from pieces so that this committed file does
        # not itself contain an absolute path prefix, which the repository-wide
        # scan in probe_working_directories.py would otherwise report.
        home_patterns = tuple("/" + name + "/" for name in ("home", "Users"))
        for line in lines:
            if any(pattern in line for pattern in home_patterns):
                FAILURES.append(f"`{printed}` printed an absolute home-directory path")
                break

    out.rule("Help text")
    result = run(("--help",))
    out(f"exit status: {result.returncode}")
    for line in result.stdout.splitlines():
        out(f"  {line}")
    if result.returncode != 0:
        FAILURES.append("`--help` did not exit zero")

    out.rule("Result")
    if FAILURES:
        out(f"FAILED {len(FAILURES)} check(s):")
        for failure in FAILURES:
            out(f"  - {failure}")
    else:
        out("Every command exited with its expected status and printed no absolute path.")
    out("")
    out(f"written to {out.save()}")
    if FAILURES:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
