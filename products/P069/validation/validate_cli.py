"""Run every documented command line and record exactly what it printed.

    python validation/validate_cli.py > validation/validate_cli_output.txt

Every command quoted in README.md appears here, so that a README number and
the program's behaviour cannot drift apart without this file changing.
"""

from __future__ import annotations

import subprocess
import sys

from _common import ROOT, rule

COMMANDS = (
    ["report", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json", "--classes", "fixtures/classes.json"],
    ["uncovered", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json"],
    ["uncovered", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json", "--strength", "3"],
    ["occupancy", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json", "--blocks", "3"],
    ["propose", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json", "--pool", "fixtures/candidate_pool.json",
     "--count", "3", "--bootstrap", "16"],
    ["benchmark", "--space", "fixtures/space.json", "--campaign",
     "fixtures/campaign_executed.json", "--pool", "fixtures/candidate_pool.json",
     "--start", "20", "--steps", "20", "--pool-size", "200", "--seeds", "12",
     "--with-reference"],
)
HEAD_LINES = 14


def main() -> int:
    """Run each command and print its exit status and output."""
    out = sys.stdout
    failures = 0
    env = {"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "HOME": "/tmp"}
    out.write("command line, every documented invocation\n")
    out.write("=" * 78 + "\n")
    for args in COMMANDS:
        argv = [sys.executable, "-m", "scenariocov", *args]
        result = subprocess.run(
            argv, cwd=ROOT, capture_output=True, text=True, check=False, env=env
        )
        printed = " ".join(["python", "-m", "scenariocov", *args])
        out.write(rule(printed) + "\n")
        out.write(f"exit status: {result.returncode}\n")
        if result.returncode != 0:
            failures += 1
            out.write(result.stderr)
        lines = result.stdout.rstrip("\n").split("\n")
        for line in lines[:HEAD_LINES]:
            out.write(line + "\n")
        if len(lines) > HEAD_LINES:
            out.write(f"... {len(lines) - HEAD_LINES} further lines\n")
    out.write(f"\nfailures: {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
