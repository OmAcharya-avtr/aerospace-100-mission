"""Exercise every CLI path and record the exit codes.

Each command is run in a subprocess exactly as a reader would run it, and the
transcript below is what the command printed. Exit codes: 0 success, 1 a
finding with --fail-on-findings, 2 a refused request.

Run:
    PYTHONPATH=src python validation/validate_cli.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMANDS: list[tuple[list[str], int]] = [
    (["--version"], 0),
    (["info"], 0),
    (["info", "--json"], 0),
    (["check"], 0),
    (["check", "--fail-on-findings"], 1),
    (["check", "--spec", "planted_unreachable"], 0),
    (["check", "--spec", "planted_unreachable", "--fail-on-findings"], 1),
    (["check", "--latency-budget", "60"], 0),
    (["check", "--assume-perfect-detection"], 0),
    (["reach"], 0),
    (["reach", "--spec", "planted_unreachable", "--fail-on-findings"], 1),
    (["reach", "--json", "--latency-budget", "20"], 0),
    (["ambiguity"], 0),
    (["ambiguity", "--json"], 0),
    (["ambiguity", "--spec", "planted_unreachable"], 0),
    (["benchmark", "--noise-scale", "2.0", "--n-per-fault", "120"], 0),
    (["benchmark", "--noise-scale", "2.0", "--n-per-fault", "120", "--json"], 0),
    (["policy", "--noise-scale", "2.0", "--n-per-fault", "120"], 0),
    (["policy", "--noise-scale", "2.0", "--n-per-fault", "120", "--mode",
      "coarse_pointing", "--detector", "baseline"], 0),
    (["policy", "--noise-scale", "2.0", "--n-per-fault", "120", "--sweep"], 0),
    (["check", "--spec", "no_such_declaration"], 2),
    (["reach", "--spec", "/nonexistent/path.yaml"], 2),
    (["policy", "--mode", "nowhere", "--n-per-fault", "60"], 2),
]

CHECKS: list[tuple[str, bool]] = []


def record(label: str, ok: bool) -> None:
    """Print and record one check."""
    CHECKS.append((label, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")


def main() -> None:
    """Run every command, print its transcript and check its exit code."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MPLBACKEND"] = "Agg"
    print("safemodetree CLI transcript")
    print("=" * 84)
    print(f"interpreter: {sys.executable}")
    print(f"working directory: {ROOT}")
    print()
    for argv, expected in COMMANDS:
        printable = "python -m safemodetree " + " ".join(argv)
        print("$ " + printable)
        completed = subprocess.run(
            [sys.executable, "-m", "safemodetree", *argv],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        body = completed.stdout.rstrip()
        if len(body.splitlines()) > 24:
            lines = body.splitlines()
            body = "\n".join([*lines[:12], f"... {len(lines) - 20} lines elided ...",
                              *lines[-8:]])
        if body:
            print(body)
        if completed.stderr.strip():
            print("stderr: " + completed.stderr.strip())
        print(f"exit {completed.returncode} (expected {expected})")
        record(f"{printable} exits {expected}", completed.returncode == expected)
        print()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} commands behaved as declared")
    if failed:
        print("FAILED CHECKS (recorded, not suppressed):")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("CLI TRANSCRIPT COMPLETE")


if __name__ == "__main__":
    main()
