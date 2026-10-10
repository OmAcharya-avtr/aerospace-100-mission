"""Re-run every command quoted in README.md, MODEL_CARD.md and VALIDATION.md.

A previous batch shipped a README output block for a command nobody had run.
This script exists so that cannot happen silently again: every CLI invocation
quoted anywhere in this repository appears in COMMANDS below, is executed in a
clean subprocess, and its verbatim output and exit status are printed into
``validate_cli_output.txt``.

The two commands that must exit non-zero are asserted on their return code,
not on their text. Exits 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
TIMEOUT = 580

# (arguments, expected exit status)
COMMANDS: list[tuple[list[str], int]] = [
    (["--help"], 0),
    (["--version"], 0),
    (["info"], 0),
    (["bound", "--n", "500", "--alpha", "0.1"], 0),
    (["bound", "--n", "9", "--alpha", "0.1"], 0),
    (["bound", "--n", "5", "--alpha", "0.1"], 2),
    (["baseline", "--severity", "2"], 0),
    (
        [
            "audit",
            "--replicates",
            "20",
            "--severities",
            "0",
            "1",
            "2",
            "3",
            "--models",
            "physics",
            "--methods",
            "parametric",
            "split",
            "weighted_declared",
        ],
        0,
    ),
    (
        [
            "audit",
            "--replicates",
            "20",
            "--severities",
            "3",
            "--models",
            "learned",
            "--methods",
            "split",
            "--fail-on-undercoverage",
        ],
        2,
    ),
    (["breaking-point", "--replicates", "20", "--model", "physics"], 0),
    (["strata", "--replicates", "10", "--severity", "2", "--model", "physics"], 0),
]

CHECKS: list[tuple[str, bool]] = []


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    for args, expected in COMMANDS:
        printable = " ".join(args)
        print(f"$ python -m conformalband {printable}")
        result = subprocess.run(
            [sys.executable, "-m", "conformalband", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
        )
        for row in result.stdout.rstrip().splitlines():
            print(f"  {row}")
        if result.stderr.strip():
            for row in result.stderr.rstrip().splitlines():
                print(f"  [stderr] {row}")
        print(f"  -> exit status {result.returncode} (expected {expected})")
        CHECKS.append((f"python -m conformalband {printable}", result.returncode == expected))
        print()
    failed = [label for label, ok in CHECKS if not ok]
    print(f"summary: {len(CHECKS) - len(failed)} of {len(CHECKS)} commands exited as expected")
    if failed:
        print("COMMANDS WITH AN UNEXPECTED EXIT STATUS:")
        for label in failed:
            print(f"  {label}")
    assert not failed, failed
    print("EVERY QUOTED COMMAND RAN AND EXITED AS DOCUMENTED")


if __name__ == "__main__":
    main()
