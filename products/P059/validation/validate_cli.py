"""Run the CLI in a subprocess and record exactly what it prints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")

COMMANDS = [
    ["--help"],
    ["--version"],
    ["systems"],
    ["invariant", "--system", "scalar_invariant_equals_x"],
    ["invariant", "--system", "scalar_empty"],
    ["invariant", "--system", "nilpotent_2d"],
    ["invariant", "--system", "attitude_loop"],
    ["invariant", "--system", "slow_pair", "--max-iter", "6", "--no-geometry"],
    ["algebra"],
    [
        "tolerance-sweep",
        "--system",
        "attitude_loop",
        "--tolerances",
        "1e-9",
        "3e-3",
        "4e-3",
        "--max-iter",
        "40",
    ],
]


def main() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    for args in COMMANDS:
        printable = " ".join(args)
        print(f"$ python -m invariantset {printable}")
        res = subprocess.run(
            [sys.executable, "-m", "invariantset", *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        for row in res.stdout.rstrip().splitlines():
            print(f"  {row}")
        if res.stderr.strip():
            for row in res.stderr.rstrip().splitlines():
                print(f"  [stderr] {row}")
        print(f"  -> exit status {res.returncode}")
        print()


if __name__ == "__main__":
    main()
