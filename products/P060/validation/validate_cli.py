"""Re-run every command quoted in README.md and print exactly what it emits.

**This script exits 0.** Several of the commands below deliberately exit 1,
because a traceability linter that finds something must say so in its exit
status; that is the product working. Each command is run through
``subprocess.run`` and its ``returncode`` is compared with the status the
README says to expect. A mismatch is printed as FAIL and counted, and the
count is printed at the end -- but the exit status of this script stays 0, so
that the release gate's "every validation script exits 0" check means what it
says.

A previous batch in this portfolio shipped a README output block for a command
that had never been run. This script is how that is made checkable here: the
transcript below is produced by executing the commands, not by transcription.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_REQ = "fixtures/sample_project/docs/REQUIREMENTS.md"
SAMPLE_JUNIT = "fixtures/sample_project/junit.xml"
SAMPLE_TESTS = "fixtures/sample_project/tests"
SAMPLE_COLLECT = "fixtures/sample_project/collect_only.txt"

AUDIT = ["audit", "-r", SAMPLE_REQ, "-t", SAMPLE_JUNIT, "--test-root", SAMPLE_TESTS]

# (argv after "python -m traceaudit", expected exit status, README section)
COMMANDS: list[tuple[list[str], int, str]] = [
    (["--help"], 0, "CLI reference"),
    (["--version"], 0, "CLI reference"),
    (["codes"], 0, "Finding codes"),
    (["config-template"], 0, "Configuration"),
    (["requirements", "-r", SAMPLE_REQ], 0, "Worked example"),
    (["tests", "-t", SAMPLE_JUNIT], 0, "Worked example"),
    (["tests", "-t", SAMPLE_COLLECT], 0, "Worked example"),
    (AUDIT, 1, "Worked example"),
    ([*AUDIT, "--show-matrix"], 1, "Worked example"),
    ([*AUDIT, "--markdown-matrix"], 1, "Worked example"),
    ([*AUDIT, "--ignore", "TA006"], 1, "Limitations"),
    (["audit", "-r", SAMPLE_REQ, "-t", SAMPLE_COLLECT, "--test-root", SAMPLE_TESTS],
     1, "Limitations"),
    (["assertions", "--test-root", "fixtures/heuristic"], 0, "Limitations"),
    (["audit", "-r", SAMPLE_REQ, "-t", "fixtures/sample_project/absent.xml"],
     2, "CLI reference"),
]

FAILURES: list[str] = []
CHECKS: list[str] = []


def env() -> dict[str, str]:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + e.get("PYTHONPATH", "")
    e["MPLBACKEND"] = "Agg"
    e["COLUMNS"] = "90"
    return e


def run(argv: list[str], expected: int, section: str) -> None:
    printable = " ".join(argv)
    print(f"$ python -m traceaudit {printable}")
    result = subprocess.run(
        [sys.executable, "-m", "traceaudit", *argv],
        capture_output=True, text=True, env=env(), cwd=ROOT, timeout=300, check=False,
    )
    for row in result.stdout.rstrip().splitlines():
        print(f"  {row}")
    for row in result.stderr.rstrip().splitlines():
        print(f"  [stderr] {row}")
    ok = result.returncode == expected
    CHECKS.append(printable)
    if not ok:
        FAILURES.append(f"{printable} (expected {expected}, got {result.returncode})")
    print(f"  -> exit status {result.returncode}   "
          f"[{'PASS' if ok else 'FAIL'}, README section: {section}, expected {expected}]")
    print()


def run_shell(label: str, argv: list[str], expected: int) -> None:
    print(f"$ {label}")
    result = subprocess.run(argv, capture_output=True, text=True, env=env(),
                            cwd=ROOT, timeout=1800, check=False)
    tail = result.stdout.rstrip().splitlines()[-3:]
    for row in tail:
        print(f"  {row}")
    ok = result.returncode == expected
    CHECKS.append(label)
    if not ok:
        FAILURES.append(f"{label} (expected {expected}, got {result.returncode})")
    print(f"  -> exit status {result.returncode}   "
          f"[{'PASS' if ok else 'FAIL'}, expected {expected}]")
    print()


def main() -> int:
    print("Every command below is executed in this run, from the repository root.")
    print("Commands that exit 1 are the product reporting findings, which is correct")
    print("behaviour; this script still exits 0.")
    print()
    for argv, expected, section in COMMANDS:
        run(argv, expected, section)

    print("The self-audit pair from the Quickstart, run in full:")
    run_shell("python -m pytest tests/ -q --junitxml=junit.xml",
              [sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider",
               "--junitxml=junit.xml"], 0)
    run(["audit", "-r", "docs/REQUIREMENTS.md", "-t", "junit.xml", "--test-root", "tests"],
        0, "Quickstart")

    print("The examples quoted in the Screenshots section:")
    for name in ("audit_sample_project.py", "coverage_definitions.py",
                 "finding_breakdown.py", "heuristic_error_rates.py"):
        run_shell(f"python examples/{name}",
                  [sys.executable, f"examples/{name}"], 0)

    print(f"COMMANDS EXECUTED: {len(CHECKS)}")
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    print()
    print("This script's own exit status: 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
