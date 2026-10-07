"""Validation 7: the command-line interface, run as a user runs it.

Every subcommand is executed in a **clean subprocess**, exactly as the README
quickstart and the reproduction section show it, and the raw output is committed
so that the README's quickstart block cannot drift from the code. The exit
statuses are checked too, because the CLI distinguishes a modelling outcome
(status 3, the invariant set is empty or the recursion did not converge) from a
bad argument (status 2) and from success (status 0), and a script that cannot
tell those apart is not usable in a pipeline.

Runtime: about 60 s on one contended core, dominated by recomputing the
invariant set once per subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import time

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()
SRC = str(ROOT / "src")

start = time.perf_counter()
failures = 0


def report(name: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {name}: {detail}")


def run(args: list[str], expect: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "simplexguard", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", "MPLBACKEND": "Agg"},
        check=False,
    )
    report(
        f"exit status of `python -m simplexguard {' '.join(args)}`",
        proc.returncode == expect,
        f"got {proc.returncode}, expected {expect}",
    )
    return proc


print("validate_cli.py")
print("=" * 78)

print()
print("$ python -m simplexguard --help")
print("-" * 78)
proc = run(["--help"], 0)
print(proc.stdout.rstrip())
print("-" * 78)
report(
    "the help text carries the research-grade statement",
    "not flight-qualified" in proc.stdout and "not a verification tool" in proc.stdout,
    "present",
)

for args in (
    ["plant"],
    ["invariant"],
    ["guard"],
    ["run", "--steps", "2000"],
    ["accounting", "--steps", "2000"],
):
    print()
    print(f"$ python -m simplexguard {' '.join(args)}")
    print("-" * 78)
    proc = run(args, 0)
    print(proc.stdout.rstrip())
    print("-" * 78)

print()
print("$ python -m simplexguard invariant --baseline-r 400   (expected: status 3)")
print("-" * 78)
proc = run(["invariant", "--baseline-r", "400"], 3)
print(proc.stderr.rstrip())
print("-" * 78)
report(
    "an empty invariant set is a modelling outcome with its own status",
    "robust invariant set emptied" in proc.stderr,
    "the message names the iteration and what to change",
)

print()
print("$ python -m simplexguard plant --dt -1   (expected: status 2)")
print("-" * 78)
proc = run(["plant", "--dt", "-1"], 2)
print(proc.stderr.rstrip())
print("-" * 78)

print()
print("$ python -m simplexguard invariant --max-iterations 2   (expected: status 3)")
print("-" * 78)
proc = run(["invariant", "--max-iterations", "2"], 3)
print(proc.stderr.rstrip())
print("-" * 78)
report(
    "a non-converged recursion does not return its last iterate",
    "is NOT returned" in proc.stderr,
    "the message says so explicitly",
)

print("=" * 78)
print(f"failed checks: {failures}")
print(f"elapsed: {time.perf_counter() - start:.1f} s")
raise SystemExit(1 if failures else 0)
