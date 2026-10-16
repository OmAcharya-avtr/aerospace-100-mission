"""The documented exit-code contract, measured with real subprocesses.

Exit codes are the part of this package a continuous-integration job depends
on, so they are measured by running the program rather than by calling
``main`` in process.
"""

from __future__ import annotations

import os
import subprocess
import sys

from _support import ROOT, SPECS, Emitter

MINIMAL = SPECS / "minimal_consistent.yaml"
PLANTED = SPECS / "planted_defects.yaml"
WORKED = SPECS / "smallsat_fc.yaml"

# (expected exit code, arguments, what the case demonstrates)
CASES: list[tuple[int, list[str], str]] = [
    (0, ["check", str(MINIMAL)], "consistent specification"),
    (0, ["check", str(MINIMAL), "--strict"], "consistent specification, strict"),
    (0, ["check", str(MINIMAL), "--format", "json"], "consistent, json"),
    (0, ["check", str(MINIMAL), "--format", "markdown"], "consistent, markdown"),
    (1, ["check", str(PLANTED)], "planted fixture, 7 errors"),
    (1, ["check", str(PLANTED), "--strict"], "planted fixture, strict"),
    (1, ["check", str(PLANTED), "--format", "json"], "planted fixture, json"),
    (1, ["check", str(WORKED)], "worked example, 1 error"),
    (1, ["check", str(WORKED), "--strict"], "worked example, strict"),
    (0, ["check", str(WORKED), "--tol-ms", "1000"], "tolerance swallows the overrun"),
    (1, ["check", str(WORKED), "--tol-ms", "1e-9"], "default tolerance does not"),
    (0, ["latency", str(WORKED)], "timing table only"),
    (0, ["latency", str(PLANTED)], "timing table of the planted fixture"),
    (0, ["mermaid", str(WORKED)], "mermaid flowchart"),
    (0, ["codes"], "check catalogue"),
    (0, ["simulate", str(MINIMAL), "--seeds", "2", "--trials", "200"], "simulation agrees"),
    (
        0,
        ["simulate", str(WORKED), "--task", "TASK-FDIR", "--seeds", "2", "--trials", "200"],
        "simulation of one task",
    ),
    (2, ["check", "does-not-exist.yaml"], "missing file"),
    (2, ["check", str(ROOT)], "a directory, not a file"),
    (2, ["simulate", str(MINIMAL), "--task", "absent"], "unknown task"),
    (2, ["simulate", str(MINIMAL), "--seeds", "0"], "zero seeds"),
    (2, [], "no subcommand"),
    (2, ["nonsense"], "unknown subcommand"),
    (2, ["check", str(MINIMAL), "--format", "latex"], "unknown format"),
    (0, ["check", "--help"], "help for check"),
    (0, ["--help"], "help"),
]


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m watchdogspec`` with ``src`` on the path."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "watchdogspec", *args],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        check=False,
    )


def main() -> int:
    """Measure the exit-code contract. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("watchdogspec validation: command line exit-code contract")
    emit()
    emit("  exit 0  internally consistent, or an informational subcommand")
    emit("  exit 1  at least one error finding (or any finding with --strict)")
    emit("  exit 2  the specification could not be read or parsed, or bad arguments")
    emit()
    emit(f"  {'expected':>8} {'actual':>7}  case")
    mismatches = 0
    for expected, args, label in CASES:
        result = run(args)
        ok = result.returncode == expected
        mismatches += 0 if ok else 1
        emit(f"  {expected:>8} {result.returncode:>7}  {label}{'' if ok else '   MISMATCH'}")
    emit()
    emit(f"  subprocesses run       : {len(CASES)}")
    emit(f"  exit-code mismatches   : {mismatches}")
    emit.check("every exit code matched the documented contract", mismatches == 0)

    emit()
    emit("  quoted output fragments, re-run here so the README cannot drift")
    minimal_tail = run(["check", str(MINIMAL)]).stdout.rstrip("\n").splitlines()[-6:]
    for line in minimal_tail:
        emit(f"    | {line}")
    emit.check(
        "the consistent specification prints a CONSISTENT verdict",
        any("CONSISTENT" in line for line in minimal_tail),
    )
    worked = run(["check", str(WORKED)]).stdout
    emit()
    for line in worked.splitlines():
        if line.strip().startswith(("WS003", "WS005")):
            emit(f"    | {line}")
    emit.check("the worked example prints WS003 and WS005", "WS003" in worked and "WS005" in worked)

    emit()
    emit("  the simulate subcommand over the worked example")
    simulated = run(
        ["simulate", str(WORKED), "--seeds", "4", "--trials", "500"]
    )
    for line in simulated.stdout.splitlines():
        emit(f"    | {line}")
    emit.check("the simulate subcommand exits 0", simulated.returncode == 0)
    emit.check(
        "no bound violation was reported",
        simulated.stdout.count("bound violations        : 0") == 4,
    )

    emit()
    emit("  gitignore check on every file this product is supposed to ship")
    tracked = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and ".hypothesis" not in path.parts
        and ".ruff_cache" not in path.parts
        and not path.name.endswith(".pyc")
    )
    emit(f"    files that must be committed : {len(tracked)}")
    ignored: list[str] = []
    for path in tracked:
        probe = subprocess.run(
            ["git", "check-ignore", "-v", str(path)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        if probe.returncode == 0:
            ignored.append(f"{path.relative_to(ROOT)} <- {probe.stdout.strip()}")
    for line in ignored:
        emit(f"    IGNORED: {line}")
    emit.check(
        "no file this product ships is excluded by a gitignore pattern",
        not ignored,
        f"{len(ignored)} ignored",
    )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
