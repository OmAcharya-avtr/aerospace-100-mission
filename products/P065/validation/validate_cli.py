"""Exit-code contract, measured by running real subprocesses.

The CLI contract is: 0 when the command succeeded and no blocking finding was
raised, 1 when at least one blocking finding was raised, 2 when the command
could not run. A tool whose whole purpose is to fail a build has to be held to
that contract by execution, not by reading the source.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from _support import ROOT, Emitter

CLEAN = "examples/cases/complete_minimal.yaml"
WORKED = "examples/cases/p024_bdot_detumble.yaml"
PLANTED = "examples/cases/planted_gaps.yaml"
P024_REQ = "fixtures/p024_bdot/docs/REQUIREMENTS.md"
P024_JUNIT = "fixtures/p024_bdot/junit.xml"
PLANTED_REQ = "fixtures/planted/docs/REQUIREMENTS.md"
PLANTED_JUNIT = "fixtures/planted/junit.xml"


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m hazardtrace`` from the repository root."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "-m", "hazardtrace", *argv],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


def main() -> int:
    """Run every documented invocation. Returns a process exit code."""
    emit = Emitter(__file__)
    emit("hazardtrace validation: CLI exit-code contract (real subprocesses)")
    emit()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        warning_only = tmp_path / "warning_only.yaml"
        warning_only.write_text(
            "meta:\n  title: warning only\nlosses:\n  - id: L-1\n    description: d\n",
            encoding="utf-8",
        )
        malformed = tmp_path / "malformed.yaml"
        malformed.write_text("meta:\n  title: t\nlosses:\n  - id: L-1\n", encoding="utf-8")

        cases: list[tuple[list[str], int, str]] = [
            (["--help"], 0, "help"),
            (["--version"], 0, "version"),
            ([], 2, "no subcommand is an argparse usage error"),
            (["codes"], 0, "the check catalogue"),
            (["check", CLEAN], 0, "a clean case"),
            (["check", CLEAN, "--strict"], 0, "a clean case, strict"),
            (["check", CLEAN, "-f", "json"], 0, "a clean case as JSON"),
            (["check", CLEAN, "-f", "markdown"], 0, "a clean case as markdown"),
            (["check", PLANTED], 1, "the planted case without cross-links"),
            (
                ["check", PLANTED, "-r", PLANTED_REQ, "-t", PLANTED_JUNIT],
                1,
                "the planted case with cross-links",
            ),
            (
                ["check", PLANTED, "-r", PLANTED_REQ, "-t", PLANTED_JUNIT, "-f", "json"],
                1,
                "the planted case as JSON",
            ),
            (["check", WORKED], 1, "the worked example without cross-links"),
            (
                ["check", WORKED, "-r", P024_REQ, "-t", P024_JUNIT],
                1,
                "the worked example with cross-links",
            ),
            ([
                "check", WORKED, "-r", P024_REQ, "-t", P024_JUNIT, "--strict",
            ], 1, "the worked example, strict"),
            (["check", str(warning_only)], 0, "a warning-only case is not blocking"),
            (
                ["check", str(warning_only), "--strict"],
                1,
                "a warning-only case blocks under --strict",
            ),
            (["grid", PLANTED], 0, "grid always exits 0"),
            (["grid", WORKED], 0, "grid on the worked example"),
            (["coverage", WORKED, "-r", P024_REQ], 0, "coverage always exits 0"),
            (
                ["diagram", WORKED, "--mermaid", str(tmp_path / "s.mmd")],
                0,
                "diagram to Mermaid",
            ),
            (
                ["diagram", WORKED, "--png", str(tmp_path / "s.png")],
                0,
                "diagram to PNG",
            ),
            (["diagram", WORKED, "--mermaid", "-"], 0, "diagram to stdout"),
            (["diagram", WORKED], 2, "diagram with no output cannot run"),
            (["check", "examples/cases/absent.yaml"], 2, "a missing analysis file"),
            (["check", str(malformed)], 2, "a malformed analysis file"),
            (["check", CLEAN, "-r", "docs/absent.md"], 2, "a missing requirements file"),
            (["check", CLEAN, "-t", "absent.xml"], 2, "a missing junit report"),
            (
                ["check", CLEAN, "-r", P024_REQ, "--requirement-pattern", "REQ-("],
                2,
                "an invalid requirement pattern",
            ),
        ]

        emit(f"   {'invocation':<66}{'want':<6}{'got':<6}verdict")
        wrong = 0
        for argv, expected, _label in cases:
            result = run(argv)
            mark = "ok" if result.returncode == expected else "MISMATCH"
            if result.returncode != expected:
                wrong += 1
            shown = (" ".join(argv) or "(none)").replace(str(tmp_path), "<tmp>")
            if len(shown) > 64:
                shown = shown[:63] + "…"
            emit(f"   {shown:<66}{expected:<6}{result.returncode:<6}{mark}")
        emit()
        emit(f"   invocations run         : {len(cases)}")
        emit(f"   exit code mismatches    : {wrong}")
        emit()
        emit.check(
            "every invocation matched its documented exit code",
            wrong == 0,
            f"{len(cases) - wrong}/{len(cases)} matched",
        )

        # -- content assertions on the two reports ------------------------
        planted = run(["check", PLANTED, "-r", PLANTED_REQ, "-t", PLANTED_JUNIT])
        worked = run(["check", WORKED, "-r", P024_REQ, "-t", P024_JUNIT])
        emit()
        emit("   content of the two reports")
        emit(f"     planted verdict line  : {planted.stdout.splitlines()[-4].strip()}")
        emit(f"     worked verdict line   : {worked.stdout.splitlines()[-4].strip()}")
        missing = [
            f"HT{n:03d}" for n in range(1, 17) if f"HT{n:03d}" not in planted.stdout
        ]
        emit.check(
            "every check code appears in the planted report",
            not missing,
            ",".join(missing) if missing else "all 16 present",
        )
        emit.check(
            "the planted report says 14 error and 2 warning",
            "14 error, 2 warning" in planted.stdout,
        )
        emit.check(
            "the worked report says 11 error and 0 warning",
            "11 error, 0 warning" in worked.stdout,
        )
        emit.check(
            "a cannot-run failure writes to stderr and not stdout",
            run(["check", "examples/cases/absent.yaml"]).stderr.startswith("hazardtrace:"),
        )
    return emit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
