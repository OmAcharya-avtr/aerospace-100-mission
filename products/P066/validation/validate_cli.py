"""The CLI exit-status contract, measured with real subprocesses.

    0  the command ran and raised no finding of severity error
    1  the command ran and raised at least one error finding
    2  the command could not run

Nothing here is mocked: every row below is a ``python -m budgetroll``
process that was started, waited for and had its status read.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _support import CASES, ROOT, Emitter

ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": str(ROOT)}


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "budgetroll", *args],
        capture_output=True,
        text=True,
        env=ENV,
        cwd=ROOT,
        timeout=180,
    )


def main() -> int:
    """Run every documented invocation and compare the status with the contract."""
    out = Emitter(__file__)
    out("budgetroll validation: CLI exit-status contract")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use.")
    out()

    cases = []
    for command in ("check", "rollup", "sensitivity", "compare"):
        cases.append(([command, str(CASES / "clean_mass.yaml")], 0))
        cases.append(([command, str(CASES / "cubesat_link.yaml")], 0))
        cases.append(([command, str(CASES / "cubesat_power.yaml")], 1))
        cases.append(([command, str(CASES / "planted_findings.yaml")], 1))
    for fmt in ("text", "json", "markdown", "full"):
        cases.append((["rollup", "--format", fmt, str(CASES / "clean_mass.yaml")], 0))
        cases.append((["rollup", "--format", fmt, str(CASES / "planted_findings.yaml")], 1))
    cases.append((["codes"], 0))
    cases.append((["--help"], 0))
    cases.append(([], 2))
    cases.append((["nosuchcommand", str(CASES / "clean_mass.yaml")], 2))
    cases.append((["check", str(CASES / "absent.yaml")], 2))

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.yaml"
        bad.write_text("policies: [\n", encoding="utf-8")
        cases.append((["check", str(bad)], 2))
        empty = Path(tmp) / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        cases.append((["check", str(empty)], 2))
        unknown = Path(tmp) / "unknown.yaml"
        unknown.write_text("budget: {}\npolicies: {}\nsurprise: 1\n", encoding="utf-8")
        cases.append((["check", str(unknown)], 2))

        out(f"1. {len(cases)} real subprocesses against the documented contract")
        mismatches = 0
        for args, expected in cases:
            result = run(*args)
            shown = " ".join(
                a.replace(str(ROOT) + "/", "").replace(tmp + "/", "<tmp>/") for a in args
            ) or "(no arguments)"
            agree = result.returncode == expected
            mismatches += 0 if agree else 1
            out(f"   [{'OK ' if agree else 'BAD'}] exit {result.returncode} "
                f"(expected {expected})  python -m budgetroll {shown}")
        out()
        out.check(
            "every invocation matched the documented exit status",
            mismatches == 0,
            f"{mismatches} mismatches out of {len(cases)}",
        )

    out()
    out("2. The scope statement appears in the output of every reporting command")
    for command in ("check", "rollup", "sensitivity", "compare"):
        result = run(command, str(CASES / "clean_mass.yaml"))
        out.check(
            f"'{command}' prints the not-flight-qualified statement",
            "not flight-qualified" in result.stdout,
        )
    result = run("codes")
    out.check(
        "'codes' prints the not-flight-qualified statement",
        "not flight-qualified" in result.stdout,
    )

    out()
    out("3. Errors go to stderr and carry the offending key path")
    result = run("check", str(CASES / "absent.yaml"))
    out.check("a missing file writes to stderr", bool(result.stderr.strip()))
    out.check(
        "the message names the missing file",
        "no such budget declaration" in result.stderr
        and result.stderr.strip().endswith("absent.yaml"),
        result.stderr.strip().replace(str(ROOT) + "/", ""),
    )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
