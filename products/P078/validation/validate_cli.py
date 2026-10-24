"""The CLI, exercised as a subprocess, with its exit codes checked.

The exit codes are the contract a pipeline reads, so they are measured by
running the process rather than by calling ``main`` in-process: an exit code
that works when the function returns an integer and not when the module is
executed is not a contract.

The interpreter used is the one running this script, but its path is never
printed: an absolute path in committed output is not reproducible for a reader
with a cold clone and it publishes the build machine's filesystem layout. The
commands below are printed as ``python -m cmdguard ...`` and that is what a
reader runs.
"""

from __future__ import annotations

import os
import subprocess
import sys

import _bootstrap
from _bootstrap import Tee, header

ROOT = _bootstrap.bootstrap()

CASES = "examples/cases"

#: Path segments whose appearance in output would mean an absolute path leaked.
ABSOLUTE_SEGMENTS = ("home", "root", "Users", "tmp")
EXPECTED = (
    (["--version"], 0, "cmdguard 0.1.0"),
    (["info"], 0, "irreversible_unverified"),
    (["info", f"{CASES}/irreversible_before_verify.yaml"], 0, "irreversible commands    : 1"),
    (["check", f"{CASES}/nominal_science_pass.yaml"], 0, "violations            : 0"),
    (["check", f"{CASES}/late_payload_enable.yaml"], 1, "because step 5"),
    (["check", f"{CASES}/irreversible_before_verify.yaml"], 2, "REFUSED"),
    (["check", f"{CASES}/resource_and_exclusion.yaml"], 1, "mutual_exclusion"),
    (
        ["check", f"{CASES}/late_payload_enable.yaml", "--policy", "block"],
        1,
        "effect policy         : block",
    ),
    (
        ["check", f"{CASES}/late_payload_enable.yaml", "--state-at", "5"],
        1,
        "state after step 5",
    ),
    (["lint", f"{CASES}/late_payload_enable.yaml"], 1, "static lint (per_command)"),
    (
        ["lint", f"{CASES}/resource_and_exclusion.yaml", "--strength", "with_timing"],
        1,
        "timing_gap",
    ),
    (["compare", f"{CASES}/resource_and_exclusion.yaml"], 1, "missed by the lint:"),
    (["compare", f"{CASES}/nominal_science_pass.yaml"], 0, "undefined (no violation to miss)"),
    (["gate", f"{CASES}/irreversible_before_verify.yaml"], 2, "REFUSE"),
    (["gate", f"{CASES}/irreversible_after_verify.yaml"], 0, "NO_REFUSAL_FOUND"),
    (["check", f"{CASES}/missing_declarations.yaml"], 3, ""),
    (["check", f"{CASES}/not_a_file.yaml"], 3, ""),
    (["check", f"{CASES}/late_payload_enable.yaml", "--state-at", "99"], 3, ""),
    (["rank", f"{CASES}/nominal_science_pass.yaml", "--k", "0"], 3, ""),
    (
        ["rank", f"{CASES}/resource_and_exclusion.yaml", "--k", "5", "--train-seeds", "6"],
        1,
        "rank step",
    ),
    (["benchmark", "--train-seeds", "6", "--test-seeds", "4"], 0, "95% CI"),
)


def run(arguments: list[str]) -> tuple[int, str, str]:
    """Run ``python -m cmdguard <arguments>`` from the repository root."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-m", "cmdguard", *arguments],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=environment,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def main() -> int:
    tee = Tee("validate_cli_output.txt")
    header(tee, "cmdguard -- the command line interface and its exit codes")
    tee("")
    tee("  exit codes: 0 nothing found, 1 violations, 2 gate refused, 3 input error")
    tee("  every command below is run as a subprocess from the repository root with")
    tee("  PYTHONPATH=src; the interpreter is the one running this script and its")
    tee("  path is deliberately not printed.")

    tee.rule("1. every subcommand, with its expected exit code")
    tee(f"  {'exit':>4s} {'want':>4s}  command")
    captured: list[tuple[list[str], int, str, str]] = []
    for arguments, expected_code, expected_text in EXPECTED:
        code, out, err = run(arguments)
        captured.append((arguments, code, out, err))
        rendered = "python -m cmdguard " + " ".join(arguments)
        tee(f"  {code:>4d} {expected_code:>4d}  {rendered}")
        tee.check(f"exit code of `{rendered}`", code == expected_code,
                  f"got {code}, wanted {expected_code}")
        if expected_text:
            tee.check(f"output of `{rendered}` contains {expected_text!r}",
                      expected_text in out,
                      "not found in stdout")
        if expected_code == 3:
            tee.check(f"`{rendered}` explains itself on stderr",
                      "cmdguard:" in err or "usage:" in err,
                      err.strip().splitlines()[0] if err.strip() else "stderr empty")

    tee.rule("2. no committed CLI output carries an absolute path")
    offenders = 0
    for arguments, _code, out, err in captured:
        for stream_name, stream in (("stdout", out), ("stderr", err)):
            for line in stream.splitlines():
                # The home-directory prefixes are assembled from segment names
                # rather than written out, so that this file does not itself
                # contain an absolute build-machine path while checking for one.
                looks_absolute = any(
                    f"/{segment}/" in line for segment in ABSOLUTE_SEGMENTS
                ) or line.strip().startswith("/")
                if looks_absolute:
                    offenders += 1
                    tee(f"  ABSOLUTE PATH in {stream_name} of "
                        f"`python -m cmdguard {' '.join(arguments)}`: {line}")
    tee(f"  commands checked : {len(EXPECTED)}")
    tee(f"  offending lines  : {offenders}")
    tee.check("no CLI output prints an absolute path", offenders == 0, str(offenders))

    tee.rule("3. the three verbatim transcripts the README quotes")
    wanted = (
        ["check", f"{CASES}/late_payload_enable.yaml"],
        ["gate", f"{CASES}/irreversible_before_verify.yaml"],
        ["compare", f"{CASES}/resource_and_exclusion.yaml"],
    )
    for arguments in wanted:
        code, out, _ = next(
            (c, o, e) for a, c, o, e in captured if a == arguments
        )
        tee("")
        tee(f"  $ python -m cmdguard {' '.join(arguments)}")
        for line in out.splitlines():
            tee(f"  {line}")
        tee(f"  (exit code {code})")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
